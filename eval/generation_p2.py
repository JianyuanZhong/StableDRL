import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes token-wise entropy for a categorical distribution parameterized by logits.
    Returns a tensor of shape [B, T].
    """
    dist_cat = torch.distributions.Categorical(logits=logits.to(torch.float32))
    return dist_cat.entropy()


def sample_and_score(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens using Gumbel-Max while computing confidence from CLEAN logits.
    Returns (x0, logp), where logp are log-probs of sampled tokens under clean logits.
    """
    clean_logits = logits.to(torch.float32)
    logp_all = F.log_softmax(clean_logits, dim=-1)

    if temperature and temperature > 0.0:
        # Standard Gumbel-Max: argmax((logits / T) + G)
        u = torch.rand_like(clean_logits, dtype=torch.float32)
        g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        noised = clean_logits + g * float(temperature)
        x0 = torch.argmax(noised, dim=-1)
    else:
        x0 = torch.argmax(clean_logits, dim=-1)

    x0_logp = logp_all.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return x0.long(), x0_logp


def stochastic_sample_from_categorical(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Back-compat wrapper; use sample_and_score implementation
    return sample_and_score(logits, temperature)


def kappa_fn(t: float, schedule: str = "linear") -> float:
    """
    Scheduling function κ(t) in [0, 1]. Controls how many tokens to update.
    - linear: κ(t) = t
    - cosine: κ(t) = 0.5 * (1 - cos(pi * t))
    - sqrt: κ(t) = sqrt(t)
    """
    if schedule == "linear":
        return float(t)
    if schedule == "cosine":
        return float(0.5 * (1.0 - np.cos(np.pi * t)))
    if schedule == "sqrt":
        return float(np.sqrt(t))
    # default
    return float(t)


def topk_lowest_masking(score: torch.Tensor, num_to_mask: torch.Tensor) -> torch.Tensor:
    """
    Select positions with the lowest scores to (re)mask/update.
    - score: [B, T], where higher means less likely to mask; positions to exclude should be +inf
    - num_to_mask: [B, 1] or [B], number of tokens to mask per batch item
    Returns a boolean mask [B, T] with True at positions chosen to mask.
    """
    if num_to_mask.dim() == 2 and num_to_mask.size(1) == 1:
        num_to_mask = num_to_mask.squeeze(1)

    batch_size = score.shape[0]
    mask = torch.zeros_like(score, dtype=torch.bool)
    # Replace NaNs if any
    safe_score = torch.nan_to_num(score, nan=float("inf"))

    for b in range(batch_size):
        k = int(num_to_mask[b].item())
        if k <= 0:
            continue
        # Clamp to available positions (exclude +inf which denotes fixed)
        valid = torch.isfinite(safe_score[b])
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            continue
        k = min(k, valid_count)
        # Get indices of k lowest scores among valid positions
        # Set invalid positions to +inf to avoid selection
        row = safe_score[b]
        row = torch.where(valid, row, torch.full_like(row, float("inf")))
        _, idx = torch.topk(row, k=k, largest=False)
        mask[b, idx] = True
    return mask


@torch.no_grad()
def generate_p2(
    model,
    prompt: torch.Tensor,
    tokenizer,
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    planner_model=None,
    planner_mode: str = "self",  # "self" or "bert"
    kappa_schedule_name: str = "linear",
    eta: float = 1.0,
    score_type: str = "confidence",  # "confidence", "random", or "pa"
    pa_lambda: float = 0.0,
    track_trajectory: bool = False,
):
    """
    Path Planning (P2) sampling for Masked Diffusion Models.

    - Self-Planning: Uses the denoiser (model) confidence for both masked and unmasked tokens.
    - BERT-Planning: For tokens that were unmasked in the previous step, uses planner_model's
      confidence to decide remasking; masked tokens still use the denoiser.

    Args match generate.py where possible.
    Returns the completed sequence tensor [B, prompt_len + gen_length].
    """
    device = prompt.device
    # Silence linter for intentionally unused parameters retained for API parity
    _ = tokenizer
    _ = block_length

    with torch.autocast(device_type="cuda"):
        batch_size = prompt.shape[0]
        prompt_len = prompt.shape[1]
        total_len = prompt_len + gen_length

        x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt.clone()

        # Fixed positions: prompt only
        fix_mask = torch.zeros_like(x, dtype=torch.bool)
        fix_mask[:, :prompt_len] = True

        # Setup for Position-Aware (PA) sampler if enabled
        if score_type == "pa":
            if pa_lambda < 0.0:
                raise ValueError("pa_lambda must be >= 0.0")
            # Additive positional penalty (shape [1, total_len])
            positional_penalty = float(pa_lambda) * torch.arange(
                total_len, device=device, dtype=torch.float32
            ).unsqueeze(0)

        # Initialize trajectory accumulators if requested
        if track_trajectory:
            # store counts over batch for each step (rows) and decoding position (cols)
            unmask_counts = torch.zeros((steps, gen_length), dtype=torch.long, device="cpu")
            # cumulative (ever unmasked up to and including this step)
            unmask_cum_counts = torch.zeros((steps, gen_length), dtype=torch.long, device="cpu")
            remask_counts = torch.zeros((steps, gen_length), dtype=torch.long, device="cpu")
            # track which generated positions have EVER been unmasked per example
            ever_unmasked = torch.zeros((batch_size, gen_length), dtype=torch.bool, device=device)

        dt = 1.0 / float(max(1, steps))
        for step_idx in tqdm(range(1, steps + 1), disable=(dist.get_rank() != 0)):
            t = step_idx * dt
            kappa_t = kappa_fn(t, schedule=kappa_schedule_name)

            last_mask = (x == mask_id)

            # Classifier-free guidance
            if cfg_scale > 0.0:
                un_x = x.clone()
                # Unconditional pass: mask prompt only
                un_x[fix_mask] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits_cat = model(x_cat).logits
                logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits = logits.to(torch.float32)
            x0, logp = sample_and_score(logits, temperature)

            # Eligible for remasking: unmasked and not prompt
            unmask_candidates = (~last_mask) & (~fix_mask)

            if planner_mode == "bert" and planner_model is not None:
                planner_logits = planner_model(x0).logits.to(torch.float32)
                planner_logp = F.log_softmax(planner_logits, dim=-1).gather(
                    -1, x0.unsqueeze(-1)
                ).squeeze(-1)
                logp = torch.where(unmask_candidates, planner_logp, logp)

            # Score computation
            if score_type == "pa":
                # Decoupled Scoring:
                # - Planning (for masked tokens): positional awareness to prefer left-to-right
                # - Correction (for unmasked tokens): pure confidence with eta controlling frequency
                score_plan = logp - positional_penalty
                score_correct = logp * float(eta)
                score = torch.where(unmask_candidates, score_correct, score_plan)
            elif score_type == "confidence":
                score = logp
                if eta != 1.0:
                    score = torch.where(unmask_candidates, score * eta, score)
            elif score_type == "random":
                score = torch.log(torch.rand_like(logp))
            else:
                raise ValueError(f"Invalid score_type: {score_type}")

            # Never mask prompt positions
            score = score.masked_fill(fix_mask, float("inf"))

            allowed_counts = (~fix_mask).sum(dim=1, keepdim=True)
            num_to_mask = (allowed_counts.float() * (1.0 - float(kappa_t))).long()

            mask_sel = topk_lowest_masking(score, num_to_mask)

            # Apply masking to selected low-score positions
            x[mask_sel] = mask_id

            # Previously masked but not selected now become x0
            mask_to_x0 = last_mask & (~mask_sel)
            x[mask_to_x0] = x0[mask_to_x0]

            # Record unmask and remask events within the generation span only
            if track_trajectory:
                # positions within generated continuation
                if gen_length > 0:
                    # Unmask events: tokens that transition from mask -> token at this step
                    unmask_events = mask_to_x0[:, prompt_len:]
                    # Update cumulative ever-unmasked state
                    ever_unmasked = ever_unmasked | unmask_events
                    # Remask events: tokens that were visible last step and are masked now
                    remask_events = (mask_sel & (~last_mask) & (~fix_mask))[:, prompt_len:]
                    # Sum over batch dimension to get frequency per position
                    unmask_counts[step_idx - 1] += unmask_events.sum(dim=0).to("cpu")
                    unmask_cum_counts[step_idx - 1] += ever_unmasked.sum(dim=0).to("cpu")
                    remask_counts[step_idx - 1] += remask_events.sum(dim=0).to("cpu")

        # Final fill for any remaining masks
        remaining_mask = (x == mask_id)
        if remaining_mask.any():
            x[remaining_mask] = x0[remaining_mask]

        if track_trajectory:
            traj = {
                "unmask_counts": unmask_counts.tolist(),
                "unmask_cum_counts": unmask_cum_counts.tolist(),
                "remask_counts": remask_counts.tolist(),
            }
            return x, traj
        else:
            return x



@torch.no_grad()
def generate_ars(
    model,
    prompt: torch.Tensor,
    tokenizer=None,
    gen_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    # EB-PA progression parameters
    gamma: float = 0.1,
    pa_lambda: float = 0.1,
    # Adaptive correction parameters
    remask_threshold_R: float = 1.5,
    max_steps: int = 100,
    cfg_scale: float = 0.0,
    track_trajectory: bool = False,
):
    """
    Adaptive Refinement Sampler (ARS) with decoupled scoring and EB-PA progression.

    - Phase 1 (Correction): Identify low-confidence tokens anywhere in the sequence and remask
      using a statistical threshold based on pure confidence (log-prob).
    - Phase 2 (Progression): Among masked tokens, use PA score (logp - positional penalty)
      to order, and EB criterion (gamma) to adaptively choose how many to unmask in parallel.

    Returns either the completed sequence tensor [B, prompt_len + gen_length] or
    (sequence, trajectory) if track_trajectory is True. When tracking, trajectory contains
    step-wise unmask and remask counts over generated positions with fixed length = max_steps.
    """
    device = prompt.device
    _ = tokenizer  # kept for API parity

    batch_size, prompt_len = prompt.shape
    total_len = prompt_len + gen_length

    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()

    fix_mask = torch.zeros_like(x, dtype=torch.bool)
    fix_mask[:, :prompt_len] = True

    # Positional penalty for PA ordering
    positional_penalty = float(pa_lambda) * torch.arange(
        total_len, device=device, dtype=torch.float32
    ).unsqueeze(0)

    # Trajectory trackers
    unmask_counts = None
    unmask_cum_counts = None
    remask_counts = None
    ever_unmasked = None
    if track_trajectory:
        unmask_counts = torch.zeros((max_steps, gen_length), dtype=torch.long, device="cpu")
        unmask_cum_counts = torch.zeros((max_steps, gen_length), dtype=torch.long, device="cpu")
        remask_counts = torch.zeros((max_steps, gen_length), dtype=torch.long, device="cpu")
        ever_unmasked = torch.zeros((batch_size, gen_length), dtype=torch.bool, device=device)

    step_used = 0
    _last_logits = None
    last_x0 = None

    for step in range(max_steps):
        # Stop if everything (except prompt) is unmasked and it's not the first iteration
        current_gen_mask = (x == mask_id) & (~fix_mask)
        if step > 0 and not current_gen_mask.any():
            break

        step_used = step

        # Forward with optional CFG
        with torch.autocast(device_type="cuda"):
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[fix_mask] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits_cat = model(x_cat).logits
                logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x).logits

        logits = logits.to(torch.float32)
        # Stochastic sample (may be used for final token updates)
        x0, _ = sample_and_score(logits, temperature)
        entropy = compute_entropy(logits)
        # Deterministic confidence for ordering
        logp_all = F.log_softmax(logits, dim=-1)
        logp_max, _ = logp_all.max(dim=-1)
        _last_logits = logits
        last_x0 = x0

        # Phase 1: Adaptive Correction (Remasking)
        # remask_candidates = (~(x == mask_id)) & (~fix_mask)
        # if remask_threshold_R > 0 and remask_candidates.any():
        #     # Robust suboptimality metric: gap to best token
        #     current_logp = logp_all.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        #     logp_diff = torch.clamp(logp_max - current_logp, min=0.0)

        #     # Stats over candidates
        #     cand_values = torch.where(remask_candidates, logp_diff, torch.zeros_like(logp_diff))
        #     num_cands = remask_candidates.sum(dim=1, keepdim=True).float()
        #     safe_num_cands = torch.clamp(num_cands, min=1.0)

        #     mean_diff = cand_values.sum(dim=1, keepdim=True) / safe_num_cands
        #     variance = torch.where(remask_candidates, (logp_diff - mean_diff) ** 2, torch.zeros_like(logp_diff))
        #     variance = variance.sum(dim=1, keepdim=True) / safe_num_cands
        #     std_diff = torch.sqrt(variance + 1e-8)

        #     # Outlier threshold on large differences
        #     threshold = mean_diff + (float(remask_threshold_R) * std_diff)
        #     remask_sel = (logp_diff > threshold) & remask_candidates & (num_cands > 0)

        #     if remask_sel.any():
        #         # Track remask events in generation span
        #         if track_trajectory and gen_length > 0:
        #             remask_events = remask_sel[:, prompt_len:]
        #             remask_counts[step, :] += remask_events.sum(dim=0).to("cpu")

        #         x[remask_sel] = mask_id
        #         # Predictions are now stale; skip progression and continue
        #         continue

        # Phase 2: Adaptive Progression (EB-PA)
        current_gen_mask = (x == mask_id) & (~fix_mask)
        if not current_gen_mask.any():
            continue

        # PA score for ordering masked tokens (deterministic)
        score = logp_max - positional_penalty
        error_proxy = -score
        error_proxy = torch.where(current_gen_mask, error_proxy, torch.full_like(error_proxy, float("inf")))

        # Sort per batch by increasing error (i.e., highest PA score first)
        sorted_errors, sorted_indices = torch.sort(error_proxy, dim=-1)
        B, T = sorted_indices.shape
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        sorted_entropies = entropy[batch_indices, sorted_indices]

        valid_mask = torch.isfinite(sorted_errors)
        masked_sorted_entropies = torch.where(valid_mask, sorted_entropies, torch.zeros_like(sorted_entropies))

        acc_entropy = torch.cumsum(masked_sorted_entropies, dim=-1)
        cummax_entropy, _ = torch.cummax(masked_sorted_entropies, dim=-1)
        inclusion_mask = (acc_entropy - cummax_entropy <= float(gamma) + 1e-6) & valid_mask

        k = inclusion_mask.sum(dim=-1)
        min_k = (valid_mask.sum(dim=-1) > 0).long()
        k = torch.max(k, min_k)

        # build selection mask for top-k indices
        ranks = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        topk_mask = (ranks < k.unsqueeze(1))
        unmask_sel = torch.zeros_like(x, dtype=torch.bool)
        unmask_sel.scatter_(1, sorted_indices, topk_mask)

        # Apply predicted x0 at selected positions
        if track_trajectory and gen_length > 0:
            prev_mask = (x == mask_id)
        x[unmask_sel] = x0[unmask_sel]
        if track_trajectory and gen_length > 0:
            # Tokens that transitioned from mask -> token at this step within gen span
            unmask_events = (prev_mask & unmask_sel)[:, prompt_len:]
            ever_unmasked = ever_unmasked | unmask_events
            unmask_counts[step, :] += unmask_events.sum(dim=0).to("cpu")
            unmask_cum_counts[step, :] += ever_unmasked.sum(dim=0).to("cpu")

    # Final fill if any remain masked due to max_steps limit
    remaining_mask = (x == mask_id) & (~fix_mask)
    if remaining_mask.any() and last_x0 is not None:
        x[remaining_mask] = last_x0[remaining_mask]

    if track_trajectory:
        # Truncate to the number of used steps + 1 (since step index starts at 0)
        used = min(step_used + 1, max_steps)
        traj = {
            "unmask_counts": unmask_counts[:used, :].tolist(),
            "unmask_cum_counts": unmask_cum_counts[:used, :].tolist(),
            "remask_counts": remask_counts[:used, :].tolist(),
        }
        return x, traj
    else:
        return x

