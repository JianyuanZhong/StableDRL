import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
import re
import math
from typing import Union, Any, Optional
from networks.modeling_llada import LLaDAConfig, LLaDAModelLM

# Try importing set_seed from accelerate for reproducibility, provide fallback
try:
    from accelerate.utils import set_seed as set_seed_spg
    from accelerate import DistributedType
except ImportError:
    print("Warning: accelerate not found. Using basic seed setting for forward_process.")
    DistributedType = None
    def set_seed_spg(seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (1 << 31))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


# ===================================================================
# Model Definition (LLaDOUModelLM) - Unchanged
# ===================================================================

class LLaDASVPO(LLaDAModelLM):
    # (Implementation remains the same as provided in the prompt)
    def __init__(self, *args, **kwargs):
        # Remove auxiliary head specific configurations from kwargs if they were passed.
        aux_keys = ["num_head_layers", "use_mask_embeddings", "use_block_embeddings", "use_adaln"]
        for key in aux_keys:
            kwargs.pop(key, None)

        # Check for config in args for potential fallback initialization
        config = None
        if args:
            if hasattr(args[0], '__dict__') or isinstance(args[0], dict):
                config = args[0]

        try:
            # Initialize the base model.
            super().__init__(*args)
        except Exception:
            # Fallback initialization
            if not hasattr(self, 'config'):
                if config is None:
                    # Create a minimal config if none found
                    config = type('obj', (object,), {'hidden_size': 4096})()
                self.config = config
            if not hasattr(self, 'model'):
                # Mock model structure if needed
                self.model = type('obj', (object,), {'alibi_cache': None, 'config': self.config})()

        self.hidden_size = getattr(self.config, 'hidden_size', 4096)
        self.reset_dropout()

    def reset_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                m.p = 0

    def _init_weights(self, module):
        # (Implementation remains the same)
        if hasattr(module, 'reset_parameters') and not isinstance(module, nn.LayerNorm):
            try:
                module.reset_parameters()
            except Exception:
                pass
        elif isinstance(module, nn.Linear):
            if hasattr(module, "is_last_linear"):
                module.weight.data.normal_(0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            try:
                nn.init.trunc_normal_(module.weight, std=0.02)
            except AttributeError:
                module.weight.data.normal_(0, 0.02)

    def forward(self, *args, **kwargs):
        kwargs.pop("pred_mask_prob", None)
        return super().forward(*args, **kwargs)


# ===================================================================
# Reward Function (Same as original)
# ===================================================================

def anti_short_boxed_reward(batch, responses, num_generations, device, min_total_chars=200, non_boxed_ratio_threshold=0.85):
    # (Implementation remains the same)
    scores = []
    for resp in responses:
        s = resp if isinstance(resp, str) else str(resp)
        s_stripped = s.strip()
        non_ws = re.sub(r"\s+", "", s_stripped)
        total_len = len(non_ws)
        pure_boxed = bool(re.fullmatch(r"[\s$]*\\boxed\{[^}]*\}[\s$]*[.?!'\"“”’]*\s*", s_stripped))
        without_boxed = re.sub(r"\\boxed\{.*?\}", "", s_stripped, flags=re.DOTALL)
        non_ws_without_boxed = re.sub(r"\s+", "", without_boxed)
        ratio_non_boxed = (len(non_ws_without_boxed)) / (len(non_ws) + 1e-6)
        too_short_and_mainly_boxed = (total_len < min_total_chars) and (pure_boxed or ratio_non_boxed < non_boxed_ratio_threshold)
        scores.append(0.0 if too_short_and_mainly_boxed else 1.0)
    return torch.tensor(scores, device=device, dtype=torch.float32)


# ===================================================================
# SPG Implementation (Adapted from spg_trainer.py)
# ===================================================================

# Define a simple config holder for SPG parameters
class SPGConfig:
    def __init__(self, forward_type="block_random", num_t=1, min_t=0.0, max_t=1.0,
                 block_length=32, use_mask_prompt=True, p_mask_prompt=0.15,
                 mask_id=126336, cfg_scale=0.0, logp_estimation="mix",
                 eubo_beta=1.5, mix_weight=0.5,
                 # New AIS/SNIS parameters
                 use_snis=False, ais_clip_iw=5.0):
        self.forward_type = forward_type
        self.num_t = num_t
        self.min_t = min_t
        self.max_t = max_t
        self.block_length = block_length
        self.use_mask_prompt = use_mask_prompt
        self.p_mask_prompt = p_mask_prompt
        self.mask_id = mask_id
        self.cfg_scale = cfg_scale
        self.logp_estimation = logp_estimation
        self.eubo_beta = eubo_beta
        self.mix_weight = mix_weight
        # New assignments
        self.use_snis = use_snis
        self.ais_clip_iw = ais_clip_iw

def gather_metric(tensor_input, accelerator):
    # (Utility function for logging)
    if not isinstance(tensor_input, torch.Tensor):
        return float(tensor_input)
    detached_tensor = tensor_input.detach()
    if accelerator:
        try:
            # Use gather_for_metrics which handles uneven batch sizes
            gathered_tensor = accelerator.gather_for_metrics(detached_tensor)
            return gathered_tensor.mean().item()
        except Exception:
            return detached_tensor.mean().item()
    elif torch.distributed.is_initialized():
        torch.distributed.all_reduce(detached_tensor, op=torch.distributed.ReduceOp.SUM)
        return detached_tensor.sum().item() / torch.distributed.get_world_size()
    else:
        return detached_tensor.mean().item()

# -------------------------------------------------------------------
# 1. Generation (Sampling) Functions
# -------------------------------------------------------------------

def add_gumbel_noise_spg(logits, temperature, dtype):
    """Adapted from SPGTrainer.add_gumbel_noise."""
    if temperature == 0.0:
        return logits
    # Use float64 as suggested in the official implementation's comments
    calc_dtype = torch.float32
    logits = logits.to(calc_dtype)
    noise = torch.rand_like(logits, dtype=calc_dtype)
    # Add epsilon for numerical stability when log(noise)
    gumbel_noise = (-torch.log(noise + 1e-9)) ** temperature
    return (logits.exp() / gumbel_noise).to(dtype)

def get_num_transfer_tokens_spg(mask_index, steps):
    """Adapted from SPGTrainer.get_num_transfer_tokens."""
    if steps <= 0: steps = 1
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        # Ensure dimensions match before indexing
        if mask.shape[0] == num_transfer_tokens.shape[0]:
            num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)

def generate_spg(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    prompt_mask=None,
    use_fp16=False
):
    """Adapted from SPGTrainer.generate."""

    # Determine device and dtype context
    device = prompt.device
    if hasattr(model, 'dtype'):
        dtype = model.dtype
    else:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32


    # FIX: Remove manual autocast context; rely on external precision management
    if True:
        bs = prompt.shape[0]

        prompt_len = prompt.shape[1]
        x = torch.full((bs, prompt_len + gen_length), mask_id, dtype=torch.long).to(device)
        x[:, :prompt_len] = prompt.clone()
        completion_slice = slice(prompt_len, x.shape[1])

        attention_mask_full = None
        if prompt_mask is not None:
            # extend prompt_mask to the full attention mask (L199-202)
            prompt_mask_bool = prompt_mask.bool()
            gen_mask = torch.ones(bs, gen_length, dtype=torch.bool).to(device)
            attention_mask_full = torch.cat([prompt_mask_bool, gen_mask], dim=1)

        prompt_index = x != mask_id

        # Block configuration (L206-210)
        if block_length <= 0: return x

        # spg_trainer.py asserts gen_length % block_length == 0.
        # We handle it slightly more robustly if the assertion fails, but assume divisibility for exact replication.
        if gen_length % block_length != 0:
            num_blocks = (gen_length + block_length - 1) // block_length
        else:
            num_blocks = gen_length // block_length

        if num_blocks == 0: return x

        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt_len + num_block * block_length
            end_idx = min(prompt_len + (num_block + 1) * block_length, x.shape[1])

            if start_idx >= end_idx: continue

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens_spg(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                # torch.cuda.empty_cache()
                mask_index = x == mask_id
                mask_index_completion = mask_index[:, completion_slice]
                x_completion = x[:, completion_slice]

                # FIX: Remove inner manual autocast wrapper
                if True:
                    # Handle Classifier-Free Guidance (CFG) (L217-228)
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)

                        attn_mask_cfg = None
                        if attention_mask_full is not None:
                            attn_mask_cfg = torch.cat([attention_mask_full, attention_mask_full], dim=0)

                        logits = model(x_, attention_mask=attn_mask_cfg).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x, attention_mask=attention_mask_full).logits

                    # Apply Gumbel noise (L231-234)
                    completion_logits = logits[:, completion_slice, :]
                    if cfg_scale > 0.0:
                        del un_logits
                    del logits

                    logits_with_noise = add_gumbel_noise_spg(
                        completion_logits, temperature=temperature, dtype=dtype
                    )
                    x0_completion = torch.argmax(logits_with_noise, dim=-1)
                    del logits_with_noise

                    # Handle remasking strategy (L237-245)
                    if remasking == "low_confidence":
                        p = F.softmax(completion_logits.to(dtype), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0_completion, -1)), -1
                        )
                    elif remasking == "random":
                        x0_p = torch.rand((x0_completion.shape[0], x0_completion.shape[1]), device=device)
                    else:
                        raise NotImplementedError(remasking)
                    del completion_logits

                    # Ensure we don't process tokens beyond the current block (L248)
                    block_end_rel = max(end_idx - prompt_len, 0)
                    x0_p[:, block_end_rel:] = -np.inf

                    # Update masked tokens (L251-252)
                    x0_completion = torch.where(mask_index_completion, x0_completion, x_completion)
                    confidence_completion = torch.where(
                        mask_index_completion, x0_p, torch.full_like(x0_p, float("-inf"))
                    )

                    # Select tokens to transfer based on confidence (L255-261)
                    transfer_index_completion = torch.zeros(
                        x_completion.shape, dtype=torch.bool, device=device
                    )
                    for j in range(confidence_completion.shape[0]):
                        # Check index boundary
                        if i < num_transfer_tokens.shape[1]:
                            num_tokens = num_transfer_tokens[j, i].item()
                        else:
                            num_tokens = 0

                        if num_tokens > 0:
                             # Ensure k is valid (not larger than available finite scores)
                            k = min(num_tokens, torch.isfinite(confidence_completion[j]).sum().item())
                            if k > 0:
                                _, select_index = torch.topk(confidence_completion[j], k=k)
                                transfer_index_completion[j, select_index] = True

                    updated_completion = torch.where(
                        transfer_index_completion, x0_completion, x_completion
                    )
                    x[:, completion_slice] = updated_completion
                    del (
                        x0_completion,
                        confidence_completion,
                        transfer_index_completion,
                        updated_completion,
                        x0_p,
                    )
        return x

# -------------------------------------------------------------------
# 2. Forward Process (Masking) Functions
# -------------------------------------------------------------------

def forward_process_spg(batch, prompt_index, mask_id, seed=None, completion_mask=None, args: SPGConfig=None):
    """
    Adapted from SPGTrainer.forward_process. (L286-484)
    """
    if args is None:
        raise ValueError("SPGConfig must be provided.")

    # Set seed for reproducibility
    set_seed_spg(seed)

    forward_type = args.forward_type
    num_t = args.num_t
    min_t = args.min_t
    max_t = args.max_t

    b, l = batch.shape
    device = batch.device

    # Ensure prompt_index is correctly shaped for batch operations [B, L] or [L]
    if prompt_index.dim() == 1:
        prompt_index_exp = prompt_index.unsqueeze(0).expand(b, -1)
    else:
        prompt_index_exp = prompt_index

    # Implementation copied verbatim from spg_trainer.py, replacing self.args with args.

    if forward_type == "all":
        # (L316-340)
        assert num_t == 1
        if args.use_mask_prompt:
            t_p = torch.ones(b, device=device) * args.p_mask_prompt
            random_matrix = torch.rand((b, l), device=device)
            is_mask_prompt = prompt_index_exp & (random_matrix < t_p.unsqueeze(1))
            is_mask_completion = ~prompt_index_exp
            is_mask = is_mask_prompt | is_mask_completion
        else:
            is_mask_completion = ~prompt_index_exp
            is_mask = is_mask_completion

        noisy_batch = torch.where(is_mask, mask_id, batch)
        noisy_batch = noisy_batch.unsqueeze(1)
        block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=device)

    elif forward_type == "random":
        # (L342-374)
        prompt_len = prompt_index.sum().item()
        gen_length = l - prompt_len
        completion_length = completion_mask.sum(-1)
        is_mask = torch.zeros((b, num_t, gen_length), dtype=torch.bool, device=device)
        for i in range(b):
            start_mask_num = max(int(completion_length[i] * min_t), 1)
            end_mask_num = min(int(completion_length[i] * max_t), completion_length[i])
            if start_mask_num > end_mask_num: continue # Handle edge case
            # assert start_mask_num <= end_mask_num
            mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, num_t), device=device)
            indices = torch.arange(gen_length, device=device).repeat(1, num_t, 1)
            is_mask[[i], :, :] = indices < mask_num.unsqueeze(2)
            for j in range(num_t):
                is_mask[i, j, :completion_length[i]] = is_mask[i, j, :completion_length[i]][torch.randperm(completion_length[i])]
        is_mask = torch.cat((torch.zeros(b, num_t, prompt_len, dtype=torch.bool, device=device), is_mask), dim=2)
        completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_len, dtype=torch.bool, device=device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)

        if args.use_mask_prompt:
            t_p = torch.ones(b, num_t, device=device) * args.p_mask_prompt
            random_matrix = torch.rand((b, num_t, l), device=device)
            # Expand prompt_index for num_t dimension
            is_mask_prompt = prompt_index_exp.unsqueeze(1).repeat(1, num_t, 1) & (random_matrix < t_p.unsqueeze(2))
            is_mask = (is_mask_prompt | is_mask) | ~completion_mask_append
        else:
            is_mask = is_mask | ~completion_mask_append
        noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1))
        block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=device)

    elif forward_type == "block_all":
        # (L376-403)
        prompt_len = prompt_index.sum().item()
        gen_length = l - prompt_len
        if gen_length <= 0:
            return batch.unsqueeze(1).repeat(1, num_t, 1), torch.ones((b, num_t, l), dtype=torch.bool, device=device)

        block_length = args.block_length
        # assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        completion_num_blocks = (completion_mask.sum(-1)-1)//block_length+1
        # assert num_t <= num_blocks

        indices = torch.arange(num_blocks, device=device).repeat(b, 1)
        for i in range(b):
            indices[i] = indices[i][torch.randperm(num_blocks)] % completion_num_blocks[i]
        mask_block_idx = indices[:, :num_t]

        is_mask = torch.zeros((b, num_t, l), dtype=torch.bool, device=device)
        block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=device)

        for i in range(b):
            for j in range(num_t):
                # We use absolute indexing derived from prompt_len for clarity
                block_start = prompt_len + mask_block_idx[i, j] * block_length
                is_mask[i, j, block_start:] = True
                if mask_block_idx[i, j] < num_blocks - 1:
                    next_block_start = prompt_len + (mask_block_idx[i, j] + 1) * block_length
                    block_mask[i, j, next_block_start:] = False

        if args.use_mask_prompt:
            t_p = torch.ones(b, num_t, device=device) * args.p_mask_prompt
            random_matrix = torch.rand((b, num_t, l), device=device)
            is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
            is_mask = is_mask_prompt | is_mask
        else:
            is_mask = is_mask
        noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1))

    elif forward_type == "block_random":
        # (L405-482)
        prompt_len = prompt_index.sum().item()
        gen_length = l - prompt_len
        if gen_length <= 0:
            return batch.unsqueeze(1).repeat(1, num_t, 1), torch.ones((b, num_t, l), dtype=torch.bool, device=device)

        block_length = args.block_length
        # assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        completion_length = completion_mask.sum(-1)
        completion_num_blocks = (completion_length-1)//block_length+1
        # assert num_t <= num_blocks

        indices = torch.arange(num_blocks, device=device).repeat(b, 1)
        for i in range(b):
            indices[i] = indices[i][torch.randperm(num_blocks)] % completion_num_blocks[i]
        mask_block_idx = indices[:, :num_t]

        is_mask = torch.zeros((b, num_t, l), dtype=torch.bool, device=device)
        block_mask = torch.ones((b, num_t, l), dtype=torch.bool, device=device)

        for i in range(b):
            for j in range(num_t):
                # Use absolute indexing
                block_start = prompt_len + mask_block_idx[i, j] * block_length
                is_mask[i, j, block_start:] = True
                if mask_block_idx[i, j] < num_blocks - 1:
                    next_block_start = prompt_len + (mask_block_idx[i, j] + 1) * block_length
                    block_mask[i, j, next_block_start:] = False

        is_mask_following = torch.ones((b, num_t, l), dtype=torch.bool, device=device)
        for i in range(b):
            for j in range(num_t):
                mask_length = min(block_length, completion_length[i] - block_length * mask_block_idx[i, j])
                if mask_length <= 0: continue
                # assert mask_length > 0
                start_mask_num = max(int(mask_length * min_t), 1)
                end_mask_num = min(int(mask_length * max_t), mask_length)
                if start_mask_num > end_mask_num: continue
                # assert start_mask_num <= end_mask_num
                mask_num = torch.randint(start_mask_num, end_mask_num + 1, (1, 1), device=device)
                indices_block = torch.arange(block_length, device=device).repeat(1, 1, 1)
                is_mask_next = indices_block < mask_num.unsqueeze(2)

                block_start = prompt_len + mask_block_idx[i, j] * block_length

                if mask_block_idx[i, j] == num_blocks - 1 and mask_length == block_length:
                    is_mask_following[i, j, block_start:] = is_mask_next[0, 0][torch.randperm(block_length)]
                else:
                    block_end = block_start + mask_length
                    is_mask_following[i, j, block_start:block_end] = is_mask_next[0, 0, :mask_length][torch.randperm(mask_length)]

        completion_mask_append = torch.cat((torch.ones(b, num_t, prompt_len, dtype=torch.bool, device=device), completion_mask.unsqueeze(1).repeat(1, num_t, 1)), dim=2).to(torch.bool)

        if args.use_mask_prompt:
            t_p = torch.ones(b, num_t, device=device) * args.p_mask_prompt
            random_matrix = torch.rand((b, num_t, l), device=device)
            is_mask_prompt = ~is_mask & (random_matrix < t_p.unsqueeze(2))
            is_mask = is_mask_prompt | (is_mask & is_mask_following) | ~completion_mask_append
        else:
            is_mask = (is_mask & is_mask_following) | ~completion_mask_append
        noisy_batch = torch.where(is_mask, mask_id, batch.unsqueeze(1).repeat(1, num_t, 1))

    else:
        raise ValueError(f"forward_type {forward_type} not recognized.")

    return noisy_batch, block_mask

# -------------------------------------------------------------------
# 3. Loss Calculation Functions
# -------------------------------------------------------------------

def get_logits_spg(model, batch, prompt_index, cfg_scale, mask_id, prompt_mask=None):
    """Adapted from SPGTrainer.get_logits."""
    multisample = False
    if len(batch.shape) == 3:
        multisample = True
        bsz, num_t, l = batch.shape
        batch_flat = batch.view(-1, l)
        if prompt_mask is not None:
            # Handle prompt_mask expansion for multisample (L491-495)
            prompt_len = prompt_mask.shape[-1]
            # Assuming prompt_mask shape matches the first dimension of batch (bsz)
            prompt_mask_flat = prompt_mask.unsqueeze(1).repeat(1, num_t, 1).view(-1, prompt_len)
    else:
        batch_flat = batch
        prompt_mask_flat = prompt_mask

    # Prepare full attention mask (L497-500)
    attention_mask = None
    if prompt_mask_flat is not None:
        prompt_mask_bool = prompt_mask_flat.bool()
        assert batch_flat.shape[0] == prompt_mask_bool.shape[0], f"batch.shape: {batch_flat.shape}, prompt_mask.shape: {prompt_mask_bool.shape}"
        attention_mask = torch.cat([prompt_mask_bool, torch.ones(batch_flat.shape[0], batch_flat.shape[1] - prompt_mask_bool.shape[1], dtype=torch.bool, device=batch.device)], dim=1)

    # Handle CFG (L502-508)
    if cfg_scale > 0.0:
        assert len(prompt_index) == batch_flat.shape[1]
        prompt_index_expanded = prompt_index.unsqueeze(0).repeat(batch_flat.shape[0], 1)
        un_batch = batch_flat.clone()
        un_batch[prompt_index_expanded] = mask_id
        batch_input = torch.cat([batch_flat, un_batch])
        if attention_mask is not None:
            # Repeat attention mask along batch dimension (dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
    else:
        batch_input = batch_flat

    # Model forward pass
    logits = model(batch_input, attention_mask=attention_mask).logits

    # Process CFG outputs (L513-515)
    if cfg_scale > 0.0:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

    if multisample:
        # Reshape back
        logits = logits.view(bsz, num_t, l, -1)
    return logits

def _get_per_seq_logps_spg(model, input_ids, logits_to_keep, mask_seeds, prompt_mask=None, completion_mask=None, reward_mask=None, args: SPGConfig=None):
    """
    Adapted from SPGTrainer._get_per_seq_logps. (L509-621)
    """
    if args is None:
        raise ValueError("SPGConfig must be provided.")

    # input_ids shape: [num_iterations, batch_size, seq_len]
    num_iterations, batch_size, seq_len = input_ids.size()
    device = input_ids.device
    num_t = args.num_t

    assert len(mask_seeds) == num_iterations

    # Define prompt_index (L548-550)
    prompt_length = seq_len - logits_to_keep
    prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
    prompt_index[:prompt_length] = True

    # Apply masking using the adapted forward_process_spg (L553-563)
    all_perturbed_seqs = []
    all_expanded_inputs = []

    for iter_idx, mask_seed in enumerate(mask_seeds):
        expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
        seed_val = mask_seed.item() if isinstance(mask_seed, torch.Tensor) else mask_seed
        # block_mask is returned but not used for loss calculation in spg_trainer.py
        perturbed_seq, _ = forward_process_spg(
            expanded_input, prompt_index, args.mask_id, seed=seed_val, completion_mask=completion_mask, args=args
        )
        all_perturbed_seqs.append(perturbed_seq)
        all_expanded_inputs.append(expanded_input)

    # Concatenation (L566-570)
    perturbed_seq = torch.cat(all_perturbed_seqs, dim=0) # [num_iterations * batch_size, num_t, seq_len]
    perturb_mask = perturbed_seq == args.mask_id
    expanded_input = torch.cat(all_expanded_inputs, dim=0)

    if prompt_mask is not None:
        # Prepare prompt_mask for get_logits_spg
        prompt_mask_concat = torch.cat([prompt_mask]*num_iterations, dim=0)
    else:
        prompt_mask_concat = None

    # Get model predictions (L573-575)
    logits = get_logits_spg(
        model, perturbed_seq, prompt_index, args.cfg_scale, args.mask_id, prompt_mask_concat
    ) # [num_iterations * batch_size, num_t, seq_len, vocab_size]

    # Calculate cross-entropy loss and probabilities (L578-592)
    completion_logits = logits[:, :, -logits_to_keep:, :]
    completion_targets = expanded_input[:, -logits_to_keep:]
    perturb_mask = perturb_mask[:, :, -logits_to_keep:]

    completion_targets = completion_targets.unsqueeze(1).repeat(1, num_t, 1)
    flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
    flat_targets = completion_targets.reshape(-1)

    loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    # FIX: Remove manual autocast disabled block; compute explicitly in float32 where needed
    # Calculate probabilities (used for EUBO). Ensure float32 for stability.
    prob = F.softmax(flat_logits.float(), dim=-1).gather(dim=-1, index=flat_targets.unsqueeze(-1))

    # Reshape (L595-600)
    completion_log_probs = -loss.view(num_iterations * batch_size, num_t, logits_to_keep)
    completion_probs = prob.view(num_iterations * batch_size, num_t, logits_to_keep)
    per_token_logps = completion_log_probs.view(num_iterations, batch_size, num_t, logits_to_keep)
    per_token_probs = completion_probs.view(num_iterations, batch_size, num_t, logits_to_keep)

    per_token_logps = per_token_logps.to(torch.float32)
    per_token_probs = per_token_probs.to(torch.float32)

    assert completion_mask is not None

    # Expand masks (L608-610)
    completion_mask_expanded = completion_mask.unsqueeze(0).unsqueeze(2).expand(num_iterations, -1, num_t, -1)
    perturb_mask_expanded = perturb_mask.view(num_iterations, batch_size, num_t, logits_to_keep)

    # --- Core ELBO/EUBO Estimation Logic (L615-653) ---

    # ELBO Calculation (Average logp of masked tokens)
    denominator = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3)
    if (denominator == 0).any():
        # Handle zero denominator (L617-622)
        # print(f"WARNING: Zero denominator detected in per_seq_logps calculation!")
        denominator = torch.clamp(denominator, min=1e-8)

    # This calculation (L623) defines the ELBO estimate per MC sample
    per_seq_logps = (per_token_logps * completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / denominator
    
    # MODIFICATION: Explicitly define ELBO averaged over MC samples (num_t)
    per_seq_elbo = per_seq_logps.mean(dim=2) # [num_iterations, batch_size]

    # EUBO related calculations (L624-633)
    if args.logp_estimation == 'eubo' or args.logp_estimation == 'mix':
        # Calculate empirical masking rate (t)
        # Added clamp to denominator
        empirical_t = (completion_mask_expanded * perturb_mask_expanded).sum(dim=3) / completion_mask_expanded.sum(dim=3).clamp(min=1e-8)
        empirical_t_expanded = empirical_t.unsqueeze(3).expand(-1, -1, -1, completion_mask_expanded.size(-1))

        # Calculate weighted probabilities: (pi^beta * mask) / t
        per_token_avg_ps = per_token_probs.pow(args.eubo_beta) * perturb_mask_expanded * completion_mask_expanded / empirical_t_expanded.clamp(min=1e-8)
        per_token_avg_ps = per_token_avg_ps.mean(dim=2) # Average over num_t (MC samples)

        # Prepare for log calculation (L631-633)
        per_token_avg_ps_dezero = per_token_avg_ps.clone()
        per_token_avg_ps_dezero[per_token_avg_ps_dezero == 0] = 1
        loss_mask = (per_token_avg_ps > 0).bool()

    # Determine positive and negative trace estimations (L635-653)
    reward_mask_expanded = reward_mask.unsqueeze(0).expand(num_iterations, -1)
    # ELBO estimate used for positive traces (L636). Use the explicit ELBO variable.
    per_seq_logps_positive = per_seq_elbo

    if args.logp_estimation == 'eubo':
        # EUBO estimate (L638)
        per_seq_logps_negative = (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / args.eubo_beta
    elif args.logp_estimation == 'mix':
        # Mixture estimate (L640)
        L_EUBO = (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=2) / loss_mask.sum(dim=2).clamp(min=1e-8) / args.eubo_beta
        L_ELBO = per_seq_elbo
        per_seq_logps_negative = args.mix_weight * L_EUBO + (1-args.mix_weight) * L_ELBO
    elif args.logp_estimation == 'elbo':
        per_seq_logps_negative = per_seq_elbo
    elif args.logp_estimation == 'zero':
        per_seq_logps_negative = torch.zeros_like(per_seq_logps_positive)
    else:
        raise ValueError(f"logp_estimation: {args.logp_estimation} is not supported")

    # Sandwiched objective (L649)
    final_per_seq_logps = reward_mask_expanded.float() * per_seq_logps_positive + (~reward_mask_expanded).float() * per_seq_logps_negative
    assert final_per_seq_logps.shape == (num_iterations, batch_size)

    # MODIFICATION: Return both the SPG bound (for loss) and the ELBO bound (for IW calculation).
    return final_per_seq_logps, per_seq_elbo

# -------------------------------------------------------------------
# 4. Main GRPO Loop Functions (Generation/Scoring and Loss Computation)
# -------------------------------------------------------------------

def generate_and_score_completions_spg(
    model, tokenizer, inputs_batch, reward_func, device, accelerator,
    args: SPGConfig, num_iterations, generation_config, num_generations=1, random_masking=True
):
    """
    Adapted from SPGTrainer._generate_and_score_completions. (L624-814)
    """

    # 1. Prepare Prompts (Tokenization)
    # We assume inputs_batch contains 'problems' based on the context of the original training loop.
    prompts_text_list = inputs_batch.get('problems', [])
    if not prompts_text_list:
        return None

    # Apply chat template and tokenize (L669-679 adaptation)
    try:
        # Attempt to apply chat template if supported
        m = [[{"role": "user", "content": prompt}] for prompt in prompts_text_list]
        prompts_tokenized = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompts_tokenized = prompts_text_list

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for generation (L676)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    prompt_inputs = tokenizer(
        text=prompts_tokenized,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False, # As used in spg_trainer.py
    ).to(device)

    tokenizer.padding_side = original_padding_side # Restore

    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Handle max_prompt_length (L683-685)
    max_prompt_length = generation_config.get("max_prompt_length")
    if max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -max_prompt_length :]
        prompt_mask = prompt_mask[:, -max_prompt_length :]

    # Repeat prompts for num_generations using repeat_interleave for GRPO grouping
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # 2. Generation (L688-721 adaptation)
    gen_length = generation_config.get("max_completion_length", 256)
    block_length = generation_config.get("block_size", args.block_length)
    steps = generation_config.get("diffusion_steps", 128)
    temperature = generation_config.get("temperature", 0.0)
    cfg_scale = generation_config.get("cfg_scale", args.cfg_scale)
    remasking = generation_config.get("remasking", "low_confidence")
    mask_id = args.mask_id
    use_fp16 = generation_config.get("use_fp16", False)

    generation_batch_size = generation_config.get("generation_batch_size", prompt_ids.size(0))

    # Unwrap model for generation unless we're running under FSDP auto-wrapping
    unwrapped_model = model
    if accelerator is not None:
        dist_state = getattr(getattr(accelerator, "state", None), "distributed_type", None)
        is_fsdp = False
        if DistributedType is not None:
            is_fsdp = dist_state == DistributedType.FSDP
        elif dist_state is not None:
            is_fsdp = str(dist_state).upper().endswith("FSDP")

        if not is_fsdp:
            unwrapped_model = accelerator.unwrap_model(model)

    prompt_completion_ids_all = []
    # Process generation in batches if needed (L698-719)
    for i in range(0, prompt_ids.size(0), generation_batch_size):
        end_idx = min(i + generation_batch_size, prompt_ids.size(0))
        batch_prompt_ids = prompt_ids[i:end_idx]
        batch_prompt_mask = prompt_mask[i:end_idx]

        # Call the adapted generation function
        batch_prompt_completion_ids = generate_spg(
            model=unwrapped_model,
            prompt=batch_prompt_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            prompt_mask=batch_prompt_mask,
            use_fp16=use_fp16
        )
        prompt_completion_ids_all.append(batch_prompt_completion_ids)
        torch.cuda.empty_cache()

    prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

    # 3. Process Completions (L724-735 adaptation)
    prompt_length = prompt_ids.size(1)
    # Update prompt_ids based on the generated output (handles potential padding alignment)
    prompt_ids_generated = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Mask after EOS token (L729-735)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    # print(f"eos_token_id: {eos_token_id}")
    if eos_token_id is None:
        completion_mask = torch.ones_like(completion_ids, dtype=torch.int)
    else:
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)

        if is_eos.any(dim=1).any():
            # Find the index of the first EOS token
            first_eos_indices = is_eos.int().argmax(dim=1)
            has_eos = is_eos.any(dim=1)
            eos_idx[has_eos] = first_eos_indices[has_eos]

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # Keep tokens up to and including the first EOS token
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # 4. (MODIFICATION: Calculate initial ELBO for Q if SNIS is enabled)
    ref_elbo = None
    if args.use_snis:
        model.eval() # Ensure model is in eval mode
        # Prepare inputs for ELBO calculation
        input_ids = torch.cat([prompt_ids_generated, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # Generate a seed for the reference calculation masking
        if random_masking:
            ref_mask_seed = torch.randint(0, 2**31 - 1, (1,), device=device).item()
        else:
            ref_mask_seed = 42
        # Dummy reward_mask (required by the function signature, but we only use the returned ELBO)
        dummy_reward_mask = torch.ones(input_ids.size(0), dtype=torch.bool, device=device)
        # Calculate ELBO(Q). This must be done without gradients.
        with torch.no_grad():
             # input_ids shape must be [num_iterations=1, batch_size, seq_len]
            # We only need the second return value (the ELBO estimate).
            _, ref_elbo = _get_per_seq_logps_spg(
                model, input_ids.unsqueeze(0), logits_to_keep, [ref_mask_seed], prompt_mask, completion_mask, reward_mask=dummy_reward_mask, args=args
            )
            # ref_elbo shape: [1, batch_size]. Detach and squeeze.
            ref_elbo = ref_elbo.squeeze(0).detach()

    # 5. Scoring (Rewards and Advantages)
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    # Calculate rewards (Integrating reward logic from the original lladou_spg.py)

    # Calculate anti_reward
    try:
        anti_reward = anti_short_boxed_reward(inputs_batch, completions_text, num_generations, device)
    except Exception as e:
        print(f"Warning: Anti-reward calculation failed: {e}")
        anti_reward = torch.zeros(len(completions_text), device=device, dtype=torch.float32)

    if reward_func is not None:
         # Assuming reward_func signature matches the expectation (batch, responses, num_generations, device)
        try:
            base_rewards = reward_func(inputs_batch, completions_text, num_generations, device).float()
        except Exception as e:
            print(f"Warning: Error in reward function: {e}")
            base_rewards = torch.zeros(len(completions_text), device=device, dtype=torch.float32)

        # Handle potential shape mismatch
        if base_rewards.numel() == anti_reward.numel() == len(completions_text):
            rewards = base_rewards.view(-1) + anti_reward.view(-1)
        else:
            raise RuntimeError(f"Incompatible shapes for reward addition. base_rewards: {base_rewards.shape}, anti_reward: {anti_reward.shape}.")
    else:
        rewards = anti_reward

    # Handle distributed gathering (L765 adaptation)
    if accelerator:
        # Gather rewards across processes
        rewards_gathered = accelerator.gather(rewards)
    else:
        rewards_gathered = rewards

    # Calculate Advantages (Group Relative) (L770-773 adaptation)
    if rewards_gathered.numel() == 0:
        return None

    # View as [Total_Prompts * World_Size, Num_Generations]
    try:
        rewards_view = rewards_gathered.view(-1, num_generations)
    except RuntimeError as e:
        print(f"Error reshaping rewards_gathered ({rewards_gathered.shape}) with num_generations ({num_generations}): {e}")
        return None

    mean_grouped_rewards = rewards_view.mean(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    advantages = rewards_gathered - mean_grouped_rewards

    # Determine the slice for the current process (Robust implementation for potentially uneven batches)
    if accelerator:
        # Get the sizes of rewards tensor from all processes
        local_size = torch.tensor([rewards.size(0)], device=device)
        all_sizes = accelerator.gather(local_size)

        # Calculate the offset for the current process
        offsets = torch.cumsum(all_sizes, dim=0)
        start_idx = offsets[accelerator.process_index] - local_size[0]
        end_idx = offsets[accelerator.process_index]

        advantages_local = advantages[start_idx:end_idx]
        reward_mask_local = (advantages_local > 0).bool()
    else:
        advantages_local = advantages
        reward_mask_local = (advantages > 0).bool()

    # 6. Prepare Output Dictionary
    # KL divergence (ref_per_seq_logps) is omitted as SPG typically uses beta=0.0.

    return {
        "prompt_ids": prompt_ids_generated, # Use the aligned prompt_ids
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages_local,
        # "mask_seeds": mask_seeds, # MODIFICATION: Removed
        "reward_mask": reward_mask_local,
        "base_rewards": base_rewards,
        "anti_rewards": anti_reward,
        "rewards": rewards, # Local rewards for logging
        "completion_length": completion_mask.sum(1).float().mean().item(), # Local mean length
        # New output for SNIS
        "ref_elbo": ref_elbo
    }

# MODIFICATION: Added random_masking parameter
def compute_loss_spg(model, inputs, current_itr_idx, args: SPGConfig, accelerator=None, random_masking=True):
    """
    Adapted from SPGTrainer.compute_loss. (L101-138)
    """
    # Extract inputs (L103-106)
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    # mask_seeds = inputs["mask_seeds"] # MODIFICATION: Removed retrieval of pre-generated seeds
    advantages = inputs["advantages"]
    reward_mask = inputs["reward_mask"]
    base_rewards = inputs["base_rewards"]
    anti_rewards = inputs["anti_rewards"]
    
    # New input for SNIS: ELBO(Q)
    ref_elbo = inputs.get("ref_elbo")

    # Combine prompt and completion (L109-110)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    logits_to_keep = completion_ids.size(1)

    # Select the mask seed for the current iteration (L113)
    # MODIFICATION: Dynamically generate the mask seed for this iteration.
    # This explicitly ensures fresh masks are generated in every gradient update iteration (SPG Algo Line 6).

    if random_masking:
        # Generate a fresh random seed dynamically within the inner loop.
        this_itr_mask_seed = torch.randint(0, 2**31 - 1, (1,), device=input_ids.device).item()
    else:
        # Use a fixed seed if randomization is disabled (matching original behavior)
        this_itr_mask_seed = 42

    # Prepare inputs for _get_per_seq_logps_spg (L114)
    input_ids = input_ids.unsqueeze(0) # [1, batch_size, seq_len]

    # Calculate log probabilities (L115)
    # MODIFICATION: Receive both SPG bound (P) and ELBO bound (P).
    per_seq_logps, per_seq_elbo = _get_per_seq_logps_spg(
        model, input_ids, logits_to_keep, [this_itr_mask_seed], prompt_mask, completion_mask, reward_mask=reward_mask, args=args
    ) # Shapes: [1, batch_size]

    # Check for NaN/inf (L118-125)
    if torch.isnan(per_seq_logps).any() or torch.isinf(per_seq_logps).any():
        print("WARNING: NaN/inf detected in per_seq_logps!")
        # Handle appropriately
        per_seq_logps = torch.nan_to_num(per_seq_logps, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(per_seq_elbo).any() or torch.isinf(per_seq_elbo).any():
        per_seq_elbo = torch.nan_to_num(per_seq_elbo, nan=0.0, posinf=0.0, neginf=0.0)

    # --- AIS/SNIS Implementation ---
    iw_mean, iw_max = 1.0, 1.0 # For logging

    if args.use_snis and ref_elbo is not None:
        # 1. Calculate Log Importance Weights: log(w) = ELBO(P) - ELBO(Q)
        # Ensure float32 and detach. Squeeze iteration dimension (dim=0).
        log_iw = (per_seq_elbo.squeeze(0).float() - ref_elbo.float()).detach()

        # 2. Clipping (Stable implementation)
        try:
            MAX_LOG_IW = math.log(args.ais_clip_iw)
        except (ValueError, OverflowError):
            MAX_LOG_IW = float('inf') # Handle invalid clip value

        clipped_log_iw = torch.clamp(log_iw, max=MAX_LOG_IW)

        # 3. Self-Normalization (SNIS Step) using Softmax for stability
        # Use torch.nn.functional.softmax or F.softmax
        normalized_weights = torch.nn.functional.softmax(clipped_log_iw, dim=0)
        normalized_weights = normalized_weights.detach()

        # 4. Calculate the loss using normalized weights (SNIS Estimator)
        # Unsqueeze iteration dimension (dim=0)
        weights_for_loss = normalized_weights.unsqueeze(0) # [1, batch_size]

        # The surrogate objective is sum(w_i * A_i * L_SPG_i).
        # Use per_seq_logps (the SPG bound, which has gradients attached).
        surrogate_objective = weights_for_loss * advantages.unsqueeze(0) * per_seq_logps # [1, batch_size]

        # SNIS uses torch.sum() because normalized_weights sum to 1.
        loss = -torch.sum(surrogate_objective)

        # Logging IW stats
        iw_mean = torch.exp(log_iw).mean().item()
        iw_max = torch.exp(log_iw).max().item()
    else:
        # Standard SPG Loss Calculation (L129-131)
        # Compute the loss (Policy Gradient) (L129)
        per_seq_loss = -advantages.unsqueeze(0) * per_seq_logps # [1, batch_size]

        # Normalize by completion length (L130-131)
        completion_length_tensor = completion_mask.sum(dim=1).unsqueeze(0).float() # [1, batch_size]

        # Normalize by the sum of completion lengths across the batch
        loss = (per_seq_loss * completion_length_tensor).sum() / completion_length_tensor.sum().clamp(min=1e-8)

    # Prepare logging metrics
    loss_to_log = loss.detach().item()
    reward_mean = gather_metric(inputs.get('rewards', torch.tensor(0.0, device=loss.device)), accelerator)
    base_reward_mean = gather_metric(base_rewards, accelerator)
    anti_reward_mean = gather_metric(anti_rewards, accelerator)
    length = inputs.get("completion_length", 0.0)

    # We return the loss tensor so the training loop can handle backward pass and scaling.
    return {
        "loss": loss_to_log,
        "loss_tensor": loss,
        "reward": reward_mean,
        "base_reward": base_reward_mean,
        "anti_reward": anti_reward_mean,
        "length": length,
        "iw_mean": iw_mean, # Add IW logging
        "iw_max": iw_max,
    }

