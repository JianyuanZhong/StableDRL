import argparse
import json
import math
import os
import random
import time
import csv
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from parsers import Parser, is_equiv
from typing import Optional
plt = None  # populated lazily when plotting

import sys
# sys.path.append('/home/ywxzml3j/ywxzml3juser46/diffusion_LM/LLaDOU')
from eval.generation_p2 import generate_p2, generate_ars
from networks.llada_svpo import LLaDASVPO as LLaDOUModelLM

from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from mbpp import MBPPDataset, ensure_mbpp_problem_file
from humaneval import HumanEvalDataset, ensure_humaneval_file
from countdown import CTDDataset
from sudoku import SudokuDataset
from local_eval import evaluate_functional_correctness as he_mbpp_evaluate

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "mbpp": MBPPDataset,
    "humaneval": HumanEvalDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()
# Removed corpus log-probabilities loader: not needed for PA-sampler



def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    planner_mode="self",
    eta=1.0,
    kappa_schedule="linear",
    score_type="confidence",
    mask_id=126336,
    planner_model=None,
    pa_lambda: float = 0.0,
    sampler: str = "p2",
    gamma: float = 0.1,
    remask_R: float = 1.5,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device
    # Accumulate trajectory counts across dataset (CPU tensors). For ARS, steps is an upper bound.
    unmask_counts_total = torch.zeros((steps, gen_length), dtype=torch.long)
    unmask_cum_counts_total = torch.zeros((steps, gen_length), dtype=torch.long)
    remask_counts_total = torch.zeros((steps, gen_length), dtype=torch.long)

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        task_ids = batch.get("task_ids", None)

        if sampler == "ars":
            out = generate_ars(
                model,
                input_ids,
                tokenizer,
                gen_length=gen_length,
                temperature=temperature,
                mask_id=mask_id,
                gamma=gamma,
                pa_lambda=pa_lambda,
                remask_threshold_R=remask_R,
                max_steps=steps,
                cfg_scale=cfg_scale,
                track_trajectory=True,
            )
        else:
            out = generate_p2(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                mask_id=mask_id,
                planner_model=planner_model,
                planner_mode=planner_mode,
                kappa_schedule_name=kappa_schedule,
                eta=eta,
                score_type=score_type,
                pa_lambda=pa_lambda,
                track_trajectory=True,
            )
        # Support new return signature with trajectory
        if isinstance(out, tuple) and len(out) == 2:
            seq_out, traj = out
            # Accumulate CPU counts
            try:
                uc = torch.tensor(traj.get("unmask_counts", []), dtype=torch.long)
                ucc = torch.tensor(traj.get("unmask_cum_counts", []), dtype=torch.long)
                rc = torch.tensor(traj.get("remask_counts", []), dtype=torch.long)
                # Pad/truncate to [steps, gen_length]
                def fit(arr):
                    if arr.numel() == 0:
                        return torch.zeros((steps, gen_length), dtype=torch.long)
                    arr = arr[:steps, :gen_length]
                    pad_s = steps - arr.shape[0]
                    pad_g = gen_length - arr.shape[1]
                    if pad_s > 0 or pad_g > 0:
                        pad = torch.zeros((max(0, pad_s), arr.shape[1]), dtype=arr.dtype)
                        if pad_s > 0:
                            arr = torch.cat([arr, pad[:pad_s]], dim=0)
                        if pad_g > 0:
                            arr = torch.cat([arr, torch.zeros((arr.shape[0], pad_g), dtype=arr.dtype)], dim=1)
                    return arr
                unmask_counts_total += fit(uc)
                unmask_cum_counts_total += fit(ucc)
                remask_counts_total += fit(rc)
            except Exception:
                pass
        else:
            seq_out = out

        generated_texts = tokenizer.batch_decode(seq_out[:, -gen_length:], skip_special_tokens=False)
        example_result = []
        for j in range(len(questions)):
            item = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            if task_ids is not None:
                item["task_id"] = task_ids[j]
                item["prompt"] = prompts[j]
            example_result.append(item)
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    # Reduce totals across ranks for both processed counter and trajectories
    # Move to device for all_reduce, then back to CPU
    world = dist.get_world_size()
    # total_processed is already on device
    if world > 1:
        dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
        uc_dev = unmask_counts_total.to(device)
        ucc_dev = unmask_cum_counts_total.to(device)
        rc_dev = remask_counts_total.to(device)
        dist.all_reduce(uc_dev, op=dist.ReduceOp.SUM)
        dist.all_reduce(ucc_dev, op=dist.ReduceOp.SUM)
        dist.all_reduce(rc_dev, op=dist.ReduceOp.SUM)
        unmask_counts_total = uc_dev.to("cpu")
        unmask_cum_counts_total = ucc_dev.to("cpu")
        remask_counts_total = rc_dev.to("cpu")

    avg_wall_time = sum(wall_times) / len(wall_times) if len(wall_times) > 0 else 0.0
    denom = max(1, int(total_processed.item()))
    unmask_freq = (unmask_counts_total.float() / float(denom)).numpy().tolist()
    # Prefer cumulative frequency; fallback to step-only if not available
    unmask_cum_freq = (unmask_cum_counts_total.float() / float(denom)).numpy().tolist()
    remask_freq = (remask_counts_total.float() / float(denom)).numpy().tolist()
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "trajectory": {
            "unmask_freq": unmask_freq,
            "unmask_cum_freq": unmask_cum_freq,
            "remask_freq": remask_freq,
            "steps": steps,
            "gen_length": gen_length,
        },
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24", "mbpp", "humaneval"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=8)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--csv_path", type=str, default="eval_results/summary_p2.csv")
    # P2-specific
    parser.add_argument("--planner_mode", type=str, choices=["self", "bert"], default="self")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--kappa_schedule", type=str, choices=["linear", "cosine", "sqrt"], default="linear")
    parser.add_argument("--score_type", type=str, choices=["confidence", "random", "pa"], default="confidence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--planner_model_path", type=str, default="answerdotai/ModernBERT-large")
    # PA-sampler specific
    parser.add_argument("--pa_lambda", type=float, default=0.5)
    # Sampler selection and ARS-specific
    parser.add_argument("--sampler", type=str, choices=["p2", "ars"], default="p2")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--remask_R", type=float, default=1.5)
    args = parser.parse_args()

    # args.diffusion_steps = args.gen_length
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256, "mbpp": -1, "humaneval": -1}
    
    print('-'*100)
    print(f"Loading model from {args.checkpoint_path}")
    print('-'*100)
    model = LLaDOUModelLM.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval().requires_grad_(False).to(local_rank)

    planner_model = None
    if args.planner_mode == "bert" and len(args.planner_model_path) > 0:
        planner_model = AutoModel.from_pretrained(
            args.planner_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        # model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(local_rank)
        # if args.merge_lora:
        #     model = model.merge_and_unload()
        #     model = model.to(device=local_rank, dtype=torch.bfloat16)
        #     print(f"Rank {local_rank}: LoRA weights merged")
        model = LLaDOUModelLM.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval().requires_grad_(False).to(local_rank)    

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    # Encode settings into the filename-safe model name
    if args.sampler == "ars":
        model_name = (
            model_name
            + f"_ars-pa"
            + f"_gamma{args.gamma}"
            + f"_t{args.temperature}"
            + f"_cfg{args.cfg_scale}"
            + f"_stpa"
            + f"_pa{args.pa_lambda}"
            + f"_R{args.remask_R}"
        )
    else:
        model_name = (
            model_name
            + f"_p2-{args.planner_mode}-{args.kappa_schedule}"
            + f"_eta{args.eta}"
            + f"_t{args.temperature}"
            + f"_cfg{args.cfg_scale}"
            + f"_bl{args.block_length}"
            + f"_st{args.score_type}"
        )

    # If using PA-sampler, encode pa_lambda into the model name for file uniqueness
    if args.score_type == "pa":
        model_name = model_name + f"_pa{args.pa_lambda}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = (
        f"{args.output_dir}/{args.dataset}_{model_name}_gl{args.gen_length}_ds{args.diffusion_steps}_"
        f"{dist.get_rank()}_generations.json"
    )
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        planner_mode=args.planner_mode,
        eta=args.eta,
        kappa_schedule=args.kappa_schedule,
        score_type=args.score_type,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        mask_id=args.mask_id,
        planner_model=planner_model,
        pa_lambda=args.pa_lambda,
        sampler=args.sampler,
        gamma=args.gamma,
        remask_R=args.remask_R,
    )

    if not args.dont_save:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                        "trajectory": metrics.get("trajectory", {}),
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0 and not args.dont_save:
        os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
        pattern = f"{args.output_dir}/{args.dataset}_{model_name}_gl{args.gen_length}_ds{args.diffusion_steps}_*_generations.json"
        files = sorted(glob.glob(pattern))

        correct = 0
        processed = 0
        wall_times = []
        # For plotting trajectory heatmaps averaged across ranks/files
        traj_unmask_accum = None
        traj_remask_accum = None
        traj_count = 0

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            gens = data.get("generations", [])
            # Collect trajectory if available
            traj = data.get("metrics", {}).get("trajectory") or data.get("trajectory")
            if traj is not None:
                # Prefer cumulative frequency for plotting if present
                uf = np.array(traj.get("unmask_cum_freq", traj.get("unmask_freq", [])), dtype=np.float32)
                rf = np.array(traj.get("remask_freq", []), dtype=np.float32)
                if uf.size > 0 and rf.size > 0:
                    if traj_unmask_accum is None:
                        traj_unmask_accum = uf
                        traj_remask_accum = rf
                    else:
                        traj_unmask_accum = traj_unmask_accum + uf
                        traj_remask_accum = traj_remask_accum + rf
                    traj_count += 1
            c = 0
            p = 0
            if args.dataset == "gsm8k":
                for item in gens:
                    p += 1
                    gt = item.get("ground_truth")
                    pred = Parser.extract_answer_gsm8k(item.get("generations", ""))
                    if pred is not None and gt is not None and pred == gt:
                        c += 1
            elif args.dataset == "math":
                for item in gens:
                    p += 1
                    gt = item.get("ground_truth", "")
                    pred = Parser.extract_answer_boxed(item.get("generations", ""))
                    if pred is not None and is_equiv(pred, gt):
                        c += 1
            elif args.dataset in ["mbpp", "humaneval"]:
                # Execution-based Scoring via LLaDOU evaluator
                import tempfile
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
                        tmp_path = tf.name
                        for item in gens:
                            task_id = item.get("task_id")
                            completion = item.get("generations", "")
                            prompt = item.get("prompt_input", item.get("prompt", ""))
                            tf.write(json.dumps({
                                "task_id": task_id,
                                "prompt": prompt,
                                "completion": completion,
                            }) + "\n")

                    # Ensure problem file path
                    if args.dataset == "mbpp":
                        problem_fp = ensure_mbpp_problem_file()
                        he_mbpp_evaluate(
                            input_file=tmp_path,
                            n_workers=8,
                            problem_file=problem_fp,
                            is_mbpp=True,
                            language="python",
                            out_dir=args.output_dir,
                        )
                    else:
                        he_mbpp_evaluate(
                            input_file=tmp_path,
                            n_workers=8,
                            problem_file=ensure_humaneval_file(),
                            is_mbpp=False,
                            language="python",
                            out_dir=args.output_dir,
                        )
                    # Read scored file to compute accuracy
                    scored_path = tmp_path.replace(".jsonl", ".scored.jsonl")
                    if os.path.exists(scored_path):
                        with open(scored_path, "r", encoding="utf-8") as sf:
                            for line in sf:
                                try:
                                    rec = json.loads(line)
                                    p += 1
                                    if rec.get("passed", False):
                                        c += 1
                                except Exception:
                                    continue
                except Exception:
                    pass
                finally:
                    if p == 0:
                        p = len(gens)
            correct += c
            processed += p
            try:
                wall_times.append(float(data.get("metrics", {}).get("wall_time", 0.0)))
            except Exception:
                pass

        accuracy = (correct / processed * 100.0) if processed > 0 else 0.0
        avg_wall_time = sum(wall_times) / len(wall_times) if len(wall_times) > 0 else 0.0

        header = [
            "dataset",
            "model_name",
            "checkpoint_path",
            "gen_length",
            "diffusion_steps",
            "accuracy",
            "processed",
            "avg_wall_time_s",
        ]
        row = [
            args.dataset,
            model_name,
            args.checkpoint_path,
            args.gen_length,
            args.diffusion_steps,
            round(accuracy, 4),
            processed,
            round(avg_wall_time, 4),
        ]

        write_header = not os.path.exists(args.csv_path)
        with open(args.csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        # Save trajectory heatmaps
        try:
            # Lazy, optional import to avoid environment dependency during non-plotting runs
            import matplotlib  # type: ignore  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore  # noqa: F401
            if traj_count > 0 and traj_unmask_accum is not None and traj_remask_accum is not None:
                unmask_avg = traj_unmask_accum / float(traj_count)
                remask_avg = traj_remask_accum / float(traj_count)

                # Create figure with two subplots for unmask and remask
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
                for ax, data_arr, title, cmap in [
                    (axes[0], unmask_avg, "Unmask Frequency", "YlGn"),
                    (axes[1], remask_avg, "Remask Frequency", "Blues"),
                ]:
                    im = ax.imshow(
                        data_arr,
                        aspect="auto",
                        origin="upper",
                        interpolation="nearest",
                        cmap=cmap,
                        vmin=0.0,
                        vmax=1.0,
                        extent=[0, data_arr.shape[1], 0, data_arr.shape[0]],
                    )
                    ax.set_xlabel("Decoding Order")
                    ax.set_ylabel("Steps")
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                heatmap_path = os.path.join(
                    args.output_dir,
                    f"{args.dataset}_{model_name}_gl{args.gen_length}_ds{args.diffusion_steps}_trajectory.png",
                )
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)

                # Also save raw arrays for later analysis
                np.savez_compressed(
                    os.path.join(
                        args.output_dir,
                        f"{args.dataset}_{model_name}_gl{args.gen_length}_ds{args.diffusion_steps}_trajectory_arrays.npz",
                    ),
                    unmask_freq=unmask_avg,
                    remask_freq=remask_avg,
                )
        except Exception:
            pass

    cleanup_ddp()


