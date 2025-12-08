import click
from tqdm import tqdm
from typing import Sequence
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from networks.lladou_arpo import LLaDOUARPOModel, sample_arpo
from dataloaders.collate_fn_math import collate_fn_math, extract_answer_gsm8k, collate_fn_gsm8k
from dataloaders.math import _math_verify_equal


def judge_answer_MATH(answers: Sequence[str], responses: Sequence[str], counts):
    counts[1] += len(answers)
    for ans, res in zip(answers, responses):
        if _math_verify_equal(ans, res):
            counts[0] += 1
    return counts


def judge_answer_GSM8K(answers: Sequence[str], responses: Sequence[str], counts):
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    counts[1] += len(ext_ans)
    for ans, res in zip(ext_ans, responses):
        if _math_verify_equal(ans, res):
            counts[0] += 1
    return counts


@click.command()
@click.option("--ckpt_path", type=str, default="")
@click.option("--local_data_path", type=str, default="openai/gsm8k")
@click.option("--batch_size", type=int, default=1)
@click.option("--num_workers", type=int, default=1)
@click.option("--steps", type=int, default=256)
@click.option("--gen_length", type=int, default=256)
@click.option("--block_length", type=int, default=8)
@click.option("--task", type=str, default="gsm8k")
@click.option("--seed", type=int, default=42)
@click.option("--no_sample", type=bool, default=False)
def main(
    ckpt_path: str = "",
    local_data_path: str = "openai/gsm8k",
    batch_size: int = 1,
    num_workers: int = 1,
    steps: int = 256,
    gen_length: int = 256,
    _block_length: int = 8,
    no_sample: bool = False,
    seed: int = 42,
    **_kwargs,
):
    # Distributed setup
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    torch.manual_seed(seed)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    tokenizer.pad_token_id = 126081

    model = LLaDOUARPOModel.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)

    # load data
    if "MATH" in local_data_path:
        ds = load_dataset(local_data_path, split="test").with_format("torch")
    elif "gsm8k" in local_data_path:
        ds = load_dataset(local_data_path, split="test", data_dir="main").with_format("torch")
    else:
        raise ValueError(f"Invalid data path: {local_data_path}")

    sampler = DistributedSampler(ds, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False)
    collate_fn = collate_fn_math if "MATH" in local_data_path else collate_fn_gsm8k
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    pbar = tqdm(dl, disable=dist.get_rank() != 0)
    counts = torch.tensor([0, 0], device=device)  # correct, total

    for _, batch in enumerate(pbar):
        answers = batch["answers"]

        outputs = sample_arpo(
            model,
            batch,
            tokenizer,
            device=device,
            inference=no_sample,
            max_steps=steps,
            gen_length=gen_length,
            temperature=0.3,
        )
        responses = tokenizer.batch_decode(outputs["final_tokens"], skip_special_tokens=True)

        if "MATH" in local_data_path:
            counts = judge_answer_MATH(answers, responses, counts)
        elif "gsm8k" in local_data_path:
            counts = judge_answer_GSM8K(answers, responses, counts)

        if dist.get_rank() == 0:
            counts_list = [counts.clone() for _ in range(dist.get_world_size())]
        else:
            counts_list = None

        # gather acc
        torch.distributed.gather(counts, counts_list, dst=0)
        if dist.get_rank() == 0:
            counts_list = torch.stack(counts_list, dim=0).sum(dim=0)
            acc = counts_list[0] / counts_list[1]
            pbar.set_description(f"acc: {acc.item() * 100:.2f}%")

    if dist.get_rank() == 0:
        print(counts_list)
        print("Final Acc: ", acc)


if __name__ == "__main__":
    main()


