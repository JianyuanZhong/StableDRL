import os
import torch
from typing import List

import gzip
import json

def read_problems(evalset_file: str) -> dict:
    problems = {}
    if evalset_file.endswith(".gz"):
        with open(evalset_file, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        task = json.loads(line)
                        problems[task["task_id"]] = task
    else:
        with open(evalset_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    task = json.loads(line)
                    problems[task["task_id"]] = task
    return problems

def format_HumanEval_prompt_zero_shot(problems):
    task_ids = list(problems.keys())
    for task_id in task_ids:
        problem = problems[task_id]
        function_name = problem['entry_point']
        prompt = problem['prompt'].rstrip()
        # Preserve the original code skeleton for execution
        problems[task_id]['prompt_code'] = prompt
        HumanEval_prompt = f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

Here is the function to complete:
```python
{prompt}
```
"""
        problems[task_id]['prompt'] = HumanEval_prompt
    return problems


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL_PATH = os.path.join(_CUR_DIR, "datasets", "HumanEval.jsonl.gz")


def ensure_humaneval_file(local_path: str = HUMAN_EVAL_PATH) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    # Prefer the official Hugging Face dataset
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("openai/openai_humaneval", split="test")
        with gzip.open(local_path, "wt", encoding="utf-8") as gzfp:
            for ex in ds:
                rec = {
                    "task_id": ex.get("task_id"),
                    "prompt": ex.get("prompt"),
                    "entry_point": ex.get("entry_point"),
                    "test": ex.get("test"),
                    # Keep canonical_solution for downstream tools that may expect it
                    "canonical_solution": ex.get("canonical_solution", ""),
                }
                gzfp.write(json.dumps(rec))
                gzfp.write("\n")
        return local_path
    except Exception:
        pass
    # As a last resort, try to copy any existing local fallback
    repo_root = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
    fallback = os.path.join(repo_root, "LLaDOU", "datasets", "HumanEval.jsonl.gz")
    try:
        if os.path.exists(fallback) and os.path.getsize(fallback) > 0:
            with open(fallback, "rb") as src, open(local_path, "wb") as dst:
                dst.write(src.read())
            return local_path
    except Exception:
        pass
    # Fallback to provided path even if missing; caller may handle
    return local_path


class HumanEvalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_examples: int = 0, add_reasoning: bool = False, subsample: int = -1):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning

        # Ensure we have the official HumanEval problems cached locally
        local_he_path = ensure_humaneval_file(HUMAN_EVAL_PATH)
        problems = read_problems(local_he_path)
        problems = format_HumanEval_prompt_zero_shot(problems)

        self.task_ids: List[str] = sorted(list(problems.keys()))
        self.prompts: List[str] = [problems[tid]["prompt"] for tid in self.task_ids]

        # Optional subsample handling
        import numpy as np
        if subsample != -1:
            assert subsample <= len(self.task_ids)
            indices = np.random.choice(len(self.task_ids), subsample, replace=False)
            self.task_ids = [self.task_ids[i] for i in indices]
            self.prompts = [self.prompts[i] for i in indices]

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        prompt = self.prompts[idx]
        return prompt, task_id

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        task_ids = [item[1] for item in batch]
        input_ids = self.tokenizer(prompts, padding_side="left", return_tensors="pt", padding="longest").input_ids
        questions = ["humaneval_task" for _ in batch]
        answers = [None for _ in batch]
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts, "task_ids": task_ids}


