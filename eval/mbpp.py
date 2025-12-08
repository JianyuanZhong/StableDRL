import os
import json
import torch
from typing import List, Dict

from datasets import load_dataset

def format_MBPP_prompt_zero_shot(problem, need_code=False):
    function_name = problem["test_list"][0].split(' ')[1].split('(')[0]
    import ast
    first_assert = problem["test_list"][0]
    num_args = 0
    try:
        tree = ast.parse(first_assert.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == function_name:
                num_args = len(node.args)
                break
    except Exception:
        num_args = 2

    param_names = ", ".join([f"input_param_{i+1}" for i in range(num_args)])
    function_declaration = f"def {function_name}({param_names}):"
    # Align prompt style with HumanEval: instruction + code block with skeleton
    return f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

Here is the function to complete:
```python
{function_declaration}
    \"\"\"{problem["text"]}\n    \"\"\"
```
"""


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MBPP_TEST_JSONL = os.path.join(_CUR_DIR, "datasets", "mbpp_test.jsonl")


def ensure_mbpp_problem_file(path: str = MBPP_TEST_JSONL, split: str = "sanitized") -> str:
    """
    Ensure a local problems jsonl exists for evaluator. Build from HF Muennighoff/mbpp.

    The evaluator expects a jsonl where each line has at least:
      {"task_id": int, "test": List[str]}

    For MBPP we join the provided test_list (and optional test_setup_code) per example.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    # Always build from HF to guarantee verifiability
    ds = load_dataset("Muennighoff/mbpp", split="test") if split == "full" else load_dataset("Muennighoff/mbpp", "sanitized", split="test")

    with open(path, "w", encoding="utf-8") as fout:
        for row in ds:
            task_id = int(row.get("task_id", row.get("id", 0)))
            # For sanitized: use test_imports (list) + test_list (list)
            imports = row.get("test_imports", []) or []
            if not isinstance(imports, list):
                imports = []
            test_list = row.get("test_list", []) or []
            if not isinstance(test_list, list):
                test_list = []
            tests: List[str] = []
            # Normalize imports: ensure strings and valid imports
            for imp in imports:
                s = str(imp).strip()
                if s:
                    tests.append(s)
            for t in test_list:
                s = str(t).strip()
                if s:
                    tests.append(s)

            problem: Dict = {"task_id": task_id, "test": tests}
            fout.write(json.dumps(problem) + "\n")

    return path


class MBPPDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_examples: int = 0, add_reasoning: bool = False, subsample: int = -1):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning

        # Use sanitized split for clearer prompts/tests; users can switch by editing ensure_mbpp_problem_file
        self.ds = load_dataset("Muennighoff/mbpp", "sanitized", split="test")

        # Build prompts and ids
        self.task_ids: List[int] = []
        self.prompts: List[str] = []
        for row in self.ds:
            self.task_ids.append(int(row.get("task_id", row.get("id", 0))))
            # Convert HF row into our zero-shot prompt
            problem = {
                "text": row.get("prompt", row.get("text", "")),
                "test_list": row.get("test_list", []),
                "code": row.get("code", ""),
            }
            self.prompts.append(format_MBPP_prompt_zero_shot(problem))

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
        # For compatibility with evaluator, include placeholders for fields not used by MBPP
        questions = ["mbpp_task" for _ in batch]
        answers = [None for _ in batch]
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts, "task_ids": task_ids}


