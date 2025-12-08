import re
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

try:
    from torch_utils.distributed import get_rank, get_world_size
except Exception:
    def get_rank():
        return 0

    def get_world_size():
        return 1

try:
    from dataloaders.sampler import InfiniteSampler
except Exception:
    try:
        from .sampler import InfiniteSampler  # type: ignore
    except Exception:
        InfiniteSampler = None  # type: ignore

# try:
    # Prefer packaged executor that mirrors local eval behavior
from custom_humaneval.execution import check_correctness  # type: ignore
# except Exception:
#     # Fallback: minimal local executor if packaged one is unavailable
#     from d1.eval.execution_local import check_correctness  # type: ignore


def _extract_python_code(text: str) -> str:
    s = str(text)
    m = list(re.finditer(r"```python\s*([\s\S]*?)```", s, flags=re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    m = list(re.finditer(r"```\s*([\s\S]*?)```", s))
    if m:
        return m[-1].group(1).strip()
    return s.strip()


def _patch_test_code(test_code: str) -> str:
    lines: List[str] = []
    for line in str(test_code).splitlines():
        if re.match(r"\s*from\s+solution\s+import\s+", line):
            continue
        if re.match(r"\s*import\s+solution\b", line):
            continue
        lines.append(line)
    patched = "\n".join(lines)
    harness = (
        "\n\n"
        "# Auto-discover and execute pytest-style test functions\n"
        "for __name, __obj in list(globals().items()):\n"
        "    if callable(__obj) and isinstance(__name, str) and __name.startswith('test_'):\n"
        "        __obj()\n"
    )
    return patched + harness


def _build_combined_test_code(solution_code: str, test_code: str) -> str:
    code = str(solution_code).strip()
    if not code:
        # Ensure at least a valid module
        code = "# empty solution"  # pragma: no cover
    patched_tests = _patch_test_code(test_code)
    return code + "\n\n" + patched_tests


def reward_kodcode(
    batch: Dict,
    responses: List[str],
    num_generations: int,
    device,
    timeout: float = 3.0,
) -> torch.Tensor:
    test_codes: List[str] = batch.get("test_codes", [])
    task_ids: List[str] = batch.get("task_ids", [])
    # Duplicate per-generation to align 1:K mapping
    test_codes = test_codes * num_generations
    task_ids = (task_ids * num_generations) if len(task_ids) > 0 else [str(i) for i in range(len(responses))]

    rewards = torch.zeros(len(responses), device=device)
    for i, (resp, tcode, tid) in enumerate(zip(responses, test_codes, task_ids)):
        try:
            solution_code = _extract_python_code(resp)
            combined = _build_combined_test_code(solution_code, tcode)
            sample = {"test_code": combined}
            result = check_correctness(task_id=str(tid), sample=sample, language_type="python", timeout=timeout, tmp_dir=None, completion_id=i)

            if bool(result.get("passed", False)) or str(result.get("result", "")) == "passed":
                rewards[i] = 1.0
            else:
                rewards[i] = -1.0
        except Exception:
            rewards[i] = -1.0
    return rewards


def _extract_fn_from_test_info(row: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    info = row.get("test_info")
    meta: Optional[Dict] = None
    if isinstance(info, list) and len(info) > 0:
        for item in info:
            if isinstance(item, dict) and ("function_declaration" in item or "function_name" in item):
                meta = item
                break
        if meta is None and isinstance(info[0], dict):
            meta = info[0]
    elif isinstance(info, dict):
        meta = info
    if meta is None:
        return None, None, None
    fn_name = None
    try:
        fn_name = meta.get("function_name")
    except Exception:
        fn_name = None
    decl = None
    try:
        decl = meta.get("function_declaration")
    except Exception:
        decl = None
    ds = None
    try:
        ds = meta.get("docstring")
    except Exception:
        ds = None
    return decl, ds, fn_name


def _extract_fn_from_question(question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract a top-level function declaration and inline docstring from a code-like question block."""
    q = str(question)
    # Prefer the last fenced python code block if present
    m = list(re.finditer(r"```python\s*([\s\S]*?)```", q, flags=re.IGNORECASE))
    code = m[-1].group(1) if m else q
    # Find first top-level def
    mdef = re.search(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^\)]*\)\s*:\s*$", code, flags=re.MULTILINE)
    if not mdef:
        return None, None, None
    fn_name = mdef.group(1)
    # Capture the full def line
    def_line = mdef.group(0)
    # Try to capture immediate triple-quoted docstring indented one level
    post = code[mdef.end():]
    mdq = re.search(r"^[\ \t]*\"\"\"([\s\S]*?)\"\"\"", post, flags=re.MULTILINE)
    ds = mdq.group(1).strip() if mdq else None
    return def_line, ds, fn_name


def _extract_fn_from_test_code(test_code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer a function name from `from solution import ...` and synthesize a minimal declaration."""
    s = str(test_code)
    m = re.search(r"from\s+solution\s+import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)", s)
    if not m:
        return None, None, None
    names = [n.strip() for n in m.group(1).split(',') if n.strip()]
    for name in names:
        if name and name[0].islower():
            return f"def {name}():", None, name
    if len(names) > 0:
        return f"def {names[0]}():", None, names[0]
    return None, None, None


def _format_prompt_humaneval_style(row: Dict) -> Optional[str]:
    """Try to build a HumanEval-style prompt from KodCode test_info metadata.

    Prefers a function skeleton constructed from test_info, similar to humaneval.py.
    Returns None if insufficient metadata is present.
    """
    # 1) Prefer explicit test_info
    declaration, ds, function_name = _extract_fn_from_test_info(row)
    # 2) Otherwise extract from code-like question blocks
    if declaration is None:
        q = row.get("question", "")
        declaration, ds, function_name = _extract_fn_from_question(q)
    # 3) Otherwise attempt to infer from test_code imports
    if declaration is None:
        declaration, ds, function_name = _extract_fn_from_test_code(row.get("test_code", ""))
    if declaration is None:
        return None

    docstring = ""
    if isinstance(ds, str) and ds.strip():
        docstring = f"    \"\"\"{ds}\n    \"\"\""

    prompt_code = declaration
    if docstring:
        prompt_code = prompt_code + "\n" + docstring

    # Match humaneval.py wording and structure
    fn_display = function_name if function_name else "the function"
    prompt = (
        f"You are an expert Python programmer. Your task is to complete the implementation of a function named `{fn_display}`.\n\n"
        "Here is the function to complete:\n"
        "```python\n"
        f"{prompt_code}\n"
        "```\n"
    )
    return prompt


def _format_prompt(question: str, row: Optional[Dict] = None) -> str:
    # Prefer HumanEval-style structured prompt when possible
    if isinstance(row, dict):
        he_prompt = _format_prompt_humaneval_style(row)
        print(f"he_prompt: {he_prompt}")
        if isinstance(he_prompt, str) and he_prompt.strip():
            return he_prompt

    q = str(question).strip()
    prompt = (
        "You are an expert Python programmer.\n"
        "Write a correct, efficient Python solution.\n"
        "Return only valid Python code (no explanations).\n\n"
        f"Task:\n{q}\n"
    )
    return prompt


def collate_fn_kodcode(batch):
    problems: List[str] = []
    test_codes: List[str] = []
    task_ids: List[str] = []
    for row in batch:
        problems.append(_format_prompt(row.get("question", ""), row))
        test_codes.append(row.get("test_code", ""))
        task_ids.append(str(row.get("question_id", row.get("conversation_id", row.get("id", "0")))))
    return {
        "problems": problems,
        "test_codes": test_codes,
        "task_ids": task_ids,
    }


def load_kodcode_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = "train",
    num_workers: int = 8,
    max_rows: int = int(1e8),
    rank: Optional[int] = None,
    num_replicas: Optional[int] = None,
    seed: int = 112,
    subset: Optional[str] = None,
):
    """
    Load KodCode SFT-4o dataset (Python coding tasks with unit tests) and provide a reward based on test execution.

    Args:
        local_path: HuggingFace datasets path, e.g., "KodCode/KodCode-V1-SFT-4o".
        split: Dataset split to use (e.g., "train").
        subset: Optional subset filter (dataset has a "subset" column). If provided, keep only matching rows.
    Returns:
        (DataLoader, reward_fn)
    """
    ds = load_dataset(local_path, split=split)

    # Basic filtering: keep examples with question and test_code
    ds = ds.filter(lambda x: isinstance(x.get("question", None), str) and len(x["question"]) > 0)
    ds = ds.filter(lambda x: isinstance(x.get("test_code", None), str) and len(x["test_code"]) > 0)

    if subset is not None:
        ds = ds.filter(lambda x: str(x.get("subset", "")).lower() == str(subset).lower())

    # Limit rows
    n = min(len(ds), int(max_rows))
    if n < len(ds):
        ds = ds.select(range(n))

    # Shuffle for RL
    ds = ds.shuffle(seed=seed)

    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(ds, rank=rank, num_replicas=num_replicas)
    else:
        sampler = InfiniteSampler(ds, rank=get_rank(), num_replicas=get_world_size())

    dl = DataLoader(
        ds,
        collate_fn=collate_fn_kodcode,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dl, reward_kodcode


