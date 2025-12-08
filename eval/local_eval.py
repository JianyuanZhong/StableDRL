import os
import json
import gzip
import re
from typing import Iterable, Dict, List
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from humaneval import read_problems as _read_problems
from humaneval import format_HumanEval_prompt_zero_shot
from execution_local import check_correctness


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r", encoding="utf-8")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()
    return results


def _normalize_task_id(task_id):
    # Try int first, then str
    try:
        return int(task_id)
    except Exception:
        return str(task_id)


def _get_problem_safe(problems: Dict, task_id):
    # Direct
    if task_id in problems:
        return problems[task_id]
    # Normalize
    norm = _normalize_task_id(task_id)
    if norm in problems:
        return problems[norm]
    # Try str/int conversions explicitly
    try:
        as_int = int(task_id)
        if as_int in problems:
            return problems[as_int]
    except Exception:
        pass
    as_str = str(task_id)
    if as_str in problems:
        return problems[as_str]
    return None


def process_humaneval_test(sample, problems, example_test=False, is_mbpp=False, language="python"):
    match = re.search(r"```python\n(.*?)```", sample.get("completion", ""), re.DOTALL)
    sample_completion = match.group(1).strip() if match else sample.get("completion", "")

    task_id = sample["task_id"]
    problem = _get_problem_safe(problems, task_id)
    if problem is None:
        return None
    if is_mbpp:
        tests = problem.get("test") or []
        if not isinstance(tests, list):
            return None
        return sample_completion + "\n" + "\n".join(tests)

    if example_test and "example_test" in problem and problem["example_test"] != "":
        test = problem["example_test"]
    else:
        test = problem["test"]
    code = sample_completion
    # Use the raw function skeleton (prompt_code) instead of the instruction-wrapped prompt
    prompt_code = problem.get("prompt_code", problem.get("prompt", "")) 
    # Ensure a newline between skeleton and generated code to avoid '"""def' concatenation
    entry_point = problem.get("entry_point", "")
    test_string = prompt_code + "\n" + code + "\n" + test + "\n" + (f"check({entry_point})" if entry_point else "")
    return test_string


def evaluate_functional_correctness(
    input_file: str = None,
    tmp_dir: str = "./",
    n_workers: int = 32,
    timeout: float = 3.0,
    problem_file: str = None,
    out_dir: str = None,
    k: List[int] = [1, 10, 100],
    test_groundtruth: bool = False,
    example_test: bool = False,
    is_mbpp: bool = False,
    language: str = "python",
):
    problems = _read_problems(problem_file)
    if not is_mbpp:
        problems = format_HumanEval_prompt_zero_shot(problems)

    sample_jsonl = stream_jsonl_all(input_file)

    from collections import Counter
    results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()

        for sample in tqdm(sample_jsonl):
            try:
                task_id = sample["task_id"]
                lang = language
                if not is_mbpp and lang == "javascript":
                    lang = "js"
                if is_mbpp:
                    lang = "python"
                tmp_dir_ = os.path.join(tmp_dir, lang, "evaluation")
                sample["task_id"] = task_id
                test_code = process_humaneval_test(sample, problems, example_test, is_mbpp, language)
                if not test_code:
                    continue
                sample["test_code"] = test_code
                completion_id_ = sample.get("completion_id", completion_id[task_id])
                args = (task_id, sample, lang, timeout, tmp_dir_, completion_id_)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
            except Exception:
                # Skip problematic sample
                continue

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    if input_file is not None:
        input_samples = list(stream_jsonl_all(input_file))
        output_path = input_file.replace(".jsonl", ".scored.jsonl")
        task_result_map = {
            task_id: {r["completion_id"]: r["passed"] for _, r in results[task_id]}
            for task_id in results
        }

        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in input_samples:
                task_id = sample["task_id"]
                completion_id = sample.get("completion_id", 0)
                passed = task_result_map.get(task_id, {}).get(completion_id, None)
                if passed is not None:
                    new_sample = OrderedDict()
                    new_sample["passed"] = passed
                    for key, value in sample.items():
                        new_sample[key] = value
                    fout.write(json.dumps(new_sample) + "\n")
                else:
                    fout.write(json.dumps(sample) + "\n")
    return {}


