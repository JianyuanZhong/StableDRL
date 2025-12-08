import os 
import torch 
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from torch.utils.data import DataLoader
try:
    from torch_utils.distributed import get_rank, get_world_size
except ImportError:
    def get_rank():
        return 0
    def get_world_size():
        return 1
try:
    from dataloaders.sampler import InfiniteSampler
except ImportError:
    try:
        from .sampler import InfiniteSampler  # type: ignore
    except ImportError:
        InfiniteSampler = None  # type: ignore
from math_verify import parse, verify  # type: ignore
import accelerate


def _load_math_verify():
    """Dynamically load Math-Verify verify function and optional configs.

    Returns a tuple: (verify_fn, parse_fn, LatexExtractionConfig, ExprExtractionConfig)
    Any missing component will be None.
    """
    try:
        mv = import_module('math_verify')
    except (ImportError, ModuleNotFoundError):
        return None, None, None, None

    verify_fn = getattr(mv, 'verify', None)
    parse_fn = getattr(mv, 'parse', None)
    LatexExtractionConfig = getattr(mv, 'LatexExtractionConfig', None)
    ExprExtractionConfig = getattr(mv, 'ExprExtractionConfig', None)
    return verify_fn, parse_fn, LatexExtractionConfig, ExprExtractionConfig


def _simple_extract_boxed(text: str):
    """Heuristic extractor favoring the last \\boxed{...} or LaTeX block."""
    s = str(text)
    boxed_matches = list(re.finditer(r"\\boxed\{([^{}]+)\}", s))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    # Try LaTeX environments: $$...$$ | $...$ | \(...\) | \[...\]
    env_matches = list(
        re.finditer(r"\$\$([^$]+)\$\$|\$([^$]+)\$|\\\(([^)]+)\\\)|\\\[([^\]]+)\\\]", s, flags=re.DOTALL)
    )
    if env_matches:
        groups = [g for g in env_matches[-1].groups() if g is not None]
        if groups:
            return groups[0].strip()

    # Fallback to last number-like token
    number_matches = list(re.finditer(r"(-?\d+(?:\.\d+)?)", s))
    if number_matches:
        return number_matches[-1].group(1)
    return s.strip()


def _normalize_basic(expr: str):
    return str(expr).strip().strip('.').replace(',', '')


def _math_verify_equal(gold_text: str, pred_text: str, timeout_s: float = 5) -> bool:
    """Use Math-Verify parse and verify with timeout; minimal fallback on issues.

    If the verification hangs longer than ``timeout_s`` seconds or raises,
    fall back to simple normalized string equality.
    """

    # def _fallback() -> bool:
    #     return str(gold_text).strip() == str(pred_text).strip()

    # def _work() -> bool:
    #     try:
    #         # Import inside the worker so imports that hang are also timed out
    #         from math_verify import parse, verify  # type: ignore
    #     except Exception:
    #         return _fallback()

    try:
        gold_obj = parse(str(gold_text))
        pred_obj = parse(str(pred_text))
        res = verify(gold_obj, pred_obj, timeout_seconds=timeout_s)
        if isinstance(res, bool):
            return res
        if isinstance(res, dict):
            for key in ('is_correct', 'correct', 'match', 'result', 'equal'):
                if key in res:
                    return bool(res[key])
        return bool(res)
    except Exception:
        print(f"Error verifying: {gold_text} and {pred_text}")
        return False

    # try:
    #     with cf.ThreadPoolExecutor(max_workers=1) as executor:
    #         future = executor.submit(_work)
    #         return future.result(timeout=timeout_s)
    # except cf.TimeoutError:
    #     # Timed out; cancel if possible and return fallback result
    #     try:
    #         future.cancel()  # type: ignore[name-defined]
    #     except Exception:
    #         pass
    #     return _fallback()

def collate_fn_math(batch,):
    problems = []
    answers = []
    levels = []
    instruct = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
    for item in batch:
        problems.append(item['problem'] + instruct)
        answers.append(item['solution'])
        levels.append(item['level'])
    
    return {
        "problems": problems, 
        "answers": answers,
        "levels": levels,
    }
    

def collate_fn_gsm8k(batch,):
    problems = []
    answers = []
    for item in batch:
        problems.append(item['question'])
        answers.append(item['answer'])

    return {
        "problems": problems, 
        "answers": answers
    }


def try_get_level(level: str, default: int = 5):
    try:
        return int(level.split()[-1])
    except Exception:
        return default


def reward_MATH(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(answers, responses)):
        try:
            if _math_verify_equal(ans, res):
                rewards[i] += 1.0
            else:
                rewards[i] -= 1.0
        except Exception:
            rewards[i] -= 1.0

    return rewards


def load_math_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train', 
    num_workers: int = 8,
    max_level: int = None,
    only_level: int = None,
    max_rows: int = 1e8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112,
):
    ds = load_dataset(local_path, split=split)
    # level <= 2: ~1344
    if max_level is not None:
        ds = ds.filter(lambda x: try_get_level(x['level'], 5) <= max_level)
    if only_level is not None:
        ds = ds.filter(lambda x: try_get_level(x['level'], 5) == only_level)
    ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.filter(lambda x: len(x.get('problem', [])) > 0 and len(x.get('problem', '')) < 1500)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(), 
        )
    
    dl = DataLoader(
        ds, collate_fn=collate_fn_math,
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )
    
    return dl, reward_MATH


def extract_answer_gsm8k(answer: str):
    # find the last part starting with '#### xxx'
    return answer.split('####')[-1].strip()


def reward_gsm8k(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    # answer rewards
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, responses)):
        try:
            if _math_verify_equal(ans, res):
                rewards[i] += 1.0
            else:
                rewards[i] -= 1.0
        except Exception:
            rewards[i] -= 1.0

    return rewards


def collate_fn_acereason(batch,):
    problems = []
    answers = []
    instruct = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
    for item in batch:
        problems.append(item['problem'] + instruct)
        answers.append(item['answer'])

    return {
        "problems": problems,
        "answers": answers,
    }


def extract_answer_acereason(answer: str):
    return str(answer).strip()


def reward_acereason(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    ext_ans = [extract_answer_acereason(ans) for ans in answers]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, responses)):
        try:
            if _math_verify_equal(ans, res):
                rewards[i] += 1.0
            else:
                rewards[i] -= 1.0
        except Exception:
            rewards[i] -= 1.0

    return rewards


def load_acereason_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 8,
    max_rows: int = 1e8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112,
):
    jsonl_path = os.path.join(local_path, 'math.jsonl')
    if os.path.exists(jsonl_path):
        print(f'Loading from JSONL file: {jsonl_path}')
        ds = load_dataset('json', data_files=jsonl_path, split='train')
    else:
        # Fallback to original method
        print('JSONL file not found, trying load_dataset...')
        ds = load_dataset(local_path, split=split)
    
    ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.filter(lambda x: len(x.get('problem', '')) > 0 and len(x.get('problem', '')) < 1500)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas,
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(),
        )

    dl = DataLoader(
        ds, collate_fn=collate_fn_acereason,
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )

    return dl, reward_acereason

def load_gsm8k_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train', 
    num_workers: int = 8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112, 
):
    ds = load_dataset(local_path, split=split, data_dir='main')
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(), 
        )

    dl = DataLoader(
        ds, collate_fn=collate_fn_gsm8k,
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )

    return dl, reward_gsm8k


def _prepare_math_dataset(local_path: str, split: str, max_rows: int, max_level: int = None):
    ds = load_dataset(local_path, split=split)
    if max_level is not None:
        ds = ds.filter(lambda x: try_get_level(x.get('level', '5'), 5) <= max_level)
    ds = ds.filter(lambda x: len(x.get('problem', '')) > 0 and len(x.get('problem', '')) < 1500)
    ds = ds.filter(lambda x: len(x.get('problem', '').split()) <= 300)
    ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.map(
        lambda x: {
            'problem': x['problem'],
            'answer': x['solution'],
            'source': 'math',
            'use_boxed': True,
        },
        remove_columns=[c for c in ds.column_names if c not in ('problem', 'solution')],
    )
    print(f"MATH: Number of problems created: {len(ds)}")
    return ds


def _prepare_gsm8k_dataset(local_path: str, split: str, max_rows: int):
    ds = load_dataset(local_path, split=split, data_dir='main')
    ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.map(
        lambda x: {
            'problem': x['question'],
            'answer': extract_answer_gsm8k(x['answer']),
            'source': 'gsm8k',
            'use_boxed': True,
        },
        remove_columns=[c for c in ds.column_names if c not in ('question', 'answer')],
    )
    print(f"GSM8K: Number of problems created: {len(ds)}")
    return ds


def _prepare_acereason_dataset(local_path: str, split: str, max_rows: int):
    jsonl_path = os.path.join(local_path, 'math.jsonl')
    if os.path.exists(jsonl_path):
        base_ds = load_dataset('json', data_files=jsonl_path, split='train')
    else:
        base_ds = load_dataset(local_path, split=split)
    base_ds = base_ds.filter(lambda x: len(x.get('problem', '')) > 0 and len(x.get('problem', '')) < 1500)
    ds = base_ds.select(range(min(len(base_ds), max_rows)))
    ds = ds.map(
        lambda x: {
            'problem': x['problem'],
            'answer': x['answer'],
            'source': 'acereason',
            'use_boxed': True,
        },
        remove_columns=[c for c in ds.column_names if c not in ('problem', 'answer')],
    )
    return ds


def collate_fn_mixed(batch,):
    problems = []
    answers = []
    sources = []
    instruct = r"(Please put the final answer in \\boxed{} tag, i.e. $\\boxed{answer here}$)"
    for item in batch:
        text = item['problem'] + instruct if item.get('use_boxed', False) else item['problem']
        problems.append(text)
        answers.append(item['answer'])
        sources.append(item.get('source', 'unknown'))

    return {
        'problems': problems,
        'answers': answers,
        'sources': sources,
    }


def reward_mixed(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(answers, responses)):
        try:
            if _math_verify_equal(ans, res):
                rewards[i] += 1.0
            else:
                rewards[i] -= 1.0
        except Exception:
            rewards[i] -= 1.0
    return rewards


def load_mixed_datasets_and_reward(
    math_path: str = None,
    gsm8k_path: str = None,
    acereason_path: str = None,
    batch_size: int = 1,
    split: str = 'train',
    num_workers: int = 8,
    max_rows_per_dataset: int = 1e8,
    math_max_level: int = 4,
    mixing_strategy: str = 'concat',  # 'concat' or 'interleave'
    sampling_weights: list = None,  # used when mixing_strategy='interleave'
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112,
):
    datasets_list = []

    if math_path is not None:
        try:
            datasets_list.append(_prepare_math_dataset(math_path, split, int(max_rows_per_dataset), math_max_level))
        except Exception:
            pass
    if gsm8k_path is not None:
        try:
            datasets_list.append(_prepare_gsm8k_dataset(gsm8k_path, split, int(max_rows_per_dataset)))
        except Exception:
            pass
    if acereason_path is not None:
        try:
            datasets_list.append(_prepare_acereason_dataset(acereason_path, split, int(max_rows_per_dataset)))
        except Exception:
            pass

    if len(datasets_list) == 0:
        raise ValueError('No datasets provided to load_mixed_datasets_and_reward')

    if mixing_strategy == 'interleave' and len(datasets_list) > 1:
        if sampling_weights is None:
            sampling_weights = [1.0 / len(datasets_list)] * len(datasets_list)
        ds = interleave_datasets(datasets_list, probabilities=sampling_weights)
    else:
        ds = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]

    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)

    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas,
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(),
        )

    dl = DataLoader(
        ds, collate_fn=collate_fn_mixed,
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )

    return dl, reward_mixed
