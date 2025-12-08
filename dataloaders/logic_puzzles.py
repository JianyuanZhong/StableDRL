import os
import re
import numpy as np
import torch
from datasets import load_dataset
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
        from .sampler import InfiniteSampler
    except ImportError:
        InfiniteSampler = None

# ===================================================================
# Countdown Utilities & Reward
# ===================================================================

def extract_solution_countdown(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None

def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract numbers from equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        
        # Check if all numbers in equation are in available_numbers
        # We need to handle counts (multiset)
        avail_counts = {}
        for n in available_numbers:
            avail_counts[n] = avail_counts.get(n, 0) + 1
            
        for n in numbers_in_eq:
            if avail_counts.get(n, 0) > 0:
                avail_counts[n] -= 1
            else:
                return False
        return True
    except:
        return False

def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None

def compute_score_countdown(solution_str, ground_truth, score=1.0, format_score=0.1):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution_countdown(solution_str)
    
    if equation is None:
        return 0.0

    if not validate_equation(equation, numbers):
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            return format_score

        if abs(result - target) < 1e-5:
            return score
        else:
            return format_score
    except:
        return format_score

def reward_countdown(prompts, completions, num_generations, device):
    """
    Args:
        prompts: list of dicts or strings. The collate_fn ensures we have 'target' and 'numbers'
                 if we pass the batch object. But the signature here is (prompts, completions, ...).
                 However, usually in these frameworks, 'prompts' might be just the text.
                 Wait, the train loop passes `inputs_batch` as first arg to reward_func if defined.
                 But `generate_and_score_completions_spg` calls:
                 `base_rewards = reward_func(inputs_batch, completions_text, num_generations, device)`
    """
    # Based on networks/llada_svpo.py:
    # base_rewards = reward_func(inputs_batch, completions_text, num_generations, device)
    
    # inputs_batch corresponds to the batch returned by collate_fn
    batch = prompts 
    responses = completions
    
    targets = batch['targets'] # List of targets
    numbers_list = batch['numbers'] # List of number lists
    
    # Expand if num_generations > 1
    # The batch elements are repeated in the generation loop (interleaved)
    # But here we receive the original batch?
    # In llada_svpo.py:
    #   prompts_text_list = inputs_batch.get('problems', [])
    #   ...
    #   prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    #   ...
    #   rewards calculation is done on completions_text which has size [batch_size * num_generations]
    
    # So we need to expand targets and numbers
    targets_expanded = []
    numbers_expanded = []
    
    bs = len(targets)
    for i in range(bs):
        for _ in range(num_generations):
            targets_expanded.append(targets[i])
            numbers_expanded.append(numbers_list[i])
            
    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": targets_expanded[i], "numbers": numbers_expanded[i]}
        scores.append(compute_score_countdown(response, ground_truth))
        
    return torch.tensor(scores, device=device)

def collate_fn_countdown(batch):
    problems = []
    targets = []
    numbers_list = []
    
    instruct = r" Please put the reasoning in <reasoning> tags and the final equation in <answer> tags, e.g. <answer> 1 + 1 </answer>."
    
    for item in batch:
        # item['input'] is numbers string "30,100,93"
        # item['output'] is target string "23"
        
        nums_str = item['input']
        target_str = item['output']
        
        try:
            nums = [int(n.strip()) for n in nums_str.split(',')]
            target = int(target_str.strip())
        except:
            continue
            
        prompt = f"Given numbers {nums}, find an equation that equals {target}.{instruct}"
        problems.append(prompt)
        targets.append(target)
        numbers_list.append(nums)
        
    return {
        "problems": problems,
        "targets": targets,
        "numbers": numbers_list
    }

def _prepare_countdown_dataset(local_path, split, max_rows):
    # Depending on format, it might be a jsonl file or directory containing it
    if os.path.isdir(local_path):
        # Look for jsonl files
        files = [f for f in os.listdir(local_path) if f.endswith('.jsonl')]
        if files:
            data_files = os.path.join(local_path, files[0])
            ds = load_dataset('json', data_files=data_files, split='train') # usually split is train for local files
        else:
             raise ValueError(f"No jsonl file found in {local_path}")
    elif local_path.endswith('.jsonl'):
         ds = load_dataset('json', data_files=local_path, split='train')
    else:
         ds = load_dataset(local_path, split=split)

    ds = ds.select(range(min(len(ds), max_rows)))
    return ds

def load_countdown_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 8,
    max_rows: int = 1e8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 42,
):
    ds = _prepare_countdown_dataset(local_path, split, int(max_rows))
    ds = ds.shuffle(seed=seed)
    
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(ds, rank=rank, num_replicas=num_replicas, shuffle=True, seed=seed)
    else:
        sampler = InfiniteSampler(ds, rank=get_rank(), num_replicas=get_world_size(), shuffle=True, seed=seed)
        
    dl = DataLoader(
        ds, collate_fn=collate_fn_countdown,
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    return dl, reward_countdown

# ===================================================================
# Sudoku Utilities & Reward
# ===================================================================

def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None

def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if len(puzzle) < 16:
        return 0.0

    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        # Check if ground_truth is long enough
        if len(ground_truth) < 16:
             return 0.0

        correct_cells = 0
        for i in empty_indices:
            if i < len(solution_str) and i < len(ground_truth):
                if solution_str[i] == ground_truth[i]:
                    correct_cells += 1
        return correct_cells / len(empty_indices)
    return 0.0

def reward_sudoku(prompts, completions, num_generations, device):
    batch = prompts
    responses = completions
    
    puzzles = batch['puzzles']
    solutions = batch['solutions']
    
    puzzles_expanded = []
    solutions_expanded = []
    
    bs = len(puzzles)
    for i in range(bs):
        for _ in range(num_generations):
            puzzles_expanded.append(puzzles[i])
            solutions_expanded.append(solutions[i])
            
    scores = []
    for i, response in enumerate(responses):
        puzzle = puzzles_expanded[i]
        ground_truth = solutions_expanded[i]
        
        extracted = extract_answer_sudoku(response)
        score = validate_sudoku_solution(extracted, ground_truth, puzzle)
        scores.append(score)
        
    return torch.tensor(scores, device=device)

def collate_fn_sudoku(batch):
    problems = []
    puzzles = []
    solutions = []
    
    instruct = r" Please put the reasoning in <reasoning> tags and the solution string in <answer> tags."
    
    for item in batch:
        puzzle = str(item['Puzzle'])
        if len(puzzle) < 16:
            puzzle = puzzle.zfill(16)

        solution = str(item['Solution'])
        if len(solution) < 16:
            solution = solution.zfill(16)
        
        # Format puzzle for prompt (optional: make it look like a grid)
        # 4x4 sudoku
        # Row 1: x x x x
        # ...
        
        prompt = f"Solve this 4x4 Sudoku puzzle: {puzzle}. Use 0 for empty cells.{instruct}"
        
        problems.append(prompt)
        puzzles.append(puzzle)
        solutions.append(solution)
        
    return {
        "problems": problems,
        "puzzles": puzzles,
        "solutions": solutions
    }

def _prepare_sudoku_dataset(local_path, split, max_rows):
    # Expecting csv
    if os.path.isdir(local_path):
        files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
        if files:
            data_files = os.path.join(local_path, files[0])
            ds = load_dataset('csv', data_files=data_files, split='train')
        else:
             raise ValueError(f"No csv file found in {local_path}")
    elif local_path.endswith('.csv'):
        ds = load_dataset('csv', data_files=local_path, split='train')
    else:
        ds = load_dataset(local_path, split=split)
        
    ds = ds.select(range(min(len(ds), max_rows)))
    return ds

def load_sudoku_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 8,
    max_rows: int = 1e8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 42,
):
    ds = _prepare_sudoku_dataset(local_path, split, int(max_rows))
    ds = ds.shuffle(seed=seed)
    
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(ds, rank=rank, num_replicas=num_replicas, shuffle=True, seed=seed)
    else:
        sampler = InfiniteSampler(ds, rank=get_rank(), num_replicas=get_world_size(), shuffle=True, seed=seed)
        
    dl = DataLoader(
        ds, collate_fn=collate_fn_sudoku,
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    
    return dl, reward_sudoku

