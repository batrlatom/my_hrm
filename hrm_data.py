import torch
from sudoku import Sudoku
import random
from typing import Dict, Tuple

# Sudoku constants
SUDOKU_BASE = 3
SUDOKU_SIZE = SUDOKU_BASE * SUDOKU_BASE  # 9
SUDOKU_SEQ_LEN = SUDOKU_SIZE * SUDOKU_SIZE  # 81
VOCAB_SIZE = 10  # 0 = blank, 1-9 = digits
MASK_TOKEN = 0


def flatten_board(board):
    """Flatten 9x9 board into length-81 list, replacing None with 0."""
    flat = []
    for row in board:
        for cell in row:
            flat.append(cell if cell is not None else MASK_TOKEN)
    return flat


def generate_sudoku_batch(batch_size: int, difficulty: float = 0.5):
    """
    Generate real Sudoku puzzles and solutions.
    Inputs: puzzle digits with blanks -> 0, digits 1-9 kept as 1-9.
    Targets: solved digits (1-9); no blanks in solutions.
    """
    inputs = []
    targets = []

    for _ in range(batch_size):
        puzzle = Sudoku(SUDOKU_BASE).difficulty(difficulty)
        solution = puzzle.solve()

        puzzle_flat = flatten_board(puzzle.board)
        solution_flat = flatten_board(solution.board)

        inputs.append(puzzle_flat)
        targets.append(solution_flat)

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets


def generate_sudoku_dataset(num_samples: int, difficulty: float = 0.5, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a full dataset of Sudoku puzzles/solutions in one go.
    Returns tensors on CPU: (num_samples, 81).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    return generate_sudoku_batch(num_samples, difficulty=difficulty)


def save_sudoku_dataset(path: str, num_samples: int, difficulty: float = 0.5, seed: int = None) -> Dict[str, torch.Tensor]:
    inputs, targets = generate_sudoku_dataset(num_samples, difficulty=difficulty, seed=seed)
    payload = {
        "inputs": inputs,
        "targets": targets,
        "difficulty": difficulty,
        "num_samples": num_samples,
        "seed": seed,
    }
    torch.save(payload, path)
    return payload


def load_sudoku_dataset(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if not all(k in data for k in ("inputs", "targets")):
        raise ValueError("Dataset file missing required keys 'inputs' and 'targets'.")
    return data
