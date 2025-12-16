import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from my_hrm.hrm_data import SUDOKU_SEQ_LEN, VOCAB_SIZE, generate_sudoku_batch, load_sudoku_dataset
    from my_hrm.hrm_model import HRMConfig, HierarchicalReasoningModel
except ImportError:
    from hrm_data import SUDOKU_SEQ_LEN, VOCAB_SIZE, generate_sudoku_batch, load_sudoku_dataset
    from hrm_model import HRMConfig, HierarchicalReasoningModel


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    total_steps: int = 2000
    log_interval: int = 100
    difficulty: float = 0.5

    # HRM settings (mirror official defaults, scaled down)
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 2
    L_layers: int = 2
    hidden_size: int = 256
    expansion: float = 4.0
    num_heads: int = 4
    pos_encodings: str = "rope"  # "rope" or "learned"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int = 4
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"

    dataset_path: str = ""  # optional path to a pre-generated dataset (.pt from hrm_make_dataset.py)
    eval_interval: int = 500  # steps between eval passes (0 to disable)
    eval_batches: int = 30  # number of batches averaged during eval
    run_root: str = "runs/sudoku"  # base folder for saving checkpoints
    save_every: int = 0  # steps between checkpoint saves (0 to disable periodic saves)
 

def build_model(cfg: TrainingConfig, device: torch.device) -> HierarchicalReasoningModel:
    hrm_cfg = HRMConfig(
        batch_size=cfg.batch_size,
        seq_len=SUDOKU_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        H_cycles=cfg.H_cycles,
        L_cycles=cfg.L_cycles,
        H_layers=cfg.H_layers,
        L_layers=cfg.L_layers,
        hidden_size=cfg.hidden_size,
        expansion=cfg.expansion,
        num_heads=cfg.num_heads,
        pos_encodings=cfg.pos_encodings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        halt_max_steps=cfg.halt_max_steps,
        halt_exploration_prob=cfg.halt_exploration_prob,
        forward_dtype=cfg.forward_dtype,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
    )
    return HierarchicalReasoningModel(hrm_cfg).to(device)


def run_eval(
    model: HierarchicalReasoningModel,
    cfg: TrainingConfig,
    device: torch.device,
    dataset_inputs: Optional[torch.Tensor],
    dataset_targets: Optional[torch.Tensor],
    step: int,
):
    """Simple eval loop averaging accuracies over a few batches."""
    model.eval()
    token_acc_sum = 0.0
    board_acc_sum = 0.0
    batches = max(1, cfg.eval_batches)

    with torch.no_grad():
        for _ in range(batches):
            if dataset_inputs is not None:
                idx = torch.randint(0, dataset_inputs.size(0), (cfg.batch_size,), device="cpu")
                tokens = dataset_inputs[idx]
                targets = dataset_targets[idx]
            else:
                tokens, targets = generate_sudoku_batch(cfg.batch_size, difficulty=cfg.difficulty)

            batch = {
                "inputs": tokens.to(device),
                "puzzle_identifiers": torch.zeros(cfg.batch_size, dtype=torch.long, device=device),
            }
            labels = targets.to(device)

            carry = model.initial_carry(batch)
            outputs = None
            for _ in range(cfg.halt_max_steps):
                carry, outputs = model(carry, batch)
                if carry.halted.all():
                    break
            preds = outputs["logits"].argmax(dim=-1)
            token_acc_sum += (preds == labels).float().mean().item()
            board_acc_sum += (preds == labels).all(dim=1).float().mean().item()

    print(
        f"[Eval @ step {step:04d}] Token Acc: {100 * token_acc_sum / batches:.1f}% | "
        f"Board Acc: {100 * board_acc_sum / batches:.2f}%"
    )
    model.train()


def prepare_run_dir(run_root: str) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                n = int(child.name.split("_")[1])
                max_n = max(max_n, n)
            except (IndexError, ValueError):
                continue
    run_dir = root / f"run_{max_n + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(run_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, name: str):
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path = run_dir / name
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")


def train_hrm_logic(cfg: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    run_dir = prepare_run_dir(cfg.run_root)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Run directory: {run_dir}")

    print("Training Hierarchical Reasoning Model (HRM)...")
    print(
        f"H_cycles: {cfg.H_cycles} | L_cycles: {cfg.L_cycles} | halt_max_steps: {cfg.halt_max_steps} | difficulty: {cfg.difficulty}"
    )

    dataset_inputs = dataset_targets = None
    dataset_meta = {}
    if cfg.dataset_path:
        ds = load_sudoku_dataset(cfg.dataset_path)
        dataset_inputs = ds["inputs"]
        dataset_targets = ds["targets"]
        dataset_meta = {k: v for k, v in ds.items() if k not in ("inputs", "targets")}
        if dataset_inputs.size(0) < cfg.batch_size:
            raise ValueError(f"Dataset has only {dataset_inputs.size(0)} samples, smaller than batch_size={cfg.batch_size}.")
        print(f"Loaded dataset from {cfg.dataset_path} ({dataset_inputs.size(0)} samples, difficulty={dataset_meta.get('difficulty', 'n/a')}, seed={dataset_meta.get('seed', 'n/a')})")

    for step in range(cfg.total_steps):
        if dataset_inputs is not None:
            idx = torch.randint(0, dataset_inputs.size(0), (cfg.batch_size,), device="cpu")
            tokens = dataset_inputs[idx]
            targets = dataset_targets[idx]
        else:
            tokens, targets = generate_sudoku_batch(cfg.batch_size, difficulty=cfg.difficulty)
        batch: Dict[str, torch.Tensor] = {
            "inputs": tokens.to(device),
            "puzzle_identifiers": torch.zeros(cfg.batch_size, dtype=torch.long, device=device),
        }
        labels = targets.to(device)

        optimizer.zero_grad()

        carry = model.initial_carry(batch)
        outputs = None
        for _ in range(cfg.halt_max_steps):
            carry, outputs = model(carry, batch)
            if carry.halted.all():
                break

        assert outputs is not None
        logits = outputs["logits"]
        q_halt_logits = outputs["q_halt_logits"]
        q_continue_logits = outputs["q_continue_logits"]

        # Token loss over all positions
        lm_loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))

        # Sequence correctness label for Q head (all tokens correct)
        with torch.no_grad():
            seq_correct = (logits.argmax(dim=-1) == labels).all(dim=1).float()

        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits, seq_correct)
        q_continue_loss = 0.0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"])

        loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                token_acc = (preds == labels).float().mean().item() * 100
                board_acc = (preds == labels).all(dim=1).float().mean().item() * 100
            print(
                f"Step {step:04d} | Loss: {loss.item():.4f} | Token Acc: {token_acc:.1f}% | "
                f"Board Acc: {board_acc:.2f}% | Halted: {carry.halted.float().mean().item():.2f}"
            )
        if cfg.eval_interval > 0 and (step + 1) % cfg.eval_interval == 0:
            run_eval(model, cfg, device, dataset_inputs, dataset_targets, step + 1)
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            save_checkpoint(run_dir, model, optimizer, step + 1, name=f"step_{step + 1:04d}.pt")

    print("\n--- Inference Test ---")
    model.eval()
    if dataset_inputs is not None and dataset_inputs.size(0) >= 2:
        idx = torch.randint(0, dataset_inputs.size(0), (2,), device="cpu")
        test_tokens = dataset_inputs[idx]
        test_targets = dataset_targets[idx]
    else:
        test_tokens, test_targets = generate_sudoku_batch(2, difficulty=cfg.difficulty)
    test_batch = {
        "inputs": test_tokens.to(device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device=device),
    }
    with torch.no_grad():
        carry = model.initial_carry(test_batch)
        out = None
        for _ in range(cfg.halt_max_steps):
            carry, out = model(carry, test_batch)
            if carry.halted.all():
                break
        assert out is not None
        preds = out["logits"].argmax(dim=-1)

    print(f"Input (0 is blank):\n{test_tokens[0].tolist()}")
    print(f"Target:\n{test_targets[0].tolist()}")
    print(f"HRM Prediction:\n{preds[0].cpu().tolist()}")
    save_checkpoint(run_dir, model, optimizer, cfg.total_steps, name="final.pt")


if __name__ == "__main__":
    cfg = TrainingConfig()
    train_hrm_logic(cfg)
