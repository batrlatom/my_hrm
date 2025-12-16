"""
Export and benchmark HRM:
- Eager PyTorch
- torch.compile (if available)
- TorchScript (trace)

Reports latency and peak memory (CUDA only) on a random input.
Optionally loads a saved checkpoint to export trained weights.
"""

import argparse
import time
from typing import Optional, Tuple

import torch

from hrm_train import build_model, TrainingConfig


def run_benchmark(fn, inputs, pids, warmup=3, iters=20, use_cuda=False) -> Tuple[float, Optional[int]]:
    if hasattr(fn, "eval"):
        fn.eval()
    with torch.no_grad():
        for _ in range(warmup):
            fn(inputs, pids)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            fn(inputs, pids)
        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
    avg_ms = (end - start) * 1000 / iters
    peak_mem = torch.cuda.max_memory_allocated() if use_cuda else None
    return avg_ms, peak_mem


def maybe_compile(model):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"[compile] skipped: {e}")
    return None


def load_checkpoint_if_any(model: torch.nn.Module, checkpoint: Optional[str], device: torch.device):
    if not checkpoint:
        return
    payload = torch.load(checkpoint, map_location=device)
    state_dict = None
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in payload.values()):
            state_dict = payload
    if state_dict is None:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict or a dict with 'model_state_dict'.")
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint weights from {checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export HRM model to TorchScript/ONNX and benchmark.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a saved model checkpoint (.pt/.pth).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy export input.")
    parser.add_argument("--halt-max-steps", type=int, default=2, help="halt_max_steps to configure the model for export.")
    parser.add_argument("--onnx-path", type=str, default="hrm.onnx", help="Where to write ONNX.")
    parser.add_argument("--ts-path", type=str, default="hrm_ts.pt", help="Where to write TorchScript.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    cfg = TrainingConfig(batch_size=args.batch_size, halt_max_steps=args.halt_max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_model(cfg, device).eval()
    load_checkpoint_if_any(base, args.checkpoint, device)
    for p in base.parameters():
        p.requires_grad_(False)

    inputs = torch.randint(0, 10, (args.batch_size, base.config.seq_len), dtype=torch.int32, device=device)
    pids = torch.zeros(args.batch_size, dtype=torch.int32, device=device)

    # Eager
    eager = lambda inp, pid: base.forward_infer(inp, pid)
    eager_ms, eager_mem = run_benchmark(eager, inputs, pids, use_cuda=device.type == "cuda")
    print(f"Eager: {eager_ms:.2f} ms | peak mem: {eager_mem}")

    # torch.compile
    compiled = maybe_compile(eager)
    if compiled:
        comp_ms, comp_mem = run_benchmark(compiled, inputs, pids, use_cuda=device.type == "cuda")
        print(f"torch.compile: {comp_ms:.2f} ms | peak mem: {comp_mem}")

    # TorchScript trace
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, pid):
            return self.m.forward_infer(x, pid)

    dummy_tokens = torch.zeros(1, base.config.seq_len, dtype=torch.int32, device=device)
    dummy_ids = torch.zeros(1, dtype=torch.int32, device=device)
    ts = torch.jit.trace(Wrapper(base), (dummy_tokens, dummy_ids))
    ts.save(args.ts_path)
    ts_ms, ts_mem = run_benchmark(ts, inputs, pids, use_cuda=device.type == "cuda")
    print(f"TorchScript: {ts_ms:.2f} ms | peak mem: {ts_mem} | saved to {args.ts_path}")

    # ONNX export (static shapes)
    class WrapperONNX(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, pid):
            logits, qh, qc = self.m.forward_infer_no_halt(x, pid)
            return logits, qh, qc

    with torch.no_grad():
        torch.onnx.export(
            WrapperONNX(base),
            (dummy_tokens, dummy_ids),
            args.onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["tokens", "puzzle_ids"],
            output_names=["logits", "q_halt", "q_continue"],
        )
    print(f"Saved ONNX to {args.onnx_path}")

    # Sanity check: compare eager vs TorchScript (logits)
    with torch.no_grad():
        logits_eager, _, _ = base.forward_infer(inputs, pids)
        logits_ts, _, _ = ts(inputs, pids)
    max_diff = (logits_eager - logits_ts).abs().max().item()
    mean_diff = (logits_eager - logits_ts).abs().mean().item()
    print(f"Sanity check (eager vs TS) max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")


if __name__ == "__main__":
    main()
