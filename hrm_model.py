"""
Hierarchical Reasoning Model (HRM) with ACT, rewritten to mirror the official
implementation in HRM/models/hrm/hrm_act_v1.py but using plain PyTorch modules.

Major features kept:
- Planner/worker hierarchy (H/L cycles) with input injection.
- Optional puzzle ID embeddings prepended to the sequence.
- RoPE or learned positional encodings.
- Bias-free linear layers, RMSNorm, SwiGLU MLP.
- ACT via Q-head (halt/continue logits), exploration, and target bootstrapping.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
except ImportError:
    try:
        from flash_attn import flash_attn_func  # type: ignore[import]
    except ImportError:
        flash_attn_func = None


# -----------------------------------------------------------
# Helpers and layers
# -----------------------------------------------------------

def trunc_normal_(tensor: torch.Tensor, std: float):
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, S, H, D)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q, k


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool, init_std: float):
        super().__init__(in_features, out_features, bias=bias)
        trunc_normal_(self.weight, std=init_std)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__(num_embeddings, embedding_dim)
        trunc_normal_(self.weight, std=init_std)
        self.cast_to = cast_to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dtype != self.cast_to:
            weight = weight.detach().to(self.cast_to)
        return F.embedding(x, weight)


class CastedSparseEmbedding(nn.Module):
    """
    Functional stand-in for the official sparse puzzle embedding.
    Keeps a single weight parameter but exposes the same interface/casting behavior.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        weight = torch.empty((num_embeddings, embedding_dim))
        trunc_normal_(weight, std=init_std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight.to(self.cast_to))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = max(256, round(expansion * hidden_size * 2 / 3))
        self.gate_up = CastedLinear(hidden_size, inter * 2, bias=False, init_std=1.0 / (hidden_size**0.5))
        self.down = CastedLinear(inter, hidden_size, bias=False, init_std=1.0 / (inter**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, causal: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads  # match official HRM defaults
        self.head_dim = hidden_size // num_heads
        self.output_size = self.head_dim * self.num_heads
        self.causal = causal
        self.qkv = CastedLinear(
            hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
            init_std=1.0 / (hidden_size**0.5),
        )
        self.out = CastedLinear(self.output_size, hidden_size, bias=False, init_std=1.0 / (self.output_size**0.5))

    def forward(self, hidden: torch.Tensor, cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, S, _ = hidden.shape
        qkv = self.qkv(hidden)
        qkv = qkv.view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rope(q, k, cos, sin)

        # If KV heads are smaller, repeat to match query heads
        if self.num_key_value_heads != self.num_heads:
            repeat = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        if flash_attn_func is not None and q.dtype in (torch.float16, torch.bfloat16):
            attn = flash_attn_func(q, k, v, causal=self.causal)
            if isinstance(attn, tuple):  # flash-attn version compatibility
                attn = attn[0]
            attn = attn.view(B, S, self.output_size)
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal
            )
            attn = attn.transpose(1, 2).reshape(B, S, self.output_size)

        return self.out(attn)


class HRMBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, rms_eps: float, causal: bool):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, causal=causal)
        self.mlp = SwiGLU(hidden_size, expansion=expansion)
        self.rms_eps = rms_eps

    def forward(self, hidden: torch.Tensor, cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        hidden = rms_norm(hidden + self.attn(hidden, cos_sin), eps=self.rms_eps)
        hidden = rms_norm(hidden + self.mlp(hidden), eps=self.rms_eps)
        return hidden


class ReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden: torch.Tensor, input_injection: torch.Tensor, cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        hidden = hidden + input_injection
        for layer in self.layers:
            hidden = layer(hidden, cos_sin)
        return hidden


# -----------------------------------------------------------
# HRM Core
# -----------------------------------------------------------

@dataclass
class HRMConfig:
    batch_size: int
    seq_len: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str = "rope"  # "rope" or "learned"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int = 4
    halt_exploration_prob: float = 0.1

    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 1
    forward_dtype: str = "bfloat16"


@dataclass
class InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HRMCarry:
    inner_carry: InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel(nn.Module):
    """ACT wrapper mirroring the official HRM ACT V1."""

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config  # keep for Python; TorchScript-friendly scalars are copied below
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embeddings
        self.embed_scale = self.config.hidden_size ** 0.5
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False, init_std=embed_init_std)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True, init_std=embed_init_std)

        # Puzzle embeddings
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                init_std=0.0,
                cast_to=self.forward_dtype,
            )
        else:
            self.puzzle_emb = None

        # Positional encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
            self.embed_pos = None
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
            self.rotary_emb = None
        else:
            raise NotImplementedError(f"pos_encodings={self.config.pos_encodings}")

        # Reasoning layers
        self.H_level = ReasoningModule(
            [HRMBlock(self.config.hidden_size, self.config.num_heads, self.config.expansion, self.config.rms_norm_eps, causal=False) for _ in range(self.config.H_layers)]
        )
        self.L_level = ReasoningModule(
            [HRMBlock(self.config.hidden_size, self.config.num_heads, self.config.expansion, self.config.rms_norm_eps, causal=False) for _ in range(self.config.L_layers)]
        )

        # Initial states
        h_init = torch.empty(self.config.hidden_size, dtype=self.forward_dtype)
        l_init = torch.empty(self.config.hidden_size, dtype=self.forward_dtype)
        trunc_normal_(h_init, std=1.0)
        trunc_normal_(l_init, std=1.0)
        self.register_buffer("H_init", h_init, persistent=True)
        self.register_buffer("L_init", l_init, persistent=True)

        # TorchScript-friendly scalar copies
        self.seq_len_ts = int(self.config.seq_len)
        self.halt_max_steps_ts = int(self.config.halt_max_steps)
        self.H_cycles_ts = int(self.config.H_cycles)
        self.L_cycles_ts = int(self.config.L_cycles)
        self.hidden_size_ts = int(self.config.hidden_size)
        self.puzzle_emb_ndim_ts = int(self.config.puzzle_emb_ndim)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    # ------------- Carry helpers -------------
    def _input_embeddings(self, tokens: torch.Tensor, puzzle_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # Use int32 to be TensorRT-friendly; embedding supports int32/int64 indices.
        emb = self.embed_tokens(tokens.to(torch.int32))

        if (self.puzzle_emb is not None) and (self.puzzle_emb_ndim_ts > 0) and (puzzle_ids is not None):
            puzzle_embedding = self.puzzle_emb(puzzle_ids.to(torch.int32))
            pad = self.puzzle_emb_len * self.hidden_size_ts - puzzle_embedding.shape[-1]
            if pad > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad))
            puzzle_tokens = puzzle_embedding.view(
                -1, self.puzzle_emb_len, self.hidden_size_ts
            )
            emb = torch.cat((puzzle_tokens, emb), dim=1)

        if self.embed_pos is not None:
            emb = 0.707106781 * (emb + self.embed_pos.weight.to(self.forward_dtype))

        return emb * self.embed_scale

    def empty_carry(self, batch_size: int, device: torch.device) -> InnerCarry:
        seq_total = self.seq_len_ts + self.puzzle_emb_len
        return InnerCarry(
            z_H=torch.empty(batch_size, seq_total, self.hidden_size_ts, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, seq_total, self.hidden_size_ts, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry) -> InnerCarry:
        # reset_flag: (B,)
        reset_flag = reset_flag.view(-1, 1, 1)
        return InnerCarry(
            z_H=torch.where(reset_flag, self.H_init, carry.z_H),
            z_L=torch.where(reset_flag, self.L_init, carry.z_L),
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HRMCarry:
        bsz = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return HRMCarry(
            inner_carry=self.empty_carry(bsz, device),
            steps=torch.zeros((bsz,), dtype=torch.int32, device=device),
            halted=torch.ones((bsz,), dtype=torch.bool, device=device),  # start halted so we reset in first pass
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    # ------------- Forward -------------
    def forward(self, carry: HRMCarry, batch: Dict[str, torch.Tensor]):
        # Reset halted sequences
        new_inner = self.reset_carry(carry.halted, carry.inner_carry)
        zero_steps = torch.zeros_like(carry.steps)
        new_steps = torch.where(carry.halted, zero_steps, carry.steps)
        new_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (v.dim() - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Input encoding
        tokens = new_data["inputs"]
        puzzle_ids = new_data.get("puzzle_identifiers")
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
        input_emb = self._input_embeddings(tokens, puzzle_ids)

        # H/L cycles (no grad for all but last step)
        with torch.no_grad():
            z_H, z_L = new_inner.z_H, new_inner.z_L
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    if not (h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
                if h_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, cos_sin)

        # Final grad-carry step
        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin)

        new_inner = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # (B,2)
        q_halt_logits, q_continue_logits = q_logits[:, 0], q_logits[:, 1]

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Halting logic
        with torch.no_grad():
            new_steps = new_steps + torch.ones_like(new_steps)
            is_last_step = new_steps >= self.halt_max_steps_ts
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt, next_q_cont = self.inner_forward_no_grad(new_inner, new_data)
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_cont))
                )

        return HRMCarry(new_inner, new_steps, halted, new_data), outputs

    def inner_forward_no_grad(self, inner: InnerCarry, data: Dict[str, torch.Tensor]):
        with torch.no_grad():
            tokens = data["inputs"]
            puzzle_ids = data.get("puzzle_identifiers")
            cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
            input_emb = self._input_embeddings(tokens, puzzle_ids)

            z_H, z_L = inner.z_H, inner.z_L
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    if not (h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
                if h_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, cos_sin)

            z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
            z_H = self.H_level(z_H, z_L, cos_sin)

            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            return q_logits[:, 0], q_logits[:, 1]

    @torch.jit.export
    def forward_infer(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None):
        """
        TorchScript-friendly inference that unrolls halt_max_steps with halting.
        Returns logits (B, seq_len, vocab), q_halt, q_continue.
        """
        bsz = inputs.size(0)
        device = inputs.device
        puzzle_ids = puzzle_identifiers if puzzle_identifiers is not None else torch.zeros(
            bsz, dtype=torch.int32, device=device
        )
        puzzle_ids = puzzle_ids.to(torch.int32)

        seq_total = self.seq_len_ts + self.puzzle_emb_len
        z_H = self.H_init.unsqueeze(0).expand(bsz, seq_total, -1).clone()
        z_L = self.L_init.unsqueeze(0).expand(bsz, seq_total, -1).clone()
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
        input_emb = self._input_embeddings(inputs, puzzle_ids)

        logits = torch.zeros(bsz, self.seq_len_ts, self.lm_head.out_features, device=device, dtype=self.forward_dtype)
        q_halt = torch.zeros(bsz, device=device)
        q_continue = torch.zeros(bsz, device=device)
        steps = torch.zeros(bsz, dtype=torch.int32, device=device)
        halted = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(self.halt_max_steps_ts):
            # H/L cycles
            for h_step in range(self.H_cycles_ts):
                for l_step in range(self.L_cycles_ts):
                    last_h = h_step == self.H_cycles_ts - 1
                    last_l = l_step == self.L_cycles_ts - 1
                    if not (last_h and last_l):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                if not (h_step == self.H_cycles_ts - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            q_halt = q_logits[:, 0]
            q_continue = q_logits[:, 1]

            steps = steps + torch.ones_like(steps)
            halt_now = (q_halt > q_continue) | (steps >= self.halt_max_steps_ts)
            halted = halted | halt_now
            if bool(torch.all(halted)):
                break

        return logits, q_halt, q_continue

    @torch.jit.export
    def forward_infer_no_halt(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None):
        """
        Fixed-length inference without data-dependent halting (for export/ONNX).
        Runs halt_max_steps_ts segments and ignores the halting condition.
        """
        bsz = inputs.size(0)
        device = inputs.device
        puzzle_ids = puzzle_identifiers if puzzle_identifiers is not None else torch.zeros(
            bsz, dtype=torch.int32, device=device
        )
        puzzle_ids = puzzle_ids.to(torch.int32)

        seq_total = self.seq_len_ts + self.puzzle_emb_len
        z_H = self.H_init.unsqueeze(0).expand(bsz, seq_total, -1).clone()
        z_L = self.L_init.unsqueeze(0).expand(bsz, seq_total, -1).clone()
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
        input_emb = self._input_embeddings(inputs, puzzle_ids)

        logits = torch.zeros(bsz, self.seq_len_ts, self.lm_head.out_features, device=device, dtype=self.forward_dtype)
        q_halt = torch.zeros(bsz, device=device)
        q_continue = torch.zeros(bsz, device=device)

        for _ in range(self.halt_max_steps_ts):
            for h_step in range(self.H_cycles_ts):
                for l_step in range(self.L_cycles_ts):
                    last_h = h_step == self.H_cycles_ts - 1
                    last_l = l_step == self.L_cycles_ts - 1
                    if not (last_h and last_l):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                if not (h_step == self.H_cycles_ts - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
            z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            q_halt = q_logits[:, 0]
            q_continue = q_logits[:, 1]

        return logits, q_halt, q_continue
