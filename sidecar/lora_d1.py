"""Native MLX QLoRA for LiquidAI/d1-2B's 24-layer Tri-Hybrid Architecture.

Uses `float32` master LoRA weights (`lora_a`, `lora_b`) to prevent `float16` AdamW
`eps=1e-8` underflow (`0/0 = NaN` when `lora_b=0` on step 1) while casting activations
seamlessly back to backbone `float16`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

TRI_HYBRID_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "in_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
)


def _infer_dims(base: nn.Module) -> tuple[int, int]:
    if isinstance(base, nn.QuantizedLinear):
        out_dims = int(base.scales.shape[0])
        in_dims = int(base.scales.shape[1] * base.group_size)
        return in_dims, out_dims
    if isinstance(base, nn.Linear):
        out_dims, in_dims = base.weight.shape
        return int(in_dims), int(out_dims)
    raise TypeError(f"Unsupported linear layer type for LoRA: {type(base)}")


class LoRALinear(nn.Module):
    """QLoRA / LoRA wrapper with float32 master adapter parameters for numerical stability."""

    def __init__(
        self,
        base: nn.Linear | nn.QuantizedLinear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        in_dims, out_dims = _infer_dims(base)
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        bound = 1.0 / math.sqrt(in_dims)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(in_dims, rank)).astype(mx.float32)
        self.lora_b = mx.zeros((rank, out_dims), dtype=mx.float32)

        object.__setattr__(self, "_lora_enabled", True)
        self.freeze(keys=["base"], recurse=True)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if not self._lora_enabled:
            return y
        if not self.training and self.lora_a.dtype == y.dtype and self.lora_b.dtype == y.dtype:
            z = x.astype(y.dtype) if x.dtype != y.dtype else x
            delta = (z @ self.lora_a) @ self.lora_b
            return y + (self.scale * delta)
        z = self.dropout(x) if (self.dropout is not None and self.training) else x
        delta = (z.astype(mx.float32) @ self.lora_a.astype(mx.float32)) @ self.lora_b.astype(mx.float32)
        return y + (self.scale * delta).astype(y.dtype)


def inject_tri_hybrid_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_names: Iterable[str] = TRI_HYBRID_TARGETS,
    last_n_layers: int | None = None,
) -> dict[str, int]:
    """Freeze base d1-2B model and inject LoRA across LFM3's tri-hybrid language backbone."""
    target_set = set(target_names)
    model.freeze()

    layers = model.language_model.model.layers
    total_layers = len(layers)
    start_layer = 0 if last_n_layers is None else max(0, total_layers - last_n_layers)

    replaced = 0
    for idx in range(start_layer, total_layers):
        layer = layers[idx]
        op_block = layer.self_attn if layer.is_attention_layer else layer.conv
        for attr in target_set:
            if hasattr(op_block, attr):
                sub = getattr(op_block, attr)
                if isinstance(sub, LoRALinear):
                    sub = sub.base
                if isinstance(sub, (nn.Linear, nn.QuantizedLinear)):
                    setattr(op_block, attr, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                    replaced += 1

        ff = layer.feed_forward
        for attr in target_set:
            if hasattr(ff, attr):
                sub = getattr(ff, attr)
                if isinstance(sub, LoRALinear):
                    sub = sub.base
                if isinstance(sub, (nn.Linear, nn.QuantizedLinear)):
                    setattr(ff, attr, LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout))
                    replaced += 1

    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    return {
        "lora_modules": replaced,
        "trainable_params": int(trainable),
        "total_packed_params": int(total),
        "start_layer": start_layer,
        "total_layers": total_layers,
    }


def strip_all_lora(model: nn.Module) -> None:
    """Restore raw base Linear/QuantizedLinear layers cleanly between ablations."""
    for layer in model.language_model.model.layers:
        for block in (layer.self_attn if layer.is_attention_layer else layer.conv, layer.feed_forward):
            for attr in TRI_HYBRID_TARGETS:
                if hasattr(block, attr):
                    sub = getattr(block, attr)
                    if isinstance(sub, LoRALinear):
                        setattr(block, attr, sub.base)
    model.freeze()


def iter_lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for _, mod in iter_lora_modules(model):
        object.__setattr__(mod, "_lora_enabled", bool(enabled))


def reset_lora_weights(model: nn.Module) -> None:
    for _, mod in iter_lora_modules(model):
        in_dims, _ = mod.lora_a.shape
        bound = 1.0 / math.sqrt(in_dims)
        mod.lora_a = mx.random.uniform(low=-bound, high=bound, shape=mod.lora_a.shape).astype(mx.float32)
        mod.lora_b = mx.zeros_like(mod.lora_b)
    mx.eval(model.trainable_parameters())


def get_lora_state(model: nn.Module) -> dict[str, mx.array]:
    state: dict[str, mx.array] = {}
    for name, mod in iter_lora_modules(model):
        state[f"{name}.lora_a"] = mx.array(mod.lora_a)
        state[f"{name}.lora_b"] = mx.array(mod.lora_b)
    mx.eval(state)
    return state


def set_lora_state(model: nn.Module, state: dict[str, mx.array]) -> None:
    for name, mod in iter_lora_modules(model):
        mod.lora_a = state[f"{name}.lora_a"]
        mod.lora_b = state[f"{name}.lora_b"]
    mx.eval(model.trainable_parameters())


def save_lora_adapter(model: nn.Module, save_dir: str | Path, metadata: dict | None = None) -> Path:
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = get_lora_state(model)
    weights_path = out / "adapters.safetensors"
    mx.save_safetensors(str(weights_path), state)
    if metadata is not None:
        (out / "adapter_meta.json").write_text(json.dumps(metadata, indent=2))
    return weights_path
