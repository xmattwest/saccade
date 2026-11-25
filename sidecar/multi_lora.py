"""Unified Resident Multi-LoRA Engine for `LiquidAI/d1-2B-MLX-4bit` (`0.98 GiB` DOM / `1.34 GiB` VLM / `0.00 GiB` Idle).

Serves ALL FIVE Saccade browser capabilities off a single 4-bit LFM3 backbone:
  1. `d1` System-One set-attention slot readouts (`dom_grounder`, `ui_box`, `handoff_guard`, `aftermath_tm`, or `None`)
  2. Causal `LFM3` text generation (`d1_quickjs_etl` via zero-copy `CausalD1View`)
  3. On-demand vision tower paging (`lazy_vision=True` keeps DOM-only footprint under `1.0 GiB` and pages in
     the `379 MB` vision tower + projector from local safetensors in `< 60 ms` only when screenshots arrive)
  4. Idle auto-unload (`unload()` drops active Metal memory to `0.00 GiB` and wakes transparently on next call)
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    from .lora_d1 import (
        LoRALinear,
        inject_tri_hybrid_lora,
        iter_lora_modules,
        set_lora_enabled,
    )
except ImportError:
    from lora_d1 import (
        LoRALinear,
        inject_tri_hybrid_lora,
        iter_lora_modules,
        set_lora_enabled,
    )


def normalize_lora_key_to_mlx(key: str) -> str:
    """Map PyTorch AutoModel d1-2B or LFM3-1.7B LoRA tensor keys onto MLX module paths."""
    for pfx in ("model.language_model.layers.", "model.layers.", "layers."):
        if key.startswith(pfx):
            return "language_model.model.layers." + key[len(pfx) :]
    return key


class CausalD1View:
    """Zero-copy causal generation view over a loaded `d1` model (`Lfm3VlModel` methods without duplicating weights)."""

    def __init__(self, d1_model: Any):
        self._m = d1_model

    def __getattr__(self, key: str) -> Any:
        return getattr(self._m, key)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from mlx_vlm.models.lfm3_vl.lfm3_vl import Model as Lfm3VlModel

        return Lfm3VlModel.__call__(self._m, *args, **kwargs)

    def get_input_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        from mlx_vlm.models.lfm3_vl.lfm3_vl import Model as Lfm3VlModel

        return Lfm3VlModel.get_input_embeddings(self._m, *args, **kwargs)


@dataclass
class AdapterSlot:
    name: str
    rank: int
    alpha: float
    tensors: dict[str, mx.array]
    adapter_dir: Path | None = None
    calibrated_temperature: float = 1.0
    logit_bias: float = 0.0
    abstain_tau: float = 0.55
    alert_tau: float = 0.50
    meta: dict[str, Any] = field(default_factory=dict)


class ResidentD1MultiLoRA:
    """Holds one unified 4-bit `d1-2B` backbone with FP16 hot-swap LoRAs, lazy vision paging, and idle sleep."""

    def __init__(
        self,
        model_id: str = "LiquidAI/d1-2B-MLX-4bit",
        default_rank: int = 8,
        max_pixels: int = 384 * 256,
        attach_lora_slots: bool = True,
        lazy_vision: bool = True,
        slot_dtype: Any = mx.float16,
    ):
        os.environ["D1_VISION_MAX_PIXELS"] = str(max_pixels)
        mx.set_cache_limit(512 * 1024 * 1024)

        self.model_id = model_id
        self.max_pixels = max_pixels
        self.default_rank = default_rank
        self.attach_lora_slots = attach_lora_slots
        self.lazy_vision = lazy_vision
        self.slot_dtype = slot_dtype

        self.model: Any = None
        self.processor: Any = None
        self.engine: Any = None
        self.causal_view: CausalD1View | None = None
        self._model_path: Path | None = None
        self.vision_loaded: bool = False
        self.load_ms: float = 0.0
        self.vision_page_ms: float = 0.0
        self.lora_info: dict[str, Any] = {"lora_modules": 0, "trainable_params": 0}

        self._modules: dict[str, LoRALinear] = {}
        self.adapters: dict[str, AdapterSlot] = {}
        self._adapter_dirs: dict[str, Path] = {}
        self.active_adapter: str | None = None

        self._load_backbone()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _load_backbone(self) -> float:
        import mlx_vlm

        t0 = time.perf_counter()
        self.model, self.processor = mlx_vlm.load(self.model_id)
        self.engine = self.model.engine
        self.causal_view = CausalD1View(self.model)
        mp = getattr(self.model, "model_path", None)
        self._model_path = Path(mp) if mp else None
        self.vision_loaded = True

        if self.attach_lora_slots:
            self.lora_info = inject_tri_hybrid_lora(
                self.model,
                rank=self.default_rank,
                alpha=self.default_rank * 2.0,
                dropout=0.0,
                last_n_layers=None,
            )
            set_lora_enabled(self.model, False)
            for mod in iter_lora_modules(self.model):
                mod[1].lora_a = mod[1].lora_a.astype(self.slot_dtype)
                mod[1].lora_b = mod[1].lora_b.astype(self.slot_dtype)
        else:
            self.lora_info = {"lora_modules": 0, "trainable_params": 0}

        self.model.eval()
        self._modules = dict(iter_lora_modules(self.model))
        self.active_adapter = None

        # Re-hydrate any previously registered adapter dirs after an idle wake
        for name, adir in list(self._adapter_dirs.items()):
            if name not in self.adapters:
                self._load_adapter_slot(name, adir)

        if self.lazy_vision and self._model_path is not None:
            self.evict_vision()
        else:
            mx.clear_cache()

        self.load_ms = (time.perf_counter() - t0) * 1000.0
        return self.load_ms

    def ensure_loaded(self) -> float:
        if self.model is not None:
            return 0.0
        return self._load_backbone()

    def evict_vision(self) -> bool:
        """Drop the `379 MB` vision tower + projector while keeping the `973 MB` text backbone & LoRAs resident."""
        if self.model is None or not self.vision_loaded:
            return False
        self.model.vision_tower = None
        self.model.multi_modal_projector = None
        self.vision_loaded = False
        gc.collect()
        mx.clear_cache()
        return True

    def ensure_vision_loaded(self) -> float:
        """Page `vision_tower` + `multi_modal_projector` back in from local safetensors (`~40-70 ms` on M4)."""
        self.ensure_loaded()
        if self.vision_loaded and getattr(self.model, "vision_tower", None) is not None:
            return 0.0
        if self._model_path is None:
            return 0.0

        from mlx_vlm.models.lfm3_vl.lfm3_vl import MultiModalProjector
        from mlx_vlm.models.lfm3_vl.vision import VisionModel

        t0 = time.perf_counter()
        cfg = self.model.config
        vt = VisionModel(cfg.vision_config)
        mm = MultiModalProjector(cfg)

        qcfg = getattr(cfg, "quantization", None) or {"group_size": 64, "bits": 4}
        bits = int(qcfg.get("bits", 4))
        gsize = int(qcfg.get("group_size", 64))

        vt_weights: dict[str, mx.array] = {}
        mm_weights: dict[str, mx.array] = {}
        for sf in sorted(self._model_path.glob("*.safetensors")):
            loaded = mx.load(str(sf))
            for k, v in loaded.items():
                if k.startswith("vision_tower."):
                    vt_weights[k[len("vision_tower.") :]] = v
                elif k.startswith("multi_modal_projector."):
                    mm_weights[k[len("multi_modal_projector.") :]] = v

        nn.quantize(vt, group_size=gsize, bits=bits, class_predicate=lambda p, m: f"{p}.scales" in vt_weights)
        nn.quantize(mm, group_size=gsize, bits=bits, class_predicate=lambda p, m: f"{p}.scales" in mm_weights)
        vt.load_weights(list(vt_weights.items()))
        mm.load_weights(list(mm_weights.items()))
        vt.eval()
        mm.eval()
        mx.eval(vt.parameters(), mm.parameters())

        self.model.vision_tower = vt
        self.model.multi_modal_projector = mm
        self.vision_loaded = True
        mx.clear_cache()
        self.vision_page_ms = (time.perf_counter() - t0) * 1000.0
        return self.vision_page_ms

    def unload(self) -> float:
        """Full idle sleep: release backbone, vision tower, and adapter tensors down to `~0.00 GiB`."""
        t0 = time.perf_counter()
        self.model = None
        self.processor = None
        self.engine = None
        self.causal_view = None
        self._modules.clear()
        self.adapters.clear()
        self.active_adapter = None
        self.vision_loaded = False
        gc.collect()
        mx.clear_cache()
        return (time.perf_counter() - t0) * 1000.0

    def _load_adapter_slot(self, name: str, adapter_dir: Path) -> AdapterSlot:
        meta_path = adapter_dir / "adapter_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        raw_tensors = mx.load(str(adapter_dir / "adapters.safetensors"))
        norm_tensors: dict[str, mx.array] = {}
        for k, v in raw_tensors.items():
            nk = normalize_lora_key_to_mlx(k)
            norm_tensors[nk] = v.astype(self.slot_dtype)
        mx.eval(norm_tensors)

        first_a = next(v for k, v in norm_tensors.items() if k.endswith(".lora_a"))
        rank = int(first_a.shape[1])
        alpha = float(meta.get("alpha", rank * 2.0))

        slot = AdapterSlot(
            name=name,
            rank=rank,
            alpha=alpha,
            tensors=norm_tensors,
            adapter_dir=adapter_dir,
            calibrated_temperature=float(meta.get("calibrated_temperature", 1.0)),
            logit_bias=float(meta.get("logit_bias", 0.0)),
            abstain_tau=float(meta.get("abstain_tau", 0.55)),
            alert_tau=float(meta.get("alert_tau", 0.50)),
            meta=meta,
        )
        self.adapters[name] = slot
        return slot

    def register_adapter_dir(self, name: str, adapter_dir: str | Path) -> AdapterSlot:
        adir = Path(adapter_dir)
        self._adapter_dirs[name] = adir
        self.ensure_loaded()
        return self._load_adapter_slot(name, adir)

    def activate(self, name: str | None) -> float:
        """Activate a named LoRA adapter or disable LoRA (`None` -> pristine base d1-2B). Returns swap latency in ms."""
        self.ensure_loaded()
        t0 = time.perf_counter()
        if name is None:
            if self.active_adapter is not None:
                set_lora_enabled(self.model, False)
                self.active_adapter = None
            return (time.perf_counter() - t0) * 1000.0

        if name not in self.adapters:
            set_lora_enabled(self.model, False)
            self.active_adapter = None
            return (time.perf_counter() - t0) * 1000.0

        if self.active_adapter == name:
            return (time.perf_counter() - t0) * 1000.0

        slot = self.adapters[name]
        scale = slot.alpha / max(slot.rank, 1)
        for mod_name, mod in self._modules.items():
            ka = f"{mod_name}.lora_a"
            kb = f"{mod_name}.lora_b"
            if ka in slot.tensors and kb in slot.tensors:
                mod.lora_a = slot.tensors[ka]
                mod.lora_b = slot.tensors[kb]
                mod.rank = slot.rank
                mod.alpha = slot.alpha
                mod.scale = scale
                object.__setattr__(mod, "_lora_enabled", True)
            else:
                object.__setattr__(mod, "_lora_enabled", False)
        self.active_adapter = name
        return (time.perf_counter() - t0) * 1000.0

    def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        media: list[Any] | None = None,
        adapter: str | None = None,
        abstain_below: float | None = None,
    ) -> dict[str, Any]:
        wake_ms = self.ensure_loaded()
        vis_ms = self.ensure_vision_loaded() if media else 0.0
        swap_ms = self.activate(adapter)
        t0 = time.perf_counter()
        out = self.engine.system_one(
            state,
            questions,
            media=media,
            abstain_below=abstain_below,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        out["timing_ms"] = {
            "wake_ms": round(wake_ms, 2),
            "vision_page_ms": round(vis_ms, 2),
            "swap_ms": round(swap_ms, 3),
            "infer_ms": round(infer_ms, 2),
        }
        out["active_adapter"] = self.active_adapter
        return out

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        adapter: str | None = "quickjs_etl",
        max_tokens: int = 240,
        assistant_prefix: str = "",
    ) -> tuple[str, dict[str, float]]:
        """Run causal decoding directly on the unified `d1-2B` LFM3 backbone via `CausalD1View`."""
        import mlx_vlm

        wake_ms = self.ensure_loaded()
        use_adap = adapter if (adapter and adapter in self.adapters) else None
        swap_ms = self.activate(use_adap)
        tok = getattr(self.processor, "tokenizer", self.processor)
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + assistant_prefix
        t0 = time.perf_counter()
        res = mlx_vlm.generate(
            self.causal_view,
            self.processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        text = assistant_prefix + getattr(res, "text", str(res))
        return text, {
            "wake_ms": round(wake_ms, 2),
            "swap_ms": round(swap_ms, 3),
            "infer_ms": round(infer_ms, 2),
        }

    @staticmethod
    def calibrate_prob(raw_prob: float, temperature: float = 1.0, logit_bias: float = 0.0) -> float:
        p = min(max(float(raw_prob), 1e-6), 1.0 - 1e-6)
        raw_logit = math.log(p / (1.0 - p))
        z = max(-30.0, min(30.0, (raw_logit + logit_bias) / max(temperature, 1e-4)))
        return float(1.0 / (1.0 + math.exp(-z)))
