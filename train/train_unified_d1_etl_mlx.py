#!/usr/bin/env python3
"""Train native `d1_quickjs_etl` directly on `LiquidAI/d1-2B-MLX-4bit` (`CausalD1View`).

Warm-starts from `lfm3_quickjs_etl` (`83.3%` zero-shot cross-transfer onto `d1-2B`) and fine-tunes
the 24-layer rank-8 LoRA directly against `d1-2B-MLX-4bit`'s exact 4-bit causal `language_model`
using completion-only cross-entropy (`0` quantization gap, `0` MB swap, `$0.00` cloud GPU spend).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

mx.set_cache_limit(512 * 1024 * 1024)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from saccade.sidecar.multi_lora import ResidentD1MultiLoRA
from saccade.sidecar.quickjs_etl import (
    JSON_ETL_SYSTEM_PROMPT,
    QUICKJS_SYSTEM_PROMPT,
    UnifiedD1Scripter,
)

SUITES_DIR = REPO_ROOT / "saccade/data/suites/quickjs_etl"
ADAPTERS_DIR = REPO_ROOT / "saccade/sidecar/adapters"


def format_pair(row: dict[str, Any]) -> tuple[str, str, str]:
    if row["kind"] == "schema_json":
        schema_str = json.dumps(row["json_schema"], separators=(",", ":"))[:2200]
        usr_p = f"JSON Schema:\n{schema_str}\n\nBrowser Source:\n{row['browser_source'][:2800]}"
        tgt = json.dumps(row["gt_json"], separators=(",", ":"))
        return JSON_ETL_SYSTEM_PROMPT, usr_p, tgt
    usr_p = f"Goal: {row['step_goal']}\n\nSnapshot:\n{row['snapshot'][:2800]}"
    return QUICKJS_SYSTEM_PROMPT, usr_p, row["gt_code"]


def eval_completion_nll(lm: Any, tok: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    lm.eval()
    backbone = lm.model
    embed = backbone.embed_tokens
    eos = tok.eos_token or "<|im_end|>"
    nlls, exact_seq, tok_accs = [], [], []
    for r in rows:
        sys_p, usr_p, tgt = format_pair(r)
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        p_ids = tok.encode(prefix, add_special_tokens=False)
        t_ids = tok.encode(tgt + eos, add_special_tokens=False)
        full_ids = (p_ids + t_ids)[:1280]
        comp_start = len(p_ids) - 1
        comp_len = len(full_ids) - len(p_ids)
        if comp_len <= 0:
            continue
        x = mx.array([full_ids], dtype=mx.int32)
        h = backbone(x)[0, comp_start : comp_start + comp_len, :]
        logits = embed.as_linear(h).astype(mx.float32)
        y = mx.array(full_ids[len(p_ids) :], dtype=mx.int32)
        ce = nn.losses.cross_entropy(logits, y, reduction="mean")
        preds = mx.argmax(logits, axis=-1)
        acc = mx.mean((preds == y).astype(mx.float32))
        ex = mx.all(preds == y)
        mx.eval(ce, acc, ex)
        nlls.append(float(ce))
        tok_accs.append(float(acc))
        exact_seq.append(bool(ex))
    return {
        "mean_nll": round(sum(nlls) / max(len(nlls), 1), 4),
        "token_acc": round(sum(tok_accs) / max(len(tok_accs), 1), 4),
        "exact_sequence_rate": round(sum(exact_seq) / max(len(exact_seq), 1), 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    rng = random.Random(20261122)
    mx.random.seed(20261122)

    train_rows = json.loads((SUITES_DIR / "train.json").read_text())
    val_rows = json.loads((SUITES_DIR / "val.json").read_text())
    test_rows = json.loads((SUITES_DIR / "test.json").read_text())

    d1 = ResidentD1MultiLoRA(
        "LiquidAI/d1-2B-MLX-4bit",
        default_rank=8,
        lazy_vision=True,
        slot_dtype=mx.float32,
    )
    warm_dir = ADAPTERS_DIR / "lfm3_quickjs_etl"
    if (warm_dir / "adapters.safetensors").exists():
        d1.register_adapter_dir("warm_etl", warm_dir)
        d1.activate("warm_etl")
        print(f"[Unified-ETL] Warm-started from {warm_dir} (active={mx.get_active_memory()/(1024**3):.3f} GiB)", flush=True)

    lm = d1.model.language_model
    tok = getattr(d1.processor, "tokenizer", d1.processor)
    eos = tok.eos_token or "<|im_end|>"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    base_val = eval_completion_nll(lm, tok, val_rows)
    print(f"[Unified-ETL] Warm-start val: {base_val}", flush=True)

    # Freeze base 4-bit backbone, unfreeze ONLY FP32 master LoRA parameters
    lm.freeze()
    for _, mod in d1._modules.items():
        mod.lora_a = mx.array(mod.lora_a.astype(mx.float32))
        mod.lora_b = mx.array(mod.lora_b.astype(mx.float32))
        mod.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    n_train = sum(v.size for _, v in tree_flatten(lm.trainable_parameters()))
    print(f"[Unified-ETL] Fine-tuning {n_train:,} LoRA params on unified d1-2B causal backbone...", flush=True)

    def completion_loss(lm_mod: Any, x: mx.array, flat_idx: mx.array, y_valid: mx.array) -> mx.array:
        h = lm_mod.model(x)  # [B, T, D]
        B, T, D = h.shape
        h_flat = h.reshape(B * T, D)
        h_comp = mx.take(h_flat, flat_idx, axis=0)
        logits = lm_mod.model.embed_tokens.as_linear(h_comp).astype(mx.float32)
        return nn.losses.cross_entropy(logits, y_valid, reduction="mean")

    loss_and_grad = nn.value_and_grad(lm, completion_loss)
    steps = 90
    lr_sched = optim.cosine_decay(8e-5, steps, end=1e-5)
    opt = optim.AdamW(learning_rate=lr_sched, weight_decay=0.01)

    best_score = base_val["token_acc"] + 0.5 * base_val["exact_sequence_rate"] - 0.05 * base_val["mean_nll"]
    best_tensors = {
        f"{name}.lora_a": mx.array(mod.lora_a)
        for name, mod in d1._modules.items()
    } | {
        f"{name}.lora_b": mx.array(mod.lora_b)
        for name, mod in d1._modules.items()
    }
    mx.eval(best_tensors)

    for step in range(1, steps + 1):
        lm.train()
        batch = rng.sample(train_rows, k=4)
        seqs = []
        valid_positions = []
        valid_targets = []

        raw_pairs = []
        max_len = 0
        for r in batch:
            sys_p, usr_p, tgt = format_pair(r)
            prefix = tok.apply_chat_template(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            p_ids = tok.encode(prefix, add_special_tokens=False)
            t_ids = tok.encode(tgt + eos, add_special_tokens=False)
            full_ids = (p_ids + t_ids)[:1152]
            raw_pairs.append((p_ids, full_ids))
            max_len = max(max_len, len(full_ids))

        for b_idx, (p_ids, full_ids) in enumerate(raw_pairs):
            seqs.append(full_ids + [pad_id] * (max_len - len(full_ids)))
            comp_start = len(p_ids) - 1
            comp_len = len(full_ids) - len(p_ids)
            for k in range(comp_len):
                valid_positions.append(b_idx * max_len + comp_start + k)
                valid_targets.append(full_ids[len(p_ids) + k])

        x_arr = mx.array(seqs, dtype=mx.int32)
        idx_arr = mx.array(valid_positions, dtype=mx.int32)
        y_arr = mx.array(valid_targets, dtype=mx.int32)

        loss, grads = loss_and_grad(lm, x_arr, idx_arr, y_arr)
        grads, _ = optim.clip_grad_norm(grads, max_norm=0.90)
        opt.update(lm, grads)
        mx.eval(loss, lm.trainable_parameters())

        if step % 15 == 0:
            mx.clear_cache()

        if step % 30 == 0 or step == steps:
            vm = eval_completion_nll(lm, tok, val_rows)
            sc = vm["token_acc"] + 0.5 * vm["exact_sequence_rate"] - 0.05 * vm["mean_nll"]
            print(
                f"  [Step {step:2d}/{steps}] loss={float(loss):.4f} | val_nll={vm['mean_nll']:.4f} "
                f"| tok_acc={vm['token_acc']:.3f} | exact_seq={vm['exact_sequence_rate']:.3f} "
                f"| active={mx.get_active_memory()/(1024**3):.3f} GiB",
                flush=True,
            )
            if sc >= best_score:
                best_score = sc
                best_tensors = {
                    f"{name}.lora_a": mx.array(mod.lora_a)
                    for name, mod in d1._modules.items()
                } | {
                    f"{name}.lora_b": mx.array(mod.lora_b)
                    for name, mod in d1._modules.items()
                }
                mx.eval(best_tensors)

    out_dir = ADAPTERS_DIR / "d1_quickjs_etl"
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_tensors = {k: v.astype(mx.bfloat16) for k, v in best_tensors.items()}
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), bf16_tensors)

    # Reload cleanly in FP16 inference mode (`lazy_vision=True`) and evaluate greedy generation on test split!
    del d1, lm
    mx.clear_cache()

    eval_d1 = ResidentD1MultiLoRA("LiquidAI/d1-2B-MLX-4bit", default_rank=8, lazy_vision=True)
    for name in ("dom_grounder", "ui_box", "handoff_guard", "aftermath_tm"):
        eval_d1.register_adapter_dir(name, ADAPTERS_DIR / f"d1_{name}")
    eval_d1.register_adapter_dir("quickjs_etl", out_dir)
    dom_only_gb = round(mx.get_active_memory() / (1024**3), 3)
    print(f"\n[Unified Engine Loaded] All 5 adapters resident (DOM-only, lazy vision): {dom_only_gb} GiB", flush=True)

    scripter = UnifiedD1Scripter(eval_d1, adapter_name="quickjs_etl")
    j_rows = [r for r in test_rows if r["kind"] == "schema_json"]
    c_rows = [r for r in test_rows if r["kind"] != "schema_json"]

    j_parse, j_valid, j_exact, j_ms = 0, 0, 0, []
    for i, r in enumerate(j_rows):
        res = scripter.extract_json(r["browser_source"], r["json_schema"])
        if res["parsed_ok"]:
            j_parse += 1
        if res["schema_valid"]:
            j_valid += 1
        if res["value"] == r["gt_json"]:
            j_exact += 1
        j_ms.append(res["infer_ms"])
        if (i + 1) % 25 == 0:
            mx.clear_cache()

    c_valid, c_zero_halluc, c_ms = 0, 0, []
    for i, r in enumerate(c_rows):
        res = scripter.synthesize_quickjs(r["step_goal"], r["snapshot"])
        if res["valid"]:
            c_valid += 1
        if not res["hallucinated_refs"]:
            c_zero_halluc += 1
        c_ms.append(res["infer_ms"])
        if (i + 1) % 25 == 0:
            mx.clear_cache()

    # Also measure full VLM footprint after paging in vision tower
    vis_page_ms = eval_d1.ensure_vision_loaded()
    full_vlm_gb = round(mx.get_active_memory() / (1024**3), 3)

    meta = {
        "adapter": "d1_quickjs_etl",
        "backbone": "LiquidAI/d1-2B-MLX-4bit",
        "rank": 8,
        "alpha": 16.0,
        "dom_only_resident_gb": dom_only_gb,
        "full_vlm_resident_gb": full_vlm_gb,
        "vision_page_in_ms": round(vis_page_ms, 2),
        "test_n": len(test_rows),
        "schema_json": {
            "n": len(j_rows),
            "json_parse_rate": round(j_parse / len(j_rows), 4),
            "schema_valid_rate": round(j_valid / len(j_rows), 4),
            "exact_value_match": round(j_exact / len(j_rows), 4),
            "median_infer_ms": round(sorted(j_ms)[len(j_ms) // 2], 2),
        },
        "quickjs_code": {
            "n": len(c_rows),
            "valid_syntax_and_refs_rate": round(c_valid / len(c_rows), 4),
            "zero_hallucinated_refs_rate": round(c_zero_halluc / len(c_rows), 4),
            "median_infer_ms": round(sorted(c_ms)[len(c_ms) // 2], 2),
        },
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "adapter_meta.json").write_text(json.dumps(meta, indent=2))
    (REPO_ROOT / "saccade/results/unified_d1_backbone_verification.json").write_text(json.dumps(meta, indent=2))
    print("\n=== Unified Single-Backbone Verification Summary ===", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
