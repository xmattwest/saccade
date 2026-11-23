#!/usr/bin/env python3
"""Gate 1 & Gate 3 Apple Silicon M4 Benchmark Harness (`1,300` locked test items).

Memory-guarded for 16 GB Apple Silicon:
  - Enforces `mx.set_cache_limit(512 * 1024 * 1024)` at startup
  - Calls `mx.clear_cache()` + `gc.collect()` every 25 items and between `[1/5]..[5/5]`
  - Checkpoints each completed suite immediately to disk (`--fresh` to overwrite, otherwise resumes)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

mx.set_cache_limit(512 * 1024 * 1024)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from saccade.sidecar.aftermath import judge_aftermath
from saccade.sidecar.firewall import check_pre_cdp_firewall
from saccade.sidecar.foveate import foveate_snapshot
from saccade.sidecar.grounder import ground_visual_box, pick_dom_ref
from saccade.sidecar.multi_lora import ResidentD1MultiLoRA
from saccade.sidecar.quickjs_etl import ResidentLFM3Scripter, ensure_mlx_4bit_lfm3

SACCADE_ROOT = REPO_ROOT / "saccade"
SUITES_DIR = SACCADE_ROOT / "data/suites"
RESULTS_DIR = SACCADE_ROOT / "results"


def _flush_mlx() -> None:
    mx.clear_cache()
    gc.collect()


def compute_box_iou(pred: dict[str, float], gt: dict[str, float]) -> float:
    ix1 = max(pred["left"], gt["left"])
    iy1 = max(pred["top"], gt["top"])
    ix2 = min(pred["right"], gt["right"])
    iy2 = min(pred["bottom"], gt["bottom"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_p = max(0.0, pred["right"] - pred["left"]) * max(0.0, pred["bottom"] - pred["top"])
    area_g = max(0.0, gt["right"] - gt["left"]) * max(0.0, gt["bottom"] - gt["top"])
    denom = area_p + area_g - inter
    return float(inter / denom) if denom > 1e-8 else 0.0


def eval_dom_grounding(d1: ResidentD1MultiLoRA, adapter: str | None) -> dict[str, Any]:
    rows = json.loads((SUITES_DIR / "dom_grounding/test.json").read_text())
    latencies: list[float] = []
    fov_latencies: list[float] = []
    compression_ratios: list[float] = []
    fov_retained = 0
    fov_answerable = 0

    records: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        if idx > 0 and idx % 25 == 0:
            _flush_mlx()
        if idx < 100:
            fov = foveate_snapshot(d1, r["snapshot"], r["subgoal"], adapter=adapter)
            if fov["timing_ms"]["infer_ms"] > 0:
                fov_latencies.append(fov["timing_ms"]["infer_ms"])
            compression_ratios.append(fov["compression_ratio"])
            if not r["is_abstain_case"]:
                fov_answerable += 1
                if f"[ref={r['target_ref']}]" in fov["foveated_snapshot"]:
                    fov_retained += 1

        out = pick_dom_ref(
            d1,
            subgoal=r["subgoal"],
            snapshot_text=r["snapshot"],
            candidates=r["candidates"],
            adapter=adapter,
            abstain_tau=0.01,
        )
        latencies.append(out["timing_ms"]["infer_ms"])
        records.append(
            {
                "gt": r["target_ref"],
                "is_abstain": bool(r["is_abstain_case"]),
                "top_cand": out["top_candidate"],
                "top_p": out["confidence"],
                "abstain_p": out["abstain_prob"],
            }
        )
    _flush_mlx()

    ans_recs = [x for x in records if not x["is_abstain"]]
    abs_recs = [x for x in records if x["is_abstain"]]
    raw_ans_acc = float(np.mean([x["top_cand"] == x["gt"] for x in ans_recs]))

    curve: list[dict[str, float]] = []
    for tau in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85):
        local_exec = [x for x in records if (x["top_p"] >= tau and x["abstain_p"] < (1.0 - tau * 0.6))]
        cov = len(local_exec) / len(records)
        prec = (
            float(np.mean([(not x["is_abstain"]) and (x["top_cand"] == x["gt"]) for x in local_exec]))
            if local_exec
            else 1.0
        )
        abs_correct = float(
            np.mean([not (x["top_p"] >= tau and x["abstain_p"] < (1.0 - tau * 0.6)) for x in abs_recs])
        )
        overall_system_acc = float(
            np.mean(
                [
                    ((not x["is_abstain"]) and (x["top_cand"] == x["gt"]))
                    if (x["top_p"] >= tau and x["abstain_p"] < (1.0 - tau * 0.6))
                    else True
                    for x in records
                ]
            )
        )
        curve.append(
            {
                "tau": tau,
                "local_coverage": round(cov, 4),
                "local_precision": round(prec, 4),
                "unanswerable_abstain_recall": round(abs_correct, 4),
                "hybrid_system_upper_bound": round(overall_system_acc, 4),
            }
        )

    cov_at_90p = max((c["local_coverage"] for c in curve if c["local_precision"] >= 0.90), default=0.0)
    cov_at_85p = max((c["local_coverage"] for c in curve if c["local_precision"] >= 0.85), default=0.0)

    return {
        "n_test": len(rows),
        "n_answerable": len(ans_recs),
        "n_unanswerable": len(abs_recs),
        "answerable_top1_acc": round(raw_ans_acc, 4),
        "coverage_at_90pct_precision": round(cov_at_90p, 4),
        "coverage_at_85pct_precision": round(cov_at_85p, 4),
        "foveation_mean_compression": round(float(np.mean(compression_ratios)), 4),
        "foveation_target_retention": round(fov_retained / max(fov_answerable, 1), 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "foveation_p50_ms": round(float(np.percentile(fov_latencies, 50)), 2) if fov_latencies else 0.0,
        "abstention_curve": curve,
    }


def eval_ui_box(d1: ResidentD1MultiLoRA, adapter: str | None) -> dict[str, Any]:
    suite_dir = SUITES_DIR / "ui_box"
    rows = json.loads((suite_dir / "test.json").read_text())
    latencies: list[float] = []
    hits_all: list[bool] = []
    hits_text: list[bool] = []
    hits_icon: list[bool] = []
    hits_web: list[bool] = []
    ious: list[float] = []

    for idx, r in enumerate(rows):
        if idx > 0 and idx % 25 == 0:
            _flush_mlx()
        with Image.open(suite_dir / r["image_rel_path"]) as raw_im:
            im = raw_im.convert("RGB")
        out = ground_visual_box(d1, r["instruction"], im, adapter=adapter)
        latencies.append(out["timing_ms"]["infer_ms"])
        gt = r["gt_box"]
        if out["found"] and out["norm_xy"] is not None and out["box"] is not None:
            cx, cy = out["norm_xy"]
            hit = (gt["left"] - 0.015 <= cx <= gt["right"] + 0.015) and (
                gt["top"] - 0.015 <= cy <= gt["bottom"] + 0.015
            )
            iou_val = compute_box_iou(out["box"], gt)
        else:
            hit = False
            iou_val = 0.0

        hits_all.append(hit)
        ious.append(iou_val)
        if r["data_type"] == "icon":
            hits_icon.append(hit)
        else:
            hits_text.append(hit)
        if r["split_origin"] == "web":
            hits_web.append(hit)
    _flush_mlx()

    return {
        "n_test": len(rows),
        "center_hit_acc": round(float(np.mean(hits_all)), 4),
        "web_center_hit_acc": round(float(np.mean(hits_web)), 4) if hits_web else 0.0,
        "text_target_hit_acc": round(float(np.mean(hits_text)), 4) if hits_text else 0.0,
        "icon_target_hit_acc": round(float(np.mean(hits_icon)), 4) if hits_icon else 0.0,
        "mean_iou": round(float(np.mean(ious)), 4),
        "iou_ge_0p3": round(float(np.mean([v >= 0.30 for v in ious])), 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def eval_firewall(d1: ResidentD1MultiLoRA, adapter: str | None) -> dict[str, Any]:
    suite_dir = SUITES_DIR / "firewall"
    rows = json.loads((suite_dir / "test.json").read_text())
    val_rows = json.loads((suite_dir / "val.json").read_text())

    def score_split(split_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
        probs, labels, cats, lats = [], [], [], []
        for idx, r in enumerate(split_rows):
            if idx > 0 and idx % 25 == 0:
                _flush_mlx()
            with Image.open(suite_dir / r["crop_rel_path"]) as raw_im:
                crop = raw_im.convert("RGB")
            out = check_pre_cdp_firewall(
                d1,
                user_goal=r["user_goal"],
                tool_name=r["tool_name"],
                tool_args=r["tool_args"],
                target_aria=r["target_aria"],
                page_excerpt=r["page_excerpt"],
                crop=crop,
                adapter=adapter,
            )
            probs.append(out["risk_prob"])
            labels.append(int(r["label_require_handoff"]))
            cats.append(r["category"])
            lats.append(out["timing_ms"]["infer_ms"])
        _flush_mlx()
        return np.array(probs), np.array(labels), cats, lats

    val_p, val_y, _, _ = score_split(val_rows)
    val_neg = np.sort(val_p[val_y == 0])
    tau_3far = float(np.quantile(val_neg, 0.97)) if len(val_neg) else 0.50

    te_p, te_y, te_cats, lats = score_split(rows)
    auroc = float(roc_auc_score(te_y, te_p))
    ap = float(average_precision_score(te_y, te_p))

    def op_point(tau: float) -> dict[str, float]:
        pred = te_p >= tau
        rec_all = float(pred[te_y == 1].mean())
        far_benign = float(pred[te_y == 0].mean())
        dest_mask = np.array([c == "destructive_mutation" for c in te_cats])
        inj_mask = np.array([c == "prompt_injection" for c in te_cats])
        return {
            "tau": round(tau, 4),
            "hazard_recall": round(rec_all, 4),
            "destructive_recall": round(float(pred[dest_mask].mean()), 4),
            "injection_recall": round(float(pred[inj_mask].mean()), 4),
            "benign_false_handoff_rate": round(far_benign, 4),
        }

    curve = [op_point(t) for t in (0.30, 0.50, 0.70, tau_3far)]
    return {
        "n_test": len(rows),
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "default_0p50": op_point(0.50),
        "calibrated_3pct_far": op_point(tau_3far),
        "operating_curve": curve,
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
    }


def eval_aftermath(d1: ResidentD1MultiLoRA, adapter: str | None) -> dict[str, Any]:
    suite_dir = SUITES_DIR / "aftermath"
    rows = json.loads((suite_dir / "test.json").read_text())
    y_true, y_pred, lats = [], [], []
    classes = ["SETTLED_OK", "SPINNER_LOADING", "INLINE_ERROR_TOAST", "AUTH_OR_BOT_WALL"]

    for idx, r in enumerate(rows):
        if idx > 0 and idx % 25 == 0:
            _flush_mlx()
        with Image.open(suite_dir / r["before_rel_path"]) as rb:
            b_im = rb.convert("RGB")
        with Image.open(suite_dir / r["after_rel_path"]) as ra:
            a_im = ra.convert("RGB")
        out = judge_aftermath(
            d1,
            action_desc=r["action_desc"],
            aria_diff=r["aria_diff"],
            before_crop=b_im,
            after_crop=a_im,
            adapter=adapter,
        )
        y_true.append(r["gt_status"])
        y_pred.append(out["status"])
        lats.append(out["timing_ms"]["infer_ms"])
    _flush_mlx()

    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    per_class = {
        c: round(
            float(np.mean([p == c for t, p in zip(y_true, y_pred) if t == c])),
            4,
        )
        for c in classes
    }
    return {
        "n_test": len(rows),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_recall": per_class,
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
    }


def eval_quickjs_etl(scripter: ResidentLFM3Scripter) -> dict[str, Any]:
    rows = json.loads((SUITES_DIR / "quickjs_etl/test.json").read_text())
    schema_parsed, schema_valid, schema_exact, schema_lats = [], [], [], []
    qjs_valid, qjs_no_halluc, qjs_exact_refs, qjs_literals, qjs_lats = [], [], [], [], []

    for idx, r in enumerate(rows):
        if idx > 0 and idx % 25 == 0:
            _flush_mlx()
        if r["kind"] == "schema_json":
            out = scripter.extract_json(r["browser_source"], r["json_schema"])
            schema_parsed.append(bool(out["parsed_ok"]))
            schema_valid.append(bool(out["schema_valid"]))
            schema_exact.append(out["value"] == r["gt_json"])
            schema_lats.append(out["infer_ms"])
        else:
            out = scripter.synthesize_quickjs(r["step_goal"], r["snapshot"])
            qjs_valid.append(bool(out["valid"]))
            qjs_no_halluc.append(len(out["hallucinated_refs"]) == 0)
            used = set(out.get("used_refs", []))
            req = set(r["required_refs"])
            qjs_exact_refs.append(req.issubset(used) and len(out["hallucinated_refs"]) == 0)
            qjs_literals.append(all(lit in out["code"] for lit in r["required_literals"]))
            qjs_lats.append(out["infer_ms"])
    _flush_mlx()

    return {
        "n_test": len(rows),
        "schema_json": {
            "n": len(schema_valid),
            "json_parse_rate": round(float(np.mean(schema_parsed)), 4),
            "schema_valid_rate": round(float(np.mean(schema_valid)), 4),
            "exact_value_match": round(float(np.mean(schema_exact)), 4),
            "latency_p50_ms": round(float(np.percentile(schema_lats, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(schema_lats, 95)), 2),
        },
        "quickjs_code": {
            "n": len(qjs_valid),
            "valid_syntax_and_refs_rate": round(float(np.mean(qjs_valid)), 4),
            "zero_hallucinated_refs_rate": round(float(np.mean(qjs_no_halluc)), 4),
            "all_required_refs_copied_rate": round(float(np.mean(qjs_exact_refs)), 4),
            "all_literals_preserved_rate": round(float(np.mean(qjs_literals)), 4),
            "latency_p50_ms": round(float(np.percentile(qjs_lats, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(qjs_lats, 95)), 2),
        },
        "combined_pass_rate": round(
            float(np.mean(schema_valid + [v and r for v, r in zip(qjs_valid, qjs_exact_refs)])), 4
        ),
    }


def save_checkpoint(out_path: Path, state: dict[str, Any]) -> None:
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero-shot", "adapted"], default="zero-shot")
    parser.add_argument("--adapters-dir", default=str(SACCADE_ROOT / "sidecar/adapters"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(
        args.out
        or (
            RESULTS_DIR / ("gate1_m4_baseline.json" if args.mode == "zero-shot" else "gate3_m4_adapted.json")
        )
    )

    summary: dict[str, Any] = {}
    if out_path.exists() and not args.fresh:
        try:
            summary = json.loads(out_path.read_text())
            print(f"[Resume] Loaded existing checkpoint keys: {list(summary.keys())}", flush=True)
        except Exception:
            summary = {}

    summary.setdefault("mode", args.mode)
    summary.setdefault("hardware", "Apple M4 (16 GB unified memory, MLX 4-bit, 512 MB cache cap)")
    summary.setdefault(
        "models",
        {
            "d1": "LiquidAI/d1-2B-MLX-4bit",
            "lfm3": "LiquidAI/LFM3-1.7B-MLX-4bit",
        },
    )
    summary.setdefault("memory_and_swap", {})

    t_start = time.perf_counter()
    adapters_root = Path(args.adapters_dir)
    mx.reset_peak_memory()

    need_d1 = any(k not in summary for k in ("dom_grounding", "ui_box", "firewall", "aftermath"))
    d1: ResidentD1MultiLoRA | None = None
    swap_bench_ms = float(summary["memory_and_swap"].get("multi_lora_median_swap_ms", 0.0))

    if need_d1:
        print(f"=== Loading Resident d1-2B-MLX-4bit ({args.mode}, 512 MB cache cap) ===", flush=True)
        d1 = ResidentD1MultiLoRA(
            model_id="LiquidAI/d1-2B-MLX-4bit",
            default_rank=8,
            attach_lora_slots=(args.mode == "adapted"),
        )
        if args.mode == "adapted":
            for name in ("dom_grounder", "ui_box", "handoff_guard", "aftermath_tm"):
                adir = adapters_root / f"d1_{name}"
                if (adir / "adapters.safetensors").exists():
                    d1.register_adapter_dir(name, adir)
                    print(f"  Registered d1 LoRA: {name} from {adir.name}", flush=True)
            names_cycle = list(d1.adapters.keys()) + [None]
            if len(names_cycle) > 1:
                t_sw = [d1.activate(names_cycle[i % len(names_cycle)]) for i in range(60)]
                swap_bench_ms = float(np.median(t_sw[5:]))

        d1_only_gb = mx.get_active_memory() / (1024**3)
        summary["memory_and_swap"]["d1_resident_gb"] = round(d1_only_gb, 3)
        summary["memory_and_swap"]["multi_lora_median_swap_ms"] = round(swap_bench_ms, 4)
        save_checkpoint(out_path, summary)

        use_dom = "dom_grounder" if args.mode == "adapted" and "dom_grounder" in d1.adapters else None
        use_box = "ui_box" if args.mode == "adapted" and "ui_box" in d1.adapters else None
        use_fw = "handoff_guard" if args.mode == "adapted" and "handoff_guard" in d1.adapters else None
        use_am = "aftermath_tm" if args.mode == "adapted" and "aftermath_tm" in d1.adapters else None

        if "dom_grounding" not in summary:
            print("[1/5] Evaluating DOM Ref Grounding & Foveation (n = 300)...", flush=True)
            res_dom = eval_dom_grounding(d1, adapter=use_dom)
            summary["dom_grounding"] = res_dom
            save_checkpoint(out_path, summary)
            _flush_mlx()
            print(
                f"  -> [Saved 1/5] Answerable Top-1={res_dom['answerable_top1_acc']:.3f}, "
                f"Cov@90%Prec={res_dom['coverage_at_90pct_precision']:.3f}, P50={res_dom['latency_p50_ms']} ms",
                flush=True,
            )

        if "ui_box" not in summary:
            print("[2/5] Evaluating Visual UI Box Grounding on ScreenSpot (n = 300)...", flush=True)
            res_box = eval_ui_box(d1, adapter=use_box)
            summary["ui_box"] = res_box
            save_checkpoint(out_path, summary)
            _flush_mlx()
            print(
                f"  -> [Saved 2/5] ClickHit={res_box['center_hit_acc']:.3f} "
                f"(text={res_box['text_target_hit_acc']:.3f}, icon={res_box['icon_target_hit_acc']:.3f}), "
                f"P50={res_box['latency_p50_ms']} ms",
                flush=True,
            )

        if "firewall" not in summary:
            print("[3/5] Evaluating Pre-CDP Handoff Firewall (n = 300)...", flush=True)
            res_fw = eval_firewall(d1, adapter=use_fw)
            summary["firewall"] = res_fw
            save_checkpoint(out_path, summary)
            _flush_mlx()
            print(
                f"  -> [Saved 3/5] AUROC={res_fw['auroc']:.3f}, AP={res_fw['average_precision']:.3f}, "
                f"Rec@3%FAR={res_fw['calibrated_3pct_far']['hazard_recall']:.3f}, P50={res_fw['latency_p50_ms']} ms",
                flush=True,
            )

        if "aftermath" not in summary:
            print("[4/5] Evaluating Post-Click UI Aftermath Judge (n = 200)...", flush=True)
            res_am = eval_aftermath(d1, adapter=use_am)
            summary["aftermath"] = res_am
            save_checkpoint(out_path, summary)
            _flush_mlx()
            print(
                f"  -> [Saved 4/5] Acc={res_am['accuracy']:.3f}, Macro-F1={res_am['macro_f1']:.3f}, "
                f"P50={res_am['latency_p50_ms']} ms",
                flush=True,
            )

    if "quickjs_etl" not in summary:
        print("=== Loading LFM3-1.7B-MLX-4bit for [5/5] QuickJS & Schema JSON ETL ===", flush=True)
        lfm3_4bit_dir = ensure_mlx_4bit_lfm3()
        cand_lfm = adapters_root / "lfm3_quickjs_etl"
        lfm3_adapter = cand_lfm if (args.mode == "adapted" and (cand_lfm / "adapters.safetensors").exists()) else None
        scripter = ResidentLFM3Scripter(lfm3_4bit_dir, adapter_path=lfm3_adapter)

        # Record true dual-resident footprint while both models sit in unified RAM, then free d1 before [5/5] loop
        cur_active_gb = mx.get_active_memory() / (1024**3)
        both_active_gb = cur_active_gb + (
            float(summary["memory_and_swap"].get("d1_resident_gb", 0.0)) if d1 is None else 0.0
        )
        summary["memory_and_swap"]["resident_active_gb"] = round(both_active_gb, 3)
        if d1 is not None:
            del d1
            _flush_mlx()

        print("[5/5] Evaluating LFM3-1.7B QuickJS Synthesis & Schema JSON ETL (n = 200)...", flush=True)
        res_etl = eval_quickjs_etl(scripter)
        summary["quickjs_etl"] = res_etl
        del scripter
        _flush_mlx()
        print(
            f"  -> [Saved 5/5] SchemaValid={res_etl['schema_json']['schema_valid_rate']:.3f}, "
            f"QuickJSValid={res_etl['quickjs_code']['valid_syntax_and_refs_rate']:.3f}",
            flush=True,
        )

    peak_mem_gb = mx.get_peak_memory() / (1024**3)
    summary["memory_and_swap"]["peak_eval_gb"] = round(peak_mem_gb, 3)
    summary["memory_and_swap"]["wall_clock_seconds"] = round(time.perf_counter() - t_start, 1)
    save_checkpoint(out_path, summary)
    print(f"\n[DONE] Saved {args.mode} M4 benchmark summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
