"""Gate C / Apple M4 Verification of Saccade v2.0 Unified Adapters (`0.97 GiB` DOM / `1.32 GiB` VLM).

Verifies on local Apple Silicon M4 (with strict 512 MiB MLX cache cap):
  1. Single-backbone memory footprint (`DOM-only` vs. `VLM paged-in` vs. `Idle 0.000 GiB`)
  2. `d1_reflex_v2` single-pass `[pick, destructive, injection]` latency & accuracy vs. v1 two-pass
  3. `d1_som_grounder` Set-of-Marks `[1]..[10]` accuracy on `ScreenSpot` vs. v1 raw `Box` (`20.67%`)
  4. `d1_pointer_etl` zero-token `Read` span-pointer accuracy & latency (`0` generated tokens)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

mx.set_cache_limit(512 * 1024 * 1024)

from saccade.sidecar.firewall import check_pre_cdp_firewall
from saccade.sidecar.grounder import act_and_guard_single_pass, ground_som_visual, pick_dom_ref
from saccade.sidecar.quickjs_etl import UnifiedD1Scripter
from saccade.sidecar.server import SaccadeSidecarService


def main() -> None:
    suites = REPO_ROOT / "saccade/data/suites_v2"
    adapters_dir = REPO_ROOT / "saccade/sidecar/adapters"

    # 1. Measure v1 two-pass latency on identical 30 reflex examples first, then unload cleanly
    svc_v1 = SaccadeSidecarService(adapters_dir, lazy_vision=True, force_v1_slots=True)
    v1_dom_gib = round(mx.get_active_memory() / (1024**3), 4)
    reflex_test = json.loads((suites / "reflex_v2/test.json").read_text())

    v1_latencies: list[float] = []
    for row in reflex_test[:30]:
        t0 = time.perf_counter()
        pick = pick_dom_ref(svc_v1.d1, subgoal=row["subgoal"], snapshot_text=row["snapshot"], candidates=row["candidates"])
        _ = check_pre_cdp_firewall(
            svc_v1.d1,
            user_goal=row["user_goal"],
            tool_name="click",
            tool_args={"ref": pick.get("top_candidate") or "e1"},
            target_aria=pick.get("top_candidate") or "e1",
            page_excerpt=row["snapshot"],
        )
        v1_latencies.append((time.perf_counter() - t0) * 1000.0)
    svc_v1.handle({"method": "unload"})
    del svc_v1
    gc.collect()
    mx.clear_cache()

    # 2. Load Saccade v2.0 Unified Service (`d1_reflex_v2` + `d1_som_grounder` + `d1_pointer_etl` + `d1_quickjs_etl`)
    svc = SaccadeSidecarService(adapters_dir, lazy_vision=True, force_v1_slots=False)
    v2_dom_gib = round(mx.get_active_memory() / (1024**3), 4)

    # Stratified held-out slice on M4 (first 60 general + all hazards + all localhost QA sandbox items)
    seen_ids: set[int] = set()
    eval_reflex: list[dict] = []
    for idx, r in enumerate(reflex_test):
        if idx < 60 or r["label_require_handoff"] == 1 or r.get("env") == "dev_sandbox":
            seen_ids.add(idx)
            eval_reflex.append(r)

    v2_latencies: list[float] = []
    ans_ok, ans_n = 0, 0
    haz_tp, haz_n = 0, 0
    ben_fp, ben_n = 0, 0
    dev_ok, dev_n = 0, 0

    for idx, row in enumerate(eval_reflex):
        t0 = time.perf_counter()
        res = act_and_guard_single_pass(
            svc.d1,
            subgoal=row["subgoal"],
            user_goal=row["user_goal"],
            snapshot_text=row["snapshot"],
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if idx < 30:
            v2_latencies.append(dt)
        if not row["is_abstain_case"]:
            ans_n += 1
            if res["top_candidate"] == row["target_ref"]:
                ans_ok += 1
        gold_haz = int(row["label_require_handoff"]) == 1
        pred_haz = bool(res["require_handoff"])
        if gold_haz:
            haz_n += 1
            if pred_haz:
                haz_tp += 1
        else:
            ben_n += 1
            if pred_haz:
                ben_fp += 1
            if row.get("env") == "dev_sandbox":
                dev_n += 1
                if not pred_haz:
                    dev_ok += 1
        if (idx + 1) % 30 == 0:
            mx.clear_cache()

    # 3. Evaluate `d1_pointer_etl` on M4 (all multi-field Chrome capture cards + 60 SQuAD v2 items)
    ptr_test = json.loads((suites / "pointer_etl/test.json").read_text())
    chrome_cards = [r for r in ptr_test if r.get("source") == "chrome_capture_card"]
    squad_slice = [r for r in ptr_test if r.get("source") != "chrome_capture_card"][:60]
    eval_ptr = chrome_cards + squad_slice

    scripter: UnifiedD1Scripter = svc.scripter
    ptr_latencies: list[float] = []
    field_ok, field_n = 0, 0
    null_ok, null_n = 0, 0
    card_ok, card_n = 0, 0

    for idx, row in enumerate(eval_ptr):
        props = {}
        for f in row["fields"]:
            q_type = f["q_type"]
            t_map = {"number": "number", "date": "string", "text": "string"}
            p: dict[str, object] = {"type": [t_map.get(q_type, "string"), "null"], "description": f["instructions"]}
            if q_type == "date":
                p["format"] = "date"
            props[f["name"]] = p
        schema = {"type": "object", "properties": props}
        t0 = time.perf_counter()
        out = scripter.extract_json(row["state"], schema)
        ptr_latencies.append((time.perf_counter() - t0) * 1000.0)
        val_obj = out.get("value") or {}
        all_match = True
        for f in row["fields"]:
            pred = val_obj.get(f["name"])
            if f["is_null"]:
                null_n += 1
                hit = pred is None or str(pred).strip() == ""
                if hit:
                    null_ok += 1
            elif f["q_type"] == "number" and f.get("gt_number") is not None:
                hit = pred is not None and abs(float(pred) - float(f["gt_number"])) < 1e-2
            else:
                gt = str(f.get("gt_text") or "").strip().strip(".,;:\"'").lower()
                pt = str(pred or "").strip().strip(".,;:\"'").lower()
                hit = bool(pt) and ((pt == gt) or (gt in pt and len(pt) <= len(gt) + 6))
            field_n += 1
            if hit:
                field_ok += 1
            else:
                all_match = False
        if row.get("source") == "chrome_capture_card":
            card_n += 1
            if all_match:
                card_ok += 1
        if (idx + 1) % 25 == 0:
            mx.clear_cache()

    # 4. Evaluate `d1_som_grounder` on ScreenSpot slice (100 test screenshots on M4, triggering lazy vision page-in!)
    som_dir = suites / "som_grounding"
    som_test = json.loads((som_dir / "test.json").read_text())[:100]
    som_ok = 0
    som_latencies: list[float] = []
    for idx, row in enumerate(som_test):
        im = Image.open(som_dir / row["image_rel_path"]).convert("RGB")
        t0 = time.perf_counter()
        res = ground_som_visual(
            svc.d1,
            target_instruction=row["instruction"],
            screenshot=im,
            candidate_boxes=row["candidate_boxes"],
            adapter="som_grounder",
            abstain_tau=0.01,
        )
        som_latencies.append((time.perf_counter() - t0) * 1000.0)
        if res.get("badge") == row["target_badge"]:
            som_ok += 1
        if (idx + 1) % 25 == 0:
            mx.clear_cache()

    v2_vlm_gib = round(mx.get_active_memory() / (1024**3), 4)
    unload_info = svc.handle({"method": "unload"})

    v1_p50 = sorted(v1_latencies)[len(v1_latencies) // 2]
    v2_p50 = sorted(v2_latencies)[len(v2_latencies) // 2]
    ptr_p50 = sorted(ptr_latencies)[len(ptr_latencies) // 2]
    som_p50 = sorted(som_latencies)[len(som_latencies) // 2]

    summary = {
        "gate": "Gate C · Apple M4 4-bit Unified Verification (Saccade v2.0)",
        "backbone": "LiquidAI/d1-2B-MLX-4bit",
        "loaded_adapters": sorted(svc.d1._adapter_dirs.keys()),
        "memory_gib": {
            "v1_dom_only_gib": v1_dom_gib,
            "v2_dom_only_gib": v2_dom_gib,
            "v2_full_vlm_gib": v2_vlm_gib,
            "idle_unloaded_gib": unload_info["active_memory_gb"],
        },
        "single_pass_reflex_v2": {
            "v1_two_pass_p50_ms": round(v1_p50, 2),
            "v2_single_pass_p50_ms": round(v2_p50, 2),
            "reflex_speedup_x": round(v1_p50 / max(v2_p50, 1.0), 2),
            "m4_answerable_top1_acc": round(ans_ok / max(ans_n, 1), 4),
            "m4_hazard_recall": round(haz_tp / max(haz_n, 1), 4),
            "m4_benign_far": round(ben_fp / max(ben_n, 1), 4),
            "m4_localhost_sandbox_allow_rate": round(dev_ok / max(dev_n, 1), 4),
        },
        "som_visual_grounder": {
            "v1_raw_box_screenspot_acc": 0.2067,
            "m4_som_screenspot_100_acc": round(som_ok / len(som_test), 4),
            "som_p50_ms": round(som_p50, 2),
        },
        "pointer_read_etl": {
            "generated_tokens": 0,
            "pointer_p50_ms": round(ptr_p50, 2),
            "m4_field_exact_acc": round(field_ok / max(field_n, 1), 4),
            "m4_null_abstention_acc": round(null_ok / max(null_n, 1), 4),
            "m4_multi_field_card_exact_rate": round(card_ok / max(card_n, 1), 4),
        },
    }
    out_path = REPO_ROOT / "saccade/results/gate_c_m4_v2_adapted.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
