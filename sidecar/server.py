#!/usr/bin/env python3
"""Unix-Socket Unified LFM Sidecar Daemon v2.0 (`/tmp/saccade-lfm.sock`) for `saccade` Rust Relay.

Architecture (v2.0 Unified Single-Backbone):
  - Single 4-bit `LiquidAI/d1-2B-MLX-4bit` instance (`~0.97 GiB` DOM-only / `~1.32 GiB` VLM / `0.00 GiB` Idle).
  - Single-pass `act_and_guard` (`d1_reflex_v2`): answers `[pick, destructive, injection]` in ONE forward pass!
  - Set-of-Marks (`d1_som_grounder`): numbered `[1]..[12]` badge visual grounding for `<canvas>` & unlabeled icons.
  - Zero-token Span-Pointer `Read` (`d1_pointer_etl`) + Table Alignment + Causal `d1_quickjs_etl` cascade.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from saccade.sidecar.aftermath import AlgorithmRTimeMachineBuffer, judge_aftermath
from saccade.sidecar.firewall import check_pre_cdp_firewall
from saccade.sidecar.foveate import foveate_snapshot
from saccade.sidecar.grounder import (
    act_and_guard_single_pass,
    ground_som_visual,
    ground_visual_box,
    pick_dom_ref,
)
from saccade.sidecar.multi_lora import ResidentD1MultiLoRA
from saccade.sidecar.quickjs_etl import UnifiedD1Scripter

DEFAULT_SOCK = "/tmp/saccade-lfm.sock"
DEFAULT_IDLE_UNLOAD_SEC = float(os.environ.get("SACCADE_IDLE_UNLOAD_SECONDS", "600"))


def _decode_img(val: str | None) -> Image.Image | None:
    if not val:
        return None
    p = Path(val)
    if len(val) < 512 and p.exists():
        return Image.open(p).convert("RGB")
    raw = base64.b64decode(val)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class SaccadeSidecarService:
    def __init__(
        self,
        adapters_dir: Path,
        lazy_vision: bool = True,
        idle_unload_sec: float = DEFAULT_IDLE_UNLOAD_SEC,
        force_v1_slots: bool = False,
    ):
        mx.set_cache_limit(512 * 1024 * 1024)
        self._req_count = 0
        self.last_active_ts = time.monotonic()
        self.idle_unload_sec = idle_unload_sec
        self.idle_unloads = 0
        self.force_v1_slots = force_v1_slots

        self.d1 = ResidentD1MultiLoRA(
            model_id="LiquidAI/d1-2B-MLX-4bit",
            default_rank=8,
            lazy_vision=lazy_vision,
        )

        # Prefer consolidated v2 adapters (`reflex_v2`, `som_grounder`, `pointer_etl`) while retaining v1 fallback
        has_v2 = (not force_v1_slots) and (adapters_dir / "d1_reflex_v2/adapters.safetensors").exists()
        if has_v2:
            for name in ("reflex_v2", "handoff_guard", "som_grounder", "pointer_etl", "aftermath_tm"):
                adir = adapters_dir / f"d1_{name}"
                if (adir / "adapters.safetensors").exists():
                    self.d1.register_adapter_dir(name, adir)
        else:
            for name in ("dom_grounder", "ui_box", "handoff_guard", "aftermath_tm"):
                adir = adapters_dir / f"d1_{name}"
                if (adir / "adapters.safetensors").exists():
                    self.d1.register_adapter_dir(name, adir)

        etl_dir = adapters_dir / "d1_quickjs_etl"
        if not (etl_dir / "adapters.safetensors").exists():
            etl_dir = adapters_dir / "lfm3_quickjs_etl"
        if (etl_dir / "adapters.safetensors").exists():
            self.d1.register_adapter_dir("quickjs_etl", etl_dir)

        self.scripter = UnifiedD1Scripter(self.d1, adapter_name="quickjs_etl")
        self.tm_buffer = AlgorithmRTimeMachineBuffer(capacity=120)

        self.trigger_counts: Counter[str] = Counter()
        self.trigger_events: list[dict[str, Any]] = []

    def check_idle_unload(self) -> bool:
        if (
            self.idle_unload_sec > 0
            and self.d1.is_loaded
            and (time.monotonic() - self.last_active_ts) >= self.idle_unload_sec
        ):
            self.d1.unload()
            self.idle_unloads += 1
            return True
        return False

    def _record_trigger(self, method: str, adapter: str | None, result: dict[str, Any], in_chars: int = 0) -> None:
        self.trigger_counts[method] += 1
        t_info = result.get("timing_ms") or {}
        ev = {
            "seq": sum(self.trigger_counts.values()),
            "ts_unix": round(time.time(), 3),
            "method": method,
            "adapter": result.get("adapter_used", adapter),
            "mode": result.get("mode"),
            "in_chars": in_chars,
            "wake_ms": float(t_info.get("wake_ms", 0.0)),
            "vision_page_ms": float(t_info.get("vision_page_ms", 0.0)),
            "swap_ms": float(t_info.get("swap_ms", 0.0)),
            "infer_ms": float(t_info.get("infer_ms", result.get("infer_ms", 0.0))),
            "active_memory_gb": round(mx.get_active_memory() / (1024**3), 3),
            "vision_loaded": self.d1.vision_loaded,
        }
        if "ref" in result:
            ev["ref"] = result.get("ref")
            ev["confidence"] = result.get("confidence")
            ev["escalate"] = result.get("escalate_to_frontier")
        if "require_handoff" in result:
            ev["require_handoff"] = result.get("require_handoff")
            ev["risk_prob"] = result.get("risk_prob")
        if method in ("ground_box", "ground_som"):
            ev["click_xy"] = result.get("click_xy")
            ev["badge"] = result.get("badge")
            ev["confidence"] = result.get("confidence")
        if method in ("extract_json", "quickjs"):
            ev["valid"] = result.get("schema_valid", result.get("valid"))
        if len(self.trigger_events) < 2000:
            self.trigger_events.append(ev)

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        self._req_count += 1
        if self._req_count % 25 == 0:
            mx.clear_cache()
        method = req.get("method", "")
        params = req.get("params") or {}

        if method == "ping":
            return {
                "ok": True,
                "version": "2.0" if "reflex_v2" in self.d1._adapter_dirs else "1.0",
                "unified_single_backbone": True,
                "model_loaded": self.d1.is_loaded,
                "vision_loaded": self.d1.vision_loaded,
                "active_memory_gb": round(mx.get_active_memory() / (1024**3), 3),
                "registered_adapters": sorted(self.d1._adapter_dirs.keys()),
                "lfm3_loaded": "quickjs_etl" in self.d1._adapter_dirs,
                "idle_unloads": self.idle_unloads,
                "trigger_counts": dict(self.trigger_counts),
                "time_machine_episodes": self.tm_buffer.seen,
            }

        if method == "unload":
            unload_ms = self.d1.unload()
            return {
                "unloaded": True,
                "unload_ms": round(unload_ms, 2),
                "active_memory_gb": round(mx.get_active_memory() / (1024**3), 3),
            }

        if method == "evict_vision":
            evicted = self.d1.evict_vision()
            return {
                "evicted": evicted,
                "vision_loaded": self.d1.vision_loaded,
                "active_memory_gb": round(mx.get_active_memory() / (1024**3), 3),
            }

        if method == "telemetry_summary":
            total_infer = sum(e["infer_ms"] for e in self.trigger_events)
            total_swap = sum(e["swap_ms"] for e in self.trigger_events)
            return {
                "total_triggers": sum(self.trigger_counts.values()),
                "counts_by_method": dict(self.trigger_counts),
                "total_lfm_infer_ms": round(total_infer, 2),
                "total_lora_swap_ms": round(total_swap, 3),
                "active_memory_gb": round(mx.get_active_memory() / (1024**3), 3),
                "vision_loaded": self.d1.vision_loaded,
                "events": self.trigger_events,
            }

        if method == "telemetry_reset":
            self.trigger_counts.clear()
            self.trigger_events.clear()
            return {"reset": True}

        self.last_active_ts = time.monotonic()

        if method == "foveate":
            fast_flag = bool(params.get("fast", True))
            snap = params["snapshot"]
            res = foveate_snapshot(
                self.d1 if not fast_flag else None,
                snapshot_text=snap,
                subgoal=params.get("subgoal", "Inspect main interactive controls"),
                keep_threshold=float(params.get("keep_threshold", 0.25)),
                redact_pii=bool(params.get("redact_pii", True)),
                fast=fast_flag,
            )
            self._record_trigger("foveate", None if fast_flag else "reflex_v2", res, in_chars=len(snap))
            return res

        if method == "pick_ref":
            snap = params["snapshot"]
            res = pick_dom_ref(
                self.d1,
                subgoal=params["subgoal"],
                snapshot_text=snap,
                abstain_tau=params.get("abstain_tau"),
            )
            self._record_trigger("pick_ref", res.get("adapter_used"), res, in_chars=len(snap))
            return res

        if method == "act_and_guard":
            snap = params["snapshot"]
            subgoal = params["subgoal"]
            user_goal = params.get("user_goal", subgoal)
            page_url = params.get("page_url", "")

            # Saccade v2.0 fast path: ONE single packed forward pass (`[pick, destructive, injection]`)!
            if "reflex_v2" in self.d1.adapters:
                joint = act_and_guard_single_pass(
                    self.d1,
                    subgoal=subgoal,
                    user_goal=user_goal,
                    snapshot_text=snap,
                    page_url=page_url,
                    abstain_tau=params.get("abstain_tau"),
                    alert_tau=params.get("alert_tau"),
                )
                self._record_trigger("act_and_guard_joint", "reflex_v2", joint, in_chars=len(snap))
                return joint

            # Legacy v1 two-pass fallback when only separate `dom_grounder` + `handoff_guard` slots exist
            pick = pick_dom_ref(self.d1, subgoal=subgoal, snapshot_text=snap, abstain_tau=params.get("abstain_tau"))
            self._record_trigger("pick_ref", "dom_grounder", pick, in_chars=len(snap))
            if pick.get("escalate_to_frontier") or not pick.get("ref"):
                return pick
            chosen_ref = pick["ref"]
            fw = check_pre_cdp_firewall(
                self.d1,
                user_goal=user_goal,
                tool_name=params.get("tool_name", "click"),
                tool_args={"ref": chosen_ref, **(params.get("tool_args") or {})},
                target_aria=chosen_ref,
                page_excerpt=snap,
                page_url=page_url,
            )
            self._record_trigger("firewall", "handoff_guard", fw, in_chars=len(snap))
            return {
                **pick,
                "require_handoff": fw["require_handoff"],
                "risk_prob": fw["risk_prob"],
                "handoff_message": fw["handoff_message"],
                "firewall_timing_ms": fw["timing_ms"],
            }

        if method in ("ground_box", "ground_som"):
            img = _decode_img(params["image"])
            assert img is not None, f"{method} requires `image` path or base64"
            cand_boxes = params.get("candidate_boxes")
            if (not self.force_v1_slots) and (method == "ground_som" or cand_boxes is not None or "som_grounder" in self.d1.adapters):
                res = ground_som_visual(
                    self.d1,
                    target_instruction=params["instruction"],
                    screenshot=img,
                    candidate_boxes=cand_boxes,
                )
                self._record_trigger("ground_som", res.get("adapter_used"), res)
                return res
            use_adap = "ui_box" if "ui_box" in self.d1.adapters else None
            res = ground_visual_box(self.d1, target_instruction=params["instruction"], screenshot=img, adapter=use_adap)
            self._record_trigger("ground_box", use_adap, res)
            return res

        if method == "firewall":
            crop = _decode_img(params.get("crop"))
            excerpt = params.get("page_excerpt", "")
            res = check_pre_cdp_firewall(
                self.d1,
                user_goal=params.get("user_goal", "General browsing"),
                tool_name=params["tool_name"],
                tool_args=params.get("tool_args") or {},
                target_aria=params.get("target_aria", ""),
                page_excerpt=excerpt,
                page_url=params.get("page_url", ""),
                crop=crop,
                alert_tau=params.get("alert_tau"),
            )
            self._record_trigger("firewall", res.get("adapter_used"), res, in_chars=len(excerpt))
            return res

        if method == "aftermath":
            b_img = _decode_img(params.get("before_image"))
            a_img = _decode_img(params.get("after_image"))
            res = judge_aftermath(
                self.d1,
                action_desc=params.get("action_desc", ""),
                aria_diff=params.get("aria_diff", ""),
                before_crop=b_img,
                after_crop=a_img,
            )
            self._record_trigger("aftermath", "aftermath_tm", res)
            return res

        if method == "record_hindsight":
            ep = self.tm_buffer.observe(
                domain=params.get("domain", "unknown"),
                state=params.get("state", ""),
                pred_probs=params.get("pred_probs") or {},
                realized_status=params["realized_status"],
            )
            return {
                "recorded": True,
                "surprise": round(ep.surprise, 4),
                "reservoir_size": len(self.tm_buffer.reservoir),
                "seen_total": self.tm_buffer.seen,
            }

        if method == "quickjs":
            snap = params["snapshot"]
            res = self.scripter.synthesize_quickjs(step_goal=params["step_goal"], snapshot_excerpt=snap)
            self._record_trigger("quickjs", "quickjs_etl", res, in_chars=len(snap))
            return res

        if method == "extract_json":
            ctx_txt = params["context"]
            res = self.scripter.extract_json(context_text=ctx_txt, json_schema=params["json_schema"])
            self._record_trigger("extract_json", res.get("mode", "quickjs_etl"), res, in_chars=len(ctx_txt))
            return res

        return {"error": f"unknown_method:{method}"}


def serve_unix_socket(sock_path: str, service: SaccadeSidecarService) -> None:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(32)

    stop = False

    def _sig(_s: int, _f: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    srv.settimeout(0.5)
    print(
        f"[saccade-sidecar-v2] Unified single-backbone listening on {sock_path} "
        f"(active={mx.get_active_memory() / (1024**3):.3f} GiB, adapters={sorted(service.d1._adapter_dirs)})",
        flush=True,
    )

    last_idle_check = time.monotonic()
    try:
        while not stop:
            now = time.monotonic()
            if (now - last_idle_check) >= 5.0:
                last_idle_check = now
                if service.check_idle_unload():
                    print(
                        f"[saccade-sidecar-v2] Unloaded weights after {service.idle_unload_sec:.0f}s idle "
                        f"(active={mx.get_active_memory() / (1024**3):.3f} GiB)",
                        flush=True,
                    )
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            with conn:
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    if b"\n" in buf:
                        line, _ = buf.split(b"\n", 1)
                        t0 = time.perf_counter()
                        try:
                            req = json.loads(line.decode("utf-8"))
                            resp = {"ok": True, "result": service.handle(req)}
                        except Exception as exc:
                            resp = {"ok": False, "error": str(exc)}
                        resp["ipc_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                        break
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=os.environ.get("SACCADE_SOCKET", DEFAULT_SOCK))
    parser.add_argument(
        "--adapters-dir",
        default=os.environ.get("SACCADE_ADAPTERS_DIR", str(REPO_ROOT / "saccade/sidecar/adapters")),
    )
    parser.add_argument("--eager-vision", action="store_true")
    parser.add_argument("--force-v1", action="store_true")
    parser.add_argument("--idle-unload-sec", type=float, default=DEFAULT_IDLE_UNLOAD_SEC)
    args = parser.parse_args()

    svc = SaccadeSidecarService(
        Path(args.adapters_dir),
        lazy_vision=not args.eager_vision,
        idle_unload_sec=args.idle_unload_sec,
        force_v1_slots=args.force_v1 or (os.environ.get("SACCADE_FORCE_V1") == "1"),
    )
    serve_unix_socket(args.socket, svc)


if __name__ == "__main__":
    main()
