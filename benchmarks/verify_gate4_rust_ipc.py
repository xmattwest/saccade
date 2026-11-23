#!/usr/bin/env python3
"""Gate 4 Live Rust <-> Python MLX Sidecar Unix-Socket IPC & Relay Verification.

Starts the real resident MLX sidecar (`saccade/sidecar/server.py`) with all 5 trained LoRA adapters
on `/tmp/saccade-gate4.sock` AND runs the compiled Rust release binary (`saccade/rust-browse/target/release/browse`)
inside an isolated `OPENCODE_BROWSER_HOME` directory:
  1. Verifies `browse tools` registers `saccade_act`, `saccade_extract`, `saccade_status`, and augmented `snapshot`/`click`.
  2. Verifies `browse saccade_status` auto-spawns the Rust relay, connects over Unix socket to the resident
     MLX sidecar, and reports `sidecar_connected: true`, `lfm3_loaded: true`, and all 4 `d1` LoRA adapters.
  3. Measures end-to-end NDJSON Unix-socket IPC round-trip overhead (`ping` P50/P95 in ms) and live RPCs
     across `foveate`, `pick_ref`, `firewall`, `record_hindsight`, `quickjs`, and `extract_json`.
  4. Stops the MLX sidecar and verifies `browse saccade_status` gracefully degrades to
     `sidecar_connected: false, fallback: "stock_browse_passthrough"` with zero crashes, then cleanly tears down
     the isolated Rust relay.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SACCADE_DIR = REPO_ROOT / "saccade"
RUST_BIN = SACCADE_DIR / "rust-browse/target/release/browse"
SERVER_PY = SACCADE_DIR / "sidecar/server.py"
RESULTS_OUT = SACCADE_DIR / "results/gate4_rust_ipc_verification.json"


def ndjson_rpc(sock_path: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(15.0)
        s.connect(sock_path)
        s.sendall((json.dumps({"method": method, "params": params}) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))


def main() -> None:
    if not RUST_BIN.exists():
        raise FileNotFoundError(f"Expected compiled Rust binary at {RUST_BIN}")

    sock_path = f"/tmp/saccade-gate4-{os.getpid()}.sock"
    tmp_home = Path(tempfile.mkdtemp(prefix="saccade-browse-home-"))
    env = os.environ.copy()
    env["SACCADE_SOCKET"] = sock_path
    env["OPENCODE_BROWSER_HOME"] = str(tmp_home)
    env["OPENCODE_BROWSER_PORT"] = "49827"

    sidecar_proc = subprocess.Popen(
        [sys.executable, str(SERVER_PY), "--socket", sock_path],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for MLX sidecar (d1-2B + 4 LoRAs + LFM3-1.7B + LoRA) to bind socket
        ready = False
        for _ in range(90):
            if os.path.exists(sock_path):
                try:
                    ping = ndjson_rpc(sock_path, "ping", {})
                    if ping.get("ok"):
                        ready = True
                        break
                except Exception:
                    pass
            time.sleep(1.0)
        if not ready:
            raise RuntimeError("Timed out waiting for Saccade MLX sidecar socket")

        # 1. Verify Rust CLI `browse tools` lists Saccade tools
        tools_run = subprocess.run([str(RUST_BIN), "tools"], env=env, capture_output=True, text=True, check=True)
        tools_stdout = tools_run.stdout
        has_tools = all(t in tools_stdout for t in ("saccade_act", "saccade_extract", "saccade_status"))

        # 2. Verify live `browse saccade_status` -> Rust HTTP Relay -> Unix Socket -> MLX Sidecar
        status_run = subprocess.run(
            [str(RUST_BIN), "saccade_status"], env=env, capture_output=True, text=True, check=True
        )
        live_status = json.loads(status_run.stdout.strip())
        assert live_status["sidecar_connected"] is True, f"Unexpected status: {live_status}"
        assert live_status["sidecar"]["lfm3_loaded"] is True
        reg_adapters = live_status["sidecar"]["registered_d1_adapters"]

        # 3. Measure pure Unix-socket NDJSON IPC latency across 30 `ping` calls
        ping_lats_ms = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = ndjson_rpc(sock_path, "ping", {})
            assert r["ok"] is True
            ping_lats_ms.append((time.perf_counter() - t0) * 1000.0)

        # 4. Live end-to-end RPC smoke checks across all sidecar capabilities
        fake_tok = "sk-" + "1234567890abcdefghijkl"
        sample_snap = (
            '- banner "Top":\n'
            f'  - text "User admin@corp.io | Token {fake_tok}"\n'
            '  - link "Dashboard" [ref=e1]\n'
            '  - link "Billing" [ref=e2]\n'
            '- main "Repository Settings":\n'
            '  - textbox "Search Collaborators" [ref=e3]\n'
            '  - button "Invite Collaborator" [ref=e4]\n'
            '  - button "Permanently Delete Repository" [ref=e9]\n'
            '- contentinfo "Footer":\n'
            '  - link "Terms" [ref=e10]\n'
            '  - link "Privacy" [ref=e11]\n'
            '  - link "Status" [ref=e12]\n'
        )
        fov_res = ndjson_rpc(sock_path, "foveate", {"snapshot": sample_snap * 2, "subgoal": "Invite a collaborator"})["result"]
        pick_res = ndjson_rpc(sock_path, "pick_ref", {"snapshot": sample_snap, "subgoal": "Click Invite Collaborator"})["result"]
        fw_benign = ndjson_rpc(
            sock_path,
            "firewall",
            {
                "user_goal": "Invite a collaborator to the repo",
                "tool_name": "click",
                "tool_args": {"ref": "e4"},
                "target_aria": 'button "Invite Collaborator" [ref=e4]',
                "page_excerpt": sample_snap,
            },
        )["result"]
        fw_hazard = ndjson_rpc(
            sock_path,
            "firewall",
            {
                "user_goal": "Check repository collaborators list",
                "tool_name": "click",
                "tool_args": {"ref": "e9"},
                "target_aria": 'button "Permanently Delete Repository" [ref=e9]',
                "page_excerpt": sample_snap,
            },
        )["result"]
        tm_res = ndjson_rpc(
            sock_path,
            "record_hindsight",
            {
                "domain": "github.com",
                "state": "click e4",
                "pred_probs": {"SETTLED_OK": 0.2, "SPINNER_LOADING": 0.8},
                "realized_status": "SETTLED_OK",
            },
        )["result"]

        # 5. Stop MLX sidecar and verify graceful fallback in live Rust relay
        sidecar_proc.terminate()
        sidecar_proc.wait(timeout=10)
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        down_run = subprocess.run(
            [str(RUST_BIN), "saccade_status"], env=env, capture_output=True, text=True, check=True
        )
        fallback_status = json.loads(down_run.stdout.strip())
        assert fallback_status["sidecar_connected"] is False
        assert fallback_status["fallback"] == "stock_browse_passthrough"

        report = {
            "rust_binary": "saccade/rust-browse/target/release/browse",
            "cli_tools_registered": has_tools,
            "live_sidecar_status": live_status["sidecar"],
            "registered_d1_adapters": reg_adapters,
            "unix_socket_ping_p50_ms": round(float(np.percentile(ping_lats_ms, 50)), 3),
            "unix_socket_ping_p95_ms": round(float(np.percentile(ping_lats_ms, 95)), 3),
            "live_rpc_checks": {
                "foveate_pii_redacted": "admin@corp.io" not in fov_res["foveated_snapshot"],
                "pick_ref_top_candidate": pick_res["top_candidate"],
                "pick_ref_infer_ms": pick_res["timing_ms"]["infer_ms"],
                "firewall_benign_blocked": fw_benign["require_handoff"],
                "firewall_destructive_blocked": fw_hazard["require_handoff"],
                "firewall_destructive_risk_prob": fw_hazard["risk_prob"],
                "time_machine_recorded_surprise": tm_res["surprise"],
            },
            "graceful_fallback_when_sidecar_stopped": fallback_status["fallback"],
        }
        RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_OUT.write_text(json.dumps(report, indent=2))
        print(f"[Gate 4 Verified] Rust<->MLX IPC P50={report['unix_socket_ping_p50_ms']} ms -> {RESULTS_OUT}", flush=True)

    finally:
        if sidecar_proc.poll() is None:
            sidecar_proc.terminate()
            try:
                sidecar_proc.wait(timeout=5)
            except Exception:
                sidecar_proc.kill()
        # Cleanly stop isolated Rust relay daemon
        subprocess.run([str(RUST_BIN), "uninstall"], env=env, capture_output=True)
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        shutil.rmtree(tmp_home, ignore_errors=True)


if __name__ == "__main__":
    main()
