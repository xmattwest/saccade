# Saccade

Saccade is a `1.00 GiB` local browser reflex layer for Claude Code, Codex, Antigravity, OpenCode, Cursor, and 30+ MCP coding agents. Built in Rust on [`anomalyco/browse`](https://github.com/anomalyco/browse) (`MIT`) and powered on Apple Silicon by a single 4-bit [`LiquidAI/d1-2B-MLX-4bit`](https://huggingface.co/LiquidAI/d1-2B-MLX-4bit) backbone with six `FP16` LoRA adapters, it handles routine clicks, form fills, schema-validated JSON extraction, multi-page pagination, PII scrubbing, and destructive-action checks on-device instead of shipping raw DOM trees to a frontier model on every step.

- **Overview & Live Chrome Telemetry:** [https://shelltoho.me/tools/saccade/](https://shelltoho.me/tools/saccade/)
- **LoRA Adapter Weights (`96 MB`, `MIT`):** [https://huggingface.co/xmattwest/saccade-d1-2b-adapters](https://huggingface.co/xmattwest/saccade-d1-2b-adapters)
- **Backbone Interpretability Study (*Inside LFM3*):** [https://shelltoho.me/pubs/inside-lfm3/](https://shelltoho.me/pubs/inside-lfm3/)

---

## What It Saves in Live Chrome

Measured end-to-end against headless Google Chrome over CDP across six multi-step web scenarios (Sprint issue search and filtering, `127.0.0.1` QA test database reset, unauthorized live repo deletion trap, prompt-injected support ticket exfiltration trap, unlabeled `<svg>` toolbar icon click, 4-field subscription card extraction, and a 3-page invoice directory sweep):

| Live Chrome Benchmark Metric | Stock `browse` | Saccade v1.0 | **Saccade v2.0** | Savings / Gain |
| :--- | :---: | :---: | :---: | :---: |
| **End-to-End Wall-Clock Time** | `25.01 s` | `18.81 s` | **`13.16 s`** | **`1.90x` faster (`-11.85 s`)** |
| **Cloud Tokens Consumed** | `3,396` (`10` turns) | `1,393` | **`1,046`** | **`-69.2%` (`-70.6%` across 20 SaaS apps)** |
| **Resident Unified RAM (DOM / VLM / Idle)** | `0 GiB` | `0.99 / 1.34 / 0.00 GiB` | **`1.00 / 1.35 / 0.00 GiB`** | Auto-spawns in `1.9 s`, `0 GiB` after 10m idle |
| **Single `saccade_act` Avg Latency** | `2,210 ms` (cloud) | `1,213 ms` (2 passes) | **`990 ms` (`~608 ms` LFM)** | `[pick, destructive, injection]` in 1 pass |
| **4-Field Subscription Card JSON ETL** | `2,175 ms` (cloud) | `1,216 ms` (drifted) | **`482 ms` (`0` output tokens)** | **`2.52x` faster + `BOS0` `null` abstention** |
| **3-Page Paginated Table Sweep (`6` rows)** | `5` cloud turns (`~10.9 s`) | `5` tool calls | **`1` `saccade_sweep` call (`3.93 s`)** | Zero per-page cloud round-trips |
| **Unlabeled `<svg>` Icon Hit (`ScreenSpot`)** | Escalates (`1,250` tok) | `20.67%` (raw `Box`) | **`87.67%` (Set-of-Marks `[1]..[12]`)** | **`+67.0 pp`** visual grounding accuracy |
| **Live Prod Delete & Prompt Injection Traps** | `2 / 2` executed | `2 / 2` blocked | **`2 / 2` blocked (`127.0.0.1` QA allowed)** | Pre-CDP `handoff()` pause, `0` false local blocks |

---

## Install in Three Commands (macOS Apple Silicon)

Prerequisites: Rust toolchain (`cargo`), [`uv`](https://docs.astral.sh/uv/), and the [`browse` Chrome extension](https://github.com/anomalyco/browse) paired with your browser.

```bash
# 1. Build the Rust relay & CLI:
cargo build --manifest-path rust-browse/Cargo.toml --release -p browse

# 2. Download the 96 MB LoRA adapter suite:
uvx --from huggingface_hub hf download xmattwest/saccade-d1-2b-adapters --local-dir sidecar/adapters

# 3. Install into all detected MCP coding agents (Claude Code, Codex, Antigravity, OpenCode, Cursor, etc.):
SACCADE_ENGINE_BIN="$PWD/bin/saccade-engine" ./rust-browse/target/release/browse install --yes
```

On your agent's first browser call, the Rust relay automatically spawns `bin/saccade-engine` over a `0.07 ms` Unix socket (`/tmp/saccade-lfm.sock`), keeps everyday DOM memory at `1.00 GiB`, pages in the `379 MB` vision tower from NVMe in `27 to 42 ms` only when screenshots arrive, and unloads weights back to `0.00 GiB` after 10 minutes of inactivity.

---

## How Coding Agents Use Saccade

Saccade preserves 100% backward compatibility with every `browse` tool (`execute`, `navigate`, `snapshot`, `click`, `fill`, `screenshot`, `tabs`, `handoff`) while adding four local fast paths:

1. **`saccade_act({ tab, step, value?, goal?, visualFallback? })`**
   Runs ARIA `eXX` ref selection and the environment-aware pre-CDP firewall (`[pick, destructive, injection]`) in one forward pass (`~608 ms` LFM inference). If confidence falls below its calibrated threshold, it returns a `< 0.5 ms` Rust BM25-foveated, PII-redacted snapshot so the frontier model picks the exact `eXX` ref without a wasted round-trip. Passing `visualFallback: true` stamps numbered amber `[1]..[12]` Set-of-Marks badges with automatic toolbar zoom for `<canvas>` and unlabeled `<svg>` controls.

2. **`saccade_extract({ tab, schema, focus? })`**
   Extracts schema-validated JSON locally through a three-tier cascade:
   - Zero-token `d1` ARIA table column alignment (`~380 to 460 ms`, lossless cell values).
   - Zero-token `d1` span-pointer `Read` (`text`, `number`, `date`, `BOS0` `null` abstention in `~480 ms`, `0` generated tokens).
   - Causal `d1_quickjs_etl` decoding (`CausalD1View` on the shared backbone) for nested objects.

3. **`saccade_sweep({ tab, schema, focus?, nextStep?, maxPages? })`**
   Loops `saccade_extract -> single-pass Next click -> 120 ms settle -> JSON row deduplication` locally across up to `maxPages` (`default: 3, max: 10`), stopping automatically when pagination ends.

4. **Sub-Millisecond Native Rust Foveation & Pre-CDP Firewall (`snapshot` & `click`/`fill`/`eval`)**
   Passing `subgoal` to `snapshot` prunes `75 to 85%` of unrelated navbars, sidebars, and footers and scrubs emails, SSNs, payment cards, and API tokens (`sk-...`, `ghp_...`, `AKIA...`) in `< 0.5 ms` with zero LLM latency. Direct mutating calls pass through `d1_handoff_guard` before Chrome dispatches them, pausing unauthorized production deletes or indirect prompt injections via `browse.handoff()` while allowing explicit `localhost` / `127.0.0.1` QA test resets.

---

## Architecture

```
Frontier Coding Agent (Claude Code / Codex / Antigravity / OpenCode / Cursor)
       │  MCP stdio (`execute`, `saccade_act`, `saccade_extract`, `saccade_sweep`, `snapshot`)
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rust `browse` Relay (`crates/browse-relay/src/saccade.rs`)                   │
│  • `< 0.5 ms` native BM25 ARIA-subtree foveator & zero-regex PII scrubber    │
│  • Pre-CDP `handoff()` gate + `saccade_sweep` multi-page deduplicating loop  │
│  • On-demand `saccade-engine` auto-spawner (`0.07 ms` Unix socket IPC)       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ NDJSON over `/tmp/saccade-lfm.sock`
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Unified Apple Silicon MLX Engine (`LiquidAI/d1-2B-MLX-4bit`)                 │
│  • `1.00 GiB` DOM-only  │  `1.35 GiB` with lazy `27 ms` VLM  │  `0.00 GiB` idle │
│  • 6 hot-swapped `FP16` LoRA slots (`0.17 ms` avg swap):                     │
│    `reflex_v2` · `som_grounder` · `pointer_etl` · `handoff_guard`            │
│    `aftermath_tm` · `quickjs_etl` (`CausalD1View`)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Controls

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `SACCADE_SOCKET` | `/tmp/saccade-lfm.sock` | Unix domain socket path between Rust `browse` and `saccade-engine`. |
| `SACCADE_ENGINE_BIN` | Auto-detected | Explicit path to `bin/saccade-engine` launcher. |
| `SACCADE_IDLE_UNLOAD_SEC` | `600` | Seconds of inactivity before unloading weights to `0.00 GiB` (`0` disables). |
| `SACCADE_DISABLE` | `0` | Set `1` to bypass Saccade completely and run stock `browse`. |

---

## Reproducing Benchmarks & Tests

```bash
# Unit & hygiene tests:
uv run pytest -q
cargo test --manifest-path rust-browse/Cargo.toml -p browse-core -p browse-relay

# Live 3-arm Headless Chrome benchmark (Stock vs. Saccade v1.0 vs. Saccade v2.0):
BROWSE_TEST_CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
SACCADE_ENGINE_BIN="$PWD/bin/saccade-engine" \
SACCADE_RUN_LIVE_V2=1 \
SACCADE_V2_BENCH_OUT="$PWD/results/live_browser_v2_comparison.json" \
cargo test --manifest-path rust-browse/Cargo.toml -p browse-script --test live_saccade_v2_bench -- --nocapture --test-threads=1
```

---

## License & Acknowledgments

MIT License. Built on top of [`anomalyco/browse`](https://github.com/anomalyco/browse) (`MIT`) and [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B).
