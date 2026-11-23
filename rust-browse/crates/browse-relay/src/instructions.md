browse drives the user's real Chrome-family browser through the browse extension, accelerated by Saccade's local unified LFM reflex layer, so logins, extensions and open tabs are already there and the user can watch or take over.

Tabs are the handle. `navigate` with a `url` and no `tab` opens a new background tab and returns its id as `tab`; keep it and pass `tab` to every other call (`saccade_act`, `saccade_extract`, `saccade_sweep`, `snapshot`, `click`, `fill`, `screenshot`, ...). `navigate` with `tab` goes to a URL in that tab. Use only tab ids you were given, a group label you set, or tabs the user lent (`tabs` lists them).

Prefer Saccade's local fast paths to save time and tokens:
- **Act locally (`saccade_act`):** For clear UI micro-steps (clicking a control, switching a filter, filling an input), call `saccade_act` with `{ tab, step, value?, goal? }` before pulling a raw `snapshot`. Saccade selects the exact `eXX` ref and enforces its environment-aware pre-CDP Handoff Firewall in one local pass; if ambiguous, it returns a compact PII-redacted `foveated_snapshot` so you can act on the returned ref immediately. Pass `visualFallback: true` for `<canvas>`, chart, or unlabeled-icon controls to trigger Set-of-Marks `[1]..[N]` badge grounding.
- **Extract JSON locally (`saccade_extract`):** For reading structured records, tables, or metrics, call `saccade_extract` with `{ tab, schema, focus? }` to return schema-validated JSON on-device with zero raw DOM bloat.
- **Sweep paginated tables (`saccade_sweep`):** For multi-page tables or directories, call `saccade_sweep` with `{ tab, schema, focus?, nextStep?, maxPages? }` to extract and paginate locally across `maxPages` without per-page cloud round-trips.
- **Foveated inspect (`snapshot`):** When you do inspect directly, pass `subgoal` (`snapshot` with `{ tab, subgoal }`) to prune unrelated navbars/sidebars/footers in `< 1 ms` while keeping target `eXX` refs (`full: true` returns the unpruned tree). Refs go stale after page changes: re-snapshot when needed.
- **Verify:** Check outcomes with a fresh read (`snapshot` with `subgoal`, `screenshot`, or `eval` of `location.href`).

In code mode, compose multi-step sequences in one run instead of one tool call per turn.

When only a person can proceed (login, 2FA, passkey, captcha, payment confirmation, or a Saccade Firewall handoff pause), call `handoff` with the `tab` and what the user should do, and wait. Never solve a captcha yourself and never retry-loop a blocked site.

When finished, close the tabs you opened (`tabs` with action `close`; `group` closes every tab in that group). `status` reports the relay and connected browsers (`nextStep` explains pairing if disconnected), and `saccade_status` reports local LFM memory and trigger telemetry.
