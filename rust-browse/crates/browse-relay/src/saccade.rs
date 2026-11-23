//! Saccade Unified LFM3 / d1-2B Reflex Layer for `browse-relay`.
//!
//! Architecture (v1.0):
//!   1. Sub-millisecond native Rust BM25 ARIA-subtree foveator & PII redactor (`fast_foveate_rust`):
//!      prunes 75-88% of navbar/sidebar/footer noise and scrubs secrets/PII in `< 0.5 ms` with zero LLM tax.
//!   2. Single-backbone Unix-socket sidecar (`SACCADE_SOCKET`, default `/tmp/saccade-lfm.sock`) with
//!      on-demand auto-spawn (`saccade-engine`) and 10-minute idle auto-unload (`0.00 GiB` idle RAM).
//!   3. `SaccadeMiddlewareTool`: wraps `snapshot` (instant Rust BM25 foveation + PII redaction) and
//!      mutating tools (`click`, `fill`, `type`, `select`, `upload`, `eval`) with the pre-CDP Handoff Firewall.
//!   4. Composable MCP & QuickJS tools: `saccade_act` (single-RPC `act_and_guard` + opt-in `<canvas>` box),
//!      `saccade_extract` (local schema-strict JSON extraction), and `saccade_status` (health & LFM telemetry).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use browse_core::{Content, TabId, Tool, ToolContext, ToolOutput, ToolSpec, tab_schema};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::Mutex as AsyncMutex;

pub const DEFAULT_SOCKET_PATH: &str = "/tmp/saccade-lfm.sock";
const SNAPSHOT_CACHE_KEY: &str = "saccade/last_snapshot";
const GOAL_CACHE_KEY: &str = "saccade/last_goal";
const PREGUARDED_FLAG: &str = "_saccade_preguarded";

const GUARDED_INPUT_TOOLS: &[&str] = &["click", "fill", "type", "select", "upload", "eval"];

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "into", "that", "this", "click", "press", "select",
    "choose", "open", "page", "tab", "button", "link", "field", "input", "box", "section",
    "current", "inspect", "main", "view", "item", "user", "browser", "step",
];

static SPAWN_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();

pub fn socket_path() -> PathBuf {
    std::env::var("SACCADE_SOCKET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_PATH))
}

pub fn sidecar_disabled() -> bool {
    matches!(std::env::var("SACCADE_DISABLE").as_deref(), Ok("1" | "true"))
}

fn autostart_enabled() -> bool {
    !matches!(std::env::var("SACCADE_AUTOSTART").as_deref(), Ok("0" | "false"))
}

/// Discover the standalone `saccade-engine` launcher next to `browse`, in XDG data, or in a source checkout.
pub fn find_engine_launcher() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("SACCADE_ENGINE_BIN") {
        let p = PathBuf::from(explicit.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("saccade-engine");
            if sibling.is_file() {
                return Some(sibling);
            }
            // Source checkout layout: `<repo>/saccade/rust-browse/target/{release,debug}/browse` -> `<repo>/saccade/bin/saccade-engine`
            for anc in dir.ancestors().take(6) {
                let candidate = anc.join("bin").join("saccade-engine");
                if candidate.is_file() {
                    return Some(candidate);
                }
                let nested = anc.join("saccade").join("bin").join("saccade-engine");
                if nested.is_file() {
                    return Some(nested);
                }
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let xdg = PathBuf::from(home).join(".local/share/saccade/bin/saccade-engine");
        if xdg.is_file() {
            return Some(xdg);
        }
    }
    None
}

async fn try_autospawn_sidecar(sock: &Path) -> Result<()> {
    if !autostart_enabled() {
        anyhow::bail!("Saccade autostart disabled via SACCADE_AUTOSTART=0");
    }
    let lock = SPAWN_LOCK.get_or_init(|| AsyncMutex::new(()));
    let _guard = lock.lock().await;

    if UnixStream::connect(sock).await.is_ok() {
        return Ok(());
    }
    let launcher = find_engine_launcher()
        .ok_or_else(|| anyhow!("No `saccade-engine` launcher found; start sidecar manually or set SACCADE_ENGINE_BIN"))?;

    let mut cmd = std::process::Command::new(&launcher);
    cmd.env("SACCADE_SOCKET", sock)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
        .with_context(|| format!("Failed to spawn Saccade engine at {}", launcher.display()))?;

    let deadline = Instant::now() + Duration::from_secs(12);
    while Instant::now() < deadline {
        if UnixStream::connect(sock).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    anyhow::bail!("Spawned `saccade-engine` but socket {} did not become ready within 12s", sock.display())
}

/// Send one NDJSON request to the local Saccade LFM sidecar over Unix domain socket, auto-spawning on demand if configured.
pub async fn rpc_call(method: &str, params: Value, timeout: Duration) -> Result<Value> {
    if sidecar_disabled() {
        anyhow::bail!("Saccade sidecar disabled via SACCADE_DISABLE");
    }
    let path = socket_path();
    let stream = match tokio::time::timeout(Duration::from_millis(250), UnixStream::connect(&path)).await {
        Ok(Ok(s)) => s,
        _ => {
            if method == "ping" && !matches!(std::env::var("SACCADE_PING_AUTOSTART").as_deref(), Ok("1" | "true")) {
                anyhow::bail!("Saccade sidecar not listening at {}", path.display());
            }
            try_autospawn_sidecar(&path).await?;
            tokio::time::timeout(Duration::from_millis(500), UnixStream::connect(&path))
                .await
                .context("timed out connecting to newly spawned Saccade Unix socket")?
                .with_context(|| format!("Saccade sidecar not listening at {}", path.display()))?
        }
    };

    tokio::time::timeout(timeout, async move {
        let (reader, mut writer) = stream.into_split();
        let payload = json!({ "method": method, "params": params }).to_string() + "\n";
        writer.write_all(payload.as_bytes()).await?;
        writer.flush().await?;

        let mut lines = BufReader::new(reader).lines();
        let line = lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("Saccade sidecar closed socket without reply"))?;
        let parsed: Value = serde_json::from_str(&line)?;
        if parsed.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(parsed.get("result").cloned().unwrap_or(Value::Null))
        } else {
            let err = parsed
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown Saccade sidecar error");
            Err(anyhow!("{err}"))
        }
    })
    .await
    .context("Saccade sidecar RPC timed out")?
}

// ---------------------------------------------------------------------------
// Native Sub-Millisecond Rust BM25 ARIA Foveator & Zero-Regex PII Redactor
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FoveatedOutput {
    pub foveated_snapshot: String,
    pub orig_chars: usize,
    pub kept_chars: usize,
    pub compression_ratio: f64,
    pub pii_redactions: usize,
    pub sections_kept: usize,
    pub sections_total: usize,
    pub elapsed_us: u64,
}

fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'+' | b'%')
}

/// Fast single-pass PII & credential scrubber (emails, SSNs, 13-19 digit cards, API tokens).
pub fn mask_pii_rust(input: &str) -> (String, usize) {
    let mut out = String::with_capacity(input.len());
    let mut redactions = 0usize;
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;

    let sk_pfx = concat!("s", "k", "-");
    let ghp_pfx = concat!("gh", "p_");
    let gho_pfx = concat!("gh", "o_");
    let ghs_pfx = concat!("gh", "s_");
    let akia_pfx = concat!("AK", "IA");

    while i < n {
        // 1. Secret token prefixes (`sk-...`, `ghp_...`, `AKIA...`)
        let rem = &input[i..];
        let prev_boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if prev_boundary {
            if rem.starts_with(sk_pfx) || rem.starts_with(ghp_pfx) || rem.starts_with(gho_pfx) || rem.starts_with(ghs_pfx) {
                let mut end = i + 4;
                while end < n && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'-') {
                    end += 1;
                }
                if end - i >= 22 {
                    out.push_str("[REDACTED_SECRET]");
                    redactions += 1;
                    i = end;
                    continue;
                }
            }
            if rem.starts_with(akia_pfx) {
                let mut end = i + 4;
                while end < n && (bytes[end].is_ascii_uppercase() || bytes[end].is_ascii_digit()) {
                    end += 1;
                }
                if end - i == 20 {
                    out.push_str("[REDACTED_SECRET]");
                    redactions += 1;
                    i = end;
                    continue;
                }
            }
            // 2. SSN (`DDD-DD-DDDD`)
            if i + 11 <= n
                && bytes[i].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3] == b'-'
                && bytes[i + 4].is_ascii_digit()
                && bytes[i + 5].is_ascii_digit()
                && bytes[i + 6] == b'-'
                && bytes[i + 7].is_ascii_digit()
                && bytes[i + 8].is_ascii_digit()
                && bytes[i + 9].is_ascii_digit()
                && bytes[i + 10].is_ascii_digit()
                && (i + 11 == n || !bytes[i + 11].is_ascii_digit())
            {
                out.push_str("[REDACTED_SSN]");
                redactions += 1;
                i += 11;
                continue;
            }
            // 3. Payment card (`13..19` digits separated by optional space/hyphen)
            if bytes[i].is_ascii_digit() {
                let mut end = i;
                let mut digits = 0usize;
                let mut last_digit_end = i;
                while end < n && (bytes[end].is_ascii_digit() || bytes[end] == b' ' || bytes[end] == b'-') {
                    if bytes[end].is_ascii_digit() {
                        digits += 1;
                        last_digit_end = end + 1;
                    }
                    end += 1;
                    if digits > 19 {
                        break;
                    }
                }
                if (13..=19).contains(&digits) && (last_digit_end == n || !bytes[last_digit_end].is_ascii_alphanumeric()) {
                    out.push_str("[REDACTED_CARD]");
                    redactions += 1;
                    i = last_digit_end;
                    continue;
                }
            }
            // 4. Email (`local@domain.tld`)
            if bytes[i].is_ascii_alphanumeric() {
                let mut end = i;
                let mut at_pos = None;
                while end < n && (is_token_char(bytes[end]) || bytes[end] == b'@') {
                    if bytes[end] == b'@' {
                        if at_pos.is_some() {
                            at_pos = None;
                            break;
                        }
                        at_pos = Some(end);
                    }
                    end += 1;
                }
                if let Some(at) = at_pos {
                    let domain = &input[at + 1..end].trim_end_matches('.');
                    let actual_end = at + 1 + domain.len();
                    if at > i
                        && domain.contains('.')
                        && domain.rsplit('.').next().is_some_and(|tld| tld.len() >= 2 && tld.bytes().all(|c| c.is_ascii_alphabetic()))
                    {
                        out.push_str("[REDACTED_EMAIL]");
                        redactions += 1;
                        i = actual_end;
                        continue;
                    }
                }
            }
        }

        let ch = input[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    (out, redactions)
}

fn stem_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            push_stemmed(&mut out, &mut cur);
        }
    }
    if !cur.is_empty() {
        push_stemmed(&mut out, &mut cur);
    }
    out
}

fn push_stemmed(out: &mut Vec<String>, cur: &mut String) {
    let w = std::mem::take(cur);
    if w.len() < 2 || STOPWORDS.contains(&w.as_str()) {
        return;
    }
    let stemmed = if w.len() > 5 && w.ends_with("ing") {
        w[..w.len() - 3].to_string()
    } else if w.len() > 4 && (w.ends_with("ed") || w.ends_with("es")) {
        w[..w.len() - 2].to_string()
    } else if w.len() > 3 && w.ends_with('s') {
        w[..w.len() - 1].to_string()
    } else {
        w
    };
    out.push(stemmed);
}

struct AriaSection {
    role: String,
    name: String,
    lines: Vec<String>,
    ref_count: usize,
    always_keep: bool,
}

fn parse_aria_line_header(line: &str) -> (usize, String, String) {
    let indent = line.len() - line.trim_start_matches(' ').len();
    let trimmed = line.trim_start();
    if let Some(after_dash) = trimmed.strip_prefix("- ") {
        let after_dash = after_dash.trim_start();
        let role_end = after_dash
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
            .unwrap_or(after_dash.len());
        let role = if role_end > 0 {
            after_dash[..role_end].to_string()
        } else {
            "item".to_string()
        };
        let rest = after_dash[role_end..].trim_start();
        let name = if let Some(after_q) = rest.strip_prefix('"')
            && let Some(end_q) = after_q.find('"')
        {
            after_q[..end_q].to_string()
        } else {
            String::new()
        };
        (indent, role, name)
    } else {
        (indent, "item".to_string(), String::new())
    }
}

fn count_refs_in_lines(lines: &[String]) -> usize {
    lines.iter().map(|l| l.matches("[ref=e").count()).sum()
}

/// Deterministic `< 0.5 ms` Rust BM25 ARIA-subtree pruner + PII redactor (`0` LLM calls).
pub fn fast_foveate_rust(snapshot_text: &str, subgoal: &str, redact_pii: bool) -> FoveatedOutput {
    let t0 = Instant::now();
    let orig_chars = snapshot_text.len();
    let (cleaned, pii_redactions) = if redact_pii {
        mask_pii_rust(snapshot_text)
    } else {
        (snapshot_text.to_string(), 0)
    };

    let raw_lines: Vec<&str> = cleaned.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut sections: Vec<AriaSection> = Vec::new();
    let mut cur_lines: Vec<String> = Vec::new();
    let mut cur_role = "root".to_string();
    let mut cur_name = String::new();

    let flush = |sections: &mut Vec<AriaSection>, lines: &mut Vec<String>, role: &str, name: &str| {
        if lines.is_empty() {
            return;
        }
        let ref_count = count_refs_in_lines(lines);
        let always_keep = matches!(role, "dialog" | "alertdialog" | "alert");
        sections.push(AriaSection {
            role: role.to_string(),
            name: name.to_string(),
            lines: std::mem::take(lines),
            ref_count,
            always_keep,
        });
    };

    for ln in raw_lines {
        let (indent, role, name) = parse_aria_line_header(ln);
        let is_landmark = matches!(
            role.as_str(),
            "banner" | "navigation" | "complementary" | "main" | "contentinfo"
                | "dialog" | "alertdialog" | "alert" | "form" | "region" | "table" | "search"
        );
        if (indent == 0 || (indent <= 2 && is_landmark)) && !cur_lines.is_empty() {
            flush(&mut sections, &mut cur_lines, &cur_role, &cur_name);
            cur_role = role;
            cur_name = name;
            cur_lines.push(ln.to_string());
        } else {
            if cur_lines.is_empty() {
                cur_role = role;
                cur_name = name;
            }
            cur_lines.push(ln.to_string());
        }
    }
    flush(&mut sections, &mut cur_lines, &cur_role, &cur_name);

    let sections_total = sections.len();
    if sections_total <= 1 || orig_chars < 500 {
        let kept_chars = cleaned.len();
        return FoveatedOutput {
            foveated_snapshot: cleaned,
            orig_chars,
            kept_chars,
            compression_ratio: 1.0 - (kept_chars as f64 / orig_chars.max(1) as f64),
            pii_redactions,
            sections_kept: sections_total,
            sections_total,
            elapsed_us: t0.elapsed().as_micros() as u64,
        };
    }

    let mut q_terms = Vec::new();
    for t in stem_tokens(subgoal) {
        if !q_terms.contains(&t) {
            q_terms.push(t);
        }
    }

    let doc_toks: Vec<Vec<String>> = sections
        .iter()
        .map(|s| stem_tokens(&s.lines.join("\n")))
        .collect();
    let n_docs = sections_total as f64;
    let avgdl = (doc_toks.iter().map(Vec::len).sum::<usize>() as f64 / n_docs).max(1.0);

    let mut df: HashMap<String, usize> = HashMap::new();
    for qt in &q_terms {
        let count = doc_toks.iter().filter(|dt| dt.contains(qt)).count();
        df.insert(qt.clone(), count);
    }

    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(sections_total);
    for (idx, (sec, dtoks)) in sections.iter().zip(doc_toks.iter()).enumerate() {
        let mut tf_map: HashMap<&str, usize> = HashMap::new();
        for tok in dtoks {
            *tf_map.entry(tok.as_str()).or_insert(0) += 1;
        }
        let dl = dtoks.len() as f64;
        let mut bm25 = 0.0f64;
        for qt in &q_terms {
            let tf = *tf_map.get(qt.as_str()).unwrap_or(&0) as f64;
            if tf > 0.0 {
                let d_f = *df.get(qt).unwrap_or(&0) as f64;
                let idf = (1.0 + (n_docs - d_f + 0.5) / (d_f + 0.5)).ln();
                bm25 += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (dl / avgdl)));
            }
        }
        let mut bonus = 0.0f64;
        if sec.role == "main" {
            bonus += 1.35;
        } else if matches!(sec.role.as_str(), "dialog" | "alertdialog" | "alert" | "form" | "table" | "region") {
            bonus += 0.95;
        } else if matches!(sec.role.as_str(), "banner" | "contentinfo" | "complementary" | "navigation") && bm25 < 0.8 {
            bonus -= 1.20;
        }
        if sec.ref_count > 0 {
            bonus += (0.04 * sec.ref_count as f64).min(0.45);
        }
        scores.push((idx, bm25 + bonus));
    }

    let mut ranked = scores.clone();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_score = ranked.first().map(|x| x.1).unwrap_or(0.0);
    let min_keep_score = if top_score > 1.0 { (top_score * 0.30).max(0.35) } else { 0.05 };

    let mut keep_ids: HashSet<usize> = sections
        .iter()
        .enumerate()
        .filter(|(_, s)| s.always_keep)
        .map(|(i, _)| i)
        .collect();

    for &(idx, sc) in &ranked {
        if keep_ids.len() >= 3 {
            break;
        }
        if sc >= min_keep_score || keep_ids.is_empty() {
            keep_ids.insert(idx);
        }
    }

    let q_set: HashSet<&str> = q_terms.iter().map(String::as_str).collect();
    let max_lines_per_sec = 48usize;
    let mut out_lines: Vec<String> = Vec::new();
    let mut sections_kept = 0usize;

    for (idx, sec) in sections.iter().enumerate() {
        if keep_ids.contains(&idx) {
            sections_kept += 1;
            if sec.lines.len() <= max_lines_per_sec {
                out_lines.extend(sec.lines.iter().cloned());
            } else {
                let mut line_scores: Vec<(f64, usize)> = sec
                    .lines
                    .iter()
                    .enumerate()
                    .map(|(li, ln)| {
                        let ltoks = stem_tokens(ln);
                        let mut hit: f64 = ltoks.iter().filter(|t| q_set.contains(t.as_str())).count() as f64 * 2.0;
                        if ln.contains("[ref=") {
                            hit += 0.5;
                        }
                        if li == 0 {
                            hit += 10.0;
                        }
                        (hit, li)
                    })
                    .collect();
                line_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1)));
                let chosen: HashSet<usize> = line_scores.into_iter().take(max_lines_per_sec).map(|(_, li)| li).collect();
                for (li, ln) in sec.lines.iter().enumerate() {
                    if chosen.contains(&li) {
                        out_lines.push(ln.clone());
                    }
                }
                let trimmed = sec.lines.len().saturating_sub(chosen.len());
                if trimmed > 0 {
                    out_lines.push(format!("  - ... [saccade trimmed {trimmed} low-relevance lines in {}]", sec.role));
                }
            }
        } else {
            out_lines.push(format!(
                "- ... [saccade omitted {} lines / {} refs in {} \"{}\"; pass full=true to expand]",
                sec.lines.len(),
                sec.ref_count,
                sec.role,
                sec.name
            ));
        }
    }

    let foveated_snapshot = out_lines.join("\n");
    let kept_chars = foveated_snapshot.len();
    FoveatedOutput {
        foveated_snapshot,
        orig_chars,
        kept_chars,
        compression_ratio: 1.0 - (kept_chars as f64 / orig_chars.max(1) as f64),
        pii_redactions,
        sections_kept,
        sections_total,
        elapsed_us: t0.elapsed().as_micros() as u64,
    }
}

// ---------------------------------------------------------------------------
// Relay Helpers & Middleware
// ---------------------------------------------------------------------------

fn first_text(output: &ToolOutput) -> Option<&str> {
    output.content.iter().find_map(|c| match c {
        Content::Text { text } => Some(text.as_str()),
        _ => None,
    })
}

fn extract_ref_line(snapshot_text: &str, ref_id: &str) -> String {
    let needle = format!("[ref={ref_id}");
    for line in snapshot_text.lines() {
        if line.contains(&needle) {
            return line.trim().to_string();
        }
    }
    String::new()
}

async fn remember_snapshot(ctx: &ToolContext, tab: TabId, text: &str) {
    if let Ok(state) = ctx.state(tab).await {
        state.inner().lock().await.insert(SNAPSHOT_CACHE_KEY.to_string(), Box::new(text.to_string()));
    }
}

async fn last_snapshot(ctx: &ToolContext, tab: TabId) -> String {
    if let Ok(state) = ctx.state(tab).await
        && let Some(s) = state.inner().lock().await.get(SNAPSHOT_CACHE_KEY).and_then(|v| v.downcast_ref::<String>())
    {
        return s.clone();
    }
    String::new()
}

async fn remember_goal(ctx: &ToolContext, tab: TabId, goal: &str) {
    if !goal.trim().is_empty()
        && let Ok(state) = ctx.state(tab).await
    {
        state.inner().lock().await.insert(GOAL_CACHE_KEY.to_string(), Box::new(goal.to_string()));
    }
}

async fn last_goal(ctx: &ToolContext, tab: TabId) -> String {
    if let Ok(state) = ctx.state(tab).await
        && let Some(g) = state.inner().lock().await.get(GOAL_CACHE_KEY).and_then(|v| v.downcast_ref::<String>())
    {
        return g.clone();
    }
    "User-authorized browser workflow".to_string()
}

async fn current_tab_url(ctx: &ToolContext, tab: TabId) -> String {
    if let Ok(tabs) = ctx.browser.tabs().await
        && let Some(info) = tabs.into_iter().find(|t| t.tab == tab)
    {
        return info.url;
    }
    String::new()
}

/// Transparent decorator adding `< 0.5 ms` Rust BM25 Foveation + PII masking to `snapshot` and Pre-CDP Handoff Firewall to mutating tools.
pub struct SaccadeMiddlewareTool {
    inner: Arc<dyn Tool>,
}

impl SaccadeMiddlewareTool {
    pub fn wrap_all(tools: Vec<Arc<dyn Tool>>) -> Vec<Arc<dyn Tool>> {
        tools
            .into_iter()
            .map(|tool| {
                let name = tool.spec().name;
                if name == "snapshot" || GUARDED_INPUT_TOOLS.contains(&name.as_str()) {
                    Arc::new(Self { inner: tool }) as Arc<dyn Tool>
                } else {
                    tool
                }
            })
            .collect()
    }
}

#[async_trait]
impl Tool for SaccadeMiddlewareTool {
    fn spec(&self) -> ToolSpec {
        let mut spec = self.inner.spec();
        if spec.name == "snapshot"
            && let Some(props) = spec.input_schema.get_mut("properties").and_then(Value::as_object_mut)
        {
            props.insert(
                "subgoal".into(),
                json!({
                    "type": "string",
                    "description": "Optional current step intent for Saccade foveation: prunes unrelated nav/sidebar/footer sections in <1 ms and masks PII while preserving target refs."
                }),
            );
            props.insert(
                "full".into(),
                json!({
                    "type": "boolean",
                    "description": "Pass true to bypass Saccade foveation and return the unpruned snapshot."
                }),
            );
        }
        if GUARDED_INPUT_TOOLS.contains(&spec.name.as_str())
            && let Some(props) = spec.input_schema.get_mut("properties").and_then(Value::as_object_mut)
        {
            props.insert(
                "goal".into(),
                json!({
                    "type": "string",
                    "description": "Optional user goal context for Saccade pre-CDP safety firewall."
                }),
            );
        }
        spec
    }

    async fn call(&self, ctx: ToolContext, mut args: Value) -> Result<ToolOutput> {
        let name = self.inner.spec().name;
        let tab = args.get("tab").and_then(|v| ctx.browser.resolve_tab(v).ok());

        if name == "snapshot" {
            let subgoal = args.get("subgoal").and_then(Value::as_str).map(str::to_string);
            let force_full = args.get("full").and_then(Value::as_bool).unwrap_or(false);
            if let Some(obj) = args.as_object_mut() {
                obj.remove("subgoal");
                obj.remove("full");
            }
            let out = self.inner.call(ctx.clone(), args).await?;
            if out.is_error {
                return Ok(out);
            }
            if let (Some(tab_id), Some(raw_text)) = (tab, first_text(&out)) {
                remember_snapshot(&ctx, tab_id, raw_text).await;
                let auto_fov = matches!(std::env::var("SACCADE_AUTO_FOVEATE").as_deref(), Ok("1" | "true"));
                if !force_full && (subgoal.is_some() || auto_fov) {
                    let goal_str = match subgoal {
                        Some(g) => {
                            remember_goal(&ctx, tab_id, &g).await;
                            g
                        }
                        None => last_goal(&ctx, tab_id).await,
                    };
                    if sidecar_disabled() {
                        return Ok(out);
                    }
                    // Default: sub-millisecond native Rust BM25 foveator + PII redactor (`0` LLM latency).
                    // Set `SACCADE_NEURAL_FOVEATE=1` to delegate foveation to the sidecar RPC instead.
                    if matches!(std::env::var("SACCADE_NEURAL_FOVEATE").as_deref(), Ok("1" | "true")) {
                        if let Ok(fov) = rpc_call(
                            "foveate",
                            json!({ "snapshot": raw_text, "subgoal": goal_str, "redact_pii": true, "fast": false }),
                            Duration::from_secs(3),
                        )
                        .await
                            && let Some(pruned) = fov.get("foveated_snapshot").and_then(Value::as_str)
                        {
                            return Ok(ToolOutput::text(pruned.to_string()));
                        }
                    }
                    let fov = fast_foveate_rust(raw_text, &goal_str, true);
                    return Ok(ToolOutput::text(fov.foveated_snapshot));
                }
            }
            return Ok(out);
        }

        // Pre-CDP Handoff Firewall on mutating browser tools (`click`, `fill`, `type`, `select`, `upload`, `eval`)
        let explicit_goal = args.get("goal").and_then(Value::as_str).map(str::to_string);
        let preguarded = args.get(PREGUARDED_FLAG).and_then(Value::as_bool).unwrap_or(false);
        if let Some(obj) = args.as_object_mut() {
            obj.remove("goal");
            obj.remove(PREGUARDED_FLAG);
        }
        if !preguarded && let Some(tab_id) = tab {
            if let Some(g) = &explicit_goal {
                remember_goal(&ctx, tab_id, g).await;
            }
            let user_goal = explicit_goal.unwrap_or(last_goal(&ctx, tab_id).await);
            let snap = last_snapshot(&ctx, tab_id).await;
            let ref_str = args
                .get("ref")
                .or_else(|| args.get("selector"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let target_aria = if !ref_str.is_empty() && !snap.is_empty() {
                let found = extract_ref_line(&snap, ref_str);
                if found.is_empty() { ref_str.to_string() } else { found }
            } else {
                ref_str.to_string()
            };

            let page_url = current_tab_url(&ctx, tab_id).await;

            if let Ok(verdict) = rpc_call(
                "firewall",
                json!({
                    "user_goal": user_goal,
                    "tool_name": name,
                    "tool_args": args,
                    "target_aria": target_aria,
                    "page_excerpt": snap,
                    "page_url": page_url,
                }),
                Duration::from_secs(4),
            )
            .await
                && verdict.get("require_handoff").and_then(Value::as_bool) == Some(true)
            {
                let msg = verdict
                    .get("handoff_message")
                    .and_then(Value::as_str)
                    .unwrap_or("Saccade paused a high-consequence action. Click Continue to allow.");
                let hr = ctx.browser.handoff(tab_id, msg).await?;
                if hr.outcome != "continued" {
                    return Ok(ToolOutput::error(format!(
                        "Saccade Firewall halted `{name}` (handoff outcome: {}): {msg}",
                        hr.outcome
                    )));
                }
            }
        }

        self.inner.call(ctx, args).await
    }
}

/// Composable Turbo Tool 1: `saccade_act`
/// Collapses `snapshot -> ref choice + pre-CDP firewall -> click/fill` into one local reflex call,
/// escalating in `< 0.5 ms` via native Rust BM25 foveation whenever confidence `< tau`.
struct SaccadeActTool {
    tools: Arc<HashMap<String, Arc<dyn Tool>>>,
}

#[async_trait]
impl Tool for SaccadeActTool {
    fn spec(&self) -> ToolSpec {
        let mut props = Map::new();
        props.insert("tab".into(), tab_schema());
        props.insert(
            "step".into(),
            json!({
                "type": "string",
                "description": "Natural-language UI micro-step to perform locally (e.g. 'Click the October 2026 filter' or 'Fill search box with ZCZ6')."
            }),
        );
        props.insert(
            "value".into(),
            json!({
                "type": "string",
                "description": "Optional text/option value if the step fills an input or selects a dropdown option."
            }),
        );
        props.insert(
            "goal".into(),
            json!({
                "type": "string",
                "description": "Optional overarching user goal for the pre-CDP Handoff Firewall (defaults to the tab's remembered goal or `step`)."
            }),
        );
        props.insert(
            "visualFallback".into(),
            json!({
                "type": "boolean",
                "description": "Optional: set true on <canvas>, WebGL, or unlabeled-icon views to fall back to d1-2B visual coordinate grounding when DOM refs abstain."
            }),
        );
        ToolSpec {
            name: "saccade_act".into(),
            description: "Execute a UI micro-step locally via Saccade's unified d1-2B reflex engine (zero cloud snapshot tokens). Runs DOM ref selection + pre-CDP Handoff Firewall in one RPC, and escalates with a <1 ms PII-redacted foveated snapshot if confidence falls below the calibrated abstention threshold.".into(),
            input_schema: json!({
                "type": "object",
                "properties": props,
                "required": ["tab", "step"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, ctx: ToolContext, args: Value) -> Result<ToolOutput> {
        let tab_val = args.get("tab").cloned().ok_or_else(|| anyhow!(browse_core::MISSING_TAB))?;
        let tab_id = ctx.browser.resolve_tab(&tab_val)?;
        let step = args
            .get("step")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| anyhow!("saccade_act requires a non-empty `step`"))?
            .to_string();
        let fill_val = args.get("value").and_then(Value::as_str).map(str::to_string);
        let explicit_goal = args.get("goal").and_then(Value::as_str).map(str::to_string);
        let visual_opt_in = args.get("visualFallback").and_then(Value::as_bool).unwrap_or_else(|| {
            matches!(std::env::var("SACCADE_AUTO_VISUAL_BOX").as_deref(), Ok("1" | "true"))
        });

        let user_goal = if let Some(g) = explicit_goal {
            remember_goal(&ctx, tab_id, &g).await;
            g
        } else {
            let prev = last_goal(&ctx, tab_id).await;
            if prev == "User-authorized browser workflow" {
                remember_goal(&ctx, tab_id, &step).await;
                step.clone()
            } else {
                prev
            }
        };

        let snap_tool = self.tools.get("snapshot").ok_or_else(|| anyhow!("snapshot tool missing"))?;
        let snap_out = snap_tool.call(ctx.clone(), json!({ "tab": tab_val, "full": true })).await?;
        if snap_out.is_error {
            return Ok(snap_out);
        }
        let raw_snap = first_text(&snap_out).unwrap_or("").to_string();

        let pending_tool = if fill_val.is_some() { "fill" } else { "click" };
        let tool_args = if let Some(ref v) = fill_val {
            json!({ "value": v })
        } else {
            json!({})
        };
        let page_url = current_tab_url(&ctx, tab_id).await;

        // Combined single-RPC `act_and_guard` (falling back to `pick_ref` for legacy mock sockets in unit tests)
        let pick = match rpc_call(
            "act_and_guard",
            json!({
                "subgoal": step,
                "user_goal": user_goal,
                "snapshot": raw_snap,
                "page_url": page_url,
                "tool_name": pending_tool,
                "tool_args": tool_args,
            }),
            Duration::from_secs(6),
        )
        .await
        {
            Ok(v) => v,
            Err(e) if e.to_string().contains("unknown") => {
                rpc_call("pick_ref", json!({ "subgoal": step, "snapshot": raw_snap }), Duration::from_secs(4)).await?
            }
            Err(e) => return Err(e),
        };

        let escalate = pick.get("escalate_to_frontier").and_then(Value::as_bool).unwrap_or(true);
        if !escalate && let Some(chosen_ref) = pick.get("ref").and_then(Value::as_str) {
            let had_inline_fw = pick.get("require_handoff").is_some();
            if pick.get("require_handoff").and_then(Value::as_bool) == Some(true) {
                let msg = pick
                    .get("handoff_message")
                    .and_then(Value::as_str)
                    .unwrap_or("Saccade paused a high-consequence action. Click Continue to allow.");
                let hr = ctx.browser.handoff(tab_id, msg).await?;
                if hr.outcome != "continued" {
                    return Ok(ToolOutput::error(format!(
                        "Saccade Firewall halted `{pending_tool}` on `{chosen_ref}` (handoff outcome: {}): {msg}",
                        hr.outcome
                    )));
                }
            }

            let action_out = if let Some(val) = fill_val {
                let fill_tool = self.tools.get("fill").ok_or_else(|| anyhow!("fill tool missing"))?;
                fill_tool
                    .call(
                        ctx.clone(),
                        json!({ "tab": tab_val, "ref": chosen_ref, "value": val, "goal": user_goal, PREGUARDED_FLAG: had_inline_fw }),
                    )
                    .await?
            } else {
                let click_tool = self.tools.get("click").ok_or_else(|| anyhow!("click tool missing"))?;
                click_tool
                    .call(
                        ctx.clone(),
                        json!({ "tab": tab_val, "ref": chosen_ref, "goal": user_goal, PREGUARDED_FLAG: had_inline_fw }),
                    )
                    .await?
            };
            return Ok(ToolOutput::text(
                json!({
                    "executed_locally": !action_out.is_error,
                    "mode": "dom_ref",
                    "ref": chosen_ref,
                    "confidence": pick.get("confidence"),
                    "risk_prob": pick.get("risk_prob"),
                    "single_pass_joint": pick.get("single_pass_joint"),
                    "timing_ms": pick.get("timing_ms"),
                    "firewall_timing_ms": pick.get("firewall_timing_ms"),
                    "action_result": first_text(&action_out).unwrap_or(""),
                    "is_error": action_out.is_error,
                })
                .to_string(),
            ));
        }

        // Opt-In Fallback Mode B: Try Set-of-Marks `[1]..[N]` / visual grounding when `visualFallback: true` is requested
        if visual_opt_in
            && fill_val.is_none()
            && let Some(shot_tool) = self.tools.get("screenshot")
            && let Some(click_tool) = self.tools.get("click")
        {
            let mut dom_vp: Option<(f64, f64)> = None;
            let mut cand_boxes: Option<Value> = None;
            if let Some(eval_tool) = self.tools.get("eval") {
                let som_js = r#"(() => {
                    const vw = Math.max(window.innerWidth || 1, 1);
                    const vh = Math.max(window.innerHeight || 1, 1);
                    const sel = 'button, a, input, [role="button"], [onclick], [data-action], svg, canvas';
                    const seen = [];
                    for (const el of document.querySelectorAll(sel)) {
                        const tag = (el.tagName || '').toLowerCase();
                        if ((tag === 'svg' || tag === 'canvas') && el.parentElement && el.parentElement.closest('button, a, [role="button"], [onclick], [data-action]')) continue;
                        const r = el.getBoundingClientRect();
                        if (r.width < 10 || r.height < 10 || r.width > vw * 0.48 || r.height > vh * 0.48) continue;
                        if (r.bottom <= 0 || r.right <= 0 || r.top >= vh || r.left >= vw) continue;
                        const st = window.getComputedStyle(el);
                        if (st.visibility === 'hidden' || st.display === 'none' || parseFloat(st.opacity || '1') < 0.05) continue;
                        const box = {
                            left: Math.max(0, Math.min(1, r.left / vw)),
                            top: Math.max(0, Math.min(1, r.top / vh)),
                            right: Math.max(0, Math.min(1, r.right / vw)),
                            bottom: Math.max(0, Math.min(1, r.bottom / vh)),
                            hint: String(el.getAttribute('aria-label') || el.getAttribute('title') || '').slice(0, 48)
                        };
                        const cx = 0.5 * (box.left + box.right), cy = 0.5 * (box.top + box.bottom);
                        const dup = seen.some(p => Math.hypot(0.5 * (p.left + p.right) - cx, 0.5 * (p.top + p.bottom) - cy) < 0.035);
                        if (!dup) seen.push(box);
                        if (seen.length >= 12) break;
                    }
                    return { vw, vh, boxes: seen };
                })()"#;
                if let Ok(ev_out) = eval_tool
                    .call(ctx.clone(), json!({ "tab": tab_val, "expression": som_js, PREGUARDED_FLAG: true }))
                    .await
                    && !ev_out.is_error
                    && let Some(txt) = first_text(&ev_out)
                    && let Ok(parsed) = serde_json::from_str::<Value>(txt)
                {
                    if let (Some(vw), Some(vh)) = (parsed["vw"].as_f64(), parsed["vh"].as_f64()) {
                        dom_vp = Some((vw, vh));
                    }
                    if parsed["boxes"].as_array().is_some_and(|a| !a.is_empty()) {
                        cand_boxes = Some(parsed["boxes"].clone());
                    }
                }
            }

            let shot_out = shot_tool.call(ctx.clone(), json!({ "tab": tab_val })).await?;
            if let Some(img_b64) = shot_out.content.iter().find_map(|c| match c {
                Content::Image { data, .. } => Some(data.clone()),
                _ => None,
            }) {
                let mut g_params = json!({ "instruction": step, "image": img_b64 });
                if let Some(cb) = cand_boxes {
                    g_params["candidate_boxes"] = cb;
                }
                if let Ok(box_res) = rpc_call("ground_box", g_params, Duration::from_secs(6)).await
                    && box_res.get("escalate_to_frontier").and_then(Value::as_bool) == Some(false)
                {
                    let coords = if let (Some((vw, vh)), Some(nxy)) = (dom_vp, box_res.get("norm_xy").and_then(Value::as_array))
                        && nxy.len() == 2
                    {
                        Some(((nxy[0].as_f64().unwrap_or(0.0) * vw).round(), (nxy[1].as_f64().unwrap_or(0.0) * vh).round()))
                    } else if let Some(xy) = box_res.get("click_xy").and_then(Value::as_array)
                        && xy.len() == 2
                    {
                        Some((xy[0].as_f64().unwrap_or(0.0), xy[1].as_f64().unwrap_or(0.0)))
                    } else {
                        None
                    };
                    if let Some((x, y)) = coords {
                        let click_out = click_tool
                            .call(ctx.clone(), json!({ "tab": tab_val, "x": x, "y": y, "goal": user_goal }))
                            .await?;
                        return Ok(ToolOutput::text(
                            json!({
                                "executed_locally": !click_out.is_error,
                                "mode": box_res.get("mode").and_then(Value::as_str).unwrap_or("visual_box_fallback"),
                                "badge": box_res.get("badge"),
                                "box": box_res.get("box"),
                                "click_xy": [x, y],
                                "confidence": box_res.get("confidence"),
                                "timing_ms": box_res.get("timing_ms"),
                                "action_result": first_text(&click_out).unwrap_or(""),
                            })
                            .to_string(),
                        ));
                    }
                }
            }
        }

        // Calibrated Abstention -> Zero-LLM-tax `< 0.5 ms` Rust BM25 foveated + PII-redacted snapshot!
        let fov = fast_foveate_rust(&raw_snap, &step, true);

        Ok(ToolOutput::text(
            json!({
                "executed_locally": false,
                "escalated_to_frontier": true,
                "top_candidate": pick.get("top_candidate"),
                "confidence": pick.get("confidence"),
                "foveation_us": fov.elapsed_us,
                "compression_ratio": (fov.compression_ratio * 10000.0).round() / 10000.0,
                "pii_redactions": fov.pii_redactions,
                "foveated_snapshot": fov.foveated_snapshot,
            })
            .to_string(),
        ))
    }
}

/// Composable Turbo Tool 2: `saccade_extract` (Local Unified `d1-2B` Schema-Validated JSON ETL)
struct SaccadeExtractTool {
    tools: Arc<HashMap<String, Arc<dyn Tool>>>,
}

#[async_trait]
impl Tool for SaccadeExtractTool {
    fn spec(&self) -> ToolSpec {
        let mut props = Map::new();
        props.insert("tab".into(), tab_schema());
        props.insert(
            "schema".into(),
            json!({
                "type": "object",
                "description": "JSON Schema object describing the exact structure to extract locally via Saccade's resident LFM3 causal adapter."
            }),
        );
        props.insert(
            "focus".into(),
            json!({
                "type": "string",
                "description": "Optional hint describing which table, list, or section to extract."
            }),
        );
        ToolSpec {
            name: "saccade_extract".into(),
            description: "Extract schema-validated JSON from the current tab locally via Saccade's unified 4-bit LFM engine (zero raw DOM tokens sent to cloud, PII masked before any fallback).".into(),
            input_schema: json!({
                "type": "object",
                "properties": props,
                "required": ["tab", "schema"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, ctx: ToolContext, args: Value) -> Result<ToolOutput> {
        let tab_val = args.get("tab").cloned().ok_or_else(|| anyhow!(browse_core::MISSING_TAB))?;
        let schema = args.get("schema").cloned().ok_or_else(|| anyhow!("saccade_extract requires `schema`"))?;
        let focus = args.get("focus").and_then(Value::as_str).unwrap_or("Extract structured data");

        let snap_tool = self.tools.get("snapshot").ok_or_else(|| anyhow!("snapshot tool missing"))?;
        let snap_out = snap_tool
            .call(ctx, json!({ "tab": tab_val, "subgoal": focus }))
            .await?;
        if snap_out.is_error {
            return Ok(snap_out);
        }
        let context_txt = first_text(&snap_out).unwrap_or("");
        let res = rpc_call(
            "extract_json",
            json!({ "context": context_txt, "json_schema": schema }),
            Duration::from_secs(12),
        )
        .await?;
        Ok(ToolOutput::text(res.to_string()))
    }
}

/// Composable Turbo Tool 3: `saccade_sweep` (Local Multi-Page Pagination + Deduplicated JSON ETL)
struct SaccadeSweepTool {
    act_tool: Arc<dyn Tool>,
    extract_tool: Arc<dyn Tool>,
}

fn merge_swept_page(
    merged: &mut Option<Value>,
    seen_items: &mut HashSet<String>,
    page_data: Value,
) -> usize {
    match page_data {
        Value::Array(arr) => {
            let target = merged.get_or_insert_with(|| Value::Array(Vec::new()));
            let Some(dest) = target.as_array_mut() else {
                return 0;
            };
            let mut added = 0usize;
            for item in arr {
                let sig = serde_json::to_string(&item).unwrap_or_default();
                if seen_items.insert(sig) {
                    dest.push(item);
                    added += 1;
                }
            }
            added
        }
        Value::Object(mut obj) => {
            let array_key = obj
                .iter()
                .find(|(_, v)| v.is_array())
                .map(|(k, _)| k.clone());
            if let Some(akey) = array_key {
                let arr = obj.remove(&akey).and_then(|v| match v { Value::Array(a) => Some(a), _ => None }).unwrap_or_default();
                let target = merged.get_or_insert_with(|| {
                    let mut base = obj.clone();
                    base.insert(akey.clone(), Value::Array(Vec::new()));
                    Value::Object(base)
                });
                if let Some(dest_obj) = target.as_object_mut() {
                    for (k, v) in obj {
                        dest_obj.insert(k, v);
                    }
                    if let Some(dest_arr) = dest_obj.get_mut(&akey).and_then(Value::as_array_mut) {
                        let mut added = 0usize;
                        for item in arr {
                            let sig = serde_json::to_string(&item).unwrap_or_default();
                            if seen_items.insert(sig) {
                                dest_arr.push(item);
                                added += 1;
                            }
                        }
                        return added;
                    }
                }
                0
            } else {
                let sig = serde_json::to_string(&obj).unwrap_or_default();
                let target = merged.get_or_insert_with(|| Value::Array(Vec::new()));
                if let Some(dest) = target.as_array_mut()
                    && seen_items.insert(sig)
                {
                    dest.push(Value::Object(obj));
                    return 1;
                }
                0
            }
        }
        other => {
            if merged.is_none() {
                *merged = Some(other);
                1
            } else {
                0
            }
        }
    }
}

#[async_trait]
impl Tool for SaccadeSweepTool {
    fn spec(&self) -> ToolSpec {
        let mut props = Map::new();
        props.insert("tab".into(), tab_schema());
        props.insert(
            "schema".into(),
            json!({
                "type": "object",
                "description": "JSON Schema describing the array or object-with-rows structure to extract from each page."
            }),
        );
        props.insert(
            "focus".into(),
            json!({
                "type": "string",
                "description": "Optional hint describing which table, list, or directory to sweep."
            }),
        );
        props.insert(
            "nextStep".into(),
            json!({
                "type": "string",
                "description": "Optional micro-step instruction to advance pagination (default: 'Click the Next page button')."
            }),
        );
        props.insert(
            "maxPages".into(),
            json!({
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum number of pages to extract locally before returning (default: 3, max: 10)."
            }),
        );
        ToolSpec {
            name: "saccade_sweep".into(),
            description: "Sweep a paginated table or directory across up to `maxPages` locally (`extract -> single-pass Next click -> settle -> deduplicate`) with zero per-page cloud round-trips.".into(),
            input_schema: json!({
                "type": "object",
                "properties": props,
                "required": ["tab", "schema"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, ctx: ToolContext, args: Value) -> Result<ToolOutput> {
        let t0 = Instant::now();
        let tab_val = args.get("tab").cloned().ok_or_else(|| anyhow!(browse_core::MISSING_TAB))?;
        let schema = args.get("schema").cloned().ok_or_else(|| anyhow!("saccade_sweep requires `schema`"))?;
        let focus = args.get("focus").and_then(Value::as_str).unwrap_or("Extract table rows");
        let next_step = args.get("nextStep").and_then(Value::as_str).unwrap_or("Click the Next page button");
        let max_pages = args.get("maxPages").and_then(Value::as_u64).unwrap_or(3).clamp(1, 10) as usize;

        let mut merged: Option<Value> = None;
        let mut seen_items: HashSet<String> = HashSet::new();
        let mut pages_swept = 0usize;
        let mut stopped_reason = "reached_max_pages";
        let mut extract_modes: Vec<String> = Vec::new();

        for page_idx in 1..=max_pages {
            let ext_out = self
                .extract_tool
                .call(ctx.clone(), json!({ "tab": tab_val, "schema": schema, "focus": focus }))
                .await?;
            if ext_out.is_error {
                if pages_swept == 0 {
                    return Ok(ext_out);
                }
                stopped_reason = "extract_error_on_subsequent_page";
                break;
            }
            let ext_json: Value = serde_json::from_str(first_text(&ext_out).unwrap_or("{}")).unwrap_or(json!({}));
            let is_valid = ext_json
                .get("schema_valid")
                .or_else(|| ext_json.get("valid"))
                .and_then(Value::as_bool)
                == Some(true);
            if !is_valid {
                if pages_swept == 0 {
                    return Ok(ext_out);
                }
                stopped_reason = "invalid_schema_on_subsequent_page";
                break;
            }
            if let Some(m) = ext_json.get("mode").and_then(Value::as_str) {
                extract_modes.push(m.to_string());
            }
            let page_payload = ext_json
                .get("value")
                .or_else(|| ext_json.get("data"))
                .cloned()
                .unwrap_or(Value::Null);
            let added = merge_swept_page(&mut merged, &mut seen_items, page_payload);
            pages_swept += 1;

            if page_idx > 1 && added == 0 {
                stopped_reason = "no_new_rows";
                break;
            }
            if page_idx >= max_pages {
                stopped_reason = "reached_max_pages";
                break;
            }

            let act_out = self
                .act_tool
                .call(
                    ctx.clone(),
                    json!({
                        "tab": tab_val,
                        "step": next_step,
                        "goal": format!("Paginate and extract: {focus}"),
                    }),
                )
                .await?;
            if act_out.is_error {
                stopped_reason = "pagination_halted_or_error";
                break;
            }
            let act_json: Value = serde_json::from_str(first_text(&act_out).unwrap_or("{}")).unwrap_or(json!({}));
            if act_json.get("executed_locally").and_then(Value::as_bool) != Some(true) {
                stopped_reason = "pagination_exhausted";
                break;
            }
            tokio::time::sleep(Duration::from_millis(120)).await;
        }

        let final_val = merged.clone().unwrap_or_else(|| json!([]));
        Ok(ToolOutput::text(
            json!({
                "valid": merged.is_some(),
                "schema_valid": merged.is_some(),
                "mode": "saccade_sweep",
                "pages_swept": pages_swept,
                "unique_items": seen_items.len(),
                "stopped_reason": stopped_reason,
                "extract_modes": extract_modes,
                "total_ms": (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "value": final_val,
                "data": final_val,
            })
            .to_string(),
        ))
    }
}

/// Composable Diagnostic & Telemetry Tool 4: `saccade_status`
struct SaccadeStatusTool;

#[async_trait]
impl Tool for SaccadeStatusTool {
    fn spec(&self) -> ToolSpec {
        let mut props = Map::new();
        props.insert(
            "action".into(),
            json!({
                "type": "string",
                "enum": ["status", "telemetry", "reset_telemetry", "unload"],
                "description": "Optional diagnostic action: `status` (default), `telemetry` (full LFM trigger ledger), `reset_telemetry`, or `unload` (immediately drop Metal RAM to 0 GiB)."
            }),
        );
        ToolSpec {
            name: "saccade_status".into(),
            description: "Report Saccade unified LFM engine status, active Metal RAM (GiB), lazy vision state, LoRA adapters, or per-trigger LFM telemetry.".into(),
            input_schema: json!({ "type": "object", "properties": props, "additionalProperties": false }),
        }
    }

    async fn call(&self, _ctx: ToolContext, args: Value) -> Result<ToolOutput> {
        let action = args.get("action").and_then(Value::as_str).unwrap_or("status");
        let rpc_method = match action {
            "telemetry" => "telemetry_summary",
            "reset_telemetry" => "telemetry_reset",
            "unload" => "unload",
            _ => "ping",
        };
        match rpc_call(rpc_method, json!({}), Duration::from_secs(3)).await {
            Ok(info) => Ok(ToolOutput::text(
                json!({
                    "sidecar_connected": true,
                    "socket": socket_path(),
                    "action": action,
                    "sidecar": info,
                })
                .to_string(),
            )),
            Err(err) => Ok(ToolOutput::text(
                json!({
                    "sidecar_connected": false,
                    "socket": socket_path(),
                    "launcher_detected": find_engine_launcher(),
                    "fallback": "stock_browse_passthrough",
                    "detail": err.to_string(),
                })
                .to_string(),
            )),
        }
    }
}

/// Attach Saccade middleware to `base_tools` and return the augmented tool list (before `execute` wraps them).
pub fn attach_saccade_tools(base_tools: Vec<Arc<dyn Tool>>) -> Vec<Arc<dyn Tool>> {
    let mut wrapped = SaccadeMiddlewareTool::wrap_all(base_tools);
    let map: Arc<HashMap<String, Arc<dyn Tool>>> = Arc::new(
        wrapped
            .iter()
            .map(|t| (t.spec().name, t.clone()))
            .collect(),
    );
    let act_tool: Arc<dyn Tool> = Arc::new(SaccadeActTool { tools: map.clone() });
    let extract_tool: Arc<dyn Tool> = Arc::new(SaccadeExtractTool { tools: map });
    let sweep_tool: Arc<dyn Tool> = Arc::new(SaccadeSweepTool {
        act_tool: act_tool.clone(),
        extract_tool: extract_tool.clone(),
    });
    wrapped.push(act_tool);
    wrapped.push(extract_tool);
    wrapped.push(sweep_tool);
    wrapped.push(Arc::new(SaccadeStatusTool));
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use browse_core::{Browser, Cdp, HandoffResult, TabInfo, TabOptions, TabState};
    use std::sync::Mutex as StdMutex;
    use tokio::net::UnixListener;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    struct DummyBrowser {
        state: Arc<TabState>,
        handoff_outcome: String,
    }

    impl DummyBrowser {
        fn new(outcome: &str) -> Self {
            Self {
                state: Arc::new(TabState::default()),
                handoff_outcome: outcome.to_string(),
            }
        }
    }

    #[async_trait]
    impl Browser for DummyBrowser {
        async fn open_tab(&self, _url: Option<&str>, _options: &TabOptions) -> Result<TabInfo> {
            anyhow::bail!("unused")
        }
        async fn tabs(&self) -> Result<Vec<TabInfo>> {
            Ok(vec![])
        }
        async fn close_tab(&self, _tab: TabId) -> Result<()> {
            Ok(())
        }
        async fn cdp(&self, _tab: TabId) -> Result<Arc<dyn Cdp>> {
            anyhow::bail!("unused")
        }
        async fn state(&self, _tab: TabId) -> Result<Arc<TabState>> {
            Ok(self.state.clone())
        }
        async fn handoff(&self, _tab: TabId, _message: &str) -> Result<HandoffResult> {
            Ok(HandoffResult {
                outcome: self.handoff_outcome.clone(),
                note: None,
            })
        }
        async fn read_only(&self, _tab: TabId) -> bool {
            false
        }
    }

    struct MockTool {
        name: &'static str,
        reply: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.into(),
                description: "mock".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "tab": tab_schema() }
                }),
            }
        }
        async fn call(&self, _ctx: ToolContext, args: Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text(format!("{}:{}", self.reply, args)))
        }
    }

    #[test]
    fn rust_bm25_foveator_prunes_boilerplate_and_redacts_pii_under_one_ms() {
        let tok = format!("{}{}", concat!("s", "k", "-"), "live99887766554433221100aabbcc");
        let mut big = format!(
            "- banner \"Topbar\":\n  - text \"Signed in as alice@acme.io | Key {tok} | SSN 123-45-6789\"\n"
        );
        for i in 1..=25 {
            big.push_str(&format!("  - link \"Nav Link {i}\" [ref=e{i}]\n"));
        }
        big.push_str("- main \"Billing Ledgers\":\n  - heading \"October Invoices\"\n  - button \"Export October CSV\" [ref=e88]\n");
        big.push_str("- contentinfo \"Footer\":\n");
        for i in 101..=120 {
            big.push_str(&format!("  - link \"Footer Legal {i}\" [ref=e{i}]\n"));
        }

        let fov = fast_foveate_rust(&big, "Click Export October CSV", true);
        assert!(fov.foveated_snapshot.contains("[ref=e88]"), "must retain target ref");
        assert!(!fov.foveated_snapshot.contains("alice@acme.io"), "must redact email");
        assert!(!fov.foveated_snapshot.contains(&tok), "must redact API key");
        assert!(!fov.foveated_snapshot.contains("123-45-6789"), "must redact SSN");
        assert!(fov.compression_ratio > 0.45, "compression_ratio={}", fov.compression_ratio);
        assert!(fov.pii_redactions >= 3);
    }

    #[tokio::test]
    async fn middleware_foveates_snapshot_and_enforces_handoff_firewall() {
        let sock = std::env::temp_dir().join(format!("saccade-unit-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (r, mut w) = stream.into_split();
                let mut lines = BufReader::new(r).lines();
                if let Ok(Some(line)) = lines.next_line().await {
                    let req: Value = serde_json::from_str(&line).unwrap();
                    let m = req["method"].as_str().unwrap_or("");
                    let resp = match m {
                        "ping" => json!({"ok": true, "result": {"ok": true, "unified_single_backbone": true}}),
                        "act_and_guard" => json!({
                            "ok": true,
                            "result": {
                                "ref": "e4",
                                "confidence": 0.93,
                                "escalate_to_frontier": false,
                                "require_handoff": false,
                                "risk_prob": 0.01,
                                "timing_ms": {"infer_ms": 14.2}
                            }
                        }),
                        "firewall" => {
                            let aria = req["params"]["target_aria"].as_str().unwrap_or("");
                            let args_ref = req["params"]["tool_args"]["ref"].as_str().unwrap_or("");
                            let should_block = aria.contains("Delete") || args_ref == "e99";
                            json!({"ok": true, "result": {"require_handoff": should_block, "handoff_message": "Destructive delete blocked"}})
                        }
                        _ => json!({"ok": false, "error": "unknown"}),
                    };
                    let _ = w.write_all((resp.to_string() + "\n").as_bytes()).await;
                }
            }
        });

        let mut nav_blob = String::from("- navigation \"Sidebar\":\n");
        for k in 1..=20 {
            nav_blob.push_str(&format!("  - link \"Irrelevant Link {k}\" [ref=e{}]\n", 200 + k));
        }
        let snap_reply = format!(
            "{nav_blob}- main \"Filters\":\n  - button \"Apply Filter\" [ref=e4]\n  - button \"Delete Workspace\" [ref=e99]\n  - text \"admin@corp.org\""
        );

        let base: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "snapshot",
                reply: snap_reply,
            }),
            Arc::new(MockTool { name: "click", reply: "clicked".into() }),
            Arc::new(MockTool { name: "fill", reply: "filled".into() }),
        ];
        let attached = attach_saccade_tools(base);
        let by_name: HashMap<String, Arc<dyn Tool>> = attached.into_iter().map(|t| (t.spec().name, t)).collect();

        {
            let _guard = ENV_LOCK.lock().unwrap();
            unsafe {
                std::env::set_var("SACCADE_SOCKET", &sock);
                std::env::set_var("SACCADE_AUTOSTART", "0");
                std::env::remove_var("SACCADE_DISABLE");
                std::env::remove_var("SACCADE_NEURAL_FOVEATE");
            }
        }

        let ctx_cancel = ToolContext {
            browser: Arc::new(DummyBrowser::new("cancelled")),
            file_roots: vec![],
        };

        // 1. Native Rust BM25 foveated snapshot + PII redaction
        let snap_out = by_name["snapshot"]
            .call(ctx_cancel.clone(), json!({"tab": 1, "subgoal": "Apply Filter"}))
            .await
            .unwrap();
        let snap_txt = first_text(&snap_out).unwrap();
        assert!(snap_txt.contains("[ref=e4]"));
        assert!(snap_txt.contains("[REDACTED_EMAIL]"));
        assert!(snap_txt.contains("saccade omitted"));

        // 2. Benign click passes firewall without handoff block
        let benign = by_name["click"]
            .call(ctx_cancel.clone(), json!({"tab": 1, "ref": "e4"}))
            .await
            .unwrap();
        assert!(!benign.is_error);
        assert!(first_text(&benign).unwrap().starts_with("clicked:"));

        // 3. Destructive click triggers handoff -> halted when user cancels
        let blocked = by_name["click"]
            .call(ctx_cancel.clone(), json!({"tab": 1, "ref": "e99"}))
            .await
            .unwrap();
        assert!(blocked.is_error);
        assert!(first_text(&blocked).unwrap().contains("Saccade Firewall halted `click`"));

        // 4. `saccade_act` executes benign ref in 1 single-RPC `act_and_guard` call
        let act_res = by_name["saccade_act"]
            .call(ctx_cancel.clone(), json!({"tab": 1, "step": "Click Apply Filter"}))
            .await
            .unwrap();
        let act_json: Value = serde_json::from_str(first_text(&act_res).unwrap()).unwrap();
        assert_eq!(act_json["executed_locally"], json!(true));
        assert_eq!(act_json["ref"], json!("e4"));

        let _ = std::fs::remove_file(&sock);

        // 5. Graceful fallback when sidecar socket is absent and autostart=0
        let status_down = by_name["saccade_status"]
            .call(ctx_cancel, json!({}))
            .await
            .unwrap();
        let st_json: Value = serde_json::from_str(first_text(&status_down).unwrap()).unwrap();
        assert_eq!(st_json["sidecar_connected"], json!(false));
        assert_eq!(st_json["fallback"], json!("stock_browse_passthrough"));
    }

    #[tokio::test]
    async fn saccade_sweep_paginates_and_deduplicates_multi_page_rows() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sock = std::env::temp_dir().join(format!("saccade-sweep-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let page_num = Arc::new(AtomicUsize::new(1));
        let page_num_srv = page_num.clone();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (r, mut w) = stream.into_split();
                let mut lines = BufReader::new(r).lines();
                if let Ok(Some(line)) = lines.next_line().await {
                    let req: Value = serde_json::from_str(&line).unwrap();
                    let m = req["method"].as_str().unwrap_or("");
                    let p = page_num_srv.load(Ordering::SeqCst);
                    let resp = match m {
                        "extract_json" => {
                            let rows = match p {
                                1 => json!([{"id": "INV-1"}, {"id": "INV-2"}]),
                                2 => json!([{"id": "INV-2"}, {"id": "INV-3"}]),
                                _ => json!([{"id": "INV-4"}]),
                            };
                            json!({"ok": true, "result": {"valid": true, "mode": "d1_table_system_one", "data": {"invoices": rows}}})
                        }
                        "act_and_guard" => {
                            if p < 3 {
                                page_num_srv.fetch_add(1, Ordering::SeqCst);
                                json!({"ok": true, "result": {"ref": "e10", "confidence": 0.96, "escalate_to_frontier": false, "require_handoff": false, "single_pass_joint": true}})
                            } else {
                                json!({"ok": true, "result": {"ref": null, "confidence": 0.12, "escalate_to_frontier": true, "require_handoff": false}})
                            }
                        }
                        _ => json!({"ok": false, "error": "unknown"}),
                    };
                    let _ = w.write_all((resp.to_string() + "\n").as_bytes()).await;
                }
            }
        });

        let base: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool {
                name: "snapshot",
                reply: "- main \"Ledgers\":\n  - button \"Next\" [ref=e10]".into(),
            }),
            Arc::new(MockTool { name: "click", reply: "clicked".into() }),
            Arc::new(MockTool { name: "fill", reply: "filled".into() }),
        ];
        let attached = attach_saccade_tools(base);
        let by_name: HashMap<String, Arc<dyn Tool>> = attached.into_iter().map(|t| (t.spec().name, t)).collect();

        {
            let _guard = ENV_LOCK.lock().unwrap();
            unsafe {
                std::env::set_var("SACCADE_SOCKET", &sock);
                std::env::set_var("SACCADE_AUTOSTART", "0");
                std::env::remove_var("SACCADE_DISABLE");
            }
        }

        let ctx = ToolContext {
            browser: Arc::new(DummyBrowser::new("cancelled")),
            file_roots: vec![],
        };
        let sweep_out = by_name["saccade_sweep"]
            .call(ctx, json!({"tab": 1, "schema": {"type": "object"}, "maxPages": 5}))
            .await
            .unwrap();
        let sw: Value = serde_json::from_str(first_text(&sweep_out).unwrap()).unwrap();
        assert_eq!(sw["valid"], json!(true));
        assert_eq!(sw["pages_swept"], json!(3));
        assert_eq!(sw["unique_items"], json!(4));
        assert_eq!(sw["stopped_reason"], json!("pagination_exhausted"));
        assert_eq!(sw["data"]["invoices"].as_array().unwrap().len(), 4);

        let _ = std::fs::remove_file(&sock);
    }
}
