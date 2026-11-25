"""Subgoal-conditioned ariaSnapshot foveation (sub-millisecond BM25 + optional d1-2B pass) & PII masking."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    from .multi_lora import ResidentD1MultiLoRA
except ImportError:
    from multi_lora import ResidentD1MultiLoRA

REF_RE = re.compile(r"\[ref=(e\d+)\b([^\]]*)\]")
LINE_RE = re.compile(r'^(?P<indent>\s*)-\s+(?P<role>[a-zA-Z0-9_-]+)(?:\s+"(?P<name>(?:\\.|[^"])*)")?')
WORD_RE = re.compile(r"[a-z0-9]{2,}")
NAV_INTENT_RE = re.compile(r"\b(?:nav|navigation|navbar|header|menu|topbar)\b", re.IGNORECASE)

STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "click", "press", "select",
    "choose", "open", "page", "tab", "button", "link", "field", "input", "box", "section",
    "current", "inspect", "main", "view", "item", "user", "browser", "step",
}

BOILERPLATE_ROLES = {"banner", "contentinfo", "complementary", "navigation"}
NAV_ROLES = {"banner", "navigation"}
PRIORITY_ROLES = {"dialog", "alertdialog", "alert", "main", "form", "table", "region"}

PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~+/=-]{20,})\b"
        ),
        "[REDACTED_SECRET]",
    ),
]


@dataclass
class SnapshotSection:
    sec_id: str
    role: str
    name: str
    lines: list[str]
    refs: list[str]
    always_keep: bool = False


def mask_pii_spans(text: str) -> tuple[str, int]:
    redactions = 0
    out = text
    for pat, repl in PII_PATTERNS:
        out, n = pat.subn(repl, out)
        redactions += n
    return out, redactions


def _stem_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for w in WORD_RE.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("es"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        toks.append(w)
    return toks


def parse_snapshot_sections(snapshot_text: str) -> list[SnapshotSection]:
    raw_lines = [ln for ln in snapshot_text.splitlines() if ln.strip()]
    if not raw_lines:
        return []

    sections: list[SnapshotSection] = []
    cur_lines: list[str] = []
    cur_role = "root"
    cur_name = ""

    def flush() -> None:
        if not cur_lines:
            return
        refs = [m.group(1) for ln in cur_lines for m in REF_RE.finditer(ln)]
        sec_id = f"s{len(sections)}"
        always = cur_role in {"dialog", "alertdialog", "alert"}
        sections.append(
            SnapshotSection(
                sec_id=sec_id,
                role=cur_role,
                name=cur_name,
                lines=list(cur_lines),
                refs=refs,
                always_keep=always,
            )
        )

    LANDMARK_SPLIT_ROLES = {
        "banner", "navigation", "complementary", "main", "contentinfo",
        "dialog", "alertdialog", "alert", "form", "region", "table", "search",
    }
    for ln in raw_lines:
        m = LINE_RE.match(ln)
        indent = len(m.group("indent")) if m else (len(ln) - len(ln.lstrip(" ")))
        role = m.group("role") if m else "item"
        name = m.group("name") or "" if m else ""
        is_split = (indent == 0) or (indent <= 2 and role in LANDMARK_SPLIT_ROLES)
        if is_split and cur_lines:
            flush()
            cur_lines = [ln]
            cur_role = role
            cur_name = name
        else:
            if not cur_lines:
                cur_role = role
                cur_name = name
            cur_lines.append(ln)
    flush()
    return sections


def fast_foveate_snapshot(
    snapshot_text: str,
    subgoal: str,
    redact_pii: bool = True,
    max_kept_sections: int = 3,
    max_lines_per_section: int = 48,
) -> dict[str, Any]:
    """Sub-millisecond BM25 + structural ARIA subtree pruner & PII redactor (`0` LLM calls)."""
    t0 = time.perf_counter()
    cleaned_text, pii_count = mask_pii_spans(snapshot_text) if redact_pii else (snapshot_text, 0)
    sections = parse_snapshot_sections(cleaned_text)
    orig_chars = len(snapshot_text)

    if len(sections) <= 1 or orig_chars < 500:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "foveated_snapshot": cleaned_text,
            "orig_chars": orig_chars,
            "kept_chars": len(cleaned_text),
            "compression_ratio": round(1.0 - (len(cleaned_text) / max(orig_chars, 1)), 4),
            "pii_redactions": pii_count,
            "sections_kept": len(sections),
            "sections_total": len(sections),
            "mode": "bm25_fast",
            "timing_ms": {"swap_ms": 0.0, "infer_ms": round(dt_ms, 3)},
        }

    q_terms = list(dict.fromkeys(_stem_tokens(subgoal)))
    wants_nav = bool(NAV_INTENT_RE.search(subgoal))
    doc_tokens = [_stem_tokens("\n".join(s.lines)) for s in sections]
    n_docs = len(sections)
    avgdl = max(1.0, sum(len(dt) for dt in doc_tokens) / n_docs)
    df: dict[str, int] = {t: sum(1 for dt in doc_tokens if t in dt) for t in q_terms}

    scores: dict[str, float] = {}
    for sec, d_toks in zip(sections, doc_tokens):
        tf_map: dict[str, int] = {}
        for tok in d_toks:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        dl = len(d_toks)
        bm25 = 0.0
        for t in q_terms:
            tf = tf_map.get(t, 0)
            if tf > 0:
                idf = math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                bm25 += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (dl / avgdl)))

        role_bonus = 0.0
        if sec.role == "main":
            role_bonus += 1.35
        elif sec.role in PRIORITY_ROLES:
            role_bonus += 0.95
        elif wants_nav and sec.role in NAV_ROLES:
            role_bonus += 0.65
        elif sec.role in BOILERPLATE_ROLES and bm25 < 0.8:
            role_bonus -= 1.20
        if sec.refs:
            role_bonus += min(0.45, 0.04 * len(sec.refs))
        scores[sec.sec_id] = bm25 + role_bonus

    ranked = sorted(sections, key=lambda s: scores[s.sec_id], reverse=True)
    top_score = scores[ranked[0].sec_id] if ranked else 0.0
    min_keep_score = max(0.35, top_score * 0.30) if top_score > 1.0 else 0.05
    keep_ids = {s.sec_id for s in sections if s.always_keep}
    for s in ranked:
        if len(keep_ids) >= max_kept_sections:
            break
        if scores[s.sec_id] >= min_keep_score or not keep_ids:
            keep_ids.add(s.sec_id)

    out_lines: list[str] = []
    kept_n = 0
    for sec in sections:
        if sec.sec_id in keep_ids:
            kept_n += 1
            if len(sec.lines) <= max_lines_per_section:
                out_lines.extend(sec.lines)
            else:
                # Within an oversized kept section, retain lines matching query terms + top interactive lines
                scored_lns = []
                for idx, ln in enumerate(sec.lines):
                    ltoks = set(_stem_tokens(ln))
                    hit = sum(2.0 for t in q_terms if t in ltoks)
                    if "[ref=" in ln:
                        hit += 0.5
                    if idx == 0:
                        hit += 10.0
                    scored_lns.append((hit, idx, ln))
                top_idx = {idx for _, idx, _ in sorted(scored_lns, key=lambda x: (-x[0], x[1]))[:max_lines_per_section]}
                for idx, ln in enumerate(sec.lines):
                    if idx in top_idx:
                        out_lines.append(ln)
                omitted_in_sec = len(sec.lines) - len(top_idx)
                if omitted_in_sec > 0:
                    out_lines.append(f"  - ... [saccade trimmed {omitted_in_sec} low-relevance lines in {sec.role}]")
        else:
            out_lines.append(
                f'- ... [saccade omitted {len(sec.lines)} lines / {len(sec.refs)} refs in {sec.role} "{sec.name}"; pass full=true to expand]'
            )

    foveated = "\n".join(out_lines)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "foveated_snapshot": foveated,
        "orig_chars": orig_chars,
        "kept_chars": len(foveated),
        "compression_ratio": round(1.0 - (len(foveated) / max(orig_chars, 1)), 4),
        "pii_redactions": pii_count,
        "sections_kept": kept_n,
        "sections_total": len(sections),
        "section_scores": {k: round(v, 3) for k, v in scores.items()},
        "mode": "bm25_fast",
        "timing_ms": {"swap_ms": 0.0, "infer_ms": round(dt_ms, 3)},
    }


def foveate_snapshot(
    d1: ResidentD1MultiLoRA | None,
    snapshot_text: str,
    subgoal: str,
    keep_threshold: float = 0.25,
    redact_pii: bool = True,
    adapter: str | None = "dom_grounder",
    fast: bool = True,
) -> dict[str, Any]:
    """Prune irrelevant ARIA sections (`fast=True` runs `< 1 ms` BM25 structural pruner; `fast=False` runs d1-2B)."""
    if fast or d1 is None:
        return fast_foveate_snapshot(snapshot_text, subgoal=subgoal, redact_pii=redact_pii)

    cleaned_text, pii_count = mask_pii_spans(snapshot_text) if redact_pii else (snapshot_text, 0)
    sections = parse_snapshot_sections(cleaned_text)
    orig_chars = len(snapshot_text)

    if len(sections) <= 2 or orig_chars < 600:
        return {
            "foveated_snapshot": cleaned_text,
            "orig_chars": orig_chars,
            "kept_chars": len(cleaned_text),
            "compression_ratio": round(1.0 - (len(cleaned_text) / max(orig_chars, 1)), 4),
            "pii_redactions": pii_count,
            "sections_kept": len(sections),
            "sections_total": len(sections),
            "mode": "d1_neural",
            "timing_ms": {"swap_ms": 0.0, "infer_ms": 0.0},
        }

    criteria: dict[str, str] = {}
    summary_rows: list[str] = []
    for sec in sections[:24]:
        preview = " ; ".join(ln.strip() for ln in sec.lines[:5])[:220]
        label_desc = f'{sec.role} "{sec.name}" ({len(sec.refs)} refs): {preview}'
        criteria[sec.sec_id] = label_desc
        summary_rows.append(f"[{sec.sec_id}] {label_desc}")

    state = f"User subgoal: {subgoal}\nPage outline sections:\n" + "\n".join(summary_rows)
    q = {
        "relevant": {
            "type": "choice",
            "multiple": True,
            "instructions": (
                "Which page outline sections are needed to accomplish the user subgoal "
                "(keep target controls, forms, active dialogs, or data tables; drop unrelated navbars/footers)?"
            ),
            "criteria": criteria,
        }
    }
    res = d1.decide(state, q, adapter=adapter)
    probs: dict[str, float] = res["answers"]["relevant"].get("probabilities", {})

    ranked_ids = sorted(probs.keys(), key=lambda k: probs.get(k, 0.0), reverse=True)
    force_top = ranked_ids[0] if ranked_ids else sections[0].sec_id

    out_lines: list[str] = []
    kept_n = 0
    for sec in sections:
        p = probs.get(sec.sec_id, 0.5)
        if sec.always_keep or p >= keep_threshold or sec.sec_id == force_top:
            out_lines.extend(sec.lines)
            kept_n += 1
        else:
            out_lines.append(
                f'- ... [saccade omitted {len(sec.lines)} lines / {len(sec.refs)} refs in {sec.role} "{sec.name}" (p={p:.2f}); pass full=true to expand]'
            )

    foveated = "\n".join(out_lines)
    return {
        "foveated_snapshot": foveated,
        "orig_chars": orig_chars,
        "kept_chars": len(foveated),
        "compression_ratio": round(1.0 - (len(foveated) / max(orig_chars, 1)), 4),
        "pii_redactions": pii_count,
        "sections_kept": kept_n,
        "sections_total": len(sections),
        "section_probs": {k: round(float(v), 4) for k, v in probs.items()},
        "mode": "d1_neural",
        "timing_ms": res["timing_ms"],
    }
