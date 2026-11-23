"""Single-Pass Joint DOM Reflex (`[pick, destructive, injection]`) + Set-of-Marks (`SoM`) Visual Grounder."""

from __future__ import annotations

import math
import re
from typing import Any

from PIL import Image

from .firewall import DESTRUCTIVE_QUESTION, INJECTION_QUESTION
from .foveate import _stem_tokens, fast_foveate_snapshot
from .multi_lora import ResidentD1MultiLoRA
from .som import propose_visual_ui_regions, stamp_som_badges

CAND_RE = re.compile(r"^(?P<indent>\s*)-\s+(?P<body>.*?)\[ref=(?P<ref>e\d+)\b(?P<rest>[^\]]*)\]")


def extract_interactive_refs(
    snapshot_text: str,
    max_candidates: int = 22,
    subgoal: str | None = None,
) -> list[dict[str, str]]:
    """Extract `[ref=eXX]` lines with immediate parent context, ranking by subgoal overlap when > `max_candidates`."""
    candidates: list[dict[str, str]] = []
    parent_stack: list[tuple[int, str]] = []
    for raw in snapshot_text.splitlines():
        if not raw.strip() or raw.startswith("URL: "):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        while parent_stack and parent_stack[-1][0] >= indent:
            parent_stack.pop()

        m = re.search(r"\[ref=(e\d+)\b([^\]]*)\]", stripped)
        if m:
            ref_id = m.group(1)
            clean_line = re.sub(r"\s*\[ref=e\d+[^\]]*\]", "", stripped).lstrip("- ").strip()
            ctx = " > ".join(p[1] for p in parent_stack[-2:])
            desc = f"{clean_line} (in {ctx})" if ctx else clean_line
            candidates.append({"ref": ref_id, "desc": desc[:160], "raw": stripped})
        else:
            label = stripped.lstrip("- ").strip()[:80]
            if label and not label.startswith("..."):
                parent_stack.append((indent, label))

    if len(candidates) <= max_candidates or not subgoal:
        return candidates[:max_candidates]

    q_terms = set(_stem_tokens(subgoal))
    scored = []
    for idx, c in enumerate(candidates):
        ctoks = set(_stem_tokens(c["desc"]))
        overlap = len(q_terms & ctoks)
        boiler = -0.5 if any(b in c["desc"].lower() for b in ("global topbar", "global footer", "legal & compliance")) else 0.0
        scored.append((overlap + boiler, -idx, idx, c))
    top_indices = sorted(idx for _, _, idx, _ in sorted(scored, reverse=True)[:max_candidates])
    return [candidates[i] for i in top_indices]


def pick_dom_ref(
    d1: ResidentD1MultiLoRA,
    subgoal: str,
    snapshot_text: str,
    candidates: list[dict[str, str]] | None = None,
    adapter: str | None = None,
    abstain_tau: float | None = None,
) -> dict[str, Any]:
    """Select the exact `e12` ref for `subgoal` in one zero-token pass, with calibrated abstention."""
    use_adap = adapter if adapter is not None else ("reflex_v2" if "reflex_v2" in d1.adapters else "dom_grounder")
    working_snap = snapshot_text
    if candidates is None and len(snapshot_text) > 1500:
        fov = fast_foveate_snapshot(snapshot_text, subgoal=subgoal, redact_pii=False, max_kept_sections=3)
        working_snap = fov["foveated_snapshot"]

    cands = candidates if candidates is not None else extract_interactive_refs(working_snap, max_candidates=22, subgoal=subgoal)
    if not cands and working_snap is not snapshot_text:
        cands = extract_interactive_refs(snapshot_text, max_candidates=22, subgoal=subgoal)
        working_snap = snapshot_text

    if not cands:
        return {
            "ref": None,
            "confidence": 0.0,
            "margin": 0.0,
            "escalate_to_frontier": True,
            "reason": "no_interactive_refs",
            "timing_ms": {"swap_ms": 0.0, "infer_ms": 0.0},
        }

    slot = d1.adapters.get(use_adap) if use_adap else None
    tau = abstain_tau if abstain_tau is not None else (slot.abstain_tau if slot else 0.45)

    criteria: dict[str, str] = {c["ref"]: c["desc"] for c in cands}
    criteria["abstain"] = "I don't know / target control not present"

    state = (
        f"Authorized user goal: {subgoal}\n"
        f"Current micro-step: {subgoal}\n"
        f"Page snapshot:\n{working_snap[:1800]}"
    )
    q = {
        "pick": {
            "type": "choice",
            "instructions": (
                "Which interactive element ref best executes the current micro-step? "
                "Pick the exact ref, or abstain if none of the listed elements match."
            ),
            "criteria": criteria,
        }
    }
    res = d1.decide(state, q, adapter=use_adap, abstain_below=tau)
    ans = res["answers"]["pick"]
    probs: dict[str, float] = ans.get("probabilities", {})

    ref_probs = sorted([(k, float(v)) for k, v in probs.items() if k != "abstain"], key=lambda kv: kv[1], reverse=True)
    top_ref, top_p = ref_probs[0] if ref_probs else (None, 0.0)
    runner_up_p = ref_probs[1][1] if len(ref_probs) > 1 else 1e-6
    abstain_p = float(probs.get("abstain", 0.0))

    denom = max(runner_up_p + abstain_p, 1e-6)
    log_margin = float(math.log(max(top_p, 1e-6) / denom))
    chosen = ans.get("choice")
    escalate = (chosen == "abstain") or (top_p < tau)

    return {
        "ref": None if escalate else top_ref,
        "top_candidate": top_ref,
        "confidence": round(top_p, 4),
        "abstain_prob": round(abstain_p, 4),
        "log_margin": round(log_margin, 4),
        "escalate_to_frontier": bool(escalate),
        "adapter_used": use_adap,
        "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        "timing_ms": res["timing_ms"],
    }


def act_and_guard_single_pass(
    d1: ResidentD1MultiLoRA,
    subgoal: str,
    user_goal: str,
    snapshot_text: str,
    page_url: str = "",
    abstain_tau: float | None = None,
    alert_tau: float | None = None,
) -> dict[str, Any]:
    """Saccade v2.0 Single-Pass Reflex: answers `[pick, destructive, injection]` in ONE `d1` forward pass!"""
    header_line = ""
    if snapshot_text.startswith("URL: "):
        header_line = snapshot_text.splitlines()[0] + "\n"
    elif page_url:
        is_local = any(h in page_url for h in ("localhost", "127.0.0.1", "/test/", "sandbox"))
        badge = "[DEV / TEST SANDBOX]" if is_local else "[PRODUCTION LIVE]"
        header_line = f"URL: {page_url} {badge}\n"

    working_snap = snapshot_text
    if len(snapshot_text) > 1500:
        fov = fast_foveate_snapshot(snapshot_text, subgoal=subgoal, redact_pii=False, max_kept_sections=3)
        working_snap = fov["foveated_snapshot"]

    cands = extract_interactive_refs(working_snap, max_candidates=22, subgoal=subgoal)
    if not cands and working_snap is not snapshot_text:
        cands = extract_interactive_refs(snapshot_text, max_candidates=22, subgoal=subgoal)
        working_snap = snapshot_text

    if not cands:
        return {
            "ref": None,
            "confidence": 0.0,
            "escalate_to_frontier": True,
            "require_handoff": False,
            "risk_prob": 0.0,
            "single_pass_joint": True,
            "reason": "no_interactive_refs",
            "timing_ms": {"swap_ms": 0.0, "infer_ms": 0.0},
        }

    use_adap = "reflex_v2" if "reflex_v2" in d1.adapters else "dom_grounder"
    slot = d1.adapters.get(use_adap)
    p_tau = abstain_tau if abstain_tau is not None else (slot.abstain_tau if slot else 0.45)
    f_tau = alert_tau if alert_tau is not None else (slot.alert_tau if (slot and use_adap == "reflex_v2") else 0.40)

    criteria: dict[str, str] = {c["ref"]: c["desc"] for c in cands}
    criteria["abstain"] = "I don't know / target control not present"

    body_snap = working_snap if working_snap.startswith("URL: ") else (header_line + working_snap)
    state = (
        f"Authorized user goal: {user_goal}\n"
        f"Current micro-step: {subgoal}\n"
        f"Page snapshot:\n{body_snap[:1900]}"
    )
    questions = {
        "pick": {
            "type": "choice",
            "instructions": (
                "Which interactive element ref best executes the current micro-step? "
                "Pick the exact ref, or abstain if none of the listed elements match."
            ),
            "criteria": criteria,
        },
        "destructive": DESTRUCTIVE_QUESTION,
        "injection": INJECTION_QUESTION,
    }
    res = d1.decide(state, questions, adapter=use_adap, abstain_below=p_tau)
    ans_p = res["answers"]["pick"]
    probs: dict[str, float] = ans_p.get("probabilities", {})

    ref_probs = sorted([(k, float(v)) for k, v in probs.items() if k != "abstain"], key=lambda kv: kv[1], reverse=True)
    top_ref, top_p = ref_probs[0] if ref_probs else (None, 0.0)
    runner_up_p = ref_probs[1][1] if len(ref_probs) > 1 else 1e-6
    abstain_p = float(probs.get("abstain", 0.0))
    log_margin = float(math.log(max(top_p, 1e-6) / max(runner_up_p + abstain_p, 1e-6)))
    chosen = ans_p.get("choice")
    escalate = (chosen == "abstain") or (top_p < p_tau)

    raw_dest = float(res["answers"]["destructive"]["noul"])
    raw_inj = float(res["answers"]["injection"]["noul"])
    risk_prob = max(raw_dest, raw_inj)
    require_handoff = risk_prob >= f_tau

    handoff_msg = None
    if require_handoff:
        if raw_inj >= raw_dest:
            handoff_msg = (
                f"Saccade Firewall paused `{subgoal[:60]}` (p={risk_prob:.2f}): potential indirect prompt injection "
                f"detected in untrusted page content. Click Continue only if intended."
            )
        else:
            handoff_msg = (
                f"Saccade Firewall paused `{subgoal[:60]}` (p={risk_prob:.2f}): unauthorized destructive or "
                f"high-consequence action on live session. Click Continue to approve."
            )

    return {
        "ref": None if escalate else top_ref,
        "top_candidate": top_ref,
        "confidence": round(top_p, 4),
        "abstain_prob": round(abstain_p, 4),
        "log_margin": round(log_margin, 4),
        "escalate_to_frontier": bool(escalate),
        "require_handoff": bool(require_handoff),
        "risk_prob": round(risk_prob, 4),
        "destructive_prob": round(raw_dest, 4),
        "injection_prob": round(raw_inj, 4),
        "handoff_message": handoff_msg,
        "single_pass_joint": True,
        "adapter_used": use_adap,
        "timing_ms": res["timing_ms"],
    }


def ground_som_visual(
    d1: ResidentD1MultiLoRA,
    target_instruction: str,
    screenshot: Image.Image,
    candidate_boxes: list[dict[str, Any]] | None = None,
    adapter: str | None = None,
    abstain_tau: float = 0.28,
) -> dict[str, Any]:
    """Set-of-Marks (`[1]..[12]`) visual grounding for `<canvas>`, charts, and unlabeled SVG icons."""
    orig_w, orig_h = screenshot.size
    boxes = candidate_boxes if candidate_boxes else propose_visual_ui_regions(screenshot, max_regions=10)
    if not boxes:
        return {
            "found": False,
            "mode": "som_badges",
            "click_xy": None,
            "confidence": 0.0,
            "escalate_to_frontier": True,
            "timing_ms": {"swap_ms": 0.0, "infer_ms": 0.0},
        }

    # Visual foveation: if DOM candidates cluster in a compact toolbar/panel (<55% width & <45% height),
    # crop with padding before resizing so small 24-48px SVG glyphs stay crisp instead of shrinking to <10px!
    min_l = min(float(b["left"]) for b in boxes)
    max_r = max(float(b["right"]) for b in boxes)
    min_t = min(float(b["top"]) for b in boxes)
    max_b = max(float(b["bottom"]) for b in boxes)
    full_centers = {
        str(i): [round(0.5 * (float(c["left"]) + float(c["right"])), 4), round(0.5 * (float(c["top"]) + float(c["bottom"])), 4)]
        for i, c in enumerate(boxes[:12], start=1)
    }
    work_img = screenshot
    work_boxes = boxes
    if (max_r - min_l) < 0.55 and (max_b - min_t) < 0.45 and orig_w >= 400 and orig_h >= 300:
        cl = max(0.0, min_l - 0.05)
        ct = max(0.0, min_t - 0.05)
        cr = min(1.0, max(max_r + 0.08, cl + 0.35))
        cb = min(1.0, max(max_b + 0.10, ct + 0.25))
        cw, ch = max(cr - cl, 1e-4), max(cb - ct, 1e-4)
        work_img = screenshot.crop((int(cl * orig_w), int(ct * orig_h), int(cr * orig_w), int(cb * orig_h)))
        work_boxes = [
            {
                **b,
                "left": max(0.0, min(1.0, (float(b["left"]) - cl) / cw)),
                "top": max(0.0, min(1.0, (float(b["top"]) - ct) / cw if False else (float(b["top"]) - ct) / ch)),
                "right": max(0.0, min(1.0, (float(b["right"]) - cl) / cw)),
                "bottom": max(0.0, min(1.0, (float(b["bottom"]) - ct) / ch)),
            }
            for b in boxes
        ]

    tagged_im, criteria, _ = stamp_som_badges(work_img, work_boxes)
    centers = full_centers
    use_adap = adapter if adapter is not None else ("som_grounder" if "som_grounder" in d1.adapters else None)
    state = f"Target UI element to click: {target_instruction}"
    q = {
        "som": {
            "type": "choice",
            "instructions": (
                "Each candidate UI control is outlined in amber with a numbered dark tag [1]..[N]. "
                "Which numbered badge marks the target UI element, or abstain if none match?"
            ),
            "criteria": criteria,
        }
    }
    res = d1.decide(state, q, media=[tagged_im], adapter=use_adap, abstain_below=abstain_tau)
    ans = res["answers"]["som"]
    probs: dict[str, float] = ans.get("probabilities", {})
    non_abs = sorted([(k, float(v)) for k, v in probs.items() if k != "abstain"], key=lambda kv: kv[1], reverse=True)
    best_badge, best_p = non_abs[0] if non_abs else (None, 0.0)
    chosen = ans.get("choice")
    escalate = (chosen == "abstain") or (best_badge is None) or (best_p < abstain_tau)

    if escalate or best_badge not in centers:
        return {
            "found": False,
            "mode": "som_badges",
            "badge": None,
            "top_badge": best_badge,
            "norm_xy": None,
            "click_xy": None,
            "confidence": round(best_p, 4),
            "escalate_to_frontier": True,
            "adapter_used": use_adap,
            "timing_ms": res["timing_ms"],
        }

    nx, ny = centers[best_badge]
    px_x = int(round(nx * orig_w))
    px_y = int(round(ny * orig_h))
    picked_box = boxes[int(best_badge) - 1]
    return {
        "found": True,
        "mode": "som_badges",
        "badge": best_badge,
        "box": {k: round(float(picked_box[k]), 4) for k in ("left", "top", "right", "bottom")},
        "norm_xy": [nx, ny],
        "click_xy": [px_x, px_y],
        "confidence": round(best_p, 4),
        "escalate_to_frontier": False,
        "adapter_used": use_adap,
        "timing_ms": res["timing_ms"],
    }


def ground_visual_box(
    d1: ResidentD1MultiLoRA,
    target_instruction: str,
    screenshot: Image.Image,
    adapter: str | None = None,
    min_confidence: float = 0.20,
    candidate_boxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Visual grounding dispatcher: uses Set-of-Marks `[1]..[12]` when `som_grounder` or `candidate_boxes` are present."""
    if candidate_boxes is not None or "som_grounder" in d1.adapters:
        return ground_som_visual(
            d1,
            target_instruction=target_instruction,
            screenshot=screenshot,
            candidate_boxes=candidate_boxes,
            adapter=adapter if adapter != "ui_box" else None,
            abstain_tau=min_confidence,
        )

    use_adapter = adapter if (adapter and adapter in d1.adapters) else None
    orig_w, orig_h = screenshot.size
    scale = min(512.0 / max(orig_w, 1), 320.0 / max(orig_h, 1), 1.0)
    bw = max(64, int(round((orig_w * scale) / 32.0)) * 32)
    bh = max(64, int(round((orig_h * scale) / 32.0)) * 32)
    bucket_img = screenshot.convert("RGB").resize((bw, bh), Image.Resampling.BILINEAR)
    q = {
        "target": {
            "type": "box",
            "instructions": f"The UI control, button, icon, or region matching: {target_instruction}",
        }
    }
    res = d1.decide(None, q, media=[bucket_img], adapter=use_adapter)
    ans = res["answers"]["target"]
    box = ans.get("box")
    conf = float(ans.get("confidence", 0.0) or 0.0)
    if not box:
        return {
            "found": False,
            "box": None,
            "norm_xy": None,
            "click_xy": None,
            "confidence": round(conf, 4),
            "escalate_to_frontier": True,
            "timing_ms": res["timing_ms"],
        }

    nx = 0.5 * (float(box["left"]) + float(box["right"]))
    ny = 0.5 * (float(box["top"]) + float(box["bottom"]))
    px_x = int(round(nx * orig_w))
    px_y = int(round(ny * orig_h))
    return {
        "found": True,
        "box": {k: round(float(v), 4) for k, v in box.items()},
        "norm_xy": [round(nx, 4), round(ny, 4)],
        "click_xy": [px_x, px_y],
        "confidence": round(conf, 4),
        "escalate_to_frontier": conf < min_confidence,
        "timing_ms": res["timing_ms"],
    }
