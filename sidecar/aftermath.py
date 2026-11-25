"""Post-click UI aftermath judge & Algorithm-R Time-Machine online reservoir buffer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from PIL import Image

try:
    from .multi_lora import ResidentD1MultiLoRA
except ImportError:
    from multi_lora import ResidentD1MultiLoRA

AFTERMATH_CRITERIA = {
    "SETTLED_OK": "The UI updated cleanly and is ready for the next step (new view, opened menu/modal, or filled field)",
    "SPINNER_LOADING": "A loading spinner, skeleton placeholder, progress bar, or 'Loading...' state is still active",
    "INLINE_ERROR_TOAST": "A validation error, red alert banner, toast error, or failed-action message appeared",
    "AUTH_OR_BOT_WALL": "A login screen, 2FA prompt, session-expired modal, or anti-bot / captcha wall appeared",
}


def stitch_before_after(before: Image.Image, after: Image.Image, panel_w: int = 320, panel_h: int = 200) -> Image.Image:
    """Pack `[before, after]` crops side-by-side into one compact image so d1-2B encodes patch diffs in 1 media slot."""
    b = before.convert("RGB").resize((panel_w, panel_h), Image.Resampling.BILINEAR)
    a = after.convert("RGB").resize((panel_w, panel_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (panel_w * 2 + 8, panel_h), (247, 246, 239))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (panel_w + 8, 0))
    return canvas


def judge_aftermath(
    d1: ResidentD1MultiLoRA,
    action_desc: str,
    aria_diff: str,
    before_crop: Image.Image | None = None,
    after_crop: Image.Image | None = None,
    adapter: str | None = "aftermath_tm",
) -> dict[str, Any]:
    media = None
    if before_crop is not None and after_crop is not None:
        media = [stitch_before_after(before_crop, after_crop)]
    elif after_crop is not None:
        media = [after_crop.convert("RGB")]

    state = (
        f"Executed browser action: {action_desc}\n"
        f"Snapshot delta (+ added / - removed):\n{aria_diff[:2000]}"
    )
    q = {
        "status": {
            "type": "choice",
            "instructions": (
                "Inspect the before/after UI view (left=before, right=after) and snapshot delta. "
                "What is the immediate post-action state of the page?"
            ),
            "criteria": AFTERMATH_CRITERIA,
        }
    }
    res = d1.decide(state, q, media=media, adapter=adapter)
    ans = res["answers"]["status"]
    status = ans["choice"]
    return {
        "status": status,
        "confidence": round(float(ans["confidence"]), 4),
        "recommended_wait_ms": 350 if status == "SPINNER_LOADING" else 0,
        "trigger_handoff": status == "AUTH_OR_BOT_WALL",
        "probabilities": {k: round(float(v), 4) for k, v in ans["probabilities"].items()},
        "timing_ms": res["timing_ms"],
    }


@dataclass
class HindsightEpisode:
    domain: str
    state: str
    pred_probs: dict[str, float]
    realized_status: str
    surprise: float


class AlgorithmRTimeMachineBuffer:
    """Unbiased Vitter Algorithm-R reservoir for zero-human-label `t + 2.0s` browser aftermath adaptation."""

    def __init__(self, capacity: int = 120, seed: int = 42):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.reservoir: list[HindsightEpisode] = []

    def observe(self, domain: str, state: str, pred_probs: dict[str, float], realized_status: str) -> HindsightEpisode:
        p_true = float(pred_probs.get(realized_status, 1e-4))
        surprise = float(1.0 - p_true)
        ep = HindsightEpisode(
            domain=domain,
            state=state,
            pred_probs=dict(pred_probs),
            realized_status=realized_status,
            surprise=surprise,
        )
        self.seen += 1
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(ep)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.capacity:
                self.reservoir[j] = ep
        return ep
