"""Set-of-Marks (SoM) Visual Badge Renderer & Fast Edge-Contour Widget Proposer (`< 4 ms` NumPy).

Used identically across:
  1. `build_v2_suites.py` (`Click-100k` training + `ScreenSpot` evaluation)
  2. `runpod_train_saccade_v2_4090.py` (RTX 4090 `d1_som_grounder` LoRA training)
  3. Live Saccade sidecar (`server.py` / `saccade.rs` visual `<canvas>` & unlabeled-icon fallback)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix1 = max(a["left"], b["left"])
    iy1 = max(a["top"], b["top"])
    ix2 = min(a["right"], b["right"])
    iy2 = min(a["bottom"], b["bottom"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a["right"] - a["left"]) * (a["bottom"] - a["top"]))
    area_b = max(1e-6, (b["right"] - b["left"]) * (b["bottom"] - b["top"]))
    return inter / (area_a + area_b - inter)


def propose_visual_ui_regions(
    img: Image.Image,
    max_regions: int = 12,
    exclude_boxes: list[dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Fast deterministic Sobel-gradient peak proposer finding salient UI buttons/icons/bars in ~3 ms."""
    small = img.convert("L").resize((192, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    mag = gx + gy

    # Score multi-scale UI patch windows (icon-sized 12x12, button-sized 28x14, card/bar-sized 20x28)
    proposals: list[tuple[float, dict[str, float]]] = []
    H, W = mag.shape
    integral = np.pad(mag, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def rect_sum(y0: int, x0: int, y1: int, x1: int) -> float:
        return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])

    for win_w, win_h, stride_x, stride_y in ((12, 12, 8, 8), (28, 14, 12, 8), (18, 26, 10, 10)):
        area = float(win_w * win_h)
        for y0 in range(2, H - win_h - 2, stride_y):
            for x0 in range(2, W - win_w - 2, stride_x):
                score = rect_sum(y0, x0, y0 + win_h, x0 + win_w) / area
                if score < 0.045:
                    continue
                b = {
                    "left": round(x0 / W, 4),
                    "top": round(y0 / H, 4),
                    "right": round((x0 + win_w) / W, 4),
                    "bottom": round((y0 + win_h) / H, 4),
                }
                proposals.append((score, b))

    proposals.sort(key=lambda x: x[0], reverse=True)
    kept: list[dict[str, float]] = []
    blocked = list(exclude_boxes or [])
    for _, cand in proposals:
        if any(box_iou(cand, ex) > 0.12 for ex in blocked):
            continue
        if any(box_iou(cand, prev) > 0.20 for prev in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_regions:
            break

    # Backfill with clean spatial grid anchors if screenshot is ultra-sparse
    if len(kept) < max_regions:
        for gy_i in (0.15, 0.38, 0.62, 0.84):
            for gx_i in (0.14, 0.38, 0.62, 0.86):
                cand = {
                    "left": round(gx_i - 0.05, 4),
                    "top": round(gy_i - 0.04, 4),
                    "right": round(gx_i + 0.05, 4),
                    "bottom": round(gy_i + 0.04, 4),
                }
                if any(box_iou(cand, ex) > 0.12 for ex in blocked):
                    continue
                if any(box_iou(cand, prev) > 0.20 for prev in kept):
                    continue
                kept.append(cand)
                if len(kept) >= max_regions:
                    break
            if len(kept) >= max_regions:
                break
    return kept


def stamp_som_badges(
    screenshot: Image.Image,
    candidates: list[dict[str, Any]],
    max_side_w: int = 512,
    max_side_h: int = 320,
) -> tuple[Image.Image, dict[str, str], dict[str, list[float]]]:
    """Render high-contrast numbered `[1]..[K]` Set-of-Marks badges onto a bounded 32px-grid image.

    Each candidate dict has `{"left": float, "top": float, "right": float, "bottom": float, "hint"?: str}`
    in normalized `[0..1]` coordinates.

    Returns `(annotated_image, choice_criteria, badge_centers_norm)`.
    """
    orig_w, orig_h = screenshot.size
    scale = min(max_side_w / max(orig_w, 1), max_side_h / max(orig_h, 1), 1.0)
    bw = max(96, int(round((orig_w * scale) / 32.0)) * 32)
    bh = max(96, int(round((orig_h * scale) / 32.0)) * 32)
    canvas = screenshot.convert("RGB").resize((bw, bh), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    criteria: dict[str, str] = {}
    centers: dict[str, list[float]] = {}

    for idx, c in enumerate(candidates[:12], start=1):
        bid = str(idx)
        x1 = int(round(float(c["left"]) * bw))
        y1 = int(round(float(c["top"]) * bh))
        x2 = max(x1 + 6, int(round(float(c["right"]) * bw)))
        y2 = max(y1 + 6, int(round(float(c["bottom"]) * bh)))

        # Crisp 2px amber outline around candidate control without obscuring interior glyph
        draw.rectangle([x1, y1, x2, y2], outline=(243, 156, 18), width=2)

        # Anchor compact dark pill tag just above/top-left of box
        tag_w = 14 if idx < 10 else 20
        tag_h = 13
        tx = max(0, min(bw - tag_w - 1, x1))
        ty = y1 - tag_h if y1 >= tag_h else min(bh - tag_h - 1, y1)
        draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=(15, 15, 15), outline=(243, 156, 18), width=1)
        draw.text((tx + 3, ty + 1), bid, fill=(255, 255, 240), font=font)

        cx = round(0.5 * (float(c["left"]) + float(c["right"])), 4)
        cy = round(0.5 * (float(c["top"]) + float(c["bottom"])), 4)
        centers[bid] = [cx, cy]

        quad_y = "top" if cy < 0.34 else ("bottom" if cy > 0.67 else "middle")
        quad_x = "left" if cx < 0.34 else ("right" if cx > 0.67 else "center")
        hint = c.get("hint") or f"UI region in {quad_y}-{quad_x}"
        criteria[bid] = f"Badge [{bid}] ({hint}, pos {int(round(cx*100))}%, {int(round(cy*100))}%)"

    criteria["abstain"] = "None of the numbered badges match the target control"
    return canvas, criteria, centers
