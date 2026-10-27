#!/usr/bin/env python3
"""Build the durable figures used by the research talk.

The figures intentionally use the saved run frames and the independent audit
frames rather than generated illustrations.  That keeps the captions tied to
evidence that another reader can inspect.

Usage:
    .venv/bin/python scripts/build_talk_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FRAMES = FIGURES / "frames"
RUN = ROOT / "data" / "runs" / "q207-temporal-ordering-2026-09-21-2"

BG = (250, 249, 246)
INK = (31, 31, 34)
MUTED = (91, 91, 98)
RULE = (205, 204, 200)
BLUE = (42, 101, 181)
LIGHT_BLUE = (185, 211, 239)
PALE_BLUE = (232, 241, 251)
ORANGE = (218, 111, 33)
RED = (188, 61, 48)
PALE_RED = (252, 232, 228)
GREEN = (51, 119, 66)
PALE_GREEN = (229, 242, 231)
AMBER = (166, 112, 20)
PALE_AMBER = (250, 240, 213)
PURPLE = (105, 78, 150)
WHITE = (255, 255, 255)


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path]
    if mono:
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ]
    elif bold:
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
    else:
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONTS = {
    "title": _font(48, bold=True),
    "subtitle": _font(26),
    "section": _font(29, bold=True),
    "body": _font(24),
    "small": _font(20),
    "tiny": _font(17),
    "mono": _font(18, mono=True),
    "body_bold": _font(24, bold=True),
    "small_bold": _font(20, bold=True),
}


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=INK, anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def wrapped_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    # Pillow's textlength is more reliable than a character count when the
    # same caption needs to fit on both macOS and Linux.
    words = text.split()
    lines: list[str] = []
    current = ""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and probe.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    fill=INK,
    spacing: int = 6,
) -> int:
    x, y = xy
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    for line in wrapped_lines(text, font, max_width):
        draw_text(draw, (x, y), line, font, fill)
        y += line_height + spacing
    return y


def fit_image(path: Path, size: tuple[int, int], background=WHITE) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def place_image(canvas: Image.Image, path: Path, box: tuple[int, int, int, int], border=RULE) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, fill=WHITE, outline=border, width=2)
    fitted = fit_image(path, (x1 - x0 - 8, y1 - y0 - 8))
    x = x0 + (x1 - x0 - fitted.width) // 2
    y = y0 + (y1 - y0 - fitted.height) // 2
    canvas.paste(fitted, (x, y))


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, fill, text_fill=WHITE) -> None:
    x, y = xy
    font = FONTS["small_bold"]
    bbox = draw.textbbox((x, y), label, font=font)
    pad_x, pad_y = 12, 6
    draw.rounded_rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        radius=8,
        fill=fill,
    )
    draw_text(draw, (x, y), label, font, text_fill)


def save(canvas: Image.Image, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    canvas.save(path, format="PNG", optimize=True)
    return path


def build_hockey_audit() -> Path:
    canvas = Image.new("RGB", (2400, 1510), BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (70, 55), "q207 — visual audit of the hockey failure", FONTS["title"])
    draw_text(
        draw,
        (72, 125),
        "Top: the scenes the agent said it found. Bottom: the scenes the benchmark labels. Similar-looking play is not enough.",
        FONTS["subtitle"],
        MUTED,
    )

    margin = 70
    gap = 28
    panel_w = (2400 - 2 * margin - 2 * gap) // 3
    image_h = 300
    top_y = 235
    bottom_y = 900
    top_files = [
        FRAMES / "event_c_agent_matched_17s.png",
        FRAMES / "event_b_agent_matched_61s.png",
        FRAMES / "event_a_agent_matched_96s.png",
    ]
    bottom_files = [
        FRAMES / "event_c_answer_key_47s.png",
        FRAMES / "event_b_answer_key_203s.png",
        FRAMES / "event_a_answer_key_118s.png",
    ]
    top_titles = [
        "Agent's Event C candidate | 17.5s",
        "Agent's Event B candidate | 61.7s",
        "Agent's Event A candidate | 96.5s",
    ]
    bottom_titles = [
        "Benchmark's Event C reference | 43–48s",
        "Benchmark's Event B reference | 202–206s",
        "Benchmark's Event A reference | 115–122s",
    ]
    top_status = [
        ("NOT VERIFIED", RED, PALE_RED, "Appears to show a Ducks puck carrier; the Honda logo alone does not verify Event C."),
        ("NOT THE BOARDS EVENT", RED, PALE_RED, "Play at/near Chicago's net, not the annotated boards battle; number 4 is not verified here."),
        ("POSSIBLE LOOK-ALIKE", AMBER, PALE_AMBER, "Play behind the net, not the asserted crease scramble; outside the annotated window."),
    ]
    bottom_status = [
        ("REFERENCE SCENE", GREEN, PALE_GREEN, "Benchmark window used by condition C; the labeled Event C instance."),
        ("REFERENCE SCENE", GREEN, PALE_GREEN, "Benchmark close-up; white Blackhawks #4 is visible in the boards battle."),
        ("REFERENCE SCENE", GREEN, PALE_GREEN, "Benchmark window; the crease scramble and Ducks goalie are visible."),
    ]

    draw_text(draw, (margin, 185), "What the agent said it found", FONTS["section"], BLUE)
    draw_text(draw, (margin, 850), "What the benchmark labels", FONTS["section"], GREEN)

    for row_y, files, titles, statuses in [
        (top_y, top_files, top_titles, top_status),
        (bottom_y, bottom_files, bottom_titles, bottom_status),
    ]:
        for i, (path, title, (label, color, pale, caption)) in enumerate(zip(files, titles, statuses)):
            x = margin + i * (panel_w + gap)
            draw.rounded_rectangle((x, row_y, x + panel_w, row_y + 545), radius=10, fill=WHITE, outline=RULE, width=2)
            draw_text(draw, (x + 20, row_y + 18), title, FONTS["body_bold"], INK)
            place_image(canvas, path, (x + 20, row_y + 62, x + panel_w - 20, row_y + 62 + image_h))
            badge(draw, (x + 22, row_y + 385), label, color)
            draw_wrapped(draw, (x + 22, row_y + 430), caption, FONTS["small"], panel_w - 44, MUTED, spacing=4)

    draw.line((margin, 1490, 2330, 1490), fill=RULE, width=2)
    draw_text(
        draw,
        (margin, 1460),
        "Plain-English takeaway: the agent found hockey-looking scenes, but did not reliably check the team, player number, puck carrier, or exact play. Keep the failure; drop the claim that its alternative order was correct.",
        FONTS["small"],
        INK,
    )
    return save(canvas, "q207_visual_audit.png")


def build_search_timeline() -> Path:
    question = json.loads((RUN / "question.json").read_text())
    manifest = json.loads((RUN / "images.json").read_text())
    trajectory = json.loads((RUN / "trajectory.json").read_text())
    duration = float(question["video"]["duration_sec"])
    evidence = [(float(e["start_sec"]), float(e["end_sec"]), e["description"].split(":", 1)[0]) for e in question["evidence"]]
    overview_count = sum(1 for item in manifest["images"] if item["tool"] == "overview")
    audit = json.loads((ROOT / "data" / "reviews" / "independent-pilot-2026-09-21" / "integrity-audit.json").read_text())[0]
    distinct_inside = sum(item["distinct_frames"] for item in audit["intervals"])
    target_windows = []
    for step in trajectory["steps"]:
        params = step["action"].get("parameters", {})
        if "start_time" in params and "end_time" in params:
            target_windows.append(
                (
                    float(params["start_time"]),
                    float(params["end_time"]),
                    step["action"]["function"],
                )
            )

    canvas = Image.new("RGB", (2400, 1390), BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (70, 55), "q207 — focused search stopped at 104s; skim ended at 110s", FONTS["title"])
    draw_text(
        draw,
        (72, 125),
        f"The video runs {duration:.0f}s. Overview sampled the whole video; the final annotated window begins at 115s.",
        FONTS["subtitle"],
        MUTED,
    )

    left = 145
    right = 2320
    axis_y = 1040
    plot_w = right - left
    scale = plot_w / duration

    def x_at(seconds: float) -> int:
        return round(left + seconds * scale)

    # Grid and axis.
    for tick in range(0, 541, 60):
        x = x_at(min(tick, duration))
        draw.line((x, 260, x, axis_y + 12), fill=(230, 229, 225), width=2)
        draw_text(draw, (x, axis_y + 28), f"{tick}s", FONTS["small"], MUTED, anchor="ma")
    draw.line((left, axis_y, right, axis_y), fill=INK, width=3)

    lanes = {
        "overview": 330,
        "targeted": 535,
        "evidence": 775,
    }
    draw_text(draw, (left, 275), "Where the agent looked", FONTS["section"], INK)
    draw_text(draw, (left, 715), "What the benchmark labels", FONTS["section"], INK)

    # Whole-video overview lane.
    y = lanes["overview"]
    draw.rounded_rectangle((left, y, right, y + 48), radius=8, fill=LIGHT_BLUE)
    draw_text(draw, (left + 18, y + 24), f"overview: {overview_count * 8} frames across the whole video", FONTS["body"], BLUE, anchor="lm")
    # Mark the three provenance samples that fell inside the annotated windows.
    for t in (43.27661, 119.486153, 204.838172):
        x = x_at(t)
        draw.ellipse((x - 8, y + 14, x + 8, y + 30), fill=ORANGE, outline=WHITE, width=2)
    draw_text(draw, (right, y + 24), "3 frames inside windows", FONTS["small"], BLUE, anchor="rm")

    # Targeted window lane. Draw skim and focus separately so the figure does
    # not turn the full-video overview into a false claim of targeted search.
    y = lanes["targeted"]
    draw_text(draw, (left, y - 40), "Detailed searches", FONTS["small_bold"], MUTED)
    focus_labels: list[str] = []
    for start, end, tool in target_windows:
        color = BLUE if tool == "focus" else (88, 137, 195)
        yy = y if tool == "skim" else y + 66
        draw.rounded_rectangle((x_at(start), yy, x_at(end), yy + 38), radius=7, fill=color)
        if tool == "skim":
            draw_text(draw, (x_at(start) + 10, yy + 19), f"skim {start:g}–{end:g}s", FONTS["small"], WHITE, anchor="lm")
        else:
            focus_labels.append(f"{start:g}–{end:g}s")
            draw_text(draw, ((x_at(start) + x_at(end)) // 2, yy + 19), "F", FONTS["small_bold"], WHITE, anchor="mm")
    draw_text(draw, (left, y + 112), "focus calls: " + ", ".join(focus_labels), FONTS["small"], BLUE)
    last_focus = max(end for start, end, tool in target_windows if tool == "focus")
    last_skim = max(end for start, end, tool in target_windows if tool == "skim")
    focus_x = x_at(last_focus)
    skim_x = x_at(last_skim)
    draw.line((focus_x, 245, focus_x, 915), fill=ORANGE, width=4)
    draw_text(draw, (focus_x + 13, 245), f"last focus look: {last_focus:g}s", FONTS["small_bold"], ORANGE)
    draw.line((skim_x, 295, skim_x, 915), fill=(144, 144, 144), width=3)
    draw_text(draw, (skim_x + 12, 295), f"skim ended: {last_skim:g}s", FONTS["small_bold"], MUTED)

    # Evidence intervals.
    y = lanes["evidence"]
    for start, end, label in evidence:
        x0, x1 = x_at(start), x_at(end)
        draw.rounded_rectangle((x0, y, x1, y + 70), radius=9, fill=RED)
        draw_text(draw, ((x0 + x1) // 2, y + 92), label, FONTS["small_bold"], RED, anchor="ma")
        draw_text(draw, ((x0 + x1) // 2, y + 122), f"{start:g}–{end:g}s", FONTS["small"], RED, anchor="ma")

    draw.rounded_rectangle((left, 1160, right, 1325), radius=10, fill=PALE_AMBER, outline=(230, 204, 145), width=2)
    draw_text(draw, (left + 22, 1190), "Interpretation", FONTS["body_bold"], AMBER)
    draw_wrapped(
        draw,
        (left + 22, 1230),
        f"Only {distinct_inside} of the {audit['A_distinct']} distinct decoded frames fell inside the three labeled intervals: one from the skim and two from the overview. A frame landing inside a labeled time window does not mean the model verified the team, player, or play described in the question.",
        FONTS["body"],
        right - left - 44,
        INK,
        spacing=5,
    )
    return save(canvas, "q207_search_timeline.png")


def build_abc_matrix() -> Path:
    canvas = Image.new("RGB", (2500, 1450), BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (70, 55), "A / B / C — what is held fixed, what changes", FONTS["title"])
    draw_text(
        draw,
        (72, 125),
        "The comparison separates frame selection from the text reports that mediate the final answer.",
        FONTS["subtitle"],
        MUTED,
    )
    x0, y0 = 70, 220
    col_widths = [330, 455, 280, 300, 260, 260, 415]
    headers = ["condition", "images shown", "reports", "system prompt", "tools", "model calls", "what it isolates"]
    row_h = 255
    table_w = sum(col_widths)
    draw.rectangle((x0, y0, x0 + table_w, y0 + 3 * row_h + 76), fill=WHITE, outline=RULE, width=2)
    x = x0
    for width, header in zip(col_widths, headers):
        draw.rectangle((x, y0, x + width, y0 + 76), fill=(239, 239, 235), outline=RULE, width=2)
        draw_wrapped(draw, (x + 16, y0 + 19), header, FONTS["small_bold"], width - 32, INK, spacing=2)
        x += width

    rows = [
        (
            "A — real pipeline",
            "Frames selected by the agent from the video.",
            "Yes: tool reports",
            "Yes",
            "Yes",
            "2 per step + answer",
            "Full system: search, reports, answer.",
            BLUE,
        ),
        (
            "B — replay",
        "A's exact bytes, order, and visible labels.",
            "No",
            "No",
            "No",
            "1 answer call",
            "Whether A's exact frames alone rescue the answer; a report-removal diagnostic.",
            PURPLE,
        ),
        (
            "C — supplied evidence",
            "Frames sampled from annotated intervals.",
            "No",
            "No",
            "No",
            "1 answer call",
            "Whether annotated frames are sufficient; a privileged diagnostic.",
            GREEN,
        ),
    ]
    for row_idx, row in enumerate(rows):
        y = y0 + 76 + row_idx * row_h
        x = x0
        for col_idx, (width, value) in enumerate(zip(col_widths, row[:-1])):
            fill = PALE_BLUE if row_idx == 0 else (PALE_AMBER if row_idx == 1 else PALE_GREEN) if col_idx == 0 else WHITE
            draw.rectangle((x, y, x + width, y + row_h), fill=fill, outline=RULE, width=2)
            font = FONTS["body_bold"] if col_idx == 0 else FONTS["body"]
            color = row[-1] if col_idx == 0 else INK
            draw_wrapped(draw, (x + 16, y + 24), value, font, width - 32, color, spacing=6)
            x += width

    footer_y = y0 + 3 * row_h + 125
    draw.rounded_rectangle((x0, footer_y, x0 + table_w, footer_y + 150), radius=10, fill=PALE_AMBER, outline=(230, 204, 145), width=2)
    draw_text(draw, (x0 + 22, footer_y + 22), "Caveat", FONTS["body_bold"], AMBER)
    draw_wrapped(
        draw,
        (x0 + 22, footer_y + 62),
        "For q207, C receives three annotated image groups in chronological order, with source times visible. Those headers do not say which group is Event A, B, or C: C shows that the answer path can produce D with helpful frames, not that recognition was automatic.",
        FONTS["body"],
        table_w - 44,
        INK,
        spacing=5,
    )
    return save(canvas, "abc_input_matrix.png")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=BLUE, width=4, dashed=False) -> None:
    x0, y0 = start
    x1, y1 = end
    if dashed:
        length = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, length // 24)
        for i in range(steps):
            if i % 2 == 0:
                t0, t1 = i / steps, min(1, (i + 1) / steps)
                draw.line((x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1), fill=fill, width=width)
    else:
        draw.line((x0, y0, x1, y1), fill=fill, width=width)
    # A compact triangular arrowhead, oriented along the final segment.
    import math

    angle = math.atan2(y1 - y0, x1 - x0)
    size = 15
    points = [
        (x1, y1),
        (x1 - size * math.cos(angle - 0.5), y1 - size * math.sin(angle - 0.5)),
        (x1 - size * math.cos(angle + 0.5), y1 - size * math.sin(angle + 0.5)),
    ]
    draw.polygon(points, fill=fill)


def box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body: str, fill=WHITE, outline=RULE, title_fill=INK) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=3)
    draw_text(draw, (x0 + 20, y0 + 18), title, FONTS["body_bold"], title_fill)
    draw_wrapped(draw, (x0 + 20, y0 + 64), body, FONTS["small"], x1 - x0 - 40, INK, spacing=5)


def build_architecture() -> Path:
    canvas = Image.new("RGB", (2700, 1800), BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (70, 55), "Why the study separates locating from answering", FONTS["title"])
    draw_text(
        draw,
        (72, 125),
        "Images leave the controller on tool calls; only text returns to the controller and the final answer step.",
        FONTS["subtitle"],
        MUTED,
    )

    # Main pipeline.
    y = 280
    h = 190
    boxes = {
        "video": (80, y, 430, y + h),
        "controller": (570, y, 1040, y + h),
        "tools": (1180, y, 1680, y + h),
        "context": (1820, y, 2320, y + h),
        "answer": (2440, y, 2640, y + h),
    }
    box(draw, boxes["video"], "Local video", "VES-Bench .mp4\nDecode locally; frames become JPEGs.", PALE_BLUE, BLUE, BLUE)
    box(draw, boxes["controller"], "Controller — text only", "Gemini 3.8 Flash\nThought call → action call\n(tool_choice = required)", PALE_BLUE, BLUE, BLUE)
    box(draw, boxes["tools"], "Vision tools", "overview: 64 frames → 8 grids\nskim / focus: up to 16 frames\nSame Gemini model; text report returns.", PALE_AMBER, ORANGE, AMBER)
    box(draw, boxes["context"], "Text context", "System prompt + thoughts +\ntool reports accumulate here.\nThis is all the answer sees.", PALE_GREEN, GREEN, GREEN)
    box(draw, boxes["answer"], "Answer", "1 call\nfinal letter", PALE_GREEN, GREEN, GREEN)
    arrow(draw, (boxes["video"][2], y + h // 2), (boxes["controller"][0], y + h // 2), BLUE)
    arrow(draw, (boxes["controller"][2], y + h // 2 - 34), (boxes["tools"][0], y + h // 2 - 34), BLUE, dashed=True)
    arrow(draw, (boxes["tools"][0], y + h // 2 + 42), (boxes["controller"][2], y + h // 2 + 42), ORANGE, dashed=True)
    arrow(draw, (boxes["controller"][2], y + h // 2 + 80), (boxes["context"][0], y + h // 2 + 80), ORANGE)
    arrow(draw, (boxes["context"][2], y + h // 2), (boxes["answer"][0], y + h // 2), GREEN)
    draw_text(draw, (1240, 215), "image path", FONTS["small_bold"], BLUE)
    draw_text(draw, (1335, 505), "text report path", FONTS["small_bold"], ORANGE)
    draw_text(draw, (2040, 215), "text only", FONTS["small_bold"], GREEN)

    # Calls and sampling annotations.
    draw.rounded_rectangle((80, 535, 2640, 675), radius=10, fill=WHITE, outline=RULE, width=2)
    draw_text(draw, (105, 560), "Per step", FONTS["body_bold"], INK)
    draw_text(draw, (310, 560), "2 model calls: thought + action", FONTS["body"], BLUE)
    draw_text(draw, (105, 620), "Run that failed", FONTS["body_bold"], INK)
    draw_text(draw, (310, 620), "7 steps × 2 + 1 answer = 15 calls", FONTS["body"], ORANGE)
    draw_text(draw, (1260, 560), "The answer step receives no image", FONTS["body_bold"], RED)
    draw_text(draw, (1260, 620), "That is why A/B/C can separate locating from reporting.", FONTS["body"], INK)

    # Recorder band.
    draw.rounded_rectangle((80, 760, 2640, 930), radius=12, fill=(244, 242, 238), outline=RULE, width=3)
    draw_text(draw, (110, 790), "Recorder — hooks every model call", FONTS["section"], INK)
    draw_wrapped(draw, (110, 845), "Writes trajectory.json, images.json with source-frame provenance, usage.json, settings.json, and events.jsonl as the run happens.", FONTS["body"], 2370, MUTED, spacing=5)

    # Branches for B/C.
    draw_text(draw, (80, 1030), "Conditions reconstructed from the saved run", FONTS["section"], INK)
    branch_y = 1110
    branch_h = 240
    conds = [
        ("A — real pipeline", "Tools choose frames; reports mediate; answer sees text only.", "Full system", BLUE, PALE_BLUE),
        ("B — exact-frame replay", "A's exact image bytes, order, and labels; no reports, prompt, or tools.", "Does the exact A image set rescue the answer?", PURPLE, (239, 232, 249)),
        ("C — annotated-frame replay", "Frames sampled from annotated intervals; no reports, prompt, or tools.", "Are the privileged annotated frames sufficient?", GREEN, PALE_GREEN),
    ]
    x_positions = [80, 940, 1800]
    for x, (title, body, isolates, color, fill) in zip(x_positions, conds):
        box(draw, (x, branch_y, x + 780, branch_y + branch_h), title, body + f"\n\nIsolates: {isolates}.", fill, color, color)
    arrow(draw, (1320, 930), (470, branch_y), BLUE, dashed=True)
    arrow(draw, (1320, 930), (1330, branch_y), PURPLE, dashed=True)
    arrow(draw, (1320, 930), (2190, branch_y), GREEN, dashed=True)

    draw.rounded_rectangle((80, 1430, 2640, 1690), radius=12, fill=PALE_AMBER, outline=(230, 204, 145), width=2)
    draw_text(draw, (110, 1460), "Interpretive rule", FONTS["section"], AMBER)
    draw_wrapped(
        draw,
        (110, 1520),
        "A misses and B misses tells us that the exact A image set is not rescued by the report-removal replay. B misses and C answers show that the annotated frames are helpful, but C is a privileged diagnostic: for q207 it still requires mapping each window to Event A, B, or C.",
        FONTS["body"],
        2370,
        INK,
        spacing=6,
    )
    return save(canvas, "architecture_overview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("audit", "timeline", "matrix", "architecture"), action="append")
    args = parser.parse_args()
    requested = set(args.only or ("audit", "timeline", "matrix", "architecture"))
    builders = {
        "audit": build_hockey_audit,
        "timeline": build_search_timeline,
        "matrix": build_abc_matrix,
        "architecture": build_architecture,
    }
    for name in ("audit", "timeline", "matrix", "architecture"):
        if name in requested:
            print(builders[name]())


if __name__ == "__main__":
    main()
