"""Check that each image a run sent shows the frames its provenance names.

Provenance records the frame indices a tool asked for. Until 2026-10-27 the tools
read frames with decord, whose seek could return a frame 1-5 s earlier than the
one asked for, with no error, so on some pilot runs provenance and pixels
disagree. This rebuilds every saved image from ffmpeg's frames at the recorded
indices (decoded from the start, no seeking), using the tools' own resize and 2x4
tiling, and compares it tile by tile with the JPEG the model received. A tile
that does not match is then matched against nearby frames, to say which moment
it actually showed.

Usage:
    .venv/bin/python scripts/audit_sent_frames.py data/runs/q15-temporal-ordering-2026-09-21-2
    .venv/bin/python scripts/audit_sent_frames.py data/runs/*
"""

import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SHORT_SIDE = 256
# Mean absolute pixel difference, 0-255. A right tile differs from its rebuild by
# JPEG noise, about 1-3; the wrong tiles found in the pilot were 19-97 away.
MAX_DIFF = 6.0


def resize_short_side(frames):
    """The tools' resize for overview and skim: shorter side to 256, bicubic."""
    _, height, width, _ = frames.shape
    scale = SHORT_SIDE / min(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack([np.array(Image.fromarray(frame).resize(size, Image.BICUBIC)) for frame in frames])


def probe(path):
    width, height, rate = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    num, den = rate.split("/")
    return int(width), int(height), int(num) / int(den)


def ffmpeg_frames(path, indices, width, height):
    """Frames at `indices`, in display order, decoded from the start by ffmpeg."""
    indices = sorted({int(index) for index in indices})
    found = {}
    for start in range(0, len(indices), 40):  # ffmpeg refuses very long select expressions
        chunk = indices[start:start + 40]
        select = "select='" + "+".join(f"eq(n\\,{index})" for index in chunk) + "'"
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vf", select, "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
        assert len(frames) == len(chunk), (len(frames), len(chunk))
        found.update(zip(chunk, frames))
    return found


def shown_frame(path, tile, asked, width, height, reach=150):
    """The frame within `reach` of `asked` that the tile matches best, compared small."""
    small = (160, int(round(160 * height / width / 2)) * 2)
    low = max(0, asked - reach)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         f"select=between(n\\,{low}\\,{asked + reach}),scale={small[0]}:{small[1]}",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    nearby = np.frombuffer(raw, np.uint8).reshape(-1, small[1], small[0], 3).astype(int)
    shrunk = np.array(Image.fromarray(tile).resize(small, Image.BICUBIC)).astype(int)
    diffs = np.abs(nearby - shrunk).mean(axis=(1, 2, 3))
    return low + int(np.argmin(diffs))


def image_entries(run_dir):
    """Images with provenance: backfilled runs keep it in provenance.json, later ones in images.json."""
    for name in ("provenance.json", "images.json"):
        path = run_dir / name
        if path.is_file():
            entries = [e for e in json.loads(path.read_text()).get("images", []) if e.get("provenance")]
            if entries:
                return entries
    return []


def video_of(run_dir):
    for name in ("question.json", "settings.json"):
        path = run_dir / name
        if path.is_file():
            data = json.loads(path.read_text())
            if data.get("video_id"):
                return data["video_id"]
            if data.get("source_run"):  # replays name the run they were built from
                return video_of(run_dir.parent / data["source_run"])
    return None


def audit(run_dir, video_dir):
    entries, video_id = image_entries(run_dir), video_of(run_dir)
    if not entries or not video_id:
        return None
    path = video_dir / f"{video_id}.mp4"
    width, height, fps = probe(path)
    truth = ffmpeg_frames(path, [i for e in entries for i in e["provenance"]["frame_indices"]], width, height)
    tiles, wrong = 0, []
    for entry in entries:
        provenance = entry["provenance"]
        saved = np.array(Image.open(BytesIO((run_dir / entry["file"]).read_bytes())).convert("RGB"))
        frames = np.stack([truth[index] for index in provenance["frame_indices"]])
        grid = provenance.get("grid") or {"rows": 1, "cols": 1}
        # Replays record no tool name, so a resized image is told apart by its size.
        if saved.shape[0] // grid["rows"] != frames.shape[1]:
            frames = resize_short_side(frames)
        tile_h, tile_w = frames.shape[1:3]
        assert saved.shape[:2] == (grid["rows"] * tile_h, grid["cols"] * tile_w), entry["file"]
        for position, index in enumerate(provenance["frame_indices"]):
            row, col = divmod(position, grid["cols"])
            tile = saved[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w]
            tiles += 1
            if np.abs(tile.astype(int) - frames[position].astype(int)).mean() > MAX_DIFF:
                shown = shown_frame(path, tile, index, width, height)
                wrong.append((entry["file"], entry.get("tool"), index / fps, shown / fps))
    return video_id, len(entries), tiles, wrong


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    args = ap.parse_args()
    for run_dir in args.run_dirs:
        result = audit(run_dir, args.video_dir)
        if result is None:
            print(f"{run_dir.name:<44} no per-image provenance")
            continue
        video_id, images, tiles, wrong = result
        print(f"{run_dir.name:<44} {video_id:<12} {len(wrong)} of {tiles} frames not the one named, "
              f"across {images} images")
        for file, tool, asked, shown in wrong:
            print(f"    {file:<26} {tool or 'replay':<8} labelled {asked:7.2f}s, shows {shown:7.2f}s "
                  f"({shown - asked:+.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
