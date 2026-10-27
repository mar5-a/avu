"""Check that a downloaded video decodes the way the tools will sample it.

`yt-dlp` will happily deliver a file that decord cannot seek in. The 720p version
of a video came down as VP9-in-mp4, every frame fetch raised `DECORDError`, and
because the agent turns a raised tool into "Tool execution failed." and carries
on, three runs answered from the question text alone while looking like ordinary
runs. Nothing in the record said the video was unreadable; this does, before any
money is spent.

It samples each video the two ways the tools do — 64 frames spread across the
whole thing (`overview`) and a 16-frame window (`focus`/`skim`) — and reports the
codec, since that is the property that decided it.

Usage:
    .venv/bin/python scripts/check_videos.py
    .venv/bin/python scripts/check_videos.py --id PAI-gQ19pbs
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def codec(path):
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def check(path):
    """Decode the two sampling shapes, or say why not."""
    from decord import VideoReader

    reader = VideoReader(str(path))
    fps = reader.get_avg_fps()
    frames = len(reader)
    whole = np.linspace(0, frames - 1, 64).astype(int)
    middle = frames // 2
    start = min(middle, max(0, frames - int(2 * fps) - 1))
    window = np.linspace(start, min(start + int(2 * fps), frames - 1), 16).astype(int)
    reader.get_batch(whole).asnumpy()
    reader.get_batch(window).asnumpy()
    height, width = reader[0].shape[:2]
    return f"{width}x{height} {frames} frames {fps:.2f} fps codec {codec(path)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--id", default=None, help="Check one video by id (filename stem).")
    args = ap.parse_args()

    paths = sorted(args.dir.glob(f"{args.id}.mp4") if args.id else args.dir.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"no videos in {args.dir}")

    failures = []
    for path in paths:
        try:
            detail = check(path)
        except Exception as error:  # noqa: BLE001 - any decode failure is the point
            failures.append(path.stem)
            print(f"FAIL  {path.stem:<16} {type(error).__name__}: {str(error)[:80]}")
        else:
            print(f"ok    {path.stem:<16} {detail}")

    print()
    if failures:
        print(f"{len(failures)} of {len(paths)} videos cannot be decoded: {', '.join(failures)}")
        print("Re-fetch with an H.264 preference, e.g.")
        print('  yt-dlp -f "bv*[vcodec^=avc1][height<=720]/bv*[vcodec^=avc1]/bv*[height<=720]/bv*" ...')
        return 1
    print(f"all {len(paths)} videos decode for both sampling shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
