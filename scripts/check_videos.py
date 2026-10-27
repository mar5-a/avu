"""Check that a downloaded video decodes the way the tools will sample it.

`yt-dlp` will happily deliver a file the tools cannot read. The 720p version of a
video came down as VP9-in-mp4, every frame fetch raised `DECORDError`, and because
the agent turns a raised tool into "Tool execution failed." and carries on, three
runs answered from the question text alone while looking like ordinary runs.

Decoding without an error is not enough either. decord, which the tools used
until 2026-10-27, jumped to a requested frame and sometimes landed one keyframe
early, returning a frame 1-9 s before the one asked for with no error, on 14 of
the 20 effort-cycle videos. The tools' reader (`videoseek/video.py`) now finds
each frame by its timestamp; this checks that it does, on every video, before any
money is spent. It asks for frames the way the agent does (the overview's 64,
then windows shaped like skim's and focus's, then single frames, all on one
reader) and requires each to be byte-for-byte the frame at that position when
the same decoder reads the file from start to finish, which involves no jumping.

Usage:
    .venv/bin/python scripts/check_videos.py
    .venv/bin/python scripts/check_videos.py --id PAI-gQ19pbs
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))

from videoseek.video import VideoReader  # noqa: E402


def codec(path):
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def digest(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def requests(frames, fps, rng):
    """Frame indices in the agent's patterns, overview first, as the agent starts."""
    yield np.linspace(0, frames - 1, 64).astype(int)
    for low, high in [(16, 120)] * 10 + [(2, 16)] * 10:  # skim-shaped, then focus-shaped
        span = int(rng.uniform(low, high) * fps)
        start = int(rng.uniform(0, max(1, frames - span - 1)))
        yield np.linspace(start, min(start + span, frames - 1) - 1, 16).astype(int)
    for _ in range(40):
        yield np.array([int(rng.uniform(0, frames - 1))])


def start_to_finish(path, wanted):
    """Digests of the frames at `wanted`, decoding in order from the first frame, and the frame count."""
    found, count = {}, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for count, frame in enumerate(container.decode(stream), start=1):
            if count - 1 in wanted:
                found[count - 1] = digest(frame.to_ndarray(format="rgb24"))
    return found, count


def check(path):
    reader = VideoReader(path)
    fps, frames = reader.get_avg_fps(), len(reader)
    jumped = {}
    for indices in requests(frames, fps, np.random.default_rng(0)):
        for index, frame in zip(indices, reader.get_batch(indices).asnumpy()):
            jumped.setdefault(int(index), set()).add(digest(frame))
    truth, decoded = start_to_finish(path, set(jumped))
    if decoded != frames:
        raise ValueError(f"reader counts {frames} frames, decoding start to finish gives {decoded}")
    wrong = sorted(index for index, seen in jumped.items() if seen != {truth[index]})
    if wrong:
        raise ValueError(
            f"{len(wrong)} of {len(jumped)} frames differ from the start-to-finish decode, "
            f"first at frame {wrong[0]} ({wrong[0] / fps:.2f}s)"
        )
    height, width = frame.shape[:2]
    return (
        f"{width}x{height} {frames} frames {fps:.2f} fps codec {codec(path)}, "
        f"{len(jumped)} distinct frames identical to the start-to-finish decode"
    )


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
            print(f"FAIL  {path.stem:<16} {type(error).__name__}: {str(error)[:120]}")
        else:
            print(f"ok    {path.stem:<16} {detail}", flush=True)

    print()
    if failures:
        print(f"{len(failures)} of {len(paths)} videos failed: {', '.join(failures)}")
        print("An error decoding: re-fetch with an H.264 preference, e.g.")
        print('  yt-dlp -f "bv*[vcodec^=avc1][height<=720]/bv*[vcodec^=avc1]/bv*[height<=720]/bv*" ...')
        print("A frame that differs: the reader is at fault, not the file; see videoseek/video.py.")
        return 1
    print(f"all {len(paths)} videos return the frames asked for, in every sampling shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
