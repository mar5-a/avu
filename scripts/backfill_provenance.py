"""Recover frame provenance for runs recorded before it was captured automatically.

Provenance written at capture time comes from the tool that built the image, so it
is authoritative. This reconstructs the same information for older runs: it
re-derives which frames each tool sampled, re-encodes those frames exactly as the
tool did, and accepts the result only where the bytes match the image that was
actually saved.

The byte check is the point. Re-deriving the sampling means restating the tools'
frame arithmetic here, and that would go quietly stale if upstream changed it; a
mismatch fails loudly instead, and the image keeps null provenance rather than a
guess. Images that cannot be reconstructed are listed and the exit status is
non-zero.

Usage:
    .venv/bin/python scripts/backfill_provenance.py --run data/runs/q257-temporal-ordering-2026-09-19-3
"""

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from decord import VideoReader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))

from config import general_config as GENERAL  # noqa: E402  (same source as the tools)

SHORT_SIDE = 256


def encode_jpeg(array) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="jpeg")
    return buffer.getvalue()


def resize_short_side(frames):
    """Mirror of the tools' resize: scale so the shorter side is 256, bicubic."""
    _, height, width, _ = frames.shape
    scale = SHORT_SIDE / min(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack(
        [np.array(Image.fromarray(frame).resize(size, Image.BICUBIC)) for frame in frames]
    )


def overview_indices(vr):
    """Mirror of `overview`: uniform sampling of the whole video, in groups of 8."""
    factor = GENERAL["frame_sampling_factor"] * GENERAL["overview_base"]
    count = factor if factor % 8 == 0 else int(np.ceil(factor / 8.0) * 8)
    return np.linspace(0, len(vr) - 1, count).astype(int)


def window_indices(vr, start_time, end_time, max_frames):
    """Mirror of `skim` and `focus`: uniform sampling of [start, end)."""
    start = int(start_time * vr.get_avg_fps())
    end = min(int(end_time * vr.get_avg_fps()), len(vr) - 1)
    count = min(int(end - start), max_frames)
    return np.linspace(start, end - 1, count).astype(int)


def rebuild_step(vr, tool, parameters):
    """The JPEG bytes each image of one step should be, and their frame indices."""
    if tool == "overview":
        indices = overview_indices(vr)
        frames = resize_short_side(vr.get_batch(indices).asnumpy())
        sheets = []
        for start in range(0, len(indices), 8):
            cells = frames[start:start + 8]
            height, width, channels = cells.shape[1:]
            sheet = (
                cells.reshape(2, 4, height, width, channels)
                .transpose(0, 2, 1, 3, 4)
                .reshape(2 * height, 4 * width, channels)
            )
            sheets.append(
                {
                    "bytes": encode_jpeg(sheet),
                    "frame_indices": [int(index) for index in indices[start:start + 8]],
                    "grid": {"rows": 2, "cols": 4},
                }
            )
        return sheets
    if tool in ("focus", "skim"):
        max_frames = GENERAL["frame_sampling_factor"] * GENERAL[f"{tool}_base"]
        indices = window_indices(vr, parameters["start_time"], parameters["end_time"], max_frames)
        frames = vr.get_batch(indices).asnumpy()
        if tool == "skim":
            frames = resize_short_side(frames)
        return [
            {
                "bytes": encode_jpeg(frames[position]),
                "frame_indices": [int(indices[position])],
                "grid": None,
            }
            for position in range(len(indices))
        ]
    return None


def backfill(run_dir: Path, video_dir: Path) -> dict:
    manifest = json.loads((run_dir / "images.json").read_text())
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    question = json.loads((run_dir / "question.json").read_text())
    actions = {step["step_id"]: step["action"] for step in trajectory["steps"]}

    video = video_dir / f"{question['video_id']}.mp4"
    if not video.is_file():
        raise SystemExit(f"video missing, cannot reconstruct: {video}")
    vr = VideoReader(str(video))
    fps = vr.get_avg_fps()

    by_step = {}
    for image in manifest["images"]:
        by_step.setdefault(image["step"], []).append(image)

    entries, matched, unmatched = [], 0, []
    for step_id, group in sorted(by_step.items()):
        tool = group[0]["tool"]
        rebuilt = rebuild_step(vr, tool, actions.get(step_id, {}).get("parameters", {}))
        if rebuilt is None:
            unmatched.extend(
                {"index": image["index"], "reason": f"tool {tool!r} not reconstructible"}
                for image in group
            )
            continue
        if len(rebuilt) != len(group):
            unmatched.extend(
                {"index": image["index"], "reason": f"{len(group)} images but {len(rebuilt)} rebuilt"}
                for image in group
            )
            continue
        for image, frames in zip(group, rebuilt):
            saved = (run_dir / image["file"]).read_bytes()
            ok = (
                hashlib.sha256(frames["bytes"]).hexdigest() == image["sha256"]
                and frames["bytes"] == saved
            )
            if ok:
                matched += 1
                provenance = {
                    "frame_indices": frames["frame_indices"],
                    "timestamps_sec": [
                        round(index / fps, 6) for index in frames["frame_indices"]
                    ],
                    "grid": frames["grid"],
                    "reconstructed": True,
                }
            else:
                unmatched.append({"index": image["index"], "reason": "re-encoded bytes differ"})
                provenance = None
            entries.append(
                {
                    "index": image["index"],
                    "file": image["file"],
                    "step": image["step"],
                    "tool": tool,
                    "sha256": image["sha256"],
                    "provenance": provenance,
                }
            )

    return {
        "source": (
            "reconstructed by scripts/backfill_provenance.py; accepted only where "
            "re-encoded bytes matched the saved image"
        ),
        "video_id": question["video_id"],
        "avg_fps": fps,
        "total_frames": len(vr),
        "images_total": len(manifest["images"]),
        "images_matched": matched,
        "images_unmatched": unmatched,
        "images": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payload = backfill(args.run, args.video_dir)
    out = args.out or (args.run / "provenance.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.run.name}: {payload['images_matched']}/{payload['images_total']} images "
          f"reconstructed byte for byte ({payload['avg_fps']:.2f} fps, "
          f"{payload['total_frames']} frames)")
    print(f"wrote {out}")
    for failure in payload["images_unmatched"]:
        print(f"  UNMATCHED image {failure['index']}: {failure['reason']}")
    return 1 if payload["images_unmatched"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
