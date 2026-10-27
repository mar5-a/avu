"""Uniform baseline: one answer from evenly spaced frames, with no agent.

The agent arms choose where to look, then answer from their own text reports.
This asks whether that buys anything over the simplest system: N frames spread
evenly across the whole video, each labelled with its time, then the question,
in one call with no tools and no system prompt. If it matches the agent on these
questions, the agent's search is not what gets them right.

Frames are sampled and encoded the way `skim` does — shorter side 256, JPEG, a
time label before each frame — so every frame here is one the agent could have
been shown. N is fixed at 128 before any result: about the number of distinct
frames a pilot agent run decoded (123 to 143 on the ordering questions).

The question block is the one replays B and C send: the planner's verbatim query
plus their answer line. Nothing else is text except the time labels and one
header giving the video's span, which `verify_spec` enforces.

Default is a dry run: sample, verify, print what it would send, and call nothing.

Usage:
    .venv/bin/python scripts/uniform_direct.py --question-id 257
    .venv/bin/python scripts/uniform_direct.py --question-id 257 --live
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import general_config, init_config  # noqa: E402
from videoseek.recorder import recorder  # noqa: E402
from videoseek.utils import call_llm_api, extract_thought  # noqa: E402

from replay_b import ANSWER_INSTRUCTION  # noqa: E402
from replay_c import sample_interval  # noqa: E402
from run_one_question import allocate_run_dir, build_query, load_question, question_record  # noqa: E402
from ves_score import extract_letter  # noqa: E402

UNIFORM_FRAMES = 128


def build_spec(row, video, num_frames=UNIFORM_FRAMES):
    """Sample the whole video evenly and lay out the request, frames kept apart."""
    from videoseek.video import VideoReader

    vr = VideoReader(str(video))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    encoded, timestamps, indices = sample_interval(vr, 0.0, duration, num_frames, fps)
    query = build_query(row)
    blocks = [
        {"type": "text", "text": f"Video segment (0.0s - {duration:.1f}s):\n", "source": "header"}
    ]
    for position, (index, timestamp, raw) in enumerate(zip(indices, timestamps, encoded)):
        # Label first, then the frame, as `skim` lays them out: a label after its
        # frame would read as the next frame's.
        blocks.append({"type": "text", "text": f"{timestamp:.1f}s", "source": "frame timestamp"})
        blocks.append(
            {
                "type": "image",
                "position": position,
                "frame_index": int(index),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    blocks.append(
        {"type": "text", "text": f"Question:\n{query}\n\n{ANSWER_INSTRUCTION}", "source": "question"}
    )
    spec = {
        "experiment": "uniform",
        "question_id": row["id"],
        "task": row["task"],
        "fps": fps,
        "total_frames": len(vr),
        "duration_sec": round(duration, 3),
        "frames": num_frames,
        "frame_indices": [int(index) for index in indices],
        "blocks": blocks,
    }
    return spec, encoded


def verify_spec(spec, frames, row):
    """Problems with the assembled request. Empty means it may be sent."""
    problems = []
    images = [b for b in spec["blocks"] if b["type"] == "image"]
    texts = [b for b in spec["blocks"] if b["type"] == "text"]

    # 1. Layout: one header, then a label before every frame, then the question.
    expected = ["header"] + ["timestamp", "image"] * spec["frames"] + ["question"]
    actual = [
        "image" if b["type"] == "image"
        else "timestamp" if b["source"] == "frame timestamp"
        else b["source"]
        for b in spec["blocks"]
    ]
    if actual != expected:
        problems.append(f"blocks do not follow the uniform layout ({len(actual)} blocks)")

    # 2. The frames span the whole video at an even step, in order, and each
    #    label is that frame's own time.
    indices = spec["frame_indices"]
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    if indices[0] != 0 or indices[-1] < spec["total_frames"] - 2 - max(gaps, default=0):
        problems.append(f"frames do not span the video ({indices[0]} to {indices[-1]})")
    if gaps and (min(gaps) < 0 or max(gaps) - min(gaps) > 1):
        problems.append(f"frames are not evenly spaced (steps {min(gaps)} to {max(gaps)})")
    for label, image in zip(texts[1:-1], images):
        if label["text"] != f"{image['frame_index'] / spec['fps']:.1f}s":
            problems.append(f"label {label['text']!r} is not frame {image['frame_index']}'s time")
    if len(frames) != len(images):
        problems.append(f"{len(frames)} frames sampled for {len(images)} image blocks")
    for raw, image in zip(frames, images):
        if hashlib.sha256(raw).hexdigest() != image["sha256"]:
            problems.append(f"frame at position {image['position']} does not match its digest")

    # 3. The question is the verbatim query plus the replays' answer line, and
    #    every other text is a time or the video's span, so none can carry an
    #    answer or an annotation.
    if texts[-1]["text"] != f"Question:\n{build_query(row)}\n\n{ANSWER_INSTRUCTION}":
        problems.append("the question block is not the verbatim query plus the answer line")
    if not re.fullmatch(r"Video segment \(0\.0s - \d+\.\ds\):\n", texts[0]["text"]):
        problems.append(f"header is not the video's span: {texts[0]['text']!r}")
    for label in texts[1:-1]:
        if not re.fullmatch(r"\d+\.\ds", label["text"]):
            problems.append(f"non-question text is not a bare timestamp: {label['text']!r}")
    return problems


def to_messages(spec, frames):
    """The wire payload, re-checking every frame against its digest."""
    content, frame_iter = [], iter(frames)
    for block in spec["blocks"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
            continue
        raw = next(frame_iter)
        if hashlib.sha256(raw).hexdigest() != block["sha256"]:
            raise SystemExit(f"frame at position {block['position']} changed; refusing to send")
        encoded = base64.b64encode(raw).decode("utf-8")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return [{"role": "user", "content": content}]


def send(spec, frames, row, video, run_dir, config):
    """Send the verified request once, and record what came back."""
    run_dir = Path(run_dir)
    query = build_query(row)
    # Written first, so a failed attempt is still readable and still comparable.
    (run_dir / "question.json").write_text(
        json.dumps(question_record(row, video, query), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recorder.start(run_dir)
    recorder.record_settings(
        {
            "model_name": config["model_name"],
            "temperature": config["temperature"],
            "reasoning_effort": config["reasoning_effort"],
            "max_tokens": config["max_tokens"],
            "seed": config["seed"],
            "tools": [],
            "api_base": config["api_base"],
            "api_key": config["api_key"],
            "api_version": config["api_version"],
        }
    )
    recorder.settings["system_prompt_sent"] = False
    recorder.settings["experiment"] = "uniform"
    recorder.settings["uniform_frames"] = spec["frames"]
    (run_dir / "uniform_request.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recorder.set_provenance(
        [
            {
                "frame_indices": [index],
                "timestamps_sec": [round(index / spec["fps"], 6)],
                "grid": None,
            }
            for index in spec["frame_indices"]
        ],
        avg_fps=spec["fps"],
        total_frames=spec["total_frames"],
    )
    response = call_llm_api(
        messages=to_messages(spec, frames),
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config["api_version"],
        max_tokens=config["max_tokens"],
        reasoning_effort=config["reasoning_effort"],
        seed=config["seed"],
        temperature=config["temperature"],
    )
    recorder.write_manifest()
    if response is None:
        raise SystemExit(f"uniform call failed; the reason is in {run_dir / 'events.jsonl'}")

    # The images that actually went out, against the frames that were sampled.
    sent = json.loads((run_dir / "images.json").read_text())["images"]
    expected_digests = [b["sha256"] for b in spec["blocks"] if b["type"] == "image"]
    verified = [image["sha256"] for image in sent] == expected_digests

    answer, answer_from = extract_thought(response)
    letter, parse = extract_letter(answer)
    expected = str(row["answer"]).strip().upper()[:1]
    payload = {
        "experiment": "uniform",
        "question_id": row["id"],
        "answer_raw": answer,
        "answer_letter": letter,
        "answer_parse": parse,
        "answer_diagnostics": {
            "answer_from": answer_from,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "reasoning": getattr(response.choices[0].message, "reasoning_content", None),
        },
        "ground_truth": expected,
        "correct": letter == expected,
        "images_sent": len(sent),
        "images_verified_against_sample": verified,
        "usage": json.loads((run_dir / "usage.json").read_text()),
    }
    (run_dir / "uniform_answer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "prediction.json").write_text(
        json.dumps({"prediction": answer}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Uniform frames, one direct answer, no agent.")
    ap.add_argument("--question-id", type=int, required=True)
    ap.add_argument("--parquet", type=Path, default=REPO / "data" / "VES-Bench.parquet")
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--output-dir", type=Path, default=REPO / "data" / "runs")
    ap.add_argument("--frames", type=int, default=UNIFORM_FRAMES)
    ap.add_argument("--reasoning-effort", dest="reasoning_effort", default=None)
    ap.add_argument("--label", default="uniform", help="Appended to the run name.")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="A directory already claimed for this run, e.g. by run_cycle.py.")
    ap.add_argument("--live", action="store_true", help="Actually send the request (costs money).")
    args = ap.parse_args()

    row = load_question(args.parquet, args.question_id)
    video = args.video_dir / f"{row['video_id']}.mp4"
    if not video.is_file():
        raise SystemExit(f"video missing: {video}")
    config = init_config(dict(general_config), args)

    spec, frames = build_spec(row, video, args.frames)
    problems = verify_spec(spec, frames, row)
    megabytes = sum(len(raw) for raw in frames) / 1e6
    print(
        f"uniform for q{row['id']}: {spec['frames']} frames over {spec['duration_sec']:.1f}s "
        f"(every {spec['duration_sec'] / spec['frames']:.1f}s), {megabytes:.1f} MB, "
        f"effort {config['reasoning_effort']}"
    )
    if problems:
        for problem in problems:
            print(f"  REFUSING: {problem}")
        return 1
    print("verification passed: even frames across the whole video, each with its own time; "
          "the question verbatim; no other text")
    if not args.live:
        print("dry run: nothing was sent. Re-run with --live to send.")
        return 0

    run_dir = args.run_dir or allocate_run_dir(args.output_dir, row, args.label)
    payload = send(spec, frames, row, video, run_dir, config)
    print(f"run dir: {run_dir}")
    print(
        f"answer: {payload['answer_letter']} (parsed as {payload['answer_parse']}) "
        f"vs expected {payload['ground_truth']}; "
        f"images verified: {payload['images_verified_against_sample']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
