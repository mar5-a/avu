"""Experiment C: answer the question from frames sampled at the annotated intervals.

A's frames are whatever the agent chose to look at, so on a question A misses they
cannot separate a locating failure from a failure to use evidence: the evidence may
simply not be in the payload. C replaces A's frames with frames the *annotators*
marked as the evidence and sends the same verbatim question. Everything else is B:
no reports, no tool history, no system prompt, nothing of A's.

    A  the frames the agent chose, plus the reports it wrote (controller is text-only)
    B  the frames the agent chose, reports removed
    C  frames from the gold intervals, reports removed

Read together they separate the two failure modes. If C fails a question B also
failed, the frames were not what stood in the way — the answerer cannot answer this
question with the evidence in front of it. If C answers a question A missed, the
failure was locating the evidence, not using it.

C samples and encodes exactly the way the pipeline's own `skim` tool does: the same
frame count, the same shorter-side-256 resize, JPEG, and a one-line timestamp label
before each frame. So a C payload is the input the agent would have received had it
aimed that call at the gold window. Nothing about the sampling is invented here.

C's payload shows each interval's bounds in its segment header, the way a real skim
call shows the bounds the agent chose, and omits the interval descriptions — those
are the answer key's own words ("Event C: From 43 to 48 seconds..."), which are
paraphrases of the question's options. The interval times are also already implicit
in the frame labels. What C never sends: `evidence[].description`, `answer`,
`note`, or any text A produced.

Default is a dry run: sample, verify, print what it would send, and call nothing.

Usage:
    .venv/bin/python scripts/replay_c.py --run data/runs/q207-temporal-ordering-2026-09-21-2
    .venv/bin/python scripts/replay_c.py --run data/runs/q207-temporal-ordering-2026-09-21-2 --live
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import general_config, init_config, prompts_config  # noqa: E402
from videoseek.recorder import recorder  # noqa: E402
from videoseek.utils import call_llm_api, extract_thought  # noqa: E402

from replay_b import ANSWER_INSTRUCTION, allocate_dir, forbidden_strings  # noqa: E402
from ves_score import extract_letter  # noqa: E402

SKIM_FRAMES = general_config["frame_sampling_factor"] * general_config["skim_base"]


def load_question(run_dir):
    return json.loads((Path(run_dir) / "question.json").read_text())


def video_path(question, video_dir):
    path = Path(video_dir) / f"{question['video_id']}.mp4"
    if not path.is_file():
        raise SystemExit(f"video missing: {path}")
    return path


def sample_interval(vr, start_sec, end_sec, num_frames, fps):
    """The frames `skim` would take from this window, and their real timestamps.

    Mirrors `tools/skim.py`: `end_frame - 1` bounds the linspace so the last frame
    is not the frame *at* the end of the window. A window too short to give the
    sampler `num_frames` distinct indices yields repeats, which is what skim does
    too; the count is reported rather than assumed.
    """
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), len(vr) - 1)
    indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    encoded, timestamps = [], []
    for frame, index in zip(frames, indices):
        resized = Image.fromarray(frame).resize(target, Image.BICUBIC)
        buffer = BytesIO()
        resized.save(buffer, format="jpeg")
        encoded.append(buffer.getvalue())
        timestamps.append(round(int(index) / fps, 6))
    return encoded, timestamps, indices.tolist()


def build_spec(run_dir, video_dir):
    """Sample every annotated interval and lay out the request C would send."""
    from decord import VideoReader

    run_dir = Path(run_dir)
    question = load_question(run_dir)
    vr = VideoReader(str(video_path(question, video_dir)))
    fps = vr.get_avg_fps()

    blocks = []
    frames = {}
    interval_records = []
    for event in question["evidence"]:
        start, end = float(event["start_sec"]), float(event["end_sec"])
        encoded, timestamps, indices = sample_interval(vr, start, end, SKIM_FRAMES, fps)
        interval_records.append(
            {
                "id": event.get("id"),
                "start_sec": start,
                "end_sec": end,
                "frames": len(encoded),
                "distinct_frames": len(set(indices)),
                "first_sec": timestamps[0],
                "last_sec": timestamps[-1],
                "frame_indices": indices,
            }
        )
        blocks.append(
            {
                "type": "text",
                "text": f"Video segment ({start:.1f}s - {end:.1f}s):\n",
                "source": f"header of interval {event.get('id')}",
            }
        )
        for index, timestamp, raw in zip(indices, timestamps, encoded):
            digest = hashlib.sha256(raw).hexdigest()
            name = f"gold/{event.get('id')}-{timestamp:.1f}s-{digest[:8]}.jpg"
            frames[name] = raw
            blocks.append(
                {
                    "type": "image",
                    "interval": event.get("id"),
                    "frame_index": index,
                    "file": name,
                    "sha256": digest,
                    "bytes": len(raw),
                }
            )
            blocks.append(
                {"type": "text", "text": f"{timestamp:.1f}s", "source": "frame timestamp"}
            )

    blocks.append(
        {
            "type": "text",
            "text": f"Question:\n{question['query_sent_to_model']}\n\n{ANSWER_INSTRUCTION}",
            "source": "question",
        }
    )
    spec = {
        "experiment": "C",
        "source_run": run_dir.name,
        "question_id": question["id"],
        "task": question["task"],
        "fps": fps,
        "intervals": interval_records,
        "blocks": blocks,
    }
    return spec, frames


def verify_spec(spec, run_dir):
    """Problems with the assembled request. Empty means it may be sent."""
    question = load_question(run_dir)
    problems = []
    texts = [b for b in spec["blocks"] if b["type"] == "text"]
    images = [b for b in spec["blocks"] if b["type"] == "image"]

    # 1. Structure: every interval's header, then its frames, each labelled.
    expected = []
    for record in spec["intervals"]:
        expected.append("header")
        for _ in range(record["frames"]):
            expected.extend(["image", "timestamp"])
    expected.append("question")
    actual = []
    for block in spec["blocks"]:
        if block["type"] == "image":
            actual.append("image")
        elif block["source"] == "question":
            actual.append("question")
        elif block["source"] == "frame timestamp":
            actual.append("timestamp")
        else:
            actual.append("header")
    if actual != expected:
        first = next(
            (i for i, (got, want) in enumerate(zip(actual, expected)) if got != want),
            min(len(actual), len(expected)),
        )
        problems.append(
            f"blocks do not follow the interval layout "
            f"({len(actual)} blocks, {len(expected)} expected, first difference at {first})"
        )

    # 2. Every frame really came from the interval it is filed under, and its
    #    label is where it came from. This is C's central claim, so it is checked
    #    on the samples themselves rather than trusted from the sampler.
    for block in images:
        record = next(r for r in spec["intervals"] if r["id"] == block["interval"])
        start_frame = int(record["start_sec"] * spec["fps"])
        end_frame = int(record["end_sec"] * spec["fps"])
        if not start_frame <= block["frame_index"] < end_frame:
            problems.append(
                f"frame {block['frame_index']} of interval {record['id']} "
                f"sits outside {record['start_sec']}-{record['end_sec']}s "
                f"({start_frame} <= i < {end_frame})"
            )
    for record in spec["intervals"]:
        if record["first_sec"] < record["start_sec"] - 0.1:
            problems.append(f"interval {record['id']} starts before its window")
        if record["last_sec"] > record["end_sec"] + 0.1:
            problems.append(f"interval {record['id']} ends after its window")

    # 3. The question block is the verbatim query plus the answer line.
    query = question["query_sent_to_model"]
    if not texts or texts[-1]["text"] != f"Question:\n{query}\n\n{ANSWER_INSTRUCTION}":
        problems.append("the question block is not the verbatim query plus the answer line")
    for option in question["options"]:
        if f"({option})" not in query:
            problems.append(f"option {option} is not in the query")

    # 4. Nothing of A's, and none of the answer key's own words, in the text C
    #    adds. Scanned over the non-question blocks only, as in B: the question
    #    block legitimately repeats the wording that the event descriptions and
    #    A's observations are both drawn from, and it is checked verbatim above.
    #    The interval descriptions are a different matter — they are the answer
    #    key's own summary of each window and must not reach C at all.
    sent = "\n".join(block["text"] for block in blocks_text_only(spec)[:-1])
    for forbidden in forbidden_strings(run_dir, question):
        if forbidden in sent:
            problems.append(f"text from A appears in C: {forbidden[:40]!r}")
    for event in question["evidence"]:
        description = event.get("description") or ""
        if description and description[:40] in sent:
            problems.append(f"an interval description reached C: {description[:40]!r}")

    # 5. Every text that is not the question is a time or a window bound, so no
    #    block anywhere can be stating an answer. The answer key's letter cannot
    #    be searched for as a substring: the question's own options list contains
    #    every letter, and the question block is verified verbatim just above.
    for block in blocks_text_only(spec)[:-1]:
        text = block["text"]
        if block["source"].startswith("header of interval"):
            if not re.fullmatch(r"Video segment \(\d+\.\ds - \d+\.\ds\):\n", text):
                problems.append(f"interval header is not a window bound: {text[:40]!r}")
            continue
        if not re.fullmatch(r"\d+\.\ds", text):
            problems.append(f"non-question text is not a bare timestamp: {text[:40]!r}")
    return problems


def blocks_text_only(spec):
    return [b for b in spec["blocks"] if b["type"] == "text"]


def write_images(frames, c_dir):
    """Write the sampled frames, so the run keeps what it sent."""
    (c_dir / "gold").mkdir(parents=True, exist_ok=True)
    for name, raw in frames.items():
        path = c_dir / name
        if path.is_file():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_bytes(raw)
    return c_dir


def to_messages(spec, c_dir):
    """The wire payload, re-reading every frame and re-checking its digest."""
    content = []
    for block in spec["blocks"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
            continue
        raw = (Path(c_dir) / block["file"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != block["sha256"]:
            raise SystemExit(
                f"{block['file']} changed since the request was assembled; refusing to send"
            )
        encoded = base64.b64encode(raw).decode("utf-8")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return [{"role": "user", "content": content}]


def send(spec, frames, run_dir, config):
    """Send the verified request once, and record what came back."""
    run_dir = Path(run_dir)
    c_dir = allocate_dir(run_dir, "C")
    write_images(frames, c_dir)
    # The frame indices each image came from, in send order, so the run's own
    # provenance says which source frame every image is.
    provenance = []
    for record in spec["intervals"]:
        for index, timestamp in zip(
            record["frame_indices"],
            [round(int(i) / spec["fps"], 6) for i in record["frame_indices"]],
        ):
            provenance.append(
                {"frame_indices": [index], "timestamps_sec": [timestamp], "grid": None}
            )
    recorder.start(c_dir)
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
    recorder.settings["experiment"] = "C"
    recorder.settings["source_run"] = run_dir.name
    (c_dir / "c_request.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recorder.set_provenance(provenance, avg_fps=spec["fps"], total_frames=None)

    messages = to_messages(spec, c_dir)
    response = call_llm_api(
        messages=messages,
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
        raise SystemExit(f"C call failed; the reason is in {c_dir / 'events.jsonl'}")

    answer, answer_from = extract_thought(response)
    question = load_question(run_dir)
    letter, how = extract_letter(answer)
    prediction = {
        "answer_text": answer,
        "answer_from": answer_from,
        "letter": letter,
        "parsed_as": how,
        "expected": question["answer"],
    }
    (c_dir / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return c_dir, prediction


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment C: gold-interval frames, no reports.")
    ap.add_argument("--run", type=Path, required=True, help="An A run directory.")
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--live", action="store_true", help="Actually send the request (costs money).")
    ap.add_argument("--out", type=Path, default=None, help="Write the request spec here.")
    args = ap.parse_args()

    config = init_config(general_config.copy(), args)
    spec, frames = build_spec(args.run, args.video_dir)
    problems = verify_spec(spec, args.run)

    frame_count = sum(r["frames"] for r in spec["intervals"])
    megabytes = sum(b["bytes"] for b in spec["blocks"] if b["type"] == "image") / 1e6
    print(
        f"C from {args.run.name}: {len(spec['intervals'])} gold intervals, "
        f"{frame_count} frames, {megabytes:.1f} MB, {len(spec['blocks'])} blocks"
    )
    for record in spec["intervals"]:
        print(
            f"    interval {record['id']}: {record['start_sec']}-{record['end_sec']}s, "
            f"{record['frames']} frames, {record['first_sec']}s to {record['last_sec']}s"
        )
    if problems:
        for problem in problems:
            print(f"  REFUSING: {problem}")
        return 1
    print("verification passed: every frame comes from its gold interval; no text of A's, "
          "no interval description, no answer letter")
    if args.out:
        args.out.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"spec written to {args.out}")
    if not args.live:
        print("dry run: nothing was sent. Re-run with --live to send.")
        return 0

    c_dir, prediction = send(spec, frames, args.run, config)
    print(f"c run dir: {c_dir}")
    print(
        f"answer: {prediction['letter']} (parsed as {prediction['parsed_as']}) "
        f"vs expected {prediction['expected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
