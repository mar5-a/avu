"""Experiment B: ask the run's own question again, from the run's own images.

Run A never showed those images to the model that answered. The controller is
text-only: the image tools produced text reports, and the final answer was
written from those reports. B holds the visual input fixed and removes the
reports — the same bytes, in the same order, with the same visible labels, and
the same question — so that a different answer can be attributed to the reports
rather than to the frames.

B's prompt is A's final-answer call with exactly one change: the images are put
where the text history was. A's answer call appended
`Question: ... Please directly provide the final answer.` to its history; B sends
that same line, preceded by the frames. A's system prompt is not sent, because
it describes tool policies for tools B cannot call.

Deliberately left out of B: everything A wrote — its tool reports, the per-call
search queries it invented, its thoughts, its final answer — and the answer key.
The request is built from `images.json` and its labels, never from
`calls[].blocks`, which faithfully records what A sent *including* the queries the
agent wrote for itself. Building from `blocks` would import A's reasoning into B.

Default is a dry run: build the request, verify it, print what it contains, and
call nothing. `--live` sends that same request, records the reply, and checks the
images that went out against A's saved bytes.

Usage:
    .venv/bin/python scripts/replay_b.py --run data/runs/q257-temporal-ordering-2026-09-19-3
    .venv/bin/python scripts/replay_b.py --run data/runs/q257-temporal-ordering-2026-09-19-3 --live
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

from config import general_config, init_config, prompts_config  # noqa: E402
from videoseek.recorder import recorder  # noqa: E402
from videoseek.utils import call_llm_api, extract_thought  # noqa: E402

from ves_score import extract_letter, provenance_map  # noqa: E402

ANSWER_INSTRUCTION = "Please directly provide the final answer."


def load_run(run_dir):
    question = json.loads((run_dir / "question.json").read_text())
    manifest = json.loads((run_dir / "images.json").read_text())
    return question, manifest


def build_spec(run_dir):
    """The request B will send: texts verbatim, images by reference and digest."""
    run_dir = Path(run_dir)
    question, manifest = load_run(run_dir)
    query = question["query_sent_to_model"]
    blocks = []
    for image in manifest["images"]:
        if image.get("label"):
            blocks.append(
                {
                    "type": "text",
                    "text": image["label"],
                    "source": f"label of image {image['index']}",
                }
            )
        blocks.append(
            {
                "type": "image",
                "index": image["index"],
                "file": image["file"],
                "sha256": image["sha256"],
                "bytes": image["bytes"],
            }
        )
    blocks.append(
        {
            "type": "text",
            "text": f"Question:\n{query}\n\n{ANSWER_INSTRUCTION}",
            "source": "question",
        }
    )
    return {
        "experiment": "B",
        "source_run": run_dir.name,
        "question_id": question["id"],
        "images": len(manifest["images"]),
        "image_bytes": sum(image["bytes"] for image in manifest["images"]),
        "blocks": blocks,
    }


def label_is_only_timestamps(label):
    """True when a label carries nothing but times and the grid's own vocabulary.

    Time labels are the only text A's frames are allowed to carry into B: they say
    when a frame came from, not what to look for.
    """
    stripped = label.replace("Timestamp Matrix", "")
    stripped = re.sub(r"[\d\s.,:\[\]]", "", stripped)
    return set(stripped) <= {"s", "-"}


def forbidden_strings(run_dir, question):
    """Distinctive text from A that must not appear anywhere in B.

    Only prefixes, and only of long strings: A's one-letter final answer would
    match any text at all, so short strings are not usable as evidence of
    leakage. The allowlist in `verify_spec` is the primary check; this one is
    a second opinion.
    """
    strings = []
    trajectory = json.loads((Path(run_dir) / "trajectory.json").read_text())
    for step in trajectory.get("steps", []):
        for key in ("thought", "observation"):
            text = step.get(key) or ""
            if len(text) > 60:
                strings.append(text[:60])
        query = (step.get("action") or {}).get("parameters", {}).get("query")
        if query:
            strings.append(query)
    final = trajectory.get("final_answer") or ""
    if len(final) > 3:
        strings.append(final)
    for event in question.get("evidence", []):
        if event.get("description"):
            strings.append(event["description"][:60])
    if question.get("note"):
        strings.append(question["note"])
    return strings


def verify_spec(spec, run_dir):
    """Problems with the assembled request. Empty means it may be sent."""
    run_dir = Path(run_dir)
    question, manifest = load_run(run_dir)
    problems = []
    texts = [block for block in spec["blocks"] if block["type"] == "text"]
    images = [block for block in spec["blocks"] if block["type"] == "image"]

    # 1. The frames and their labels, in the recorded order. Checked as a whole
    #    sequence, so a dropped or reordered label is caught rather than only a
    #    wrong one: a frame without its label loses the time it came from.
    expected = []
    for image in manifest["images"]:
        if image.get("label"):
            expected.append("label")
        expected.append("image")
    expected.append("question")
    actual = [
        "image" if block["type"] == "image"
        else "question" if block["source"] == "question"
        else "label"
        for block in spec["blocks"]
    ]
    if actual != expected:
        first = next(
            (i for i, (got, want) in enumerate(zip(actual, expected)) if got != want),
            min(len(actual), len(expected)),
        )
        problems.append(
            f"blocks do not follow the recorded labels and images "
            f"({len(actual)} blocks, {len(expected)} expected, first difference at {first})"
        )
    for image in manifest["images"]:
        block = images[image["index"]] if image["index"] < len(images) else None
        if block is None or block["sha256"] != image["sha256"]:
            problems.append(f"image {image['index']} digest disagrees with the manifest")
            continue
        path = run_dir / image["file"]
        if not path.is_file():
            problems.append(f"image {image['index']} missing: {image['file']}")
            continue
        raw = path.read_bytes()
        if len(raw) != image["bytes"] or hashlib.sha256(raw).hexdigest() != image["sha256"]:
            problems.append(f"image {image['index']} on disk no longer matches its digest")

    # 2. Every label is that image's label, and says nothing but when.
    for block in texts[:-1]:
        index = int(block["source"].rsplit(" ", 1)[-1])
        if block["text"] != manifest["images"][index].get("label"):
            problems.append(f"text before image {index} is not that image's label")
        if not label_is_only_timestamps(block["text"]):
            problems.append(f"label of image {index} carries words, not just timestamps")

    # 3. The question block is the question, not a summary of the answer.
    query = question["query_sent_to_model"]
    if texts and texts[-1]["text"] != f"Question:\n{query}\n\n{ANSWER_INSTRUCTION}":
        problems.append("the question block is not the verbatim query plus the answer line")
    for option in question["options"]:
        if f"({option})" not in query:
            problems.append(f"option {option} is not in the query")

    # 4. Nothing of A's comes along for the ride. Scanned over the labels only:
    # A's search queries are written out of the question's own wording, so the
    # question block legitimately contains those words — it is checked verbatim
    # just above instead. A label, by contrast, is allowed to be nothing but
    # timestamps, so any of A's text there is a real leak.
    sent_labels = "\n".join(block["text"] for block in texts[:-1])
    for forbidden in forbidden_strings(run_dir, question):
        if forbidden in sent_labels:
            problems.append(f"text from A appears in B's labels: {forbidden[:40]!r}")
    return problems


def to_messages(spec, run_dir):
    """The wire payload, re-reading every image and re-checking its digest."""
    run_dir = Path(run_dir)
    content = []
    for block in spec["blocks"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
            continue
        raw = (run_dir / block["file"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != block["sha256"]:
            raise SystemExit(
                f"image {block['index']} changed since the request was assembled "
                f"({digest} != {block['sha256']}); refusing to send"
            )
        encoded = base64.b64encode(raw).decode("utf-8")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return [{"role": "user", "content": content}]


def allocate_dir(run_dir, suffix):
    """A directory beside the run's own, claiming the name as it is created."""
    run_dir = Path(run_dir)
    for n in range(1, 100):
        candidate = run_dir.parent / (f"{run_dir.name}-{suffix}" if n == 1 else f"{run_dir.name}-{suffix}{n}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f"too many {suffix} runs of {run_dir.name}")


def allocate_b_dir(run_dir):
    return allocate_dir(run_dir, "B")


def send(spec, run_dir, config):
    """Send the verified request once, and record what came back.

    Recording goes through the module-level recorder, the same one `call_llm_api`
    writes to. A private `ImageRecorder()` here would collect nothing: the images
    are captured inside `utils`, which knows only about the singleton.
    """
    run_dir = Path(run_dir)
    b_dir = allocate_b_dir(run_dir)
    recorder.start(b_dir)
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
    # Recorded rather than implied: A's system prompt is deliberately not sent.
    recorder.settings["system_prompt_sent"] = False
    recorder.settings["experiment"] = "B"
    recorder.settings["source_run"] = run_dir.name

    (b_dir / "b_request.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    messages = to_messages(spec, run_dir)
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
    # The frames B sent are A's frames, so their provenance is A's too. Copied
    # only where the digests agree, which is what makes it safe to copy, and it
    # lets B be scored on its own. Runs recorded before provenance was automatic
    # keep theirs in a backfilled provenance.json instead of images.json.
    _, a_manifest = load_run(run_dir)
    backfilled = run_dir / "provenance.json"
    frames = provenance_map(run_dir, backfilled if backfilled.is_file() else None)
    by_digest = {
        image["sha256"]: frames.get(image["index"]) for image in a_manifest["images"]
    }
    for image in recorder.images:
        image["provenance"] = by_digest.get(image["sha256"])
    recorder.write_manifest()

    if response is None:
        raise SystemExit(
            f"B call failed; the reason is in {b_dir / 'events.jsonl'}"
        )
    # Read the reply the same way the agent does: Gemini has answered a
    # video-understanding request with a tool call instead of prose, leaving
    # `content` empty, and the text is then in `reasoning_content`.
    answer, answer_from = extract_thought(response)
    diagnostics = {
        "answer_from": answer_from,
        "content_is_none": response.choices[0].message.content is None,
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        # The thinking text, when the provider returns it. Those tokens are
        # generated and paid for either way, and without them B's reasoning is
        # unreadable next to A's, whose thoughts are in the trajectory.
        "reasoning": getattr(response.choices[0].message, "reasoning_content", None),
    }

    # The images that actually went out, against A's saved bytes. This is the
    # verification the plan asks for, done on the sent payload rather than on
    # the intent to send it.
    sent_manifest = json.loads((b_dir / "images.json").read_text())
    _, a_manifest = load_run(run_dir)
    mismatches = []
    for sent, original in zip(sent_manifest["images"], a_manifest["images"]):
        if sent["sha256"] != original["sha256"]:
            mismatches.append(sent["index"])
    if len(sent_manifest["images"]) != len(a_manifest["images"]):
        mismatches.append("count")

    letter, parse = extract_letter(answer)
    question, _ = load_run(run_dir)
    expected = str(question["answer"]).strip().upper()[:1]
    a_prediction = json.loads((run_dir / "prediction.json").read_text())["prediction"]
    a_letter, a_parse = extract_letter(a_prediction)
    payload = {
        "experiment": "B",
        "source_run": run_dir.name,
        "question_id": question["id"],
        "answer_raw": answer,
        "answer_letter": letter,
        "answer_parse": parse,
        "answer_diagnostics": diagnostics,
        "ground_truth": expected,
        "correct": letter == expected,
        "a_answer_letter": a_letter,
        "a_answer_parse": a_parse,
        "agreement_with_a": letter == a_letter,
        "images_sent": len(sent_manifest["images"]),
        "image_mismatches": mismatches,
        "images_verified_against_a": not mismatches,
        "usage": json.loads((b_dir / "usage.json").read_text()),
    }
    (b_dir / "b_answer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (b_dir / "prediction.json").write_text(
        json.dumps({"prediction": answer}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return b_dir, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build (and optionally send) experiment B.")
    ap.add_argument("--run", type=Path, required=True, help="A run directory.")
    ap.add_argument("--live", action="store_true", help="Actually send the request (costs money).")
    ap.add_argument("--out", type=Path, default=None, help="Write the request spec here.")
    args = ap.parse_args()

    spec = build_spec(args.run)
    problems = verify_spec(spec, args.run)
    print(f"B from {spec['source_run']}: {spec['images']} images, "
          f"{spec['image_bytes'] / 1e6:.1f} MB, "
          f"{sum(1 for b in spec['blocks'] if b['type'] == 'text')} text blocks")

    if problems:
        print("verification FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("verification passed: bytes, order, labels and question match A; "
          "no text of A's is present")

    if args.out:
        args.out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if not args.live:
        print("dry run: nothing was sent. Re-run with --live to send.")
        return 0

    config = dict(general_config)
    config.update(prompts_config)
    config = init_config(config, argparse.Namespace())
    b_dir, payload = send(spec, args.run, config)
    print(f"b run dir: {b_dir}")
    print(f"answer: {payload['answer_letter']} (parsed as {payload['answer_parse']}) "
          f"vs A {payload['a_answer_letter']} vs expected {payload['ground_truth']}")
    print(f"images verified against A: {payload['images_verified_against_a']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
