"""Score saved runs against VES-Bench's own metrics.

These are the benchmark's definitions (paper §4.3), not ones invented here:

- **F_i** — the set of *distinct* source-frame timestamps decoded for the model
  before it answered. An interval the agent proposed but never decoded
  contributes nothing, and a frame re-supplied in a later call is not counted
  twice, because it is already in the set.
- **supplied** — how many frames were handed over, counted per call, so a frame
  re-supplied later is charged again. This is the convention behind the paper's
  frame-budget figure and is deliberately different from F_i.
- **Coverage@k** — the share of questions where every annotated interval holds
  at least k timestamps from F_i. The paper's headline is k = 2.
- **Evidence Precision (EP)** — per question, |F_i ∩ union of intervals| / |F_i|,
  averaged over questions: a mean of ratios, not one pooled ratio.
- **Allocation Ratio (AR)** — EP / EP_ref. 1.0 is uniform allocation; above 1
  means evidence intervals were observed more than their share of the video.
- **ECA@k** — answered correctly while also passing Coverage@k.

One measure here is ours, not the paper's, and is labelled so in the output:

- **Claimed times** — for each event, the time the run's own answer gives for it
  ("Event C: at around 15.0s – 18.0s"), and whether that overlaps the event's
  annotated interval. Coverage says a frame was delivered inside an interval;
  this says whether the run placed the event there. Neither shows the event was
  recognised, and an answer that names no times scores nothing either way.

Two conventions the paper leaves open are fixed here and written into the output
so figures stay comparable: a frame counts as inside an interval when its exact
timestamp falls in the closed range [start, end], and timestamps are compared as
seconds rounded to six decimals.

Usage:
    .venv/bin/python scripts/ves_score.py data/runs/q257-temporal-ordering-2026-09-19-3
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))

# Benchmark constant: the macro-averaged share of video duration covered by
# annotated evidence intervals on VES-Bench. AR is measured against it.
EP_REF_VES_BENCH = 0.137

METRIC_SOURCE = (
    "VES-Bench metrics, paper section 4.3; closed interval bounds and 6-decimal "
    "timestamp comparison chosen here and stated in the output"
)


def extract_letter(text, letters="ABCD"):
    """The option letter in an answer, and how it was found.

    Checked in the order the shapes actually appear: the whole answer being the
    letter, then a closing line that is nothing but the letter (the prompt asks
    for the letter directly, and an answer that reasons through "Event A, B, C"
    still ends on the letter it chose), then a labelled answer ("Answer: B",
    "**(C)**"), then a leading letter, then a single unambiguous mention.
    Returns (None, reason) rather than guessing, because an unparsed answer is not
    a wrong answer.
    """
    if not text or not text.strip():
        return None, "empty"
    stripped = text.strip()
    bare = stripped.strip("()[]*.:# \n\t").upper()
    if len(bare) == 1 and bare in letters:
        return bare, "bare"
    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines:
        closing = lines[-1].strip().strip("()[]*.:# \t").upper()
        if len(closing) == 1 and closing in letters:
            return closing, "closing-line"
    labelled = re.search(
        # Markers arrive in either order: "(C)" and "**(B)**" both appear.
        r"(?:final\s+answer|correct\s+option|answer)\s*(?:is|:|-)?[(*\"'\s]{0,6}([A-D])\b",
        stripped,
        flags=re.IGNORECASE,
    )
    if labelled:
        return labelled.group(1).upper(), "labelled"
    # A letter wrapped in brackets, bold or not: "**(B)**", "(C)", "[D]". Only a
    # *wrapped* letter counts, which keeps the option text these answers restate
    # ("Event C -> Event B -> Event A") from reading as three answers; an answer
    # can name the sequence it chose without naming an option keyword, as q150's
    # experiment C did, and then this is the only rule that sees its letter. It
    # answers only when one letter is wrapped: "either (B) or (C)" wraps two and
    # is a hedge, not a choice.
    wrapped = set(
        re.findall(r"[(\[]\s*\**\s*([A-D])\s*\**\s*[)\]]", stripped, flags=re.IGNORECASE)
    )
    if len(wrapped) == 1:
        return wrapped.pop().upper(), "bracketed"
    leading = re.search(r"^\s*[(*\"'\s]{0,6}([A-D])[).\s*\"']", stripped, flags=re.IGNORECASE)
    if leading:
        return leading.group(1).upper(), "leading"
    # "Event C" names an event, not the option chosen, so those mentions are not
    # counted here; an answer that only ever writes events has said nothing about
    # which option it picked.
    found = set(re.findall(r"(?<!EVENT )\b([A-D])\b", stripped.upper()))
    if len(found) == 1:
        return found.pop(), "sole-mention"
    return None, "ambiguous"


def provenance_map(run_dir, provenance_path=None):
    """Image index -> frame provenance, from the run or a backfilled file."""
    if provenance_path is not None:
        entries = json.loads(Path(provenance_path).read_text())["images"]
    else:
        entries = json.loads((Path(run_dir) / "images.json").read_text())["images"]
    return {entry["index"]: entry.get("provenance") for entry in entries}


def load_observations(run_dir, provenance_path=None):
    """Every frame the run decoded, plus every frame it was handed."""
    entries = provenance_map(run_dir, provenance_path)
    manifest = json.loads((Path(run_dir) / "images.json").read_text())
    decoded, seen, gaps = [], set(), []
    supplied = 0
    for image in manifest["images"]:
        frames = entries.get(image["index"])
        timestamps = (frames or {}).get("timestamps_sec") or []
        if not timestamps:
            gaps.append(image["index"])
            continue
        supplied += len(timestamps)
        for timestamp in timestamps:
            key = round(float(timestamp), 6)
            if key not in seen:
                seen.add(key)
                decoded.append(key)
    return {
        "decoded": decoded,
        "supplied": supplied,
        "gaps": gaps,
        "images": len(manifest["images"]),
        "frame_space": manifest.get("frame_space"),
    }


def union_length(intervals):
    """Total duration covered by intervals, counting overlaps once."""
    total, current_end = 0.0, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def tool_windows(run_dir):
    """Every windowed tool call the agent made: its step, tool and requested window."""
    path = Path(run_dir) / "trajectory.json"
    if not path.is_file():
        return []
    windows = []
    for step in json.loads(path.read_text()).get("steps", []):
        action = step.get("action") or {}
        parameters = action.get("parameters") or {}
        if action.get("function") in ("focus", "skim") and "start_time" in parameters:
            windows.append(
                {
                    "step": step.get("step_id"),
                    "tool": action["function"],
                    "start_sec": float(parameters["start_time"]),
                    "end_sec": float(parameters["end_time"]),
                }
            )
    return windows


def covering_call(interval, windows):
    """The call whose window overlaps an interval most, and its signed shifts.

    Negative start shift: the window began before the interval. Negative end
    shift: it stopped before the interval ended. Both at once is the "aimed
    early" pattern — inspecting the approach to an event and clipping its tail.
    Ties go to the earlier call. An interval covered only by `overview` has no
    window of its own and reports None: overview's window is the whole video and
    says nothing about where the agent chose to look.
    """
    start, end = interval
    best = None
    for window in windows:
        overlap = min(end, window["end_sec"]) - max(start, window["start_sec"])
        if overlap <= 0:
            continue
        if best is None or (overlap, -window["step"]) > (best[0], -best[1]["step"]):
            best = (overlap, window)
    if best is None:
        return None
    window = best[1]
    return {
        "step": window["step"],
        "tool": window["tool"],
        "start_sec": window["start_sec"],
        "end_sec": window["end_sec"],
        "overlap_sec": round(best[0], 3),
        "start_shift_sec": round(window["start_sec"] - start, 3),
        "end_shift_sec": round(window["end_sec"] - end, 3),
    }


EVENT_LABEL = re.compile(r"\bEvent\s+([A-Z])\b")
SECONDS = r"(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds)\b"
TIME_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds)?\s*(?:-|–|—|to|and|until)\s*" + SECONDS,
    flags=re.IGNORECASE,
)
TIME_POINT = re.compile(SECONDS, flags=re.IGNORECASE)


def claimed_windows(sources):
    """Event letter -> the window a run's own text places it in, and where it said so.

    `sources` is [(name, text)] in order of preference. A claim is the first time
    range (or single time) after a mention of "Event X", before the next event's
    mention or the end of that line, so "Event C -> Event B -> Event A" claims
    nothing and "Event C: at around 15.0s – 18.0s" claims 15-18s. The first
    source that places an event wins.
    """
    claims = {}
    for name, text in sources:
        for line in (text or "").splitlines():
            mentions = list(EVENT_LABEL.finditer(line))
            for mention, following in zip(mentions, mentions[1:] + [None]):
                event = mention.group(1).upper()
                if event in claims:
                    continue
                segment = line[mention.end(): following.start() if following else len(line)]
                found = TIME_RANGE.search(segment)
                if found:
                    window = sorted((float(found.group(1)), float(found.group(2))))
                else:
                    point = TIME_POINT.search(segment)
                    if not point:
                        continue
                    window = [float(point.group(1))] * 2
                claims[event] = {"window_sec": window, "source": name}
    return claims


def score_claimed_times(sources, evidence):
    """Per annotated event: the run's claimed window, and whether it overlaps."""
    claims = claimed_windows(sources)
    per_event = []
    for interval in evidence:
        label = EVENT_LABEL.match((interval.get("description") or "").strip())
        if not label:
            continue
        event = label.group(1).upper()
        start, end = float(interval["start_sec"]), float(interval["end_sec"])
        claim = claims.get(event)
        record = {"event": event, "gold_sec": [start, end], "claimed_sec": None,
                  "source": None, "overlaps": None, "gap_sec": None}
        if claim:
            low, high = claim["window_sec"]
            gap = max(start - high, low - end, 0.0)
            record.update(
                claimed_sec=[low, high], source=claim["source"],
                overlaps=gap == 0.0, gap_sec=round(gap, 3),
            )
        per_event.append(record)
    return {
        "measure": "ours, not the paper's: overlap of the run's stated time with the interval",
        "events": len(per_event),
        "claimed": sum(1 for e in per_event if e["claimed_sec"] is not None),
        "overlapping": sum(1 for e in per_event if e["overlaps"]),
        "per_event": per_event,
    }


def answer_texts(run_dir, prediction):
    """The texts a run states its answer in, most direct first.

    The agent's answer is sometimes a bare letter while the planner's last
    thought lists each event's time; a direct-answer run's times, if any, are in
    its answer or the reasoning the provider returned.
    """
    run_dir = Path(run_dir)
    sources = [("answer", prediction)]
    trajectory = run_dir / "trajectory.json"
    if trajectory.is_file():
        steps = json.loads(trajectory.read_text()).get("steps") or []
        if steps:
            sources.append(("final_thought", steps[-1].get("thought")))
    for name in ("uniform_answer.json", "b_answer.json"):
        path = run_dir / name
        if path.is_file():
            reasoning = json.loads(path.read_text()).get("answer_diagnostics", {}).get("reasoning")
            sources.append(("reasoning", reasoning))
    return sources


def score_question(observation, evidence, windows=None, k_levels=(1, 2, 3)):
    """Coverage, evidence precision and per-interval detail for one question."""
    decoded = observation["decoded"]
    per_event = []
    for event in evidence:
        start, end = float(event["start_sec"]), float(event["end_sec"])
        inside = sum(1 for timestamp in decoded if start <= timestamp <= end)
        per_event.append(
            {
                "id": event.get("id"),
                "start_sec": start,
                "end_sec": end,
                "frames_observed": inside,
                "covering_call": covering_call((start, end), windows or []),
            }
        )
    intervals = [(event["start_sec"], event["end_sec"]) for event in per_event]
    inside_any = sum(
        1 for timestamp in decoded
        if any(start <= timestamp <= end for start, end in intervals)
    )
    covered = [event["covering_call"] for event in per_event if event["covering_call"]]
    # Kept apart by tool. A `focus` window is a deliberate lookup, so its shift
    # measures aim; a `skim` window is a sweep of the region the agent was
    # unsure about, sometimes most of the video, so its shift measures the sweep.
    # Pooling them lets one 100-second skim outvote every narrow lookup and
    # reports a left-shift bias that none of the lookups shows.
    by_tool = {}
    for tool in sorted({call["tool"] for call in covered}):
        group = [call for call in covered if call["tool"] == tool]
        by_tool[tool] = {
            "calls": len(group),
            "start_shift_sec_mean": mean([call["start_shift_sec"] for call in group]),
            "end_shift_sec_mean": mean([call["end_shift_sec"] for call in group]),
        }
    return {
        "decoded_frames": len(decoded),
        "supplied_frames": observation["supplied"],
        "images": observation["images"],
        "images_without_provenance": observation["gaps"],
        "per_event": per_event,
        "coverage": {k: all(e["frames_observed"] >= k for e in per_event) for k in k_levels},
        "evidence_precision": (inside_any / len(decoded)) if decoded else None,
        "evidence_union_sec": union_length(intervals),
        "window_shifts": {
            "windowed_calls": len(windows or []),
            "intervals": len(per_event),
            "intervals_with_a_covering_call": len(covered),
            "by_tool": by_tool,
        },
    }


def video_duration(video_path):
    """Duration in seconds, or None if the video is not available."""
    if not Path(video_path).is_file():
        return None
    from videoseek.video import VideoReader

    reader = VideoReader(str(video_path))
    return round(len(reader) / reader.get_avg_fps(), 3)


def score_run(run_dir, provenance_path=None, video_dir=None, ep_ref=EP_REF_VES_BENCH, k_levels=(1, 2, 3)):
    """One run's metrics, with correct-answer and reference-EP context."""
    run_dir = Path(run_dir)
    question = json.loads((run_dir / "question.json").read_text())
    prediction = json.loads((run_dir / "prediction.json").read_text())["prediction"]
    # Runs recorded before provenance was automatic keep theirs in a backfilled
    # file beside the run; say which source the numbers came from.
    backfilled = run_dir / "provenance.json"
    if provenance_path is None and backfilled.is_file():
        provenance_path = backfilled
    observation = load_observations(run_dir, provenance_path)
    if not observation["decoded"]:
        raise SystemExit(
            f"{run_dir.name}: no frame provenance. Recorded runs carry it in "
            "images.json; older ones need scripts/backfill_provenance.py first."
        )
    scored = score_question(observation, question["evidence"], tool_windows(run_dir), k_levels)
    predicted, parse = extract_letter(prediction)
    expected = str(question["answer"]).strip().upper()[:1]
    correct = predicted == expected
    union = scored["evidence_union_sec"]
    duration = video_duration(Path(video_dir) / f"{question['video_id']}.mp4") if video_dir else None
    result = {
        "run": run_dir.name,
        "question_id": question["id"],
        "task": question["task"],
        "metric_source": METRIC_SOURCE,
        "provenance_source": (
            Path(provenance_path).name if provenance_path is not None else "images.json"
        ),
        "frame_space": observation["frame_space"],
        "answer": {
            "prediction_raw": prediction,
            "prediction_letter": predicted,
            "parse": parse,
            "ground_truth": expected,
            "correct": correct,
        },
        "evidence_union_sec": union,
        "video_sec": duration,
        # What uniform sampling would have scored on this question, for context:
        # the benchmark's own EP_ref is the macro-average of this over all of them.
        "reference_ep_this_question": (union / duration) if duration else None,
        **scored,
    }
    if result["evidence_precision"] is not None:
        result["allocation_ratio"] = result["evidence_precision"] / ep_ref
    result["ECA"] = {k: bool(correct and scored["coverage"][k]) for k in k_levels}
    result["claimed_times"] = score_claimed_times(
        answer_texts(run_dir, prediction), question["evidence"]
    )
    return result

def format_report(result):
    """A short readable block, not a dump of every field."""
    lines = [
        f"{result['run']}  (question {result['question_id']}, {result['task']})",
        f"  answer: {result['answer']['prediction_raw']!r} -> "
        f"{result['answer']['prediction_letter']} (parsed as {result['answer']['parse']}), "
        f"expected {result['answer']['ground_truth']} -> "
        f"{'correct' if result['answer']['correct'] else 'WRONG'}",
        f"  frames: {result['decoded_frames']} distinct decoded, "
        f"{result['supplied_frames']} supplied over {result['images']} images",
    ]
    if result["images_without_provenance"]:
        lines.append(
            f"  WARNING: {len(result['images_without_provenance'])} images carry no "
            "provenance; coverage figures below are lower bounds"
        )
    lines.append("  per annotated interval:")
    for event in result["per_event"]:
        call = event.get("covering_call")
        window = (
            f"   window {call['tool']} {call['start_sec']:.1f}-{call['end_sec']:.1f}s "
            f"(start {call['start_shift_sec']:+.1f}s, end {call['end_shift_sec']:+.1f}s)"
            if call
            else "   no covering focus/skim window"
        )
        lines.append(
            f"    {event['start_sec']:>7.1f}-{event['end_sec']:<7.1f}s "
            f"{event['frames_observed']:>3} frames{window}"
        )
    shifts = result["window_shifts"]
    lines.append(
        f"  windows: {shifts['intervals_with_a_covering_call']}/"
        f"{shifts['intervals']} intervals covered by a windowed call"
    )
    for tool, group in sorted(shifts["by_tool"].items()):
        note = "  (sweep, not aim)" if tool == "skim" else ""
        lines.append(
            f"    {tool}: {group['calls']} call(s), mean shift "
            f"start {group['start_shift_sec_mean']:+.1f}s, "
            f"end {group['end_shift_sec_mean']:+.1f}s{note}"
        )
    coverage = ", ".join(
        f"@{k} {'pass' if ok else 'fail'}" for k, ok in sorted(result["coverage"].items())
    )
    precision = result["evidence_precision"]
    lines.append(
        f"  coverage: {coverage}   ECA: "
        + ", ".join(
            f"@{k} {'pass' if ok else 'fail'}" for k, ok in sorted(result["ECA"].items())
        )
    )
    lines.append(
        f"  evidence precision: {precision:.3f}" if precision is not None
        else "  evidence precision: n/a"
    )
    claimed = result["claimed_times"]
    details = ", ".join(
        f"{e['event']} {e['claimed_sec'][0]:.0f}-{e['claimed_sec'][1]:.0f}s vs "
        f"{e['gold_sec'][0]:.0f}-{e['gold_sec'][1]:.0f}s {'ok' if e['overlaps'] else 'miss'}"
        if e["claimed_sec"] else f"{e['event']} no time stated"
        for e in claimed["per_event"]
    )
    lines.append(
        f"  claimed times (ours): {claimed['overlapping']}/{claimed['events']} events placed "
        f"inside their interval ({claimed['claimed']} stated): {details}"
    )
    if result.get("allocation_ratio") is not None:
        share = (
            f"{result['reference_ep_this_question']:.3f}"
            if result["reference_ep_this_question"] is not None
            else "n/a (video not available)"
        )
        lines.append(
            f"  allocation ratio: {result['allocation_ratio']:.2f} "
            f"(EP_ref {EP_REF_VES_BENCH:.3f}; this question's uniform share {share})"
        )
    return "\n".join(lines)


def mean(values):
    """Average of the values that are present, or None if none are."""
    present = [value for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


def main():
    ap = argparse.ArgumentParser(description="Score runs against VES-Bench's metrics.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--provenance", type=Path, default=None,
                    help="Backfilled provenance file, for runs recorded before it was automatic.")
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--ep-ref", type=float, default=EP_REF_VES_BENCH)
    ap.add_argument("--no-write", action="store_true", help="Print only, write no score.json.")
    args = ap.parse_args()

    results = []
    for run_dir in args.run_dirs:
        result = score_run(run_dir, args.provenance, args.video_dir, args.ep_ref)
        print(format_report(result))
        print()
        if not args.no_write:
            (run_dir / "score.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        results.append(result)

    if len(results) > 1:
        n = len(results)
        print(f"across {n} questions (macro-average):")
        for k in sorted(results[0]["coverage"]):
            print(f"  Coverage@{k}: {sum(r['coverage'][k] for r in results) / n:.3f}"
                  f"   ECA@{k}: {sum(r['ECA'][k] for r in results) / n:.3f}")
        precisions = [r["evidence_precision"] for r in results if r["evidence_precision"] is not None]
        accuracy = sum(r["answer"]["correct"] for r in results) / n
        print(f"  Evidence Precision: {sum(precisions) / len(precisions):.3f}"
              f"   Accuracy: {accuracy:.3f}")
        events = sum(r["claimed_times"]["events"] for r in results)
        print(
            f"  claimed times (ours): {sum(r['claimed_times']['overlapping'] for r in results)}"
            f"/{events} events placed inside their interval, "
            f"{sum(r['claimed_times']['claimed'] for r in results)} with a time stated"
        )
        for tool in sorted({t for r in results for t in r["window_shifts"]["by_tool"]}):
            groups = [
                r["window_shifts"]["by_tool"][tool]
                for r in results
                if tool in r["window_shifts"]["by_tool"]
            ]
            starts = mean([g["start_shift_sec_mean"] for g in groups])
            ends = mean([g["end_shift_sec_mean"] for g in groups])
            note = "  (sweep, not aim)" if tool == "skim" else ""
            print(
                f"  window shift, {tool} only, {sum(g['calls'] for g in groups)} call(s): "
                f"start {starts:+.1f}s, end {ends:+.1f}s{note}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
