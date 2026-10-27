"""Pick questions for a run set by a rule fixed before any result is seen.

The rule, not the outcome, decides membership. Strata are quantiles of the span
between the first and last annotated interval — how far apart the evidence is,
which is what "connect three moments" actually stresses — plus positional picks
evenly spaced through the question file. Questions sharing a video, and questions
already run, are skipped; the only substitution allowed is a video that will not
download. Without all this, "a fixed set selected before seeing results" is a
claim rather than a fact.

Ego4D questions are excluded: their `video_path` is a bare UUID with no source
URL, so they cannot be fetched here. Any pilot from this script is YouTube-only,
which should be said out loud whenever the pilot is described.

Usage:
    .venv/bin/python scripts/select_questions.py            # the five
    .venv/bin/python scripts/select_questions.py --json
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]

# (task, stratum, position through that list, how many). Positions are fractions
# of the list length, so picks spread across the question file rather than
# clustering at one end.
FIVE = (
    ("temporal_ordering", "all", 0.25, 1),
    ("temporal_ordering", "spread", 0.50, 1),
    ("temporal_ordering", "tight", 0.50, 1),
    ("event_counting", "all", 0.50, 1),
    ("event_counting", "spread", 0.50, 1),
)


def evidence_span(row):
    """Seconds from the start of the first annotated interval to the end of the last."""
    events = row["evidence"]
    return max(event["end_sec"] for event in events) - min(
        event["start_sec"] for event in events
    )


def already_run():
    """Question ids that already have a run directory."""
    runs = REPO / "data" / "runs"
    ids = set()
    if runs.is_dir():
        for path in runs.iterdir():
            name = path.name
            if name.startswith("q"):
                try:
                    ids.add(int(name[1:].split("-")[0]))
                except ValueError:
                    continue
    return ids


def stratum(rows, name):
    """Quantile strata, so they exist whatever the distribution looks like."""
    ordered = sorted(rows, key=evidence_span)
    if name == "all":
        return ordered
    if name == "spread":
        return ordered[int(0.75 * len(ordered)):]
    if name == "tight":
        return ordered[: max(1, int(0.25 * len(ordered)))]
    raise SystemExit(f"unknown stratum {name!r}")


def select(rows, spec, exclude=()):
    """One question per line of `spec`, never two from the same video.

    `exclude` holds question ids whose video would not download: re-running with
    those ids excluded is how the rule replaces a pick mechanically instead of by
    hand.
    """
    done = already_run() | set(exclude)
    eligible = sorted(
        (
            row
            for row in rows
            if row["dataset"] == "youtube" and row["id"] not in done
        ),
        key=lambda row: row["id"],
    )
    chosen, videos = [], set()
    for task, name, position, count in spec:
        pool = stratum([row for row in eligible if row["task"] == task], name)
        if not pool:
            raise SystemExit(f"empty stratum: {task}/{name}")
        index = min(int(round(position * (len(pool) - 1))), len(pool) - 1)
        picked = 0
        # Walk forward from the position, then wrap, until the count is met or
        # the pool runs out: substitution is mechanical, never a judgement call.
        for offset in range(len(pool)):
            row = pool[(index + offset) % len(pool)]
            if row["video_id"] in videos:
                continue
            videos.add(row["video_id"])
            chosen.append({**row, "_stratum": f"{task}/{name}"})
            picked += 1
            if picked == count:
                break
        if picked < count:
            raise SystemExit(f"stratum {task}/{name} could not fill {count} distinct videos")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=REPO / "data" / "VES-Bench.parquet")
    ap.add_argument("--json", action="store_true", help="Print ids and metadata as JSON.")
    ap.add_argument(
        "--exclude",
        default="",
        help="Comma-separated question ids to skip, for videos that will not download.",
    )
    args = ap.parse_args()

    exclude = [int(part) for part in args.exclude.split(",") if part.strip()]
    rows = pq.read_table(args.parquet).to_pylist()
    chosen = select(rows, FIVE, exclude)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": row["id"],
                        "task": row["task"],
                        "video_id": row["video_id"],
                        "stratum": row["_stratum"],
                        "intervals": len(row["evidence"]),
                        "evidence_span_sec": round(evidence_span(row), 1),
                    }
                    for row in chosen
                ],
                indent=2,
            )
        )
        return 0

    for row in chosen:
        spans = ", ".join(
            f"{event['start_sec']:.0f}-{event['end_sec']:.0f}s" for event in row["evidence"]
        )
        print(
            f"q{row['id']:<4} {row['task']:<18} {row['_stratum']:<28} "
            f"video {row['video_id']:<12} {len(row['evidence'])} intervals, "
            f"span {evidence_span(row):.0f}s: {spans}"
        )
    print("\nids:", " ".join(str(row["id"]) for row in chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
