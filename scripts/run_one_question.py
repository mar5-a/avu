"""Run one VES-Bench question through VideoSeek and save the run.

The benchmark's answer key and evidence annotations are written to the run
directory for our own inspection and are never placed in the prompt. The model
sees the question stem and the four options, and nothing else.

Usage:
    .venv/bin/python scripts/run_one_question.py --question-id 257
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))

from videoseek.cli import main as cli_main  # noqa: E402


def load_question(parquet: Path, question_id: int) -> dict:
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        if row["id"] == question_id:
            return row
    raise SystemExit(f"question id {question_id} not found in {parquet}")


def build_query(row: dict) -> str:
    """The prompt text: question stem plus options, in the format the agent expects."""
    options = "\n".join(f"({k}) {v}" for k, v in row["options"].items())
    return (
        f"{row['question']}\n{options}\n"
        "Please directly answer with the best option's letter from the given "
        "choices directly (A, B, C, or D)."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-id", type=int, default=257)
    ap.add_argument("--parquet", type=Path, default=REPO / "data" / "VES-Bench.parquet")
    ap.add_argument("--video-dir", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--output-dir", type=Path, default=REPO / "data" / "runs")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    row = load_question(args.parquet, args.question_id)
    video = args.video_dir / f"{row['video_id']}.mp4"
    if not video.is_file():
        raise SystemExit(f"video missing: {video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query = build_query(row)

    print(f"question {row['id']}  task={row['task']}  source={row['dataset']}")
    print(f"video    {video.name} ({video.stat().st_size / 1e6:.1f} MB)")
    spans = ", ".join(f"{e['start_sec']:.0f}-{e['end_sec']:.0f}s" for e in row["evidence"])
    print(f"evidence {spans}")
    print("-" * 70)

    argv = [
        "--video_path", str(video),
        "--user_query", query,
        "--output_dir", str(args.output_dir),
        "--verbose",
    ]
    if args.max_steps is not None:
        argv += ["--max_steps", str(args.max_steps)]

    rc = cli_main(argv)

    # cli.main names the run directory itself; the newest one is ours.
    run_dir = max(args.output_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    (run_dir / "question.json").write_text(
        json.dumps(
            {
                "note": "Answer key and evidence. Never sent to the model.",
                "id": row["id"],
                "task": row["task"],
                "dataset": row["dataset"],
                "source_question_id": row["source_question_id"],
                "video_id": row["video_id"],
                "question": row["question"],
                "options": row["options"],
                "answer": row["answer"],
                "evidence": row["evidence"],
                "query_sent_to_model": query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("-" * 70)
    print(f"run dir: {run_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
