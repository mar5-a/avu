"""Run one VES-Bench question through VideoSeek and save the run.

The benchmark's answer key and evidence annotations are written to the run
directory for our own inspection and are never placed in the prompt. The model
sees the question stem and the four options, and nothing else.

The run directory is claimed here and passed down, rather than discovered
afterwards by looking at what appeared in the output directory: with two runs
in flight that guess can name the other run's directory.

Usage:
    .venv/bin/python scripts/run_one_question.py --question-id 257
"""

import argparse
import json
import sys
import time
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


def video_properties(path: Path) -> dict:
    """Frames, rate, duration and pixel size of the source video.

    Recorded because nothing else in a run captures it, and the frames the model
    saw depend on it: `focus` sends frames unresized, so a video that was
    downloaded at a lower resolution is a caveat on any fine-detail claim.
    """
    from decord import VideoReader

    reader = VideoReader(str(path))
    fps = reader.get_avg_fps()
    frames = len(reader)
    height, width = reader[0].shape[:2]
    return {
        "frames": frames,
        "fps": round(fps, 3),
        "duration_sec": round(frames / fps, 2),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
    }


def run_name(row: dict) -> str:
    """A readable run directory name: `q257-temporal-ordering-2026-09-19`."""
    return f"q{row['id']}-{row['task'].replace('_', '-')}-{time.strftime('%Y-%m-%d')}"


def allocate_run_dir(output_dir: Path, row: dict) -> Path:
    """Claim today's run name for this question, suffixing if it is taken.

    Claimed with `exist_ok=False` so that two runs starting together cannot both
    believe they own the same directory.
    """
    base = run_name(row)
    for n in range(1, 1000):
        candidate = output_dir / (base if n == 1 else f"{base}-{n}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f"too many runs of {base} under {output_dir}")


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

    run_dir = allocate_run_dir(args.output_dir, row)
    argv = [
        "--video_path", str(video),
        "--user_query", query,
        "--output_dir", str(args.output_dir),
        "--run_dir", str(run_dir),
        "--verbose",
    ]
    if args.max_steps is not None:
        argv += ["--max_steps", str(args.max_steps)]

    try:
        rc = cli_main(argv)
    finally:
        # The answer key is kept next to the run whatever happened to it, so a
        # failed attempt is still readable and still comparable.
        (run_dir / "question.json").write_text(
            json.dumps(
                {
                    "note": "Answer key and evidence. Never sent to the model.",
                    "id": row["id"],
                    "task": row["task"],
                    "dataset": row["dataset"],
                    "source_question_id": row["source_question_id"],
                    "video_id": row["video_id"],
                    "video": video_properties(video),
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
    # A run whose tools raised still produces an answer, from whatever was left —
    # which may be nothing but the question. Say so rather than let it pass as a
    # quiet model mistake.
    manifest = run_dir / "images.json"
    if manifest.is_file():
        for error in json.loads(manifest.read_text()).get("tool_errors") or []:
            print(
                f"WARNING: {error['tool']} raised {error['error_type']}: "
                f"{error['error'][:120]} — this run's answer may rest on no evidence"
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
