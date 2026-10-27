"""Run one cycle's planned runs in order, never starting one that could pass the cap.

Each finished run's cost — LiteLLM's estimate, from its own usage.json — is
appended to the cycle's ledger. Before every run the guard asks whether the spend
so far plus that run's worst case would pass the cycle cap. The worst case is the
per-run cap (`max_run_cost_usd`), at which a run stops itself before its next
call, so the cycle can pass its cap by at most one call's overshoot. A run whose
calls could not be priced, or that left no usage record, stops the cycle: the
guard cannot see what it spent. The AI Studio dashboard is the real bill.

Arms, all with the event definitions shown to the workers:
    medium   the agent, every call at reasoning effort "medium"
    high     the agent, every call at "high"
    uniform  128 evenly spaced frames, one direct answer, at "high"

Runs are ordered question by question, all arms of one question before the next,
so a cycle stopped by the cap leaves complete pairs rather than one arm finished.
A (question, arm, attempt) already in the ledger is skipped, so re-running the
same command resumes; repeats of a question are asked for with --attempt 2.

Default is a dry run: print the plan and the guard's arithmetic, run nothing.

Usage:
    .venv/bin/python scripts/run_cycle.py --cycle effort --questions 101,202
    .venv/bin/python scripts/run_cycle.py --cycle effort --questions 101,202 --live
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import general_config  # noqa: E402

from run_one_question import allocate_run_dir, load_question  # noqa: E402
from ves_score import extract_letter  # noqa: E402

CYCLE_CAP_USD = 12.0
ARMS = {
    "medium": ("run_one_question.py", ["--reasoning-effort", "medium"]),
    "high": ("run_one_question.py", ["--reasoning-effort", "high"]),
    "uniform": ("uniform_direct.py", ["--reasoning-effort", "high", "--live"]),
}


def read_ledger(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spent_so_far(entries):
    """Total estimated spend, and whether any run's spend is not fully known."""
    total = sum(entry["cost_usd"] or 0.0 for entry in entries)
    blind = [
        entry for entry in entries
        if entry["cost_usd"] is None or entry["cost_unavailable_calls"]
    ]
    return total, blind


def may_start(spent, worst_case, cap):
    """The guard: start a run only if even its worst case keeps the cycle in budget."""
    return spent + worst_case <= cap


def run_entry(question_id, arm, attempt, run_dir, exit_code, row):
    """One ledger line, read from what the run left on disk."""
    usage_path = run_dir / "usage.json"
    usage = json.loads(usage_path.read_text()) if usage_path.is_file() else None
    prediction_path = run_dir / "prediction.json"
    prediction = (
        json.loads(prediction_path.read_text())["prediction"]
        if prediction_path.is_file() else None
    )
    letter, parse = extract_letter(prediction) if prediction is not None else (None, "missing")
    return {
        "question_id": question_id,
        "arm": arm,
        "attempt": attempt,
        "run_dir": run_dir.name,
        "exit_code": exit_code,
        "cost_usd": usage["cost_usd"] if usage else None,
        "cost_unavailable_calls": usage["cost_unavailable_calls"] if usage else None,
        "calls": usage["calls"] if usage else None,
        "calls_failed": usage["calls_failed"] if usage else None,
        "answer_letter": letter,
        "answer_parse": parse,
        "ground_truth": str(row["answer"]).strip().upper()[:1],
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a cycle's runs under a spending cap.")
    ap.add_argument("--cycle", required=True, help="Cycle name; its ledger is data/cycles/<name>/.")
    ap.add_argument("--questions", required=True, help="Comma-separated question ids, in run order.")
    ap.add_argument("--arms", default="medium,high,uniform")
    ap.add_argument("--attempt", type=int, default=1, help="2 or more for repeats.")
    ap.add_argument("--cap", type=float, default=CYCLE_CAP_USD, help="Cycle cap in USD.")
    ap.add_argument("--parquet", type=Path, default=REPO / "data" / "VES-Bench.parquet")
    ap.add_argument("--output-dir", type=Path, default=REPO / "data" / "runs")
    ap.add_argument("--live", action="store_true", help="Actually run (costs money).")
    args = ap.parse_args()

    questions = [int(part) for part in args.questions.split(",") if part.strip()]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")
    worst_case = general_config["max_run_cost_usd"]
    cycle_dir = REPO / "data" / "cycles" / args.cycle
    ledger_path = cycle_dir / "ledger.jsonl"
    entries = read_ledger(ledger_path)
    done = {(e["question_id"], e["arm"], e["attempt"]) for e in entries}
    planned = [
        (q, arm) for q in questions for arm in arms if (q, arm, args.attempt) not in done
    ]

    spent, blind = spent_so_far(entries)
    print(
        f"cycle {args.cycle}: {len(planned)} run(s) to go, attempt {args.attempt}; "
        f"spent ${spent:.2f} of ${args.cap:.2f}; worst case per run ${worst_case:.2f}"
    )
    if blind:
        print(f"STOP: {len(blind)} earlier run(s) have spend the guard cannot see: "
              + ", ".join(entry["run_dir"] for entry in blind))
        return 1
    if not args.live:
        for q, arm in planned:
            print(f"  would run q{q} {arm}")
        print("dry run: nothing was run. Re-run with --live to run.")
        return 0

    cycle_dir.mkdir(parents=True, exist_ok=True)
    for q, arm in planned:
        if not may_start(spent, worst_case, args.cap):
            print(f"STOP before q{q} {arm}: ${spent:.2f} spent + ${worst_case:.2f} worst case "
                  f"would pass the ${args.cap:.2f} cap")
            return 1
        row = load_question(args.parquet, q)
        label = arm if args.attempt == 1 else f"{arm}{args.attempt}"
        run_dir = allocate_run_dir(args.output_dir, row, label)
        script, extra = ARMS[arm]
        with open(run_dir / "run.log", "w", encoding="utf-8") as log:
            exit_code = subprocess.run(
                [sys.executable, str(REPO / "scripts" / script), "--question-id", str(q),
                 "--run-dir", str(run_dir), *extra],
                stdout=log, stderr=subprocess.STDOUT,
            ).returncode
        entry = run_entry(q, arm, args.attempt, run_dir, exit_code, row)
        with open(ledger_path, "a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry) + "\n")
        print(
            f"  q{q} {arm:<8} {entry['answer_letter'] or '-'} vs {entry['ground_truth']}  "
            f"exit {exit_code}  "
            + (f"${entry['cost_usd']:.3f}" if entry["cost_usd"] is not None else "cost unknown")
            + f"  -> {run_dir.name}"
        )
        if entry["cost_usd"] is None or entry["cost_unavailable_calls"]:
            print("STOP: this run's spend is not fully known, so the guard cannot continue")
            return 1
        spent += entry["cost_usd"]
    print(f"cycle {args.cycle}: done; spent ${spent:.2f} of ${args.cap:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
