import argparse
import json
import time
from pathlib import Path
from typing import Optional

from config import general_config, init_config, prompts_config
from .agent import VideoSeekAgent
from .recorder import recorder


def build_agent_config(args: argparse.Namespace) -> dict:
    config = {}
    config.update(general_config)
    config.update(prompts_config)
    config = init_config(config, args)
    return config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="videoseek")
    p.add_argument("--video_path", required=True, help="YouTube URL or local path to video.")
    p.add_argument("--user_query", required=True, help="Question/query towards this video.")
    p.add_argument("--subtitle_path", default=None, help="Local path to subtitle file.")
    p.add_argument(
        "--output_dir",
        default="./output/",
        help="Directory to write outputs (default: ./output/).",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help=(
            "Explicit directory for this run's artifacts. Defaults to "
            "<output_dir>/<video_id>_<timestamp>, which two concurrent runs of the "
            "same video can collide on."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Print agent step logs.")

    # Allow overriding general.yaml keys (optional)
    p.add_argument("--model_name", default=general_config['model_name'], help="Model name.")
    p.add_argument("--api_base", default=general_config['api_base'], help="API base.")
    p.add_argument("--api_key", default=general_config['api_key'], help="API key.")
    p.add_argument("--api_version", default=general_config['api_version'], help="API version.")
    p.add_argument("--reasoning_effort", default=general_config['reasoning_effort'], help="Reasoning effort of the LLM.")
    p.add_argument("--seed", type=int, default=general_config['seed'], help="Seed.")
    p.add_argument("--temperature", type=float, default=general_config['temperature'], help="Temperature.")
    p.add_argument("--max_tokens", type=int, default=general_config['max_tokens'], help="Max output tokens of the LLM.")
    p.add_argument("--max_steps", type=int, default=general_config['max_steps'], help="Max steps of the VideoSeek agent.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_agent_config(args)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        video_id = args.video_path.split("/")[-1].split(".")[0]
        run_id = f"{video_id}_{int(time.time())}"
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    recorder.start(run_dir)
    recorder.record_settings(config)

    # Run the agent
    agent = VideoSeekAgent(
        config=config,
        video_path=args.video_path,
        subtitle_path=args.subtitle_path,
        output_dir=str(output_dir),
        tools=config["tools"],
        verbose=args.verbose,
    )
    traj_dict = None
    try:
        traj = agent.run(args.user_query)
        traj_dict = traj.to_dict()
    except Exception as e:
        # A crash on step 7 must not cost the six steps that already happened:
        # the partial transcript is what separates a model mistake from a broken
        # run. Recorded, then the exception continues on its way.
        traj_dict = {
            "question": args.user_query,
            "steps": [step.to_dict() for step in agent.trajectory_steps],
            "final_answer": agent.final_answer,
            "finish_reason": "error",
            "error": f"{type(e).__name__}: {e}",
        }
        raise
    finally:
        recorder.write_manifest()
        (run_dir / "trajectory.json").write_text(
            json.dumps(traj_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    prediction = traj_dict.get("final_answer") or ""

    print(f"Question: {args.user_query}")
    print(f"Prediction: {prediction}")

    (run_dir / "prediction.json").write_text(json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
