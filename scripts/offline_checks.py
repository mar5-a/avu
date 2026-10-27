"""Offline checks: no network, no API key, no cost.

Each check exercises one claim the recording, replay and scoring depend on, and
fails loudly when that claim stops holding. They were written after a review
found that the record was weaker than it looked: images were saved but the frame
they showed was not, retries and failures never reached disk, and a run that
crashed lost its transcript.

    .venv/bin/python scripts/offline_checks.py

Exits non-zero if any check fails.
"""

import base64
import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "videoseek"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm  # noqa: E402
from litellm.types.utils import Usage  # noqa: E402
from videoseek.video import VideoReader  # noqa: E402

import videoseek.tools.focus as focus_tool  # noqa: E402
import videoseek.tools.overview as overview_tool  # noqa: E402
import videoseek.tools.skim as skim_tool  # noqa: E402
import videoseek.utils as utils  # noqa: E402
from videoseek.cli import main as cli_main  # noqa: E402
from videoseek.recorder import ImageRecorder, recorder  # noqa: E402
from videoseek.utils import call_llm_api, retry_with_exponential_backoff  # noqa: E402

import run_cycle  # noqa: E402
import uniform_direct  # noqa: E402
from replay_b import build_spec, label_is_only_timestamps, send, verify_spec  # noqa: E402
from ves_score import score_claimed_times, score_run  # noqa: E402

FPS = 30
SECONDS = 3
TOOLS_CONFIG = {
    "model_name": "gemini/gemini-3.8-flash",
    "api_base": None,
    "api_key": None,
    "api_version": None,
    "max_tokens": 256,
    "reasoning_effort": "low",
    "seed": 1,
    "temperature": 1.0,
}


class FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content

    def json(self):
        return {"content": self.content, "tool_calls": self.tool_calls}


def fake_response(content=None, tool_calls=None, usage=None, finish_reason="stop", reasoning=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(content, tool_calls, reasoning), finish_reason=finish_reason
            )
        ],
        usage=usage,
    )


def fake_usage(prompt=10, completion=5, reasoning=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=None if prompt is None else prompt + completion,
        completion_tokens_details=(
            SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
        ),
    )


@contextlib.contextmanager
def patch_attr(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def read_events(run_dir):
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_test_video(path, seconds=SECONDS, fps=FPS):
    """A short clip with moving content, so neighbouring frames differ."""
    if not Path(path).is_file():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             f"testsrc2=size=320x240:rate={fps}:duration={seconds}",
             "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
    return Path(path)


def recording_stub(content):
    """A stand-in for `call_llm_api` that records exactly as the real one does.

    The network is what gets stubbed; the recording path under test is the real
    one, reached through the same calls the real function makes.
    """

    def call(*args, **kwargs):
        recorder.record_request(kwargs)
        recorder.capture(kwargs.get("messages") or [])
        response = fake_response(content=content, usage=fake_usage())
        recorder.record_usage(response)
        return response

    return call


def encode(array):
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="jpeg")
    return buffer.getvalue()


def check_block_structure(tmp):
    """A captured call keeps every block, including the text before the first image."""
    payload = base64.b64encode(b"\xff\xd8\xff\xe0not-really-a-jpeg").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The video segment is located at 0.0s - 3.0s:\n"},
                {"type": "text", "text": "[0.0s - 1.0s]:\nTimestamp Matrix:\n[[0.0s, 1.0s]]\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
                {"type": "text", "text": "Please generate descriptions for each frame."},
            ],
        }
    ]
    record = ImageRecorder()
    record.start(tmp)
    record.set_context(step=1, tool="overview")
    record.record_request(
        {
            "messages": messages,
            "tools": [{"function": {"name": "overview"}}],
            "tool_choice": "required",
            "temperature": 1.0,
            "max_tokens": 32768,
            "return_json": True,
        }
    )
    record.set_provenance(
        [{"frame_indices": [0, 30], "timestamps_sec": [0.0, 1.0], "grid": {"rows": 2, "cols": 4}}],
        avg_fps=30.0,
        total_frames=90,
    )
    record.capture(messages)
    record.write_manifest()

    manifest = json.loads((Path(tmp) / "images.json").read_text())
    blocks = manifest["calls"][0]["blocks"]
    assert [block["type"] for block in blocks] == ["text", "text", "image", "text"], blocks
    assert blocks[0]["text"].startswith("The video segment"), "text before the first image was dropped"
    assert manifest["images"][0]["label"] == messages[0]["content"][1]["text"]
    assert manifest["images"][0]["provenance"]["frame_indices"] == [0, 30]
    request = manifest["calls"][0]["request"]
    assert request["tools"] == ["overview"], request
    assert request["max_tokens"] == 32768, request
    assert request["json_output"] is True, request


def check_usage_unknown_is_not_zero(tmp):
    """A token field the provider did not report is unknown, never a zero."""
    record = ImageRecorder()
    record.start(tmp)
    record.record_usage(fake_response(usage=fake_usage()))
    assert record.usage["calls"] == 1 and record.usage["prompt_tokens"] == 10
    assert record.usage["complete"] is True

    record.record_usage(fake_response(usage=None))
    assert record.usage["calls_without_usage"] == 1, record.usage
    assert record.usage["prompt_tokens"] == 10, "a missing usage must not add a zero"
    assert record.usage["complete"] is False

    record.record_usage(fake_response(usage=fake_usage(prompt=None)))
    assert record.usage["fields_missing"]["prompt_tokens"] == 1, record.usage
    assert record.usage["incomplete_calls"] == 1 and record.usage["complete"] is False
    assert record.usage["prompt_tokens"] == 10, record.usage

    # Cost on its own recorder: an unpriced response is counted as unpriced, and
    # a priced one must not be lost among those.
    priced_dir = Path(tmp) / "priced"
    priced_dir.mkdir()
    priced = ImageRecorder()
    priced.start(priced_dir)
    priced.record_usage(fake_response(usage=fake_usage()))
    assert priced.usage["cost_unavailable_calls"] == 1, priced.usage
    priced.record_usage(
        litellm.ModelResponse(
            model="gemini/gemini-3.8-flash",
            usage=Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
        )
    )
    assert priced.usage["cost_unavailable_calls"] == 1, "an unpriced call was recorded as priced"
    assert abs(priced.usage["cost_usd"] - 0.001125) < 1e-9, priced.usage
    assert priced.usage["complete"] is True, "an unknown price must not spoil the token totals"


def check_failure_recording(tmp):
    """Retries, give-ups and unrecoverable failures all reach disk."""
    recorder.start(tmp)

    def refused(*args, **kwargs):
        raise RuntimeError("boom: refused by the provider")

    # Not a retryable error by the wrapper's own rules: one attempt, then out.
    wrapper = retry_with_exponential_backoff(
        refused, initial_delay=0, exponential_base=1, jitter=False, max_retries=2
    )
    assert wrapper() is None, "the wrapper should return None once it gives up"
    events = [e for e in read_events(tmp) if e["kind"] == "attempt"]
    assert [e["status"] for e in events] == ["failed"], events
    assert events[-1]["error_type"] == "RuntimeError"
    assert "refused by the provider" in events[-1]["error"]
    assert events[-1]["elapsed_sec"] is not None

    def rate_limited(*args, **kwargs):
        raise RuntimeError("Rate limit exceeded")

    wrapper = retry_with_exponential_backoff(
        rate_limited, initial_delay=0, exponential_base=1, jitter=False, max_retries=1
    )
    assert wrapper() is None
    statuses = [e["status"] for e in read_events(tmp) if e["kind"] == "attempt"]
    assert statuses == ["failed", "retry", "gave_up"], statuses
    retries = [e for e in read_events(tmp) if e["kind"] == "attempt" and e["status"] == "retry"]
    assert retries and retries[0]["delay"] is not None, "no retry delay recorded"

    recorder.write_manifest()
    usage = json.loads((Path(tmp) / "usage.json").read_text())
    assert usage["calls_failed"] == 2, usage
    assert usage["complete"] is False, usage


def check_call_failure_is_recorded(tmp):
    """A call that dies returns None as before, but leaves the reason on disk."""
    recorder.start(tmp)

    def refused(*args, **kwargs):
        raise RuntimeError("boom: refused by the provider")

    with patch_attr(utils, "completion", refused):
        assert call_llm_api(
            model_name="gemini/gemini-3.8-flash", api_base=None,
            messages=[{"role": "user", "content": "hi"}],
        ) is None
    recorder.write_manifest()
    usage = json.loads((Path(tmp) / "usage.json").read_text())
    assert usage["calls_failed"] == 1 and usage["calls"] == 0, usage
    events = [e for e in read_events(tmp) if e["kind"] == "attempt"]
    assert events and events[-1]["status"] == "failed"
    # Without this the check passed on any failure, including a call that never
    # reached the provider because its arguments were wrong.
    assert "refused by the provider" in events[-1]["error"], events[-1]


def check_high_demand_is_retried(tmp):
    """Gemini's 503 "high demand" is retried, not turned into a failed tool."""
    recorder.start(tmp)
    # The message a smoke-test run got on 2026-10-27, which upstream did not retry.
    overloaded = litellm.ServiceUnavailableError(
        message='GeminiException - {"error": {"code": 503, "message": "This model is currently '
        'experiencing high demand. Spikes in demand are usually temporary. Please try again '
        'later.", "status": "UNAVAILABLE"}}',
        llm_provider="gemini",
        model="gemini/gemini-3.8-flash",
    )
    outcomes = [overloaded]

    def busy_once(*args, **kwargs):
        if outcomes:
            raise outcomes.pop()
        return fake_response(content="fine", usage=fake_usage())

    with patch_attr(utils, "completion", busy_once), patch_attr(utils.time, "sleep", lambda s: None):
        response = call_llm_api(
            model_name="gemini/gemini-3.8-flash", api_base=None,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert response is not None, "a 503 still ends the call"
    attempts = [e for e in read_events(tmp) if e["kind"] == "attempt"]
    assert [e["status"] for e in attempts] == ["retry"], attempts
    assert attempts[0]["error_type"] == "ServiceUnavailableError", attempts[0]
    recorder.write_manifest()
    usage = json.loads((Path(tmp) / "usage.json").read_text())
    assert usage["calls"] == 1 and usage["calls_failed"] == 0, usage
    return "one 503, one retry, then the answer"


def check_crash_keeps_transcript(tmp):
    """A run that dies mid-flight still leaves a readable, partial record.

    The run does not die on the provider error itself: `call_llm_api` returns None
    when it gives up and the agent then trips over the None. So the trajectory
    records one error and `events.jsonl` records the cause — this checks both, and
    deliberately does not require the trajectory to name the provider.
    """
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"

    def refused(*args, **kwargs):
        raise RuntimeError("boom: refused by the provider")

    with patch_attr(utils, "completion", refused):
        try:
            cli_main(
                [
                    "--video_path", str(video),
                    "--user_query", "What colour is the pattern?\n(A) red\n(B) blue\n",
                    "--output_dir", str(tmp),
                    "--run_dir", str(run_dir),
                ]
            )
            raise AssertionError("the run was expected to fail")
        except Exception:  # noqa: BLE001 - any failure, the artifacts are the subject
            pass

    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    assert trajectory["finish_reason"] == "error", trajectory
    assert trajectory["error"], trajectory
    assert trajectory["steps"] == [], trajectory
    attempts = [e for e in read_events(run_dir) if e["kind"] == "attempt"]
    assert attempts and "refused by the provider" in attempts[-1]["error"], attempts
    usage = json.loads((run_dir / "usage.json").read_text())
    assert usage["calls_failed"] == 1, usage
    settings = json.loads((run_dir / "settings.json").read_text())
    assert settings["model_name"], settings
    assert "api_key" not in settings and "api_key_present" in settings, settings
    return f"died on {trajectory['error'].split(':', 1)[0]}, cause recorded in events.jsonl"


def check_provenance_identifies_frames(tmp):
    """Recorded provenance names the frames that are actually inside each image."""
    video = make_test_video(tmp / "clip.mp4")
    reader = VideoReader(str(video))
    fps = reader.get_avg_fps()
    run_dir = tmp / "prov"
    run_dir.mkdir()
    recorder.start(run_dir)

    overview_content = json.dumps({"frames": [{"timestamp": "1.0s", "description": "x"}]})
    recorder.set_context(step=1, tool="overview")
    with patch_attr(overview_tool, "call_llm_api", recording_stub(overview_content)):
        overview_tool.execute_overview(
            TOOLS_CONFIG, {"vr": reader, "subtitles": []}
        )
    recorder.set_context(step=2, tool="focus")
    with patch_attr(focus_tool, "call_llm_api", recording_stub("a description")):
        focus_tool.execute_focus(
            TOOLS_CONFIG,
            {"query": "q", "start_time": 1.0, "end_time": 2.0, "vr": reader, "subtitles": []},
        )
    recorder.set_context(step=3, tool="skim")
    with patch_attr(skim_tool, "call_llm_api", recording_stub("a description")):
        skim_tool.execute_skim(
            TOOLS_CONFIG,
            {"query": "q", "start_time": 0.0, "end_time": 3.0, "vr": reader, "subtitles": []},
        )
    recorder.write_manifest()
    manifest = json.loads((run_dir / "images.json").read_text())

    checked = {"focus": 0, "skim": 0, "overview": 0}
    for image in manifest["images"]:
        provenance = image["provenance"]
        assert provenance, f"image {image['index']} has no provenance"
        saved = (run_dir / image["file"]).read_bytes()
        if image["tool"] in ("focus", "skim"):
            # One frame per image: re-encoding it must reproduce the exact bytes.
            frames = reader.get_batch([provenance["frame_indices"][0]]).asnumpy()
            if image["tool"] == "skim":
                frames = resize_short_side(frames)
            assert hashlib.sha256(encode(frames[0])).hexdigest() == image["sha256"], (
                f"{image['tool']} image {image['index']} is not the frame it claims"
            )
            checked[image["tool"]] += 1
            continue

        # A grid: crop each cell out of the saved image and confirm the frame the
        # provenance names is the one that best matches that cell. Cropping the
        # sent image rather than rebuilding it is what makes this independent of
        # the layout code it is checking.
        claimed = provenance["frame_indices"]
        rows, cols = provenance["grid"]["rows"], provenance["grid"]["cols"]
        sheet = np.array(Image.open(run_dir / image["file"]).convert("RGB"))
        height, width, _ = sheet.shape
        cell_height, cell_width = height // rows, width // cols
        frames = resize_short_side(reader.get_batch(claimed).asnumpy())
        for position, frame_index in enumerate(claimed):
            row, column = divmod(position, cols)
            cell = sheet[
                row * cell_height:(row + 1) * cell_height,
                column * cell_width:(column + 1) * cell_width,
            ]
            match = np.array(
                Image.fromarray(frames[position]).resize((cell_width, cell_height), Image.BICUBIC)
            )
            difference = np.abs(cell.astype(int) - match.astype(int)).mean()
            assert difference < 12, (
                f"overview cell {position} does not show frame {frame_index} "
                f"(mean absolute difference {difference:.1f})"
            )
            checked["overview"] += 1

    assert checked["overview"] == 64, checked
    assert checked["focus"] == 16, checked
    assert checked["skim"] == 16, checked
    return f"{sum(checked.values())} images verified against their source frames"


def resize_short_side(frames):
    """Mirror of the tools' resize: scale so the shorter side is 256, bicubic."""
    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack(
        [np.array(Image.fromarray(frame).resize(size, Image.BICUBIC)) for frame in frames]
    )


def check_end_to_end_run(tmp):
    """A whole agent run, scripted offline, then scored against a known answer."""
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    focus_arguments = json.dumps(
        {"query": "What is shown?", "start_time": 1.0, "end_time": 2.0}
    )
    scripted = [
        fake_response(content="I will inspect the middle of the clip.", usage=fake_usage()),
        fake_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "focus", "arguments": focus_arguments},
                }
            ],
            finish_reason="tool_calls",
            usage=fake_usage(),
        ),
        fake_response(content="A moving test pattern with a counter.", usage=fake_usage()),
        fake_response(content="I can answer now.", usage=fake_usage()),
        fake_response(
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "answer", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
            usage=fake_usage(),
        ),
        fake_response(content="B", usage=fake_usage()),
    ]

    def scripted_completion(*args, **kwargs):
        assert scripted, "the agent asked for more responses than the script provides"
        return scripted.pop(0)

    with patch_attr(utils, "completion", scripted_completion):
        cli_main(
            [
                "--video_path", str(video),
                "--user_query", "What colour is the pattern?\n(A) red\n(B) blue\n",
                "--output_dir", str(tmp),
                "--run_dir", str(run_dir),
            ]
        )

    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    assert trajectory["finish_reason"] == "stop", trajectory
    assert trajectory["final_answer"] == "B", trajectory
    assert [step["action"]["function"] for step in trajectory["steps"]] == ["focus", "answer"]

    (run_dir / "question.json").write_text(
        json.dumps(
            {
                "note": "Answer key and evidence. Never sent to the model.",
                "id": 1,
                "task": "temporal_ordering",
                "dataset": "synthetic",
                "video_id": "clip",
                "question": "What colour is the pattern?",
                "options": {"A": "red", "B": "blue"},
                "answer": "B",
                "evidence": [{"id": 1, "start_sec": 1.0, "end_sec": 2.0, "description": "synthetic"}],
                "query_sent_to_model": "What colour is the pattern?\n(A) red\n(B) blue\n",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scored = score_run(run_dir, video_dir=tmp)
    # The scripted focus covers 1.0-2.0s and the annotation is 1.0-2.0s, so every
    # decoded frame is inside it and the answer is right by construction.
    assert scored["decoded_frames"] == 16, scored
    assert scored["evidence_precision"] == 1.0, scored
    assert all(scored["coverage"].values()), scored["coverage"]
    assert scored["answer"]["correct"] is True, scored["answer"]
    assert all(scored["ECA"].values()), scored["ECA"]

    usage = json.loads((run_dir / "usage.json").read_text())
    assert usage["calls"] == 6, usage
    assert usage["cost_unavailable_calls"] == 6, "unknown cost must be reported as unknown"
    assert usage["complete"] is True, usage
    return "6 scripted calls, coverage@3 and ECA@3 satisfied by construction"


def check_run_dir_is_claimed(tmp):
    """Two runs of the same question cannot claim the same directory.

    The name is claimed by creating it, not by looking first and creating after:
    with two runs in flight, the look-then-create version lets both believe they
    own the same directory.
    """
    from run_one_question import allocate_run_dir, run_name

    row = {"id": 257, "task": "temporal_ordering"}
    claimed = [allocate_run_dir(tmp, row) for _ in range(3)]
    assert len({path.name for path in claimed}) == 3, claimed
    assert claimed[0].name == run_name(row), claimed[0]
    assert claimed[1].name == f"{run_name(row)}-2", claimed[1]
    assert all(path.is_dir() for path in claimed), claimed

    # A directory that already exists must be left alone rather than taken over.
    (tmp / "existing").mkdir()
    assert allocate_run_dir(tmp, {"id": 1, "task": "t"}).name != "existing"
    labelled = allocate_run_dir(tmp, row, "high")
    assert labelled.name == f"{run_name(row)}-high", labelled
    return f"{', '.join(path.name for path in claimed)}"


def check_b_send_records_what_it_sent(tmp):
    """Sending B records B's own images, usage and events — offline.

    This check exists because of a real bug: the first live B call answered
    correctly and wrote a manifest containing no images at all. The sending code
    had built a private `ImageRecorder()`, while the images are captured inside
    `utils` against the module-level recorder. Only a check that runs the sending
    path itself can catch that; verifying the request beforehand cannot.
    """
    source = REPO / "data" / "runs" / "q257-temporal-ordering-2026-09-19-3"
    if not source.is_dir():
        return "skipped: the recorded run is not present"
    run = tmp / source.name
    shutil.copytree(source, run)
    spec = build_spec(run)
    assert not verify_spec(spec, run)
    config = {
        "model_name": "gemini/gemini-3.8-flash",
        "api_base": None,
        "api_key": None,
        "api_version": None,
        "max_tokens": 256,
        "reasoning_effort": "medium",
        "seed": 1,
        "temperature": 1.0,
    }
    with patch_attr(
        utils,
        "completion",
        lambda *a, **k: fake_response(
            content="(A)", usage=fake_usage(), reasoning="The tag reads $229."
        ),
    ):
        b_dir, payload = send(spec, run, config)

    recorded = json.loads((b_dir / "images.json").read_text())["images"]
    original = json.loads((run / "images.json").read_text())["images"]
    assert len(recorded) == 72, f"B recorded {len(recorded)} images, not 72"
    assert [i["sha256"] for i in recorded] == [i["sha256"] for i in original], (
        "the images B recorded are not the ones A saved"
    )
    assert recorded[0]["provenance"], "B's images are missing the provenance it was sent"
    assert payload["images_verified_against_a"] is True, payload["image_mismatches"]
    usage = json.loads((b_dir / "usage.json").read_text())
    assert usage["calls"] == 1 and usage["prompt_tokens"] == 10, usage
    assert read_events(b_dir), "B wrote no events"
    assert payload["answer_letter"] == "A" and payload["correct"] is True, payload
    assert payload["answer_diagnostics"]["reasoning"] == "The tag reads $229.", (
        "B's thinking text was not kept"
    )
    return "72 images captured, digests matched, usage, events and reasoning written"


def check_window_offsets(tmp=None):
    """The window shift is measured, and its sign means what it claims."""
    from ves_score import covering_call, score_question

    windows = [
        {"step": 1, "tool": "focus", "start_sec": 10.0, "end_sec": 15.0},
        {"step": 2, "tool": "focus", "start_sec": 32.0, "end_sec": 37.0},
    ]
    # A window ending before its interval: both shifts negative.
    early = covering_call((14.0, 18.0), windows)
    assert early["step"] == 1, early
    assert early["start_shift_sec"] == -4.0 and early["end_shift_sec"] == -3.0, early
    # A window aligned at the end: the end shift is zero, not negative.
    aligned = covering_call((34.0, 37.0), windows)
    assert aligned["step"] == 2 and aligned["end_shift_sec"] == 0.0, aligned
    # An interval no windowed call touched reports nothing, rather than zero.
    assert covering_call((100.0, 104.0), windows) is None
    # A window wider than its interval shifts the other way at both ends.
    wide = covering_call(
        (34.0, 37.0), [{"step": 9, "tool": "skim", "start_sec": 30.0, "end_sec": 40.0}]
    )
    assert wide["start_shift_sec"] == -4.0 and wide["end_shift_sec"] == 3.0, wide
    # Ties go to the earlier call, so scoring the same run twice gives one answer.
    tie = covering_call(
        (14.0, 18.0),
        [
            {"step": 3, "tool": "focus", "start_sec": 9.0, "end_sec": 15.0},
            {"step": 2, "tool": "focus", "start_sec": 10.0, "end_sec": 15.0},
        ],
    )
    assert tie["step"] == 2, tie

    scored = score_question(
        {"decoded": [14.5, 15.5, 16.5, 35.0], "supplied": 4, "images": 4, "gaps": []},
        [
            {"id": 1, "start_sec": 14.0, "end_sec": 18.0},
            {"id": 2, "start_sec": 34.0, "end_sec": 37.0},
            {"id": 3, "start_sec": 100.0, "end_sec": 104.0},
        ],
        windows,
    )
    shifts = scored["window_shifts"]
    assert shifts["intervals_with_a_covering_call"] == 2, shifts
    assert shifts["by_tool"]["focus"]["calls"] == 2, shifts
    assert shifts["by_tool"]["focus"]["start_shift_sec_mean"] == -3.0, shifts  # (-4 + -2) / 2
    assert shifts["by_tool"]["focus"]["end_shift_sec_mean"] == -1.5, shifts  # (-3 + 0) / 2

    # A broad skim is not averaged together with the narrow lookups. Pooling them
    # let one 100-second sweep outvote every focused call and reported a
    # left-shift bias that no focused call showed.
    mixed = score_question(
        {"decoded": [14.5, 100.5], "supplied": 2, "images": 2, "gaps": []},
        [
            {"id": 1, "start_sec": 14.0, "end_sec": 18.0},
            {"id": 2, "start_sec": 100.0, "end_sec": 104.0},
        ],
        [
            {"step": 1, "tool": "focus", "start_sec": 10.0, "end_sec": 15.0},
            {"step": 2, "tool": "skim", "start_sec": 90.0, "end_sec": 110.0},
        ],
    )["window_shifts"]["by_tool"]
    assert mixed["focus"]["start_shift_sec_mean"] == -4.0, mixed
    assert mixed["skim"]["start_shift_sec_mean"] == -10.0, mixed
    assert mixed["skim"]["end_shift_sec_mean"] == 6.0, mixed
    return "sign convention pinned: -4/-3 early, 0 aligned, 2 of 3 covered, skim kept apart"


def check_letter_parsing(tmp=None):
    """The answer letter is read the way the model writes it, or not at all."""
    from ves_score import extract_letter

    assert extract_letter("B") == ("B", "bare")
    assert extract_letter("**(B)**") == ("B", "bare")
    # The prompt asks for the letter, and a reasoned answer still ends on one.
    assert extract_letter("Event C -> Event B -> Event A.\n\nC") == ("C", "closing-line")
    assert extract_letter("Therefore the order is C, B, A.\n\nAnswer: B") == ("B", "labelled")
    assert extract_letter("The correct answer is **(B)**.") == ("B", "labelled")
    assert extract_letter("(A)") == ("A", "bare")
    assert extract_letter("D. Because the third event happens last.")[0] == "D"
    # An answer can name the sequence it chose without naming an option keyword.
    # q150's experiment C wrote "The correct chronological order of the events is
    # **(B)** (Event C -> Event B -> Event A)." and the letters of the restated
    # event names outvoted the wrapped letter.
    assert extract_letter(
        "The correct chronological order of the events is **(B)** (Event C -> Event B -> Event A)."
    ) == ("B", "bracketed")
    # Nothing conclusive must stay inconclusive rather than guess.
    assert extract_letter("Events A, B and C are hard to order.") == (None, "ambiguous")
    assert extract_letter("The sequence is Event C -> Event B -> Event A.") == (None, "ambiguous")
    assert extract_letter("The answer is either (B) or (C), hard to tell.") == (None, "ambiguous")
    assert extract_letter("") == (None, "empty")
    return "bare, closing-line, labelled, bracketed and leading read; ambiguous stays None"


def check_string_parameters(tmp):
    """A model that sends numbers as strings must not break a tool.

    One live run arrived with `"start_time": "100"`, the sampling arithmetic
    raised before any call, and the run answered from the question text alone.
    """
    video = make_test_video(tmp / "clip.mp4")
    reader = VideoReader(str(video))
    run_dir = tmp / "strings"
    run_dir.mkdir()
    recorder.start(run_dir)
    recorder.set_context(step=1, tool="skim")
    with patch_attr(skim_tool, "call_llm_api", recording_stub("a description")):
        outcome = skim_tool.execute_skim(
            TOOLS_CONFIG,
            {"query": "q", "start_time": "1", "end_time": "2", "vr": reader, "subtitles": []},
        )
    recorder.write_manifest()
    manifest = json.loads((run_dir / "images.json").read_text())
    assert outcome == "a description", outcome
    assert manifest["image_count"] == 16, manifest["image_count"]
    assert manifest["tool_errors"] == [], manifest["tool_errors"]
    return "16 frames sampled from string start/end times"


def check_tool_error_is_visible(tmp):
    """A tool that raises says so, in the observation and in the run's record.

    The three pilot runs that answered with no evidence looked exactly like
    ordinary runs: `Tool execution failed.` and nothing else.
    """
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    bad_arguments = json.dumps({"query": "q", "start_time": "not-a-number", "end_time": 2.0})
    scripted = [
        fake_response(content="I will look closely.", usage=fake_usage()),
        fake_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "focus", "arguments": bad_arguments},
                }
            ],
            finish_reason="tool_calls",
            usage=fake_usage(),
        ),
        fake_response(content="I cannot look, so I will answer.", usage=fake_usage()),
        fake_response(
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "answer", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
            usage=fake_usage(),
        ),
        fake_response(content="A", usage=fake_usage()),
    ]

    def scripted_completion(*args, **kwargs):
        return scripted.pop(0)

    with patch_attr(utils, "completion", scripted_completion):
        cli_main(
            [
                "--video_path", str(video),
                "--user_query", "What colour is the pattern?\n(A) red\n(B) blue\n",
                "--output_dir", str(tmp),
                "--run_dir", str(run_dir),
            ]
        )

    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    observation = trajectory["steps"][0]["observation"]
    assert "Tool execution failed: ValueError" in observation, observation
    manifest = json.loads((run_dir / "images.json").read_text())
    assert manifest["tool_errors"], "the raised tool was not recorded"
    assert manifest["tool_errors"][0]["tool"] == "focus", manifest["tool_errors"]
    assert manifest["tool_errors"][0]["error_type"] == "ValueError", manifest["tool_errors"]
    return "reason kept in the observation and in images.json"


def check_replay_b_request(tmp=None):
    """B is assembled from the images and labels, and tampering is caught."""
    run = REPO / "data" / "runs" / "q257-temporal-ordering-2026-09-19-3"
    if not run.is_dir():
        return "skipped: the recorded run is not present"
    spec = build_spec(run)
    assert not verify_spec(spec, run), verify_spec(spec, run)
    assert spec["images"] == 72, spec["images"]
    labels = [b for b in spec["blocks"] if b["type"] == "text" and b["source"] != "question"]
    assert len(labels) == spec["images"], len(labels)
    assert all(label_is_only_timestamps(b["text"]) for b in labels)

    tampered = json.loads(json.dumps(spec))
    tampered["blocks"][0]["text"] = "Snapdragon 821 processor"
    assert any("carries words" in problem for problem in verify_spec(tampered, run))

    tampered = json.loads(json.dumps(spec))
    tampered["blocks"][1]["sha256"] = "0" * 64
    assert verify_spec(tampered, run), "a digest that disagrees with the manifest was accepted"

    tampered = json.loads(json.dumps(spec))
    tampered["blocks"].pop(0)
    assert verify_spec(tampered, run), "a missing label was accepted"
    return "72 images, and three kinds of tampering rejected"


def check_replay_c_request(tmp):
    """C samples the annotated windows, and its leak check ignores the question.

    Both assertions here are regressions. C's frames must come from the gold
    interval they are filed under, or C does not test what it claims. And the
    leak scan must not read the question's own wording as a leak: the benchmark's
    event descriptions and the question's event lines are the same sentences, so
    scanning the question block flagged every question and C refused to run.
    """
    from replay_c import SKIM_FRAMES, build_spec as build_c_spec, verify_spec as verify_c_spec

    # The test clip is 3s at 30fps; both windows are wide enough to sample.
    shared = "A red crate is pushed across the floor while the counter climbs."
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    run_dir.mkdir()
    query = (
        "In what order do these events occur?\n"
        f"Event A: {shared} slowly.\n"
        "Event B: The red dial turns all the way clockwise.\n"
        "(A) Event A then Event B\n(B) Event B then Event A\n"
        "(C) Neither\n(D) Both at once\n"
        "Please directly answer with the best option's letter (A, B, C, or D)."
    )
    (run_dir / "question.json").write_text(
        json.dumps(
            {
                "id": 999,
                "task": "temporal_ordering",
                "video_id": "clip",
                "query_sent_to_model": query,
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": "B",
                "evidence": [
                    {"id": 1, "start_sec": 0.2, "end_sec": 0.9, "description": shared},
                    {"id": 2, "start_sec": 1.5, "end_sec": 2.5,
                     "description": "The red dial turns clockwise as the frame brightens."},
                ],
            }
        ),
        encoding="utf-8",
    )
    # C is scanned against the run's own trajectory for text of A's, so the
    # fixture carries one, as every real run does.
    (run_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "question": query,
                "steps": [
                    {
                        "step_id": 1,
                        "thought": "The crate moves before the dial turns, so I should look near the start.",
                        "observation": "Two frames show the crate crossing the floor while the counter climbs.",
                        "action": {"function": "focus", "parameters": {"query": "when does the crate move"}, "id": "c1"},
                    }
                ],
                "final_answer": "B",
                "finish_reason": "stop",
            }
        ),
        encoding="utf-8",
    )

    spec, frames = build_c_spec(run_dir, tmp)
    assert not verify_c_spec(spec, run_dir), verify_c_spec(spec, run_dir)
    assert len(spec["intervals"]) == 2, spec["intervals"]
    assert all(r["frames"] == SKIM_FRAMES for r in spec["intervals"])
    fps = spec["fps"]
    for block in spec["blocks"]:
        if block["type"] != "image":
            continue
        record = next(r for r in spec["intervals"] if r["id"] == block["interval"])
        assert int(record["start_sec"] * fps) <= block["frame_index"] < int(record["end_sec"] * fps), block

    # A frame filed under the wrong window is refused.
    tampered = json.loads(json.dumps(spec))
    target = next(b for b in tampered["blocks"] if b["type"] == "image")
    target["frame_index"] = int(tampered["intervals"][0]["end_sec"] * fps) + 5
    assert any("sits outside" in p for p in verify_c_spec(tampered, run_dir))

    # An interval description in text C adds is refused; the same words inside
    # the question block are what the question is, and are not.
    tampered = json.loads(json.dumps(spec))
    stamp = next(
        b for b in tampered["blocks"]
        if b["type"] == "text" and b.get("source") == "frame timestamp"
    )
    stamp["text"] = shared
    assert verify_c_spec(tampered, run_dir), "an interval description was accepted"
    return "2 windows sampled from their own intervals; a mismatched frame and a leaked description both refused"


def check_replay_c_cli(tmp):
    """The C CLI runs end to end: --live builds, writes frames, records a run.

    This exercises `main()`, not just `build_spec` and `send`. The defect it exists
    for lived only in `main`: a local frame count reassigned the sampled-bytes
    dictionary before `send()` was called, so `write_images` received an int. A
    check that calls `build_spec` and `send` directly walks past that, and a dry
    run returns before `send` in any case — which is why the CLI was broken while
    every offline check still passed.
    """
    import replay_c

    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    run_dir.mkdir()
    (run_dir / "question.json").write_text(
        json.dumps(
            {
                "id": 998,
                "task": "temporal_ordering",
                "video_id": "clip",
                "query_sent_to_model": "In what order?\n(A) one\n(B) two\n(C) three\n(D) four\n"
                                       "Please directly answer with the best option's letter (A, B, C, or D).",
                "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
                "answer": "B",
                "evidence": [
                    {"id": 1, "start_sec": 0.2, "end_sec": 0.9, "description": "first"},
                    {"id": 2, "start_sec": 1.5, "end_sec": 2.5, "description": "second"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trajectory.json").write_text(
        json.dumps({"steps": [], "final_answer": "B"}), encoding="utf-8"
    )

    with patch_attr(utils, "completion", lambda *a, **k: fake_response(content="(B)", usage=fake_usage())):
        with patch_attr(
            sys, "argv",
            ["replay_c.py", "--run", str(run_dir), "--video-dir", str(tmp), "--live"],
        ):
            code = replay_c.main()
    assert code == 0, f"the C CLI returned {code}"

    c_dir = tmp / "run-C"
    assert c_dir.is_dir(), "the C CLI created no run directory"
    gold = sorted((c_dir / "gold").glob("*.jpg"))
    assert len(gold) == 32, f"wrote {len(gold)} sampled frames, not 32"
    manifest = json.loads((c_dir / "images.json").read_text())
    assert manifest["image_count"] == 32, manifest["image_count"]
    assert all(image["provenance"] for image in manifest["images"]), "no provenance recorded"
    prediction = json.loads((c_dir / "prediction.json").read_text())
    assert prediction["letter"] == "B" and prediction["expected"] == "B", prediction
    assert read_events(c_dir), "the C CLI wrote no events"
    return "32 sampled frames written, recorded with provenance, answer read back"


HOCKEY_STEM = (
    "Based on the visual segments provided in the video, in what sequence do these three scenes occur?\n"
    "Event A: Several players crowd the goal crease area in a physical scramble.\n"
    "Event B: Blackhawks player number 4 and a Ducks player battle along the boards.\n"
    "Event C: A Blackhawks player controls the puck as a Ducks player pursues from behind."
)
HOCKEY_OPTIONS = "(A) Event A -> Event B -> Event C\n(B) Event C -> Event A -> Event B\n"


def tool_call(call_id, name, arguments):
    return fake_response(
        tool_calls=[
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
        finish_reason="tool_calls",
        usage=fake_usage(),
    )


def check_workers_get_question_and_effort(tmp):
    """skim and focus see the event definitions; every call carries the run's effort.

    The pilot's hockey run sent a worker "Does Event A or Event B occur right after
    Event C?" with nothing saying what those events were, and upstream fixed skim
    at "low" whatever the run asked for.
    """
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    scripted = [
        fake_response(content="Skim the clip first.", usage=fake_usage()),
        tool_call("call_1", "skim", {"query": "Where is Event A?", "start_time": 0.5, "end_time": 2.5}),
        fake_response(content="A test pattern.", usage=fake_usage()),
        fake_response(content="Now look closely.", usage=fake_usage()),
        tool_call("call_2", "focus", {
            "query": "Does Event A or Event B occur right after Event C?",
            "start_time": 1.0, "end_time": 2.0,
        }),
        fake_response(content="No relevant content found.", usage=fake_usage()),
        fake_response(content="I can answer.", usage=fake_usage()),
        tool_call("call_3", "answer", {}),
        fake_response(content="B", usage=fake_usage()),
    ]

    def scripted_completion(*args, **kwargs):
        return scripted.pop(0)

    with patch_attr(utils, "completion", scripted_completion):
        cli_main(
            [
                "--video_path", str(video),
                "--user_query", f"{HOCKEY_STEM}\n{HOCKEY_OPTIONS}",
                "--question_context", HOCKEY_STEM,
                "--reasoning_effort", "high",
                "--output_dir", str(tmp),
                "--run_dir", str(run_dir),
            ]
        )

    manifest = json.loads((run_dir / "images.json").read_text())
    for tool in ("skim", "focus"):
        call = next(c for c in manifest["calls"] if c["tool"] == tool)
        prompt = call["blocks"][-1]["text"]
        assert HOCKEY_STEM in prompt, f"{tool} did not receive the event definitions"
        assert "(A) Event A -> Event B -> Event C" not in prompt, f"{tool} received the options"
    requests = [e for e in read_events(run_dir) if e["kind"] == "request"]
    efforts = {e["reasoning_effort"] for e in requests}
    assert efforts == {"high"}, f"calls ran at {efforts}, not only 'high'"
    trajectory_text = (run_dir / "trajectory.json").read_text()
    assert "question_context" not in trajectory_text, "the context leaked into the trajectory"
    settings = json.loads((run_dir / "settings.json").read_text())
    assert settings["question_context_sent"] is True and settings["reasoning_effort"] == "high"

    # Without a context the worker prompt is upstream's, unchanged.
    reader = VideoReader(str(video))
    bare_dir = tmp / "bare"
    bare_dir.mkdir()
    recorder.start(bare_dir)
    with patch_attr(focus_tool, "call_llm_api", recording_stub("x")):
        focus_tool.execute_focus(
            TOOLS_CONFIG,
            {"query": "q", "start_time": 1.0, "end_time": 2.0, "vr": reader, "subtitles": []},
        )
    recorder.write_manifest()
    bare = json.loads((bare_dir / "images.json").read_text())["calls"][0]["blocks"][-1]["text"]
    assert "Context:" not in bare and "Question:\nq\n" in bare, bare
    return f"{len(requests)} calls, all at 'high'; skim and focus saw the definitions, not the options"


def check_effort_reaches_gemini(tmp=None):
    """LiteLLM turns each effort into a Gemini thinking level rather than dropping it."""
    from litellm.utils import get_optional_params

    for effort in ("medium", "high"):
        params = get_optional_params(
            model="gemini-3.8-flash", custom_llm_provider="gemini",
            reasoning_effort=effort, temperature=1.0,
        )
        level = (params.get("thinkingConfig") or {}).get("thinkingLevel")
        assert level == effort, f"{effort} became {params.get('thinkingConfig')}"
    return "medium and high arrive as thinkingLevel medium and high"


def check_code_snapshot(tmp):
    """Every run records the commit it ran on and keeps any uncommitted diff."""
    record = ImageRecorder()
    record.start(tmp)
    record.record_settings(dict(TOOLS_CONFIG))
    record.write_manifest()
    code = json.loads((Path(tmp) / "settings.json").read_text())["code"]
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert code["head"] == head, code
    diff_path = Path(tmp) / "code.diff"
    if code["diff_sha256"]:
        assert diff_path.is_file(), "a dirty tree's diff was not kept"
        digest = hashlib.sha256(diff_path.read_text().encode("utf-8")).hexdigest()
        assert digest == code["diff_sha256"], "code.diff does not match its recorded hash"
    else:
        assert not diff_path.exists(), "a clean tree wrote a diff"
    return f"head {head[:7]}, dirty={code['dirty']}, {len(code['untracked'])} untracked"


def check_run_budget_stops_run(tmp):
    """A run at its cost cap stops before the next call, and says why.

    The call that crosses the cap is inside a tool here, the case most likely to be
    mistaken for a failed tool: it must end the run, not become an observation.
    """
    video = make_test_video(tmp / "clip.mp4")
    run_dir = tmp / "run"
    scripted = [
        fake_response(content="Look closely.", usage=fake_usage()),
        tool_call("call_1", "focus", {"query": "q", "start_time": 1.0, "end_time": 2.0}),
        fake_response(content="never reached", usage=fake_usage()),
    ]

    def scripted_completion(*args, **kwargs):
        return scripted.pop(0)

    with patch_attr(utils, "completion", scripted_completion), \
            patch_attr(litellm, "completion_cost", lambda **kwargs: 0.4):
        try:
            cli_main(
                [
                    "--video_path", str(video),
                    "--user_query", "What colour is the pattern?\n(A) red\n(B) blue\n",
                    "--max_run_cost_usd", "0.7",
                    "--output_dir", str(tmp),
                    "--run_dir", str(run_dir),
                ]
            )
            raise AssertionError("the run was expected to stop at its cap")
        except Exception as error:  # noqa: BLE001 - the stop itself is the subject
            assert type(error).__name__ == "RunBudgetExceeded", error

    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    assert trajectory["finish_reason"] == "error" and "RunBudgetExceeded" in trajectory["error"]
    assert trajectory["steps"] == [], "the stop was recorded as a tool step"
    usage = json.loads((run_dir / "usage.json").read_text())
    assert usage["calls"] == 2 and abs(usage["cost_usd"] - 0.8) < 1e-9, usage
    assert any(e["kind"] == "budget_exceeded" for e in read_events(run_dir))
    assert json.loads((run_dir / "images.json").read_text())["tool_errors"] == []
    return "stopped after 2 calls at $0.80 against a $0.70 cap, before the focus call"


def check_cycle_guard(tmp):
    """The cycle never starts a run whose worst case could pass the cap."""
    assert run_cycle.may_start(11.0, 1.0, 12.0)
    assert not run_cycle.may_start(11.01, 1.0, 12.0)
    assert run_cycle.read_ledger(Path(tmp) / "missing.jsonl") == []
    entries = [
        {"run_dir": "a", "cost_usd": 0.2, "cost_unavailable_calls": 0},
        {"run_dir": "b", "cost_usd": 0.3, "cost_unavailable_calls": 0},
    ]
    spent, blind = run_cycle.spent_so_far(entries)
    assert abs(spent - 0.5) < 1e-9 and blind == []
    entries.append({"run_dir": "c", "cost_usd": 0.1, "cost_unavailable_calls": 2})
    entries.append({"run_dir": "d", "cost_usd": None, "cost_unavailable_calls": None})
    _, blind = run_cycle.spent_so_far(entries)
    assert [entry["run_dir"] for entry in blind] == ["c", "d"], blind
    return "$11.00 + $1 worst case may start under $12, $11.01 may not; unpriced runs stop it"


def check_uniform_request(tmp):
    """The uniform arm sends even frames, each after its own time label, then the question."""
    video = make_test_video(tmp / "clip.mp4")
    row = {
        "id": 1, "task": "temporal_ordering", "dataset": "synthetic", "source_question_id": 1,
        "video_id": "clip", "question": HOCKEY_STEM,
        "options": {"A": "Event A -> Event B -> Event C", "B": "Event C -> Event A -> Event B"},
        "answer": "B",
        "evidence": [{"id": 1, "start_sec": 1.0, "end_sec": 2.0, "description": "Event A: synthetic"}],
    }
    spec, frames = uniform_direct.build_spec(row, video, 16)
    assert uniform_direct.verify_spec(spec, frames, row) == [], uniform_direct.verify_spec(spec, frames, row)
    kinds = [block["type"] for block in spec["blocks"][:3]]
    assert kinds == ["text", "text", "image"], "a label must come before its frame"

    tampered = json.loads(json.dumps(spec))
    labels = [b for b in tampered["blocks"] if b["type"] == "text"][1:-1]
    labels[0]["text"], labels[1]["text"] = labels[1]["text"], labels[0]["text"]
    assert uniform_direct.verify_spec(tampered, frames, row), "swapped labels were accepted"

    run_dir = tmp / "run"
    run_dir.mkdir()
    with patch_attr(utils, "completion", lambda *a, **k: fake_response(content="B", usage=fake_usage())):
        payload = uniform_direct.send(spec, frames, row, video, run_dir, dict(TOOLS_CONFIG))
    assert payload["images_verified_against_sample"] and payload["answer_letter"] == "B", payload
    manifest = json.loads((run_dir / "images.json").read_text())
    sent_indices = [image["provenance"]["frame_indices"][0] for image in manifest["images"]]
    assert sent_indices == spec["frame_indices"], "provenance does not name the sampled frames"
    scored = score_run(run_dir, video_dir=tmp)
    assert scored["decoded_frames"] == 16 and scored["answer"]["correct"], scored["answer"]
    return f"16 even frames across {spec['duration_sec']}s, labelled before each frame, scored"


def check_claimed_times(tmp=None):
    """The time a run states for each event is read, and compared with its interval."""
    evidence = [
        {"start_sec": 43.0, "end_sec": 48.0, "description": "Event C: From 43 to 48 seconds..."},
        {"start_sec": 115.0, "end_sec": 122.0, "description": "Event A: Between 115 and 122..."},
        {"start_sec": 202.0, "end_sec": 206.0, "description": "Event B: From 202 to 206..."},
    ]
    hockey = (
        "- **Event C**: At around 15.0s – 18.0s, a Blackhawks player controls the puck.\n"
        "- **Event B**: At around 58.0s – 64.0s, Blackhawks player number 4 battles.\n"
        "- **Event A**: At around 92.0s – 104.0s, several players crowd the crease.\n\n"
        "Therefore, the sequence is Event C -> Event B -> Event A.\n\nC"
    )
    scored = score_claimed_times([("answer", hockey)], evidence)
    by_event = {e["event"]: e for e in scored["per_event"]}
    assert scored["claimed"] == 3 and scored["overlapping"] == 0, scored
    assert by_event["C"]["claimed_sec"] == [15.0, 18.0] and by_event["C"]["gap_sec"] == 25.0

    timer = "1. **Event C**: Between approximately 40s and 45s, the wire.\n2. **Event A**: Around 120 to 130 seconds."
    scored = score_claimed_times([("answer", "A"), ("final_thought", timer)], evidence)
    by_event = {e["event"]: e for e in scored["per_event"]}
    assert by_event["C"]["claimed_sec"] == [40.0, 45.0] and by_event["C"]["overlaps"]
    assert by_event["A"]["claimed_sec"] == [120.0, 130.0] and by_event["A"]["source"] == "final_thought"
    assert by_event["B"]["claimed_sec"] is None

    inline = "Event C (at 44s) comes before Event B (at 58s)."
    scored = score_claimed_times([("answer", inline)], evidence)
    by_event = {e["event"]: e for e in scored["per_event"]}
    assert by_event["C"]["claimed_sec"] == [44.0, 44.0] and by_event["B"]["claimed_sec"] == [58.0, 58.0]

    for silent in ("Event C -> Event B -> Event A", "Does Event A or Event B occur after Event C?", "D"):
        assert score_claimed_times([("answer", silent)], evidence)["claimed"] == 0, silent
    return "ranges, single times and fallbacks read; orderings and bare letters claim nothing"


def check_reader_returns_frame_asked_for(tmp):
    """The tools' reader returns the frame asked for, on a clip decord's seek gets wrong.

    The clip has open GOPs: the frames shown just before a keyframe are decoded
    after it and depend on the group before. YouTube encodes do this, and on this
    exact clip decord returned the wrong frame for 19 of 128 requests. Every frame
    the reader returns is compared with ffmpeg decoding the clip from the start.
    """
    clip = tmp / "open_gop.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30000/1001:duration=30", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-x264-params", "open-gop=1:keyint=90:min-keyint=20:bframes=3:b-pyramid=normal",
         str(clip)],
        check=True,
    )
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    truth = np.frombuffer(raw, np.uint8).reshape(-1, 240, 320, 3).astype(int)
    keyframes = [
        index for index, flag in enumerate(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=key_frame",
             "-of", "csv=p=0", str(clip)],
            capture_output=True, text=True, check=True,
        ).stdout.split())
        if flag == "1"
    ]
    reader = VideoReader(str(clip))
    assert len(reader) == len(truth), (len(reader), len(truth))
    rng = np.random.default_rng(0)
    requests = (
        [np.linspace(0, len(reader) - 1, 64).astype(int)]
        + [[key - back] for key in keyframes[1:] for back in (1, 2, 3)]
        + [[int(rng.uniform(0, len(reader) - 1))] for _ in range(40)]
    )
    total = 0
    for indices in requests:
        for index, frame in zip(indices, reader.get_batch(indices).asnumpy()):
            diff = np.abs(frame.astype(int) - truth[index]).mean()
            assert diff < 1.0, f"asked for frame {index}, got a frame {diff:.1f} away from it"
            total += 1
    return f"{total} requests, {3 * (len(keyframes) - 1)} of them just before an open-GOP keyframe, all exact"


CHECKS = (
    ("block structure", check_block_structure),
    ("usage accounting", check_usage_unknown_is_not_zero),
    ("failure recording", check_failure_recording),
    ("call failure recorded", check_call_failure_is_recorded),
    ("high demand is retried", check_high_demand_is_retried),
    ("crash keeps transcript", check_crash_keeps_transcript),
    ("provenance identifies frames", check_provenance_identifies_frames),
    ("end-to-end scripted run", check_end_to_end_run),
    ("run dir claimed safely", check_run_dir_is_claimed),
    ("letter parsing", check_letter_parsing),
    ("window offsets", check_window_offsets),
    ("string parameters", check_string_parameters),
    ("tool error is visible", check_tool_error_is_visible),
    ("replay B request", check_replay_b_request),
    ("replay B records what it sent", check_b_send_records_what_it_sent),
    ("replay C request", check_replay_c_request),
    ("replay C CLI", check_replay_c_cli),
    ("workers get question and effort", check_workers_get_question_and_effort),
    ("effort reaches Gemini", check_effort_reaches_gemini),
    ("code snapshot", check_code_snapshot),
    ("run budget stops a run", check_run_budget_stops_run),
    ("cycle guard", check_cycle_guard),
    ("uniform request", check_uniform_request),
    ("claimed times", check_claimed_times),
    ("reader returns the frame asked for", check_reader_returns_frame_asked_for),
)


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory(prefix="videoseek-checks-") as tmp:
        root = Path(tmp)
        for index, (name, check) in enumerate(CHECKS):
            target = root / f"{index:02d}-{name.replace(' ', '-')}"
            target.mkdir()
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    note = check(target)
            except Exception as error:  # noqa: BLE001 - a check may fail any way it likes
                message = f"{type(error).__name__}: {error}"
                frames = traceback.extract_tb(error.__traceback__)
                if frames:
                    last = frames[-1]
                    message += f"  (at {Path(last.filename).name}:{last.lineno} in {last.name})"
                failures.append((name, message))
                print(f"FAIL  {name}: {message}")
            else:
                print(f"PASS  {name}" + (f" — {note}" if note else ""))

    print()
    if failures:
        print(f"{len(failures)} of {len(CHECKS)} checks failed")
        return 1
    print(f"all {len(CHECKS)} checks passed, with no network calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
