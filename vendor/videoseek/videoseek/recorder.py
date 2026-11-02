"""Record exactly what a run sent to the model, so it can be audited and replayed.

Every model call goes through `utils.call_llm_api`, which makes that the single
place worth instrumenting: whatever content a tool builds, the bytes that
actually left the process are captured here in order, each with the text block
that immediately preceded it (its visible time label).

Three levels of detail, because they answer different questions:

- `images.json` — the bytes, their order, and their labels. This is what a
  replay reads: it can rebuild a request from `images` and their labels without
  guessing at anything.
- Provenance on each image — which decoded source frames it shows, as exact
  frame indices and unrounded timestamps, plus the grid layout for the 2x4 sheets
  `overview` merges. Scoring needs frame identity rather than visible labels:
  two frames can round to the same 0.1s, and the benchmark's evidence intervals
  are in seconds. Where a tool supplies no provenance the field is null, and a
  count mismatch is recorded as an event rather than papered over.
- `events.jsonl` — appended as the run proceeds, one line per model-call
  attempt, capture, and agent step. A run that dies mid-flight (process killed,
  API error past its retries) still leaves a readable record of what it had done.

`images.json` also keeps the full ordered block structure of every image-bearing
call. That is a faithful record of the request, not a replay template: it holds
text the agent generated for itself, such as the per-call search query, which a
replay must not inherit. Replays build from `images` and their labels.
"""

import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

# Token fields are summarised rather than guessed at: a field the provider did
# not report is counted as missing, never added to a total as a zero.
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class RunBudgetExceeded(Exception):
    """A run reached its cost cap. Raised before a call, so nothing more is spent."""


def code_snapshot():
    """The commit this code came from, and how the working tree differed from it.

    A commit hash alone says nothing about uncommitted edits, which is exactly
    what a run made mid-change would be using. The diff itself is returned so the
    caller can keep it; untracked files are listed by name only. None outside a
    git checkout.
    """
    here = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(here), *args], capture_output=True, text=True, check=True
        ).stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        diff = git("diff", "HEAD")
        untracked = git("ls-files", "--others", "--exclude-standard", ":/").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    snapshot = {
        "head": head,
        "dirty": bool(diff or untracked),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
        "untracked": untracked,
    }
    return snapshot, diff or None


class ImageRecorder:
    def __init__(self):
        self.run_dir = None
        self.images = []
        self.calls = []
        self.usage = {}
        self.settings = {}
        self.frame_space = None
        self.cost_cap_usd = None
        self._code_diff = None
        self._step = None
        self._tool = None
        self._provenance = None
        self._request = None
        self._events = None
        self._seq = 0

    @property
    def active(self):
        return self.run_dir is not None

    def start(self, run_dir):
        """Begin recording into `run_dir`. Safe to call once per run."""
        self.run_dir = Path(run_dir)
        (self.run_dir / "images").mkdir(parents=True, exist_ok=True)
        # Truncate rather than append: a rerun into an existing directory must not
        # inherit the previous run's events.
        (self.run_dir / "events.jsonl").write_text("", encoding="utf-8")
        self._events = None
        self.images = []
        self.calls = []
        self.usage = {
            "calls": 0,
            "calls_without_usage": 0,
            "calls_failed": 0,
            "incomplete_calls": 0,
            "fields_missing": {field: 0 for field in TOKEN_FIELDS},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "cost_unavailable_calls": 0,
            # Token accounting only: true when no call failed and no reported
            # field was missing. Price availability is not part of it.
            "complete": True,
        }
        self.frame_space = None
        self.tool_errors = []
        self.cost_cap_usd = None
        self._code_diff = None
        self._step = None
        self._tool = None
        self._provenance = None
        self._request = None
        self._seq = 0

    def set_cost_cap(self, cap_usd):
        """Stop this run before any call once its estimated cost reaches `cap_usd`."""
        self.cost_cap_usd = cap_usd

    def check_budget(self):
        """Raise before a call that would start once the run is at its cap.

        The check comes before a call, so the call that crosses the cap still
        completes: a run can overshoot by at most one call. Calls LiteLLM could
        not price add nothing here, so the guard is only as good as its price
        table; `cost_unavailable_calls` says when it was blind.
        """
        if not self.active or self.cost_cap_usd is None:
            return
        if self.usage["cost_usd"] >= self.cost_cap_usd:
            self._event(
                "budget_exceeded",
                cost_usd=self.usage["cost_usd"],
                cap_usd=self.cost_cap_usd,
            )
            raise RunBudgetExceeded(
                f"run cost ${self.usage['cost_usd']:.4f} reached its cap "
                f"${self.cost_cap_usd:.2f}"
            )

    def set_context(self, step, tool):
        """Attribute subsequent captures to one agent step and tool."""
        self._step = step
        self._tool = tool

    def set_provenance(self, entries, avg_fps=None, total_frames=None):
        """Frame identity behind the images of the next call, one entry per image.

        The tools know which decoded frames they sampled; this recorder only ever
        sees base64. Passing the indices across is what makes coverage scorable
        without re-decoding the video and matching JPEG bytes afterwards. For a
        merged grid, `frame_indices` is in row-major cell order, so
        `frame_indices[row * cols + col]` is the frame in that cell.
        """
        if not self.active:
            return
        self._provenance = list(entries)
        space = {"avg_fps": avg_fps, "total_frames": total_frames}
        if self.frame_space is None:
            self.frame_space = space
        elif self.frame_space != space:
            # Two different frame spaces in one run would silently corrupt every
            # timestamp; say so instead.
            self._event("frame_space_changed", was=self.frame_space, now=space)

    def record_request(self, kwargs):
        """Describe one outgoing call: shape and settings, never the API key."""
        if not self.active:
            return
        tools = kwargs.get("tools") or []
        request = {
            "messages": len(kwargs.get("messages") or []),
            "tools": [t.get("function", {}).get("name") for t in tools],
            "tool_choice": kwargs.get("tool_choice"),
            "temperature": kwargs.get("temperature"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            # The callers name this `max_tokens`; it reaches the provider as
            # `max_completion_tokens`.
            "max_tokens": kwargs.get("max_completion_tokens", kwargs.get("max_tokens")),
            "json_output": bool(kwargs.get("return_json") or kwargs.get("response_format")),
        }
        self._request = request
        self._event("request", **request)

    def capture(self, messages):
        """Save every image in `messages`, preserving order and labels."""
        if not self.active:
            return
        provenance = self._provenance
        self._provenance = None
        start_index = len(self.images)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            # Every block, from the first one: the text preceding the first image
            # is part of the request too, and dropping it made the recorded
            # structure disagree with what the model received.
            blocks = []
            label = None
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    label = block.get("text")
                    blocks.append({"type": "text", "text": label})
                elif kind == "image_url":
                    raw = base64.b64decode(block["image_url"]["url"].partition(",")[2])
                    index = len(self.images)
                    name = f"{index:04d}_{self._tool or 'unknown'}.jpg"
                    (self.run_dir / "images" / name).write_bytes(raw)
                    self.images.append(
                        {
                            "index": index,
                            "file": f"images/{name}",
                            "step": self._step,
                            "tool": self._tool,
                            "label": label,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw),
                            "provenance": None,
                        }
                    )
                    blocks.append({"type": "image", "index": index})
            if any(block["type"] == "image" for block in blocks):
                self.calls.append(
                    {
                        "step": self._step,
                        "tool": self._tool,
                        "request": self._request,
                        "blocks": blocks,
                    }
                )
        self._request = None
        captured = self.images[start_index:]
        if provenance is None:
            # Not every image-bearing call comes from a tool that knows, or
            # records, its frame indices. Null provenance is an answer, not a
            # failure; a count mismatch is a failure.
            return
        if len(provenance) == len(captured):
            for entry, frames in zip(captured, provenance):
                entry["provenance"] = frames
        else:
            self._event(
                "provenance_mismatch",
                expected=len(provenance),
                captured=len(captured),
                first_image=start_index,
            )

    def record_usage(self, response):
        """Accumulate token counts and cost from one model response."""
        if not self.active:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            self.usage["calls_without_usage"] += 1
            self.usage["complete"] = False
            self._event("usage_missing")
            return
        self.usage["calls"] += 1
        missing = [
            field
            for field in TOKEN_FIELDS
            if getattr(usage, field, None) is None
        ]
        for field in missing:
            self.usage["fields_missing"][field] += 1
        if missing:
            self.usage["incomplete_calls"] += 1
            self.usage["complete"] = False
        for field in TOKEN_FIELDS:
            value = getattr(usage, field, None)
            if value is not None:
                self.usage[field] += value
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        try:
            import litellm

            self.usage["cost_usd"] += litellm.completion_cost(completion_response=response)
        except Exception:
            # Unknown model or missing pricing: report tokens, don't invent a cost.
            # Cost has its own counter rather than flipping `complete`: the token
            # totals are still trustworthy when only the price is unknown.
            self.usage["cost_unavailable_calls"] += 1
        self._event(
            "usage",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            missing=missing,
        )

    def record_attempt(self, status, error=None, error_type=None, elapsed_sec=None, delay=None):
        """Record one model-call attempt: `retry`, `gave_up` or `failed`."""
        if not self.active:
            return
        if status != "retry":
            # The call never returned. Its usage is unknown, not zero.
            self.usage["calls_failed"] += 1
            self.usage["complete"] = False
        self._event(
            "attempt",
            status=status,
            error=None if error is None else str(error)[:2000],
            error_type=error_type,
            elapsed_sec=None if elapsed_sec is None else round(elapsed_sec, 3),
            delay=None if delay is None else round(delay, 3),
        )

    def record_tool_error(self, tool, error, error_type):
        """Record a tool that raised, or returned nothing.

        Without this, a broken tool and a model that chose badly look identical in
        the trajectory: both leave "Tool execution failed." A pilot was run where
        every frame fetch raised on an undecodable video and the answers came from
        the question text alone, which read as an ordinary run.
        """
        if not self.active:
            return
        self.tool_errors.append(
            {"tool": tool, "error": str(error)[:2000], "error_type": error_type}
        )
        self._event("tool_error", tool=tool, error=str(error)[:2000], error_type=error_type)

    def record_lost_step(self, step, thought):
        """Record a step that ended without running a tool.

        Gemini sometimes answers the call that turns the thought into a tool call
        without one. The step then left no step record, only the parser's message on
        stdout, so a run that lost steps looked like one that took fewer.
        """
        if not self.active:
            return
        self._event("lost_step", step=step, tool=None, thought=thought)

    def record_unparsed_reply(self, text, finish_reason):
        """Keep, verbatim, a model reply a tool could not parse.

        In the effort cycle four overview replies failed to parse and only the
        parser's message survived ("Extra data: line 1 column 12449"), which says
        where it stopped but not what the model wrote. The text goes to its own
        file because it runs to tens of thousands of characters.
        """
        if not self.active:
            return
        name = f"unparsed-step{self._step or 0:02d}-{self._tool or 'call'}.txt"
        (self.run_dir / name).write_text(text or "", encoding="utf-8")
        self._event(
            "unparsed_reply",
            file=name,
            chars=None if text is None else len(text),
            finish_reason=finish_reason,
        )

    def record_step(self, step):
        """Record one agent step as it completes, so a crash keeps the transcript."""
        if not self.active:
            return
        self._event("step", step=step.get("step_id"), payload=step)

    def record_finish(self, final_answer, finish_reason):
        """Record how the run ended, so the answer survives without the trajectory."""
        if not self.active:
            return
        self._event("finish", final_answer=final_answer, finish_reason=finish_reason)

    def record_settings(self, config):
        """Save the settings a run used, so two runs can be compared.

        The API key is never written — only whether one was present. The system
        prompt is stored as a digest rather than text, which is enough to tell
        whether it changed between runs without duplicating it per run.
        """
        if not self.active:
            return
        keys = (
            "model_name", "temperature", "reasoning_effort", "max_steps",
            "max_tokens", "max_retries", "seed", "frame_sampling_factor",
            "overview_base", "skim_base", "focus_base", "tools",
            "api_base", "api_version", "max_run_cost_usd",
        )
        prompt = config.get("SYSTEM_PROMPT") or ""
        self.settings = {key: config.get(key) for key in keys}
        self.settings["code"], self._code_diff = code_snapshot()
        self.settings["api_key_present"] = config.get("api_key") is not None
        self.settings["system_prompt_sha256"] = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        self.settings["system_prompt_chars"] = len(prompt)

    def write_manifest(self):
        if not self.active:
            return
        self._write(
            "images.json",
            {
                "image_count": len(self.images),
                "frame_space": self.frame_space,
                "tool_errors": self.tool_errors,
                "images": self.images,
                "calls": self.calls,
            },
        )
        self._write("usage.json", self.usage)
        self._write("settings.json", self.settings)
        if self._code_diff:
            # Kept beside the run so its exact code can be rebuilt from the
            # commit in settings.json; `diff_sha256` there identifies it.
            (self.run_dir / "code.diff").write_text(self._code_diff, encoding="utf-8")

    def _write(self, name, payload):
        (self.run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _event(self, kind, **fields):
        if not self.active:
            return
        if self._events is None:
            self._events = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        self._seq += 1
        record = {"seq": self._seq, "kind": kind, "t": round(time.time(), 3),
                  "step": self._step, "tool": self._tool}
        record.update(fields)
        self._events.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Flush per event: the point of this file is to survive a hard kill.
        self._events.flush()


recorder = ImageRecorder()
