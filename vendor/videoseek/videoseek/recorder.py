"""Record the exact images a run sends to the model, so it can be replayed.

Every model call goes through `utils.call_llm_api`, which makes that the single
place worth instrumenting: whatever content a tool builds, the bytes that
actually left the process are captured here in order, each with the text block
that immediately preceded it (its visible time label).

The manifest keeps the full ordered block structure of every image-bearing call,
not just the images, so a later request can be rebuilt without guessing at what
surrounded them.
"""

import base64
import hashlib
import json
from pathlib import Path


class ImageRecorder:
    def __init__(self):
        self.run_dir = None
        self.images = []
        self.calls = []
        self.usage = {}
        self._step = None
        self._tool = None

    @property
    def active(self):
        return self.run_dir is not None

    def start(self, run_dir):
        """Begin recording into `run_dir`. Safe to call once per run."""
        self.run_dir = Path(run_dir)
        (self.run_dir / "images").mkdir(parents=True, exist_ok=True)
        self.images = []
        self.calls = []
        self.usage = {
            "calls": 0,
            "calls_without_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "cost_unavailable_calls": 0,
        }
        self._step = None
        self._tool = None

    def set_context(self, step, tool):
        """Attribute subsequent captures to one agent step and tool."""
        self._step = step
        self._tool = tool

    def capture(self, messages):
        """Save every image in `messages`, preserving order and labels."""
        if not self.active:
            return
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            blocks = None
            label = None
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    label = block.get("text")
                    if blocks is not None:
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
                        }
                    )
                    if blocks is None:
                        blocks = []
                    blocks.append({"type": "image", "index": index})
            if blocks is not None:
                self.calls.append(
                    {"step": self._step, "tool": self._tool, "blocks": blocks}
                )

    def record_usage(self, response):
        """Accumulate token counts and cost from one model response."""
        if not self.active:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            self.usage["calls_without_usage"] += 1
            return
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self.usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        try:
            import litellm

            self.usage["cost_usd"] += litellm.completion_cost(completion_response=response)
        except Exception:
            # Unknown model or missing pricing: report tokens, don't invent a cost.
            self.usage["cost_unavailable_calls"] += 1

    def write_manifest(self):
        if not self.active:
            return
        payload = {"image_count": len(self.images), "images": self.images, "calls": self.calls}
        (self.run_dir / "images.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.run_dir / "usage.json").write_text(
            json.dumps(self.usage, ensure_ascii=False, indent=2), encoding="utf-8"
        )


recorder = ImageRecorder()
