# First phase: one video question, two answers

**Goal:** run one VES-Bench question through VideoSeek with Gemini Flash, then
ask Gemini again using the exact images from that run. Produce a small report
that lets us see what happened. Read [base-research.md](base-research.md) for why.

**Status:** VideoSeek vendored at `vendor/videoseek` (pinned commit from below).
Environment installed and verified offline. Gemini config prepared but not yet
exercised against the live API; no research runs yet.

## Progress

**Setup, first pass (2026-09-19).**

- `.venv` at the repo root, Python 3.13.9, created with `uv`. Package installed
  editable from `vendor/videoseek`. `ffmpeg` 9.0.2 installed via Homebrew.
- Three compatibility changes, all inside the vendored tree.
  - **`decord` → `decord2`.** Upstream `decord` publishes no macOS arm64 wheel and,
    more decisively, no source distribution at all, so it cannot install on this
    machine. `decord2` installs the same top-level `decord` module, so
    `from decord import VideoReader` is unchanged.
  - **Dropped `openai-whisper`.** It exists to generate subtitles and pulls in
    PyTorch, while this phase is images-only. Re-add it when subtitles are needed.
- **Gemini configuration.** `config/general.yaml` now sets
  `gemini/gemini-3.8-flash` and reads `GEMINI_API_KEY` instead of `OPENAI_API_KEY`.
  The `gemini-3.8-flash` name previously flagged as unverified is confirmed stable
  in Google's model documentation, and LiteLLM needs the `gemini/` prefix to route
  to Google. Prompts, frame sampling, `max_steps`, and the tool set are unchanged.
- **Verified with no API calls:** CLI help, the `decord` import, decoding a locally
  generated 100-frame test clip, the `focus` frame-sampling arithmetic, JPEG/base64
  image-block construction, system-prompt formatting, and the tool schemas.
- **Gemini compatibility settled offline, without a paid call.** Checked by driving
  LiteLLM's own translation code directly rather than waiting to find out in
  production. Results:
  - **`seed` would have failed the very first call.** Gemini defines no `seed`
    parameter, LiteLLM raises `UnsupportedParamsError` instead of ignoring unknown
    parameters, and `litellm.drop_params` defaults to `False`. Fixed in
    `utils.py:call_llm_api` by forwarding `seed` only for non-`gemini/` models, so
    the setting still works if other providers are added later.
  - Everything else VideoSeek sends is supported: `max_completion_tokens`,
    `reasoning_effort` (mapped to a thinking level for Gemini 3+), `temperature`,
    `tools`, `tool_choice`, `response_format`.
  - The `strict: True` flag on every tool schema is stripped by LiteLLM before the
    request reaches Google, so it is harmless.
  - Confirmed by translation rather than assumption: a VideoSeek image message
    becomes `parts[].inline_data` carrying the JPEG mime type with its interleaved
    time labels preserved in order, and the four tools become
    `function_declarations`.
  - LiteLLM warns that `temperature`/`top_p`/`top_k` still work on Gemini 3+ but are
    slated for removal, with sampling guidance to move into the system prompt.
    Revisit if the model name is bumped.
- **`.env` support added.** `config/__init__.py` calls `load_dotenv` before any
  `os.getenv`, searching upward from the package to the nearest `.env`. A variable
  already exported in the shell still wins. `.env` holds the local key and is
  gitignored; `.env.example` is the committed template. `python-dotenv` is now a
  declared dependency rather than borrowed from LiteLLM's dependency tree.
**Run 1 (question 257, `temporal_ordering`, 80.7s video) — answered A, ground truth A.**

**The answer is correct, but the run is not a valid baseline.** The agent called
`overview` at all nine steps and never once called `skim` or `focus`, the tools
meant for localizing and inspecting evidence. Steps 1-9 produced no thought text
at all; only step 10 contained reasoning. So it re-read the whole video nine times
and answered correctly from that, rather than by seeking evidence.

Root cause, established with one minimal call: **the reasoning call returns
`content=None`.** Gemini 3.8 Flash answers a video-understanding prompt by invoking
`video_intelligence:get_frame` — a Google-side server tool that appears nowhere in
LiteLLM and that VideoSeek never passes. It comes back as a `tool_calls` response
with `finish_reason="tool_calls"` and the model's actual reasoning in
`reasoning_content`. VideoSeek reads only `.content`, so its thought is empty, and
the follow-up action-selection call then has nothing to work from and picks
`overview` by default. Fixing this is the prerequisite for a meaningful run.

Bug #1, fixed: `agent.py` serialized tool arguments with `str(action.parameters)`,
which emits single quotes and is therefore not valid JSON. OpenAI tolerated it;
Gemini's converter rejects it when the history is replayed, which killed the first
live attempt outright. Now `json.dumps`, verified against LiteLLM's converter.

**Run 2 (same question) — the loop is gone.** Six steps instead of ten, real
reasoning at every step, and the agent used `focus` four times.

| | run 1 (broken) | run 2 (fixed) |
|---|---|---|
| tool sequence | overview x9, answer | overview, focus x4, answer |
| thoughts | empty at steps 1-9 | 735, 605, 315, ... chars |
| steps | 10 | 6 |
| images | 72, all overview | 72: 8 overview grids + 64 focus frames |
| answer | A (correct) | A (correct) |
| recorded cost | not captured | $0.1117, 18 calls, 115,319 in / 6,734 out |

The fallback fired only at step 1 (`thought_source=reasoning_content`); steps 2-6
produced ordinary text (`thought_source=content`). So the empty thought was
self-perpetuating — once step 1 carried real reasoning into the conversation, the
model answered in prose from then on.

**Suppressing the phantom tool is not possible.** `tool_choice="none"` does emit
`toolConfig.functionCallingConfig.mode=NONE`, but the model returns a tool call
anyway, and the name differs each time (`video_intelligence:get_frame`,
`sample_frames`, `video_toolbox:get_video_duration`). Those are not one registered
tool it is invoking — it is inventing video-tool names it believes it has. Nothing
executes, because we never grant the tool, so it never actually looked at frames on
its own. The fallback is therefore the whole fix, not a workaround for part of it.

**Usage capture is in.** Each run writes `usage.json` with call count, token totals
(prompt, completion, reasoning) and a cost from LiteLLM's price table. Two caveats:
the figure is LiteLLM's estimate, not Google's invoice — the AI Studio dashboard is
authoritative — and LiteLLM prices `gemini-3.8-flash` at $0.75/1M in and $3.75/1M
out, which may not match Google's published rate. Missing usage is counted as
unknown rather than zero.

Still open:
1. Decide whether `temperature` stays, given LiteLLM's removal warning.
2. Then the 5-question and 20-question sets.

**Next:** step 3 — replay the saved images in a fresh request (experiment B).

## 1. Get the existing agent ready

Bring VideoSeek into this project and record the code version and dependencies.
Use its existing LiteLLM connection to call Gemini Flash. Verify the exact Gemini
model name and that it accepts the images and tool requests VideoSeek sends.
Use that same model for planning, choosing tools, reading images, and answering.

Keep the agent's existing prompts, image selection, and step limits. Make only
necessary compatibility fixes and briefly explain each one. We are getting an
existing agent working, not building a new framework.

## 2. Connect one question and save the run

Read the actual `VES-Bench.parquet` question file, select one development question,
and obtain only its video. Keep the correct answer and evidence annotations
separate from anything sent to Gemini. Use images only, without audio or subtitles.
Let VideoSeek select the images; do not replace it with a whole-video upload.

Save the question, exact images sent to Gemini, where they came from in the video,
image order and visible time labels, tool reports, final answer, and why the agent
stopped. Also save the model/settings, API usage, retries, and errors. This lets us
check what it actually saw, rather than assume it saw everything it requested.

## 3. Ask again using those saved images

Send a fresh request containing the same question/options and the saved images.
Keep their bytes, grids, order, repeats, and existing time labels unchanged.
Leave out the first run's reports, answer, and generated hints. Check that the
submitted images match the saved ones exactly. If they do not fit the API limits,
report that limitation instead of dropping or changing images.

## 4. Check it and show the result

First use small offline tests to check that the saved images can be replayed
unchanged, answer labels cannot leak into requests, and API failures are recorded.
An unclear/invalid answer is different from a valid wrong answer; missing usage
is unknown, not zero. If image delivery is uncertain, mark the comparison unverified.

Before live calls, agree on a spending cap and the one-video download. Keep keys
in environment variables/secrets and keep videos, image traces, and keys out of Git.
Do not start model downloads, GPUs, or batches for this phase.

The first run is ready to review when both answers are saved in the expected option
format, the image match is verified, and a readable report shows the images, tool
reports, answers, errors, and usage. Retain failed attempts too. **The answers do
not have to be correct for this first milestone to succeed.**

## Starting points for the implementer

Previously inspected VideoSeek commit: `443f8710bd1f9023f4a52ac95cb83610ad29c5ac`.
Start with `videoseek/agent.py`, `videoseek/tools/`, and `videoseek/utils.py`.
The existing saved output needs extra image logging for our replay. The dataset's
question-file structure and Gemini compatibility still need checking in practice.
The earlier `gemini-3.8-flash` name was not verified; confirm before using it.

**First coding task:** get the pinned agent set up, prepare its Gemini configuration,
and add the one-question input and image-saving path. Check these without paid
calls first. Then add replay and the report. Record actual progress here briefly;
do not create more planning documents.
