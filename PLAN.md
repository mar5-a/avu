# First phase: one video question, two answers

**Goal:** run one VES-Bench question through VideoSeek with Gemini Flash, then
ask Gemini again using the exact images from that run. Produce a small report
that lets us see what happened. Read [base-research.md](base-research.md) for why.

**Status:** the five-question pilot is done and reviewed. The effort cycle (20 new
ordering questions, three arms, $12 cap) is set up and selected, and no paid call
has been made for it. Setup found that the frame reader returned frames from the
wrong moment on most videos, the pilot's included; that is fixed, and its effect on
the pilot is measured. See "Effort cycle" below.

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


**Second pass (2026-09-19/21).** A review found the record weaker than it looked. Fixed
and covered by `scripts/offline_checks.py` — 10 checks, no network, no cost:

- **Frame provenance**: each image records the source frames behind it (indices, unrounded
  timestamps, grid cell layout), so coverage never depends on re-matching bytes.
- **Failures are auditable**: attempts, retries and errors reach `events.jsonl` as they
  happen; a token field the provider omits counts as missing, not zero; a crashed run
  keeps its transcript, marked `finish_reason: "error"`.
- **No run-dir races**: the directory is claimed with `exist_ok=False` and passed down as
  `--run_dir`, not inferred from the output folder afterwards. `pyarrow` is declared.

**Metrics are the benchmark's (§4.3), not invented.** An earlier "Cov@1" here described a
definition that agreed with the paper only by luck. `scripts/ves_score.py` implements
Coverage@k (distinct decoded frames per interval, every interval), Evidence Precision
(macro-averaged), Allocation Ratio (EP / 13.7%) and ECA@k. Conventions it fixes and states:
closed interval bounds, six-decimal timestamps.

| run (dir suffix) | decoded / supplied | frames per interval | Cov@1/2/3 | EP | AR |
|---|---|---|---|---|---|
| `-2` — all `overview` | 64 / 576 | 4, 2, 2 | pass / pass / **fail** | 0.125 | 0.91 |
| `-3` — valid baseline | 123 / 128 | 9, 11, 8 | pass / pass / pass | 0.228 | 1.66 |

Provenance for both came from `scripts/backfill_provenance.py`, accepted only where the
re-encoded bytes matched (72/72). The unsuffixed directory is the first, crashed attempt.

**Experiment B: sent twice; both answered `A`** (matching A and the ground truth). The
second sample recorded cleanly — 72 images digest-verified against A's bytes, 80k in /
426 out, $0.0617. The first sample's capture was empty from a recorder-wiring bug, since
fixed and now covered by an offline check over the sending path. On a question A already
gets right, B agreeing is a control rather than evidence.

## Five questions (2026-09-21)

Chosen by `scripts/select_questions.py` before any answer was looked at. Strata are
quantiles of the evidence span — first to last annotated interval — within each task, plus
one positional pick through the question file; questions already run, and videos already
used, are skipped. Ego4D is excluded because its `video_path` is a bare UUID with no source
URL, so this pilot is YouTube-only and should be described that way.

| q | task | stratum | span | intervals |
|---|---|---|---|---|
| 207 | temporal_ordering | all | 163s | 3 |
| 15 | temporal_ordering | spread | 358s | 3 |
| 150 | temporal_ordering | tight | 122s | 3 |
| 528 | event_counting | all | 290s | 5 |
| 550 | event_counting | spread | 440s | 4 |

Two mechanical substitutions: q83's video is gone from YouTube, so the next qualifying
question in its stratum took the place (q150). Videos are fetched video-only — audio is
never decoded — at the best resolution up to 720p; one arrived at 360x640, and every run
records its video's frames, fps, duration and pixel size in `question.json`.

`score.json` now also records, per annotated interval, the windowed call that covered it
and that window's signed start and end shift, so the "aimed early" observation is measured
rather than felt. It changes nothing about how the agent runs.

## Pilot: the five (2026-09-21)

The first pass was wasted and the reason is worth keeping: the 720p videos had come down as
VP9-in-mp4, which this decord build cannot decode. Every frame fetch raised, and because a
raised tool becomes "Tool execution failed." and the run continues, three of the five
answered from the question text alone while looking like ordinary runs. `q207`, `q15` and
`q528` are marked `INVALID.md`, kept on disk, and were re-run; `q150` and `q550` ran on
decodable 360p files and stand.

Fixes, each with an offline check behind it: `scripts/check_videos.py` decodes every video
the two ways the tools sample it, before any money is spent; the tools coerce the numeric
parameters this model sometimes sends as strings; a raised tool now records its reason in
the observation, in `events.jsonl` and in `images.json`, and the runner prints a warning.
Videos are re-fetched preferring H.264.

**Next:** done — see below.

## Pilot result: A, B and C on the five (2026-09-21)

A got 4/5. The single miss (q207) remains a useful locating-and-verification failure,
but the visual audit does not establish that the agent found valid alternative instances
or that the benchmark is wrong. The 538s video received a whole-video overview, a
10–110s skim, and four detailed focus calls in 12–104s; 3 of its 143 distinct decoded
frames fell inside the annotated intervals (one per interval). The saved frames show why
coverage is not enough: the early center-ice candidate appears to have a Ducks puck
carrier, the 60–61s candidate is at Chicago's net rather than the annotated boards
battle, and the 96s candidate is behind the net rather than the asserted crease
scramble. The agent accepted plausible-looking scenes without reliably checking the
team, player number, puck carrier, or exact play.

B also fails q207, so removing A's reports did not rescue the exact A image set. That is
a report-removal diagnostic, not proof that the reports were accurate, irrelevant, or
the sole cause. C returns the key's D with 48 annotated-window images. C shows that the
answer path works with helpful frames, but it does not establish exactly why; the model
still had to map the windows to Event A, B, and C. Do not run a timestamp-free shuffled
C2 for ordering, because removing timestamps removes chronology. See
`docs/q207-hockey-audit.md` for the evidence and speaker-friendly wording.

| q | gold | A | B | C text |
|---|---|---|---|---|
| 207 | D | C ✗ | B ✗ | **D ✓** |
| 15 | C | C ✓ | C ✓ | C ✓ |
| 150 | B | B ✓ | B ✓ | B* |
| 528 | A | A ✓ | A ✓ | A ✓ |
| 550 | B | B ✓ | B ✓ | B ✓ |

*C's textual answer for q150 is B, but the parser stored it as `null/ambiguous`; fix the
parser before treating C as a clean 5/5. The parser has since been fixed (a bracketed-letter
rule), and q150-C's run directory carries `prediction_reparsed.json` with the corrected read,
its original record left intact. In the four A/B agreements, B's exact-frame
replay chose the same stored letter with no reports, system prompt, or tools. That is
agreement, not proof that reports contributed nothing. Coverage and correctness come
apart both ways (q528 correct with @1 failing; q550 passing @1–@3 with one of four
intervals targeted). The q257 window-offset hunch **does not replicate**: focus windows
mean start +2.8s, end +0.1s across the five. The semantic audit also found that q528's
correct letter hides disagreement about which clips A and B counted, while C can answer
counting questions by counting its separate annotated image groups.

Spend $1.85 total; A+B+C on the five was $1.59, of which q528 alone $0.49 (20-step
limit; the cap guard checks between questions, so one long question can overshoot it).

**Next:** fix the C parser and the worker prompt so an event-letter follow-up includes the
event definitions; resolve the independent-scene rule for q528; run a no-cost ambiguity
audit of the remaining questions that checks identifying details, not just recurring
visual types; then scale the batch (20 questions) under the same pre-registered selection.
Decide the temperature question with the non-determinism already recorded before reading
larger numbers as stable.

## Effort cycle: setup (2026-10-27)

**The question.** Once the setup is fixed, does more reasoning, or a much simpler
system, remove the failures? If it does, simplify. If the same human-confirmed failure
recurs across several new videos, that is the direction. Three arms run on the same 15
questions, identical code and settings apart from the arm:

| arm | what runs |
|---|---|
| medium | the agent, every call at reasoning effort `medium` |
| high | the agent, every call at `high` |
| uniform | 128 evenly spaced frames and the question, one call at `high`, no agent (`scripts/uniform_direct.py`) |

Ordering questions only: counting waits for a rule on what counts as a separate scene.
Temperature stays 1.0, so a single medium/high disagreement can be noise; disagreements
are re-run once if the budget allows. The pilot's runs used a different configuration and
are not merged into this table.

**Setup changes, each covered by an offline check (26 checks, no network):**

- **Frames are found by their timestamp.** decord, upstream's reader, jumps to the
  keyframe before a requested frame and counts forward from it. On many YouTube files
  the jump lands one keyframe earlier than decord believes, so it returns a frame 1–9s
  *before* the one asked for, under the later frame's time label, with no error. On
  the twenty it did this on 14 of 20 videos, for 35 to 225 of 424 requests each
  (measured on shrunken frames, which also flags a few right frames in still scenes;
  the other six had at most 7). The tools now read through `videoseek/video.py`
  (PyAV 19), which decodes forward until a frame's own timestamp is the one
  requested, so a bad jump costs time, never the wrong frame; where decord was right,
  the pixels are identical. `check_videos.py` makes 424 frame requests per video in
  the agent's patterns and requires every frame returned to be byte-identical to the
  same position in a start-to-finish decode. All 26 videos pass; with decord swapped
  back in it fails the phone-review video on 65 of 324 frames, and passes the timer
  tutorial's, where decord was always right. The offline check uses a synthetic clip with
  open GOPs (frames shown just before a keyframe are decoded after it, as in YouTube
  encodes), on which decord gets 19 of 128 frames wrong.

- **Workers get the event definitions.** `skim` and `focus` now see the question stem,
  without the options, beside the planner's query. In the pilot they saw the query alone:
  the hockey run's "Does Event A or Event B occur right after Event C?" reached its worker
  with the labels undefined, and other queries dropped identifying clauses.
- **`skim` follows the configured effort.** Upstream hard-coded it to `low`, so "high"
  would not have meant every call.
- **Code snapshot.** `settings.json` records git HEAD, whether the tree was dirty, and a
  hash of the diff, which is kept beside the run as `code.diff`.
- **Spending guards.** A run stops before its next call once its estimated cost reaches
  $1.00 (`max_run_cost_usd`; the pilot's priciest ordering run was $0.17).
  `scripts/run_cycle.py` starts a run only if spend so far plus $1.00 stays within $12,
  and stops the cycle if any run's cost could not be priced. These are LiteLLM estimates;
  the AI Studio dashboard is the bill.
- **Claimed times** (our measure, not the paper's) in `ves_score.py`: whether the time a
  run states for each event overlaps that event's interval. It is closer to "found the
  right moment" than frame coverage, but still not proof of recognition. On the pilot's
  ordering runs: hockey (q207) 0 of 3, timer tutorial (q150) 3 of 3, floatplane trip
  (q15) 3 of 3, phone review (q257) 2 of 3, though q257's close-ups near its miss
  were shown 2.3s early (below).
- **Gemini's 503 "high demand" is retried.** Upstream's retry did not recognise it, so
  the tool crashed on the empty result and the run carried on without that call.
- **An overview reply that fails to parse is kept** (recording only, from the repeat
  runs on). In the cycle's first attempts 4 of the 15 medium runs, and none of the 15
  high ones, lost their overview at step 1 to a reply the JSON parser rejected, and only
  the parser's message survived. The reply is now written verbatim beside the run
  (`unparsed-step01-overview.txt`, with an `unparsed_reply` event); the tool still
  raises the same error, so what the agent sees is unchanged.

**Wrong frames in the pilot.** `scripts/audit_sent_frames.py` rebuilds every image a
run sent from the frames its provenance names, and compares tile by tile with the JPEG
the model received. Provenance records the frame that was asked for, so where the two
differ, the model saw an earlier moment under a later label. Frame replays (B) re-sent
their source run's images and inherit its errors.

| run | frames not the one labelled | how early | what it changes |
|---|---|---|---|
| phone review (q257), run 3 | 38 of 128: 15 overview, 23 focus | 1.4–4.4s | Its focus close-ups labelled 12.6–15.0s showed 10.3–12.6s, so no close-up reached Event B (gold 14–18s); only the overview's 15.3–17.9s tiles, which were right, did. Its Event B miss (claimed 10–13s) was partly made on mislabelled frames and is weak evidence about the agent. Run 2: 135 of 576. |
| floatplane trip (q15), agent | 18 of 96: 10 overview, 8 focus | 1.9–4.7s | Still correct, with 3 of 3 claimed times. |
| floatplane trip (q15), C | 8 of 48 | 2.6s | 8 of C's "annotated window" frames were from 2.6s before the window. |
| hockey (q207), agent | 7 of 144: overview and skim only | 0.2–2.1s | The audit's candidates came from focus frames, all correct; its conclusion stands. |
| q550 (counting), agent | 1 of 80 | 1.5s | Negligible. |
| timer tutorial (q150), q528 (counting) | 0 | — | Unaffected. |

Each question's first run predates per-image provenance and cannot be audited. Two
pilot numbers lean on frame labels and should be re-measured before being cited:
frame coverage and Evidence Precision on the affected runs, which count the frame
asked for; and "focus windows start +2.8s late on average", because this error shows
content under labels later than its true time, which is the direction that would make
an agent aim late.

**The twenty**, from `select_questions.py --set twenty` before any run: five per quartile
of evidence span, at evenly spaced positions, with the pilot's questions and videos
excluded. The holdout is the third of every four in span order; it is not run in this
cycle, and is for checking a fix later, not for exploring.

| q | topic | span | role |
|---|---|---|---|
| 6 | news report: police on a bridge, a speech, a figure in a blanket at night | 60s | explore |
| 215 | bike cleaning: brush, pressure washer, cloth | 90s | explore |
| 98 | dinosaur documentary | 124s | holdout |
| 60 | product safety text: chemical labels, lifeline, tilt and turn | 150s | explore |
| 47 | car promo: grille lights, engine bay, driving away | 162s | explore |
| 273 | microscopy explainer animation | 164s | explore |
| 256 | wiring a wall switch: multimeter, terminal screws | 166s | holdout |
| 181 | car repair under a green car | 167s | explore |
| 100 | fashion designer profile, leather goods | 170s | explore |
| 188 | Alex Trebek tribute | 182s | explore |
| 71 | bike component showcase | 205s | holdout |
| 185 | election news: ballot counting, a wire fence, an official | 223s | explore |
| 114 | acrobatic stage performance | 234s | explore |
| 186 | Puss in Boots film scenes | 253s | explore |
| 279 | travel show: phone number, sled dogs, sea cliff | 267s | holdout |
| 9 | college basketball, players told apart by jersey number | 297s | explore |
| 127 | US passport renewal guide | 326s | explore |
| 82 | animated story: ghost, planets, a beach | 356s | explore |
| 69 | sports drama: locker room, desk, gym recovery | 416s | holdout |
| 153 | cars on city streets: orange sedan, Rolls-Royce | 487s | explore |

**Next:** commit the setup and the selection, so the twenty are on record before any
result. Then a smoke test of the first and last explore questions in span order, q6
(news, 60s span) and q153 (cars, 487s span), so it brackets cost, under all three arms:
about $1–1.5. Check the records, including `audit_sent_frames.py` on each run, then run
the other 13 on approval.

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
