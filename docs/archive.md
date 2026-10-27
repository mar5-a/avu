# Archive — pilot state, corrected

Everything durable about the five-question pilot, after the independent review
(`data/reviews/independent-pilot-2026-09-21/review.html`, summarised in
[independent-review.md](independent-review.md) and [q207-hockey-audit.md](q207-hockey-audit.md)).
This replaces the talk notes; the deck itself is in `deck/`.

## What the study is

Does a video-understanding agent fail by **missing** evidence, or by failing to **use**
evidence it saw? VES-Bench questions, VideoSeek as the agent, Gemini 3.8 Flash as the model.
The agent's image tools send frames to the model and return **text**; the controller is
text-only and writes the final answer from that text, never seeing an image. That split is
why a wrong answer has two candidate causes, and why there are three conditions:

- **A** — the pipeline as it runs: tools pick frames, text reports mediate, the answer sees text.
- **B** — one call with A's exact frames and labels, reports removed. Tests what the reports added.
- **C** — one call with frames sampled from the benchmark's annotated intervals, reports
  removed. Privileged input, used only where A's frames failed to cover the evidence.

## Results (five pre-registered questions)

| Question | Key / A / B / C | Note |
|---|---|---|
| q207 hockey broadcast, order three plays | D / C / B / **D** | The actionable failure. A accepted plausible-looking hockey without checking the details that distinguish the scenes. See below. |
| q15 bush-plane trip, order three events | C / C / C / C | Supported ordering. Annotation bounds are padded; B's letter alone does not rule out prior travel-narrative expectations. |
| q150 light-switch tutorial, order three events | B / B / B / B | **The strongest straightforward success.** A's three focus windows overlap all three annotated windows. C's stored parse said "ambiguous"; `prediction_reparsed.json` records the corrected read (B, bracketed) without touching the original. |
| q528 dogs at mirrors, count scenes | A / A / A / A | **Officially correct, evidentially unreliable.** The annotation groups separate clips into one "scene", a clearly relevant puppy clip at ~177–189s is outside every interval, and A's own reports disagree with its final list. A and B reach five via different lists. |
| q550 space-station tour, count events | B / B / B / B | Plausible success. The other three windows *were* represented in A's overview input, so "only one targeted" undersells it. The question lists the same four themes, so a question-only control would be informative. |

Macro Coverage@1 0.80 / @2 0.60 / @3 0.40; Evidence Precision 0.324. Recorded spend $1.85
($1.59 for A+B+C on the five). The review independently verified all 1,296 recorded JPEG
entries against their stored hashes, and confirmed B preserved A's image sequence and labels
on all five questions.

## Claims withdrawn

The review's corrections, which supersede earlier readings here:

1. **"The agent's alternative ordering was valid for the instances it found"** — not
   established. Recurrence is plausible, but no complete alternative sequence with every
   required attribute (team, player number, possession, action) was verified. The official
   wrong-answer label stands.
2. **"It never looked past 104s"** — inaccurate. Its `overview` sampled the whole 538s and
   its `skim` reached 110s; what stopped at 104s was the last **detailed** call.
3. **"Its frame distribution was worse than chance"** — the wrong baseline. The annotated
   windows cover 2.98% of *this* video, so ~4 of 143 frames would land in them by chance;
   A put 3. The 13.7% figure is a benchmark-wide constant, not this question's probability.
4. **"B exonerates the reports"** — too strong. B failing on A's frames shows removing the
   reports did not rescue that frame set; it does not show the reports were accurate or
   causally irrelevant. B also picked a different wrong order and hedged its own evidence
   ("#4 or #44"; the source frame supports #44, not the requested #4).
5. **"Timestamps alone reveal hockey's answer"** — wrong. C's segment headers give the
   windows' order but do not name the events; choosing option D still requires identifying
   at least two of the three clips.

Two more, found in C itself: **counting questions leak their answer through C's structure**
(16 images per annotated group, so counting headers yields the expected count — C's
correctness on q528/q550 is not evidence of counting ability), and **C's windows are in
chronological order with bounds visible**, so it tests the answering path rather than
visual recognition. Proposed follow-ups: for recognition, show each clip and ask which
description it matches; for ordering, shuffle clips but **keep** their timestamps. Do not
strip both — that removes the information the question asks about.

## The two mechanisms worth pursuing

1. **Premature acceptance of approximate matches.** Coarse search finds visually similar
   candidates, focused workers assert that they match, the text-only controller accepts and
   sorts. A stopped after its first accepted candidates, with steps left unused.
2. **The local worker never receives the question.** A's 18–32s query was "Does Event A or
   Event B occur right after Event C?" — the worker gets only the local query and its frames,
   so those event letters have no meaning in that request, and its "No relevant content
   found" is untrustworthy. Queries also dropped required clauses (the center-ice query
   omitted "as a Ducks player pursues from behind").

## Where the records are

| What | Where |
|---|---|
| Runs, images, per-run scores | `data/runs/q<id>-*` (`score.json` per run) |
| Independent review + its audit images | `data/reviews/independent-pilot-2026-09-21/` |
| Figures | `figures/` — timeline, visual audit, A/B/C input matrix, architecture, coverage vs correctness |
| Talk | `deck/q207-video-agent-evidence-talk.pptx` |
| System reference, findings, experiment notes | `notes/System.md`, `notes/Findings.md`, `notes/Experiments/` (local vault) |
| How to run and score | `scripts/run_one_question.py`, `check_videos.py`, `ves_score.py`, `replay_b.py`, `replay_c.py`; 16 offline checks in `offline_checks.py` |

`figures/q207_event_type_recurs.png` was deleted: it framed approximate look-alikes as
matches and supported a recurrence claim the review did not establish. The individual frames
it used are still in `figures/frames/`.
