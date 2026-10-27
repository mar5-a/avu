# Independent review of the five-question pilot

This page is the deck-facing summary of the independent review saved under
`data/reviews/independent-pilot-2026-09-21/`. The review inspected the saved trajectories,
the exact A/B/C model-input images, the benchmark windows, and independently decoded source
frames. It verified 1,296 recorded image-file hashes, including that B received A's exact image
sequence and labels. It made no paid calls and changed no pipeline code.

## The conclusion to carry forward

The pilot still contains a valuable hockey failure. The safe claim is not “the agent found valid
alternative instances,” and it is not “the benchmark is wrong.” The evidence supports this:

> Finding a relevant-looking scene, receiving a frame inside an annotated window, and correctly
> identifying the requested event are three different things.

The agent often stopped after the first plausible visual match. The next experiment should require
it to verify every identifying detail in the event description before it accepts a match.

## q207 — hockey ordering

| Scene | Benchmark window | A's proposed window | Independent assessment |
|---|---:|---:|---|
| Blackhawks puck carrier over the Honda center-circle logo | 43–48s | 15–18s | Early sequence appears to show a Ducks puck carrier. Logo + both teams is not enough. |
| Crowded crease with the Ducks goalie dropping down | 115–122s | 92–104s | Earlier play is a plausible look-alike, but the reviewed 96s frame is behind the net rather than the asserted crease scramble. |
| Blackhawks #4 battling a Ducks player at the boards | 202–206s | 58–64s | The answer-key close-up makes #4 visible; A's earlier wide shots do not verify the matchup. The 60–61s sequence is at Chicago's net. |

The numerical facts remain useful: A decoded 143 distinct frames, with one frame in each annotated
interval at 43.277s, 119.486s, and 204.838s. That is 3/143, approximately 2.1%. The overview
sampled the full video and the skim covered 10–110s; only the detailed focus calls stopped at 104s.

A answered `C`; B answered `B`; the key is `D`. B's failure means report removal did not rescue
this exact image set. It does not prove the reports were accurate, irrelevant, or the sole cause.
C answered `D` from 48 annotated-window images. That is a helpful-frame diagnostic, not proof of
visual recognition and not a reason to run a timestamp-free shuffled ordering test.

See [q207-hockey-audit.md](q207-hockey-audit.md) and
[q207_visual_audit.png](figures/q207_visual_audit.png) for the detailed evidence and captions.

## The other four questions

- **Floatplane / wildlife trip (q15):** the ordering is visually supported. A focused on the
  first two events; overview frames captured the late group-photo interval. Some annotation bounds
  include unrelated shots and transitions.
- **Timer-switch tutorial (q150):** the clearest success. A inspected all three scenes and their
  order is supported. C's text answer is correct (`B`), but the parser stored it as
  `null/ambiguous`; fix that before reporting C as a clean success.
- **Dogs and mirrors (q528):** the correct letter hides disagreement about which clips A and B
  counted. A confirmed a blue-vest puppy around 23–28s and another mirror interaction around
  177–189s, then omitted both from its final five-scene list; it never sampled the 50–52s interval.
  The benchmark also appears to combine separate clips in some windows. C receives five separate
  annotated image groups, so it can reach the expected count by counting groups rather than
  visually counting events. Define “independent scene” before expanding.
- **Space-station tour (q550):** the four broad themes are visually supported. Overview already
  supplied frames from the first three; only the food section needed another inspection.

The official scores should be preserved. They are useful labels for the pilot, but “correct” is not
the same as “every requested event was consistently identified.”

## Changes this review implies

1. Keep q207 as the official failure and use the corrected figures.
2. Fix the replay-C parser and the CLI issue recorded in the review.
3. Include event definitions in worker follow-up prompts; a question such as “Does Event A or
   Event B occur next?” is not answerable from bare letters alone.
4. Resolve the independent-scene rule for q528.
5. Run a no-cost ambiguity audit of the remaining questions that checks player/team/object details,
   not just recurring visual types.
6. Scale only after those checks. Do not spend on a timestamp-free shuffled C2 for an ordering
   question: removing timestamps removes chronology rather than testing recognition.

