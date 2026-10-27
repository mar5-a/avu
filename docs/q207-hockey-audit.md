# q207 — what the hockey failure actually shows

This is the human-readable companion to [the visual audit figure](figures/q207_visual_audit.png)
and [the search timeline](figures/q207_search_timeline.png). It incorporates the independent
review in `data/reviews/independent-pilot-2026-09-21/review.html`.

## The short version

The hockey miss is real, but the earlier interpretation was too confident. The agent searched
the beginning of a 538-second broadcast, accepted the first hockey scenes that sounded roughly
right, and then stopped its detailed search. Its reports did not reliably verify the details that
make the benchmark's three scenes different: which team had the puck, which player wore number
4, and whether the play was at the boards or the crease.

The safe finding is therefore:

> q207 is a search-and-verification failure. The agent found relevant-looking hockey, but it did
> not establish that its proposed scenes were the benchmark's requested scenes.

That keeps the failure without claiming that the agent's alternative order was correct or that the
benchmark is wrong.

![Six panels: the three scenes the agent proposed on top, and the three benchmark reference scenes underneath.](figures/q207_visual_audit.png)

**Text description of the figure:** The top row contains the agent's 17.5s center-ice candidate,
61.7s net/crease candidate, and 96.5s behind-the-net candidate. Their labels say “not verified,”
“not the boards event,” and “possible look-alike.” The bottom row contains the benchmark's 43–48s
center-ice reference, 202–206s #4 boards battle, and 115–122s crease reference. The figure's
message is that a scene can look relevant without satisfying the details in the question.

## What the frames show

| Scene the question asks for | Benchmark window | What A reported | Independent frame check |
|---|---:|---:|---|
| Event C — a Blackhawks player carries the puck over the Honda center-circle logo while a Ducks player follows | 43–48s | 15–18s | The early sequence appears to show a Ducks puck carrier. The Honda logo and the presence of both teams are not enough to verify Event C. |
| Event A — a crowded crease with the Ducks goalie dropping down | 115–122s | 92–104s | The earlier play is a plausible look-alike, but the reviewed 96s frame is behind the net rather than the asserted crease scramble. The event type may recur; this frame does not establish that it is the annotated instance. |
| Event B — Blackhawks #4 battles a Ducks player along the boards | 202–206s | 58–64s | The annotated close-up makes #4 visible. The earlier wide shots do not independently verify that matchup; the 60–61s sequence is at Chicago's net, not the annotated boards battle. |

The exact frames are in the top and bottom rows of `q207_visual_audit.png`. The top row is what
the agent relied on; the bottom row is the benchmark's reference evidence. The labels are
deliberately conservative: “not verified” and “possible look-alike” are not claims that the
benchmark annotation is perfect.

## What the timing result does and does not mean

The numerical coverage result is sound:

- A decoded 143 distinct frames and presented 144 image tiles.
- One distinct frame landed in each annotated interval: 43.277s, 119.486s, and 204.838s.
- That is 3 of 143 distinct decoded frames, about 2.1%.

But an in-window frame is not automatically an identified event. The 43.277s frame is before the
clearly identifiable center-circle action; the 119.486s frame is a sparse overview sample; and the
204.838s frame shows the skater after the close boards confrontation without clearly establishing
the requested #4 matchup.

The agent also did not literally stop receiving video at 104s. Its `overview` sampled the whole
538-second video, and its one `skim` covered 10–110s. What stopped at 104s was the final detailed
`focus` call. That distinction is shown in the timeline figure.

![Timeline separating the whole-video overview, the 10–110s skim, the four focus calls, and the annotated windows.](figures/q207_search_timeline.png)

**Text description of the timeline:** A pale blue bar spans the full 538 seconds for the
overview. A darker skim bar spans 10–110 seconds. Four short focus bars sit between 12 and 104
seconds. Red markers identify the benchmark windows at 43–48, 115–122, and 202–206 seconds. One
orange line marks the last focus at 104 seconds; a gray line marks the end of the skim at 110
seconds.

The annotated intervals occupy 16 of 537.77 seconds, about 3.0% of the video. Three in-window
frames versus roughly four under uniform sampling is weak evidence on its own; it is not a strong
random-sampling comparison.

## What A, B, and C support

- **A** answered `C`; the key is `D`. Its final ordering follows its own reports, but the visual
  identities underneath those reports are unreliable.
- **B** received A's exact image sequence and labels, with the reports, system prompt, and tools
  removed. It answered `B`, also wrong. That shows removing the reports did not rescue this image
  set. It does not prove the reports were accurate, irrelevant, or the sole cause of A's failure.
  B's early boards candidate is especially instructive: its diagnostic text calls out **#44**, not
  the requested **#4**.
- **C** received 48 images: 16 samples from each annotated window, with their source timestamps.
  It answered `D`. This shows that the model can produce the expected answer when given much more
  helpful images. It does not tell us whether the improvement came from frame selection, sampling
  density, reduced distraction, or the answering context; C saved no explanation.

Do not run the proposed “shuffled, timestamp-free C2” as an ordering test. Removing both the
presentation order and source timestamps removes the information needed to reconstruct chronology.
For recognition, ask which event each clip depicts. For ordering, shuffled clips must retain their
source timestamps.

## Speaker-friendly wording

Use this:

> “The agent's miss is a useful failure, but not because we proved it found alternative valid
> instances. It found plausible-looking hockey, then stopped. When we checked the details, the
> center-ice puck carrier and the #4 boards matchup were not established, and one claimed crease
> scene was actually behind the net. The lesson is that relevance, in-window coverage, and event
> identification are three different things.”

Avoid this:

- “The event types recur, so the agent's order was correct for the events it found.”
- “The benchmark annotations are underdetermined.”
- “The agent searched only 12–104s.”
- “B proves the reports did not matter.”
- “C proves the model recognised the events visually.”

Those claims go beyond the saved images and the independent check.
