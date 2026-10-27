# System architecture figure — spec for whoever draws it

This is the figure that explains why the study is designed the way it is. A generic
"LLM agent" diagram will be wrong, because the load-bearing fact is that **the component
that writes the answer never receives an image**. Everything else follows from that.

Verified against the vendored code in `vendor/videoseek/` (upstream commit `443f871`,
plus our instrumentation). Do not soften or re-order these facts to make the picture
prettier.

## The one thing the figure must communicate

Images travel **out** of the controller on tool calls and only **text** comes back. So:

- a wrong answer has (at least) two candidate causes — the agent looked in the wrong
  places, or the text it wrote about what it saw lost/confused the evidence;
- the only way to separate them is to hold one fixed and vary the other, which is what
  the three conditions do.

Give the image path and the text path different colours or line styles and keep that
convention for the whole figure. That colour split **is** the figure's argument.

## Elements, top to bottom

**1. The agent, condition A (the real pipeline)**

Flow, left to right:

| Box | Label it | Notes to draw accurately |
|---|---|---|
| Video | `VES-Bench video (.mp4, local)` | decoded locally with `decord`; the video is never uploaded — frames are sent as JPEGs |
| Controller | `Controller — text only, Gemini 3.8 Flash` | one model, two calls per step: a **thought** call (prose, no tools) then an **action** call (`tool_choice="required"`) |
| Tools | `Vision tools: overview / skim / focus` | each decodes frames and sends them as JPEGs to the same model, then returns a **text** report |
| Context | `Text context: system prompt + thoughts + tool reports` | accumulates across steps; this is all the answer call can see |
| Answer | `answer tool` → final letter | appends `Question: … Please directly provide the final answer.` to the text history and makes one more call |

Frame sampling, label exactly (these numbers are in `config/general.yaml`):

- `overview`: **64 frames** sampled across the whole video, merged **8 per image** into
  **eight 2×4 grids**, each cell captioned with its timestamp.
- `skim`: up to **16 frames** from a requested window >16s.
- `focus`: up to **16 frames** from a requested window ≤16s, about 1 frame per second.

Cost/latency point worth annotating: **2 model calls per step, plus 1 for the answer**
(the run that failed used 7 steps, so 15 calls).

**2. The instrumentation band (across the whole of condition A)**

`Recorder — hooks the single choke point every model call passes through` and writes:
`trajectory.json`, `images.json` (every image with the **source frame** it came from),
`usage.json` (tokens and cost, missing fields recorded rather than zeroed),
`settings.json`, `events.jsonl` (appended as the run happens, so a crash keeps a record).

This band is why the three conditions can be built at all: B and C are reconstructed from
the saved frames, not re-invented.

**3. The three conditions (below the pipeline, branching off the saved run)**

| Condition | Draw it as | Exact statement |
|---|---|---|
| **A** | the pipeline above | the real thing: tools pick frames, reports mediate, the answer sees text only |
| **B** | one call: A's exact frames + verbatim question | same bytes, same order, same labels; **no reports, no system prompt, no tools**. Diagnoses whether the exact A frames alone rescue the answer |
| **C** | one call: frames from the annotated intervals + verbatim question | sampled and encoded exactly as `skim` would; **no reports, no system prompt, no tools**. Diagnoses whether the privileged annotated frames are sufficient |

Annotate the logic the figure is meant to support carefully: A misses and B misses means the
exact A image set was not rescued by removing the reports. B misses and C answers means the
annotated frames were helpful, but it does not by itself isolate locating as the only cause.

**4. Honest caveat, as a footnote on the figure**

C's payload carries 16 samples from each annotated window with their source timestamps. The
windows arrive in chronological group order, but the headers do **not** say which group is
Event A, B, or C; the model still has to map the scenes to the event descriptions. C shows
that the answer path yields the key's answer with helpful frames — not exactly why, and not
that the model recognised the events visually. Put this in the caption rather than hiding it;
it is the kind of thing that gets caught in review.

## Style

- Paper figure, not slide art: thin rules, no shadows, no clip-art, no emoji. Serif
  (Computer Modern / LaTeX look) or a clean sans-serif, one family only.
- Vector preferred (PDF/SVG). If generated with matplotlib, note that **matplotlib is not
  installed in the project venv** — install it in whatever environment draws the figure,
  or hand-write TikZ, which is what the LaTeX-style look usually wants anyway.
- Fit a single column or a full double-column width with the text legible at print size.
  Two stacked panels (A on top, B/C below) work at both.
- Colour-blind safe: do not rely on red/green alone for the image/text distinction; pair
  colour with line style (solid vs dashed) or with a legend.
- Numbers on the diagram (64 frames, 16 frames, 2 calls per step) must match the table
  above; they come from `config/general.yaml`.

## Common errors to avoid

- Drawing the answer step with images flowing into it. It has none. This is the point.
- Implying one model call per step. There are two.
- Drawing the vision tools as a separate model. They call the same Gemini model; the
  split is a code boundary (controller vs tools), not a model boundary.
- Showing A's frames as "all frames". A run decodes a small sample: the pilot's failed run
  decoded **143 distinct frames** over 7 steps, 3 of which fell inside the annotated
  intervals.
- Calling condition C an "oracle" without the caveat above, or implying it uses the answer
  key's *descriptions* — C sends the intervals' frames and their timestamps, never the
  descriptions or the answer.

## Also worth drawing, in this order of value

1. **The A/B/C input matrix** — 3 rows × columns {images shown, source of images, text
   reports, system prompt, tools, model calls}. One glance at what was held fixed. Easy
   to make and hard to get wrong.
2. **Coverage vs correctness grid** — the five questions × {Coverage@1, @2, @3, correct},
   from each run's `score.json`. It is the figure for the caveat that a coverage pass does
   not mean the agent located anything. (Built: `docs/figures/coverage_vs_correctness.png`.)
3. Already built and ready: `q207_search_timeline.png`, `q207_visual_audit.png`,
   `abc_input_matrix.png`, and `architecture_overview.png`. The earlier
   `q207_event_type_recurs.png` is a superseded preliminary interpretation and should not be
   shown.
