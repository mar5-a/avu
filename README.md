# AVU: why do video agents get answers wrong?

When an AI agent answers a question about a video and gets it wrong, there are two
very different explanations. It may have **missed the evidence**: it never looked at
the moment that holds the answer. Or it may have **seen the evidence and not used it**:
the right moment was in front of it, and it still chose wrong. This project measures
which one happens, on a benchmark where people have marked where the evidence is.

Undergraduate research by Ammar Sohail at [BRAIN Lab](https://brain-lab-ai.github.io/),
KFUPM, with Dr. Muzammil Behzad. The work is exploratory and does not propose a new
method.

## How it works

The agent is [VideoSeek](https://arxiv.org/abs/2603.20185), an existing video agent,
run with Gemini 3.8 Flash. At each step a controller decides where in the video to
look. Its tools pick frames from that part of the video, Gemini describes the frames
in text, and the controller reads those notes. The controller never sees the images
itself: its final answer is written from the text notes alone.

The questions come from [VES-Bench](https://arxiv.org/abs/2608.22516), where annotators
marked the time window of each event a question depends on. The answer key and these
windows are saved beside each run and never sent to the model. Every frame the agent
is shown is recorded with its time, so we can check whether it ever looked inside the
right window.

To tell the two explanations apart, the same question is answered in several ways:

| | what the model gets |
|---|---|
| **A** | a normal agent run: frames chosen by the agent, answer written from the text notes |
| **B** | the exact images from run A, sent fresh in one request, with the notes left out |
| **C** | frames taken from the annotated windows instead, also with no notes |
| **Uniform** | 128 evenly spaced frames from the whole video in one request, no agent |

If A is wrong but B is right, the notes lost something the images had. If A and B are
both wrong but C is right, the model can answer from the right frames, which points to
the frames the agent picked or how it judged them.

## What we have found so far

These are small runs, so they are patterns to test, not results.

- **Pilot, 5 questions.** A was right on 4. B gave the same answers on those 4 and also
  missed the fifth, a hockey video. C, given frames from the annotated windows, got the
  hockey question right.
- **15 ordering questions.** The agent got 12 right at medium reasoning effort and 13
  at high. The uniform baseline got 14, at lower cost. One question apart is within
  noise at this size.
- **Most wrong answers had the evidence.** In 5 of the 6 wrong answers, frames from
  inside every event's window had been sent, and the agent accepted a similar looking
  moment elsewhere. In a bike cleaning video, asked for a close up of a water gun on
  the rear cassette at 115 to 122 seconds, it settled on an earlier wide shot of the
  bike being hosed at 28 seconds.

On this small set, that points toward the model misusing evidence more often than
missing it.

## What's next

Through an API we can change what surrounds the model, but we cannot look inside it.
The next stage moves to smaller open models such as Qwen, which run on our own GPUs,
can be inspected (for example, where attention goes before an answer), and can be fine
tuned. The first step is to repeat the measurement at scale on the benchmark's 568
YouTube questions, sorting every wrong answer into missed evidence or misused evidence.

## Running it

You need Python 3.13, ffmpeg, and a Gemini API key. Calls to Gemini cost money.
`replay_b.py`, `replay_c.py`, `uniform_direct.py` and `run_cycle.py` only send
requests when given `--live`; `run_one_question.py` always sends.

```bash
uv venv --python 3.13 .venv
uv pip install -e vendor/videoseek -r requirements.txt
cp .env.example .env   # then add your key after GEMINI_API_KEY=
```

Data stays out of Git:

- `data/VES-Bench.parquet` is the benchmark's question file, from its authors (see the
  [TRACE paper](https://arxiv.org/abs/2608.22516)).
- `data/videos/<video_id>.mp4` holds one video per question. For YouTube questions,
  `video_id` is the YouTube id. Download the video stream only, preferring H.264 up
  to 720p:

```bash
yt-dlp -f "bv*[vcodec^=avc1][height<=720]/bv*[vcodec^=avc1]/bv*[height<=720]/bv*" -o "data/videos/<video_id>.mp4" "https://www.youtube.com/watch?v=<video_id>"
```

Run each script as `.venv/bin/python scripts/<name>.py`:

| script | what it does |
|---|---|
| `offline_checks.py` | checks the recording, replay and scoring code; no network, no key, no cost |
| `check_videos.py` | checks that downloaded videos decode the way the tools read them |
| `run_one_question.py --question-id 215` | run A: the agent on one question, saved under `data/runs/` |
| `replay_b.py --run <run dir> --live` | run B from a saved A run |
| `replay_c.py --run <run dir> --live` | run C from a saved A run |
| `uniform_direct.py --question-id 215 --live` | the uniform baseline |
| `ves_score.py <run dirs>` | scores runs with the benchmark's own metrics |
| `select_questions.py` | picks questions by a rule fixed before any result is seen |
| `run_cycle.py --cycle <name> --questions 6,215 --live` | runs a batch in order and stops before it could pass a spending cap |

Each run directory keeps the question and answer key, every image sent with the frames
it came from, the text notes, the final answer, the settings, token usage and cost,
and any errors.

## Credits

VideoSeek is by Jingyang Lin and colleagues ([paper](https://arxiv.org/abs/2603.20185),
[code](https://github.com/jylins/videoseek)) and is MIT licensed. A copy is vendored in
`vendor/videoseek` with its license, and our changes to it are listed in
`vendor/videoseek-local-changes.patch`. VES-Bench is from the TRACE paper (Liu et al.,
2026, [arXiv 2608.22516](https://arxiv.org/abs/2608.22516)).
