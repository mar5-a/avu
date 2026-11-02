import os
import random
import re
import time
import httpx
from litellm import completion

from .recorder import recorder


def _failed_in_transit(error):
    """Whether the call died on the network: no connection, or the reply was cut off.

    LiteLLM wraps these in APIConnectionError, which is also its catch-all for
    errors it does not recognise; retrying one of those only repeats it. The httpx
    error it was raised while handling is what says the network was at fault, and
    unlike the message ("nodename nor servname provided" on macOS, "Name or service
    not known" on Linux) it does not depend on the operating system.
    """
    while error is not None:
        if isinstance(error, (httpx.NetworkError, httpx.RemoteProtocolError)):
            return True
        error = error.__cause__ or error.__context__
    return False


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        started = time.monotonic()

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if (
                    "rate limit" in str(e).lower()
                    or "timed out" in str(e)
                    or "Too Many Requests" in str(e)
                    or "Forbidden for url" in str(e)
                    or "the maximum usage" in str(e).lower()
                    or "server had an error" in str(e).lower()
                    or "has no attribute 'upper'" in str(e).lower()
                    or "internal" in str(e).lower()
                    # Gemini's 503 "model is currently experiencing high demand"
                    # (status UNAVAILABLE): upstream gave up at once, the tool
                    # crashed on the None, and the run carried on without it.
                    or "unavailable" in str(e).lower()
                    # A dropped connection: a failed DNS lookup ended a cycle run
                    # on its first call, which upstream did not retry.
                    or _failed_in_transit(e)
                ):
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        # The caller only sees None, so record why the call died:
                        # otherwise the reason lives in stdout and nowhere else.
                        recorder.record_attempt(
                            "gave_up",
                            error=e,
                            error_type=type(e).__name__,
                            elapsed_sec=time.monotonic() - started,
                        )
                        print("Max retries reached. Exiting.")
                        return None

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    recorder.record_attempt(
                        "retry",
                        error=e,
                        error_type=type(e).__name__,
                        elapsed_sec=time.monotonic() - started,
                        delay=delay,
                    )
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    recorder.record_attempt(
                        "failed",
                        error=e,
                        error_type=type(e).__name__,
                        elapsed_sec=time.monotonic() - started,
                    )
                    print(str(e))
                    return None

    return wrapper


@retry_with_exponential_backoff
def _call_llm_api(
    model_name: str,
    messages: list,
    api_base: str,
    api_key: str = None,
    api_version: str = None,
    max_tokens: int = 32768,
    reasoning_effort: str = "medium",
    seed: int = 42,
    temperature: float = 1.0,
    tools: list = None,
    tool_choice: str = None,
    return_json: bool = False,
) -> dict:
    params = dict(
        model=model_name,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        api_version=api_version,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format={"type": "json_object"} if return_json else None,
        timeout=900,
    )
    # Gemini defines no `seed` parameter, and LiteLLM raises UnsupportedParamsError
    # instead of ignoring it, so send it only to providers that accept it.
    if not model_name.startswith("gemini/"):
        params["seed"] = seed
    return completion(**params)


def extract_thought(response) -> tuple:
    """Return the model's plan text, and whether it needed a fallback.

    Gemini 3.8 answers VideoSeek's reasoning-only request with a function call
    instead of prose: it proposes a video tool it was never granted, and the name
    varies between calls (`video_intelligence:get_frame`, `sample_frames`, ...).
    `.content` is then None and the plan sits in `.reasoning_content`. Without
    this fallback the thought is empty, and the follow-up call that chooses a tool
    from the thought has nothing to work from, so it defaults to `overview`.
    """
    message = response.choices[0].message
    if message.content:
        return message.content, "content"
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning, "reasoning_content"
    return "", "empty"


def question_context_block(question_context) -> str:
    """The run's question as a worker sees it, or nothing when there is none.

    Upstream gives `skim` and `focus` only the query the planner wrote. A query
    such as "Does Event A or Event B occur right after Event C?" then reaches the
    worker with its labels undefined, and a paraphrased query can drop the
    detail that tells two similar scenes apart. The question stem carries the
    event definitions; the answer options stay out, because choosing an option
    is the planner's job, not a worker's.
    """
    if not question_context:
        return ""
    return (
        "Context: the overall question this video is being searched for. Use it to "
        "interpret the question below, for example what an event label refers to, "
        "but answer only the question below.\n"
        f"{question_context}\n\n"
    )


def call_llm_api(*args, **kwargs) -> dict:
    """Single entry point: record the outgoing request, then call with retries.

    Capturing here rather than inside the retried function keeps a retry from
    appending the same images to the manifest a second time. A call that never
    returns is counted by the retry wrapper, which knows why it stopped, so
    every call the agent made is accounted for exactly once.
    """
    recorder.check_budget()
    recorder.record_request(kwargs)
    recorder.capture(kwargs.get("messages") or [])
    response = _call_llm_api(*args, **kwargs)
    if response is not None:
        recorder.record_usage(response)
    return response


def load_subtitles(subtitle_path: str):
    """Parse SRT file and return list of {start_time, end_time, subtitle} dicts."""
    if subtitle_path is None or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = []
    # SRT format: index, HH:MM:SS,mmm --> HH:MM:SS,mmm, then text lines
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    blocks = re.split(r"\n\n+", content.strip())

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        match = pattern.search(block)
        if match:
            start = to_seconds(*match.groups()[:4])
            end = to_seconds(*match.groups()[4:8])
            text = block[match.end() :].strip().replace("\n", " ")
            result.append(
                {
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                    "subtitle": text,
                }
            )
    return result


def convert_to_free_form_text_representation(
    history: list[dict], content_type: str = "caption"
) -> str:
    """
    This function will form the textual representation for the entire video to be used for QA.
    It gives a good structured representation of the entire video.
    JSON types of representations are good for outputs, but free-form/ semi-structured should be better for input.
    """
    free_form_text_representation = ""
    if len(history) == 0:
        return f"No {content_type} found."
    for i in history:
        if i[content_type] is None:
            continue
        x = ""
        start_time, end_time = i["start_time"], i["end_time"]
        x += f"**Timestamp**: {start_time}s - {end_time}s\n"
        x += f"**{content_type.capitalize()}**: {i[content_type]}\n"

        free_form_text_representation += f"{x}\n"

    return free_form_text_representation
