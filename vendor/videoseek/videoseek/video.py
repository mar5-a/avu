"""Frame access for the tools, finding each frame by its own timestamp.

decord seeks to the keyframe before a requested frame and counts decoded frames
from there. On many YouTube files the seek lands one keyframe earlier than decord
believes, so it hands back a frame one keyframe gap (1-5 s) before the one asked
for, with no error. In the pilot, 18 of the 96 frames one run was shown came from
earlier moments than their time labels said. `seek_accurate` does the same.

This reader decodes forward from the seek until a frame's timestamp is the one
asked for, so a seek that lands early costs decoding time, never the wrong frame.
On the frames decord does return correctly the pixels are identical.

It covers the part of decord's `VideoReader` the tools and scripts use.
"""

import av
import numpy as np


class FrameBatch:
    """What decord's `get_batch` returns, as far as callers use it."""

    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class VideoReader:
    def __init__(self, path):
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        # Every frame the decoder will output, in display order, read from the
        # packets without decoding. Packets an edit list cuts are not frames.
        self._pts = sorted(
            packet.pts
            for packet in self._container.demux(self._stream)
            if packet.pts is not None and not packet.is_discard
        )
        self._fps = float(self._stream.average_rate)
        self._ticks_per_sec = 1 / float(self._stream.time_base)
        self._decoded = None  # the decoder, positioned just after the frame at `_last`
        self._last = None

    def __len__(self):
        return len(self._pts)

    def __getitem__(self, index):
        return self._frame(int(index))

    def get_avg_fps(self):
        return self._fps

    def get_batch(self, indices):
        return FrameBatch(np.stack([self._frame(int(index)) for index in indices]))

    def _frame(self, index):
        target = self._pts[index]
        # A frame shortly after the last one decoded is reached faster by
        # decoding on than by seeking back to a keyframe.
        if self._last is not None and self._last < target <= self._last + 2 * self._ticks_per_sec:
            frame = self._decode_to(target)
            if frame is not None:
                return frame
        # A seek that lands after the frame finds nothing; each retry starts
        # further back, the last from the first frame, which always reaches it.
        for back_sec in (0, 10, 60, None):
            start = self._pts[0] if back_sec is None else target - int(back_sec * self._ticks_per_sec)
            self._container.seek(max(start, self._pts[0]), stream=self._stream, backward=True)
            self._decoded = self._container.decode(self._stream)
            frame = self._decode_to(target)
            if frame is not None:
                return frame
        raise ValueError(f"frame {index} (pts {target}) was never decoded")

    def _decode_to(self, target):
        for frame in self._decoded:
            self._last = frame.pts
            if frame.pts == target:
                return frame.to_ndarray(format="rgb24")
            if frame.pts > target:
                return None
        self._last = None
        return None
