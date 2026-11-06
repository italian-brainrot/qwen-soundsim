"""Real-time convolution with a room impulse response.

The response is long (a stairwell tail is seconds) and the input is continuous,
so direct FIR filtering is far too expensive.  This is **uniform partitioned
frequency-domain convolution** (overlap-save): the response is cut into blocks of
``block`` samples, and for every input block the output is one accumulated
matrix-vector product of pre-computed response spectra against the history of
input spectra.  Cost per output sample grows only linearly with the tail length
and is a handful of operations for a 3 s response.

Changing the environment while sound is playing is the whole point of the app, so
a new response is installed with a short **crossfade between two convolvers** --
no clicks, no drop-outs, and the change itself becomes an audible demonstration of
what the parameters do.
"""

from __future__ import annotations

import numpy as np


class PartitionedConvolver:
    """Single-channel partitioned FFT convolution (overlap-save, uniform blocks)."""

    def __init__(self, block: int = 512):
        self.block = int(block)
        self.nfft = 2 * self.block
        self.h: np.ndarray | None = None          # (n_parts, nfft//2+1) complex
        self.n_parts = 0
        self._hist: np.ndarray | None = None      # (n_parts, nfft//2+1) input spectra
        self._pos = 0
        self._left = np.zeros(self.block)         # input staging
        self._pending = np.zeros(self.block)          # overlap-add carry-over

    # -- setup ----------------------------------------------------------
    def set_ir(self, ir: np.ndarray):
        """Split the response into blocks of ``block`` samples, each carried by a
        zero-padded transform of length 2*block (overlap-save)."""
        ir = np.asarray(ir, dtype=np.float64).ravel()
        if len(ir) == 0:
            raise ValueError("empty impulse response")
        pad = (-len(ir)) % self.block
        ir = np.concatenate([ir, np.zeros(pad)])
        blocks = ir.reshape(-1, self.block)
        padded = np.zeros((blocks.shape[0], self.nfft))
        padded[:, :self.block] = blocks
        self.h = np.fft.rfft(padded, axis=1)
        self.n_parts = self.h.shape[0]
        self._hist = np.zeros_like(self.h)
        self._pos = 0
        self._left = np.zeros(self.block)
        self._prev_spec = np.zeros(self.nfft // 2 + 1, dtype=np.complex128)

    # -- streaming ------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Consume an arbitrary-length chunk; output is delayed by one block."""
        x = np.asarray(x, dtype=np.float64)
        buf = np.concatenate([self._left, x])
        pieces = []
        i = 0
        while i + self.block <= len(buf):
            pieces.append(self._one_block(buf[i:i + self.block]))
            i += self.block
        self._left = buf[i:].copy()
        return np.concatenate(pieces) if pieces else np.empty(0)

    def _one_block(self, blk: np.ndarray) -> np.ndarray:
        """Frequency-domain overlap-add over one input block.

        With H_i = FFT(h_i, 2N) and X_k = FFT(x_k, 2N) (both zero padded), the
        circular convolution equals the linear one in full, so the accumulate
        ``A_k = SUM_i H_i X_(k-i)`` holds, for every partition pair, the part of
        the answer that falls in the current output block (first half) *and* the
        part that belongs to the next one (second half):

            out_k = first_half(IFFT(A_k)) + second_half(IFFT(A_(k-1)))
        """
        if self.h is None:
            return np.zeros(self.block)
        n = self.nfft
        spec = np.fft.rfft(np.concatenate([blk, np.zeros(self.block)]))
        self._hist[self._pos] = spec
        self._pos = (self._pos + 1) % self.n_parts
        # response partition i multiplies the input block from i steps back:
        # the earliest partition pairs with the *newest* spectrum
        order = (self._pos - 1 - np.arange(self.n_parts)) % self.n_parts
        acc = np.einsum("ij,ij->j", self.h, self._hist[order])
        y = np.fft.irfft(acc, n=n)
        out = y[:self.block] + self._pending
        self._pending = y[self.block:]
        return out

    def reset(self):
        if self._hist is not None:
            self._hist[:] = 0.0
        self._pos = 0
        self._left = np.zeros(self.block)
        self._pending = np.zeros(self.block)


class CrossfadeConvolver:
    """One convolver per channel, with glitch-free response replacement."""

    def __init__(self, channels: int = 2, block: int = 512):
        self.channels = int(channels)
        self.block = block
        self.conv = [PartitionedConvolver(block) for _ in range(self.channels)]
        self.next_conv: list[PartitionedConvolver] | None = None
        self.irs: list[np.ndarray] = []      # kept for level prediction
        self.fade_samples = 0
        self.fade_index = 0
        # The block machinery needs one block of latency; priming the output queue
        # with silence keeps a whole block of real samples ahead of the callback
        # instead of inserting zeros in the middle of the waveform.
        self._out_left = np.zeros((self.block, self.channels))

    def set_ir(self, irs, crossfade_s: float = 0.03, sample_rate: int = 48_000):
        """Install new response(s); ``irs`` is one array or one per channel."""
        if isinstance(irs, np.ndarray) and irs.ndim == 1:
            irs = [irs] * self.channels
        if self.fade_samples > 0 and self.next_conv is not None:
            active = self.next_conv
        else:
            active = self.conv
        new = [PartitionedConvolver(self.block) for _ in range(self.channels)]
        for i, c in enumerate(new):
            c.set_ir(irs[i % len(irs)])
        self.irs = [np.asarray(irs[i % len(irs)], dtype=np.float64)
                    for i in range(self.channels)]
        if self.fade_samples == 0:
            self.conv = new
            self.next_conv = None
            return
        # continue from whatever is currently active, crossfading into `new`
        self.next_conv = new
        self.fade_samples = int(crossfade_s * sample_rate)
        self.fade_index = 0
        self._old = active

    def process(self, frames: np.ndarray) -> np.ndarray:
        """frames: (n, channels) -> exactly n samples.

        The block machinery can hand back more (or fewer) samples than were fed
        in, so the surplus is queued and a deficit is zero-filled: the audio
        callback always gets the number of samples it asked for.
        """
        if frames.ndim == 1:
            frames = frames[:, None]
        n = len(frames)

        def run(convs):
            cols = [convs[c].process(frames[:, c]) for c in range(self.channels)]
            m = max(len(c) for c in cols)
            out = np.zeros((m, self.channels))
            for c, col in enumerate(cols):
                out[:len(col), c] = col
            return out

        out = run(self.conv)
        if self.next_conv is not None and self.fade_samples > 0:
            new = run(self.next_conv)
            m = max(len(out), len(new))
            if len(out) < m:
                out = np.pad(out, ((0, m - len(out)), (0, 0)))
            if len(new) < m:
                new = np.pad(new, ((0, m - len(new)), (0, 0)))
            w = np.linspace(self.fade_index / self.fade_samples,
                            min(1.0, (self.fade_index + m) / self.fade_samples), m)
            out = (1.0 - w[:, None]) * out + w[:, None] * new
            self.fade_index += m
            if self.fade_index >= self.fade_samples:
                self.conv = self.next_conv
                self.next_conv = None
                self.fade_samples = 0

        out = np.concatenate([self._out_left, out]) if self._out_left.size else out
        if len(out) > n:
            self._out_left = out[n:]
            out = out[:n]
        else:
            self._out_left = np.empty((0, self.channels))
            out = np.pad(out, ((0, n - len(out)), (0, 0)))
        return out

    def reset(self):
        for c in self.conv:
            c.reset()
        if self.next_conv:
            for c in self.next_conv:
                c.reset()

    @property
    def latency_samples(self) -> int:
        """Constant output delay of the block machinery (not the room's delay).

        One block is inherent to overlap-add, one comes from priming the output
        queue.  The player subtracts this so the *only* delay the listener hears
        is the physical one in the impulse response.
        """
        return 2 * self.block

    @property
    def ir_length_samples(self) -> int:
        if self.conv[0].h is None:
            return 0
        return self.conv[0].n_parts * self.block
