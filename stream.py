#!/usr/bin/env python3
"""Continuous streaming capture from the AD9643 ring buffer (spec section 6).

The FPGA writes the 500 MB DDR window as 16 segments of 32,768,000 B, in
order, wrapping. reg5 counts segments completely written; the host acks with
reg7. This module reads them out and either writes them to a file or discards
them, and reports what was lost.

WHY THE READ PATH LOOKS LIKE THIS
---------------------------------
Raw read()/readv()/pread() on /dev/xdma0_c2h_* HARD-FREEZES this SoC --
reproducibly, into a plain malloc'd buffer, nothing logged (bisect_dma.py
stage S4, NOTES.md #5). The spec's section 2/5 sample code uses exactly that
call. The only access pattern proven safe here is the vendor CLI's:
O_RDWR|O_TRUNC, posix_memalign(4096), chunked lseek+read -- which is what
native/xdma_shm_reader replicates verbatim. So every byte goes through that
helper. `validate_fast_dma.py` PASSED on hardware 2026-09-28 (byte-identical
to the vendor CLI up to 64 MB, 30 s soak, 0.91 GB/s) which is what makes it
usable here at all; re-run it after any kernel or driver change.

MEASURED ON THIS HOST (Orin, Gen3 x4, 2026-09-28)
-------------------------------------------------
  1 C2H channel   1.28 GB/s     mode 3 needs 1.00 -- only 1.28x, worst-case
                                single read 28.8 ms vs a 32.8 ms budget
  2 C2H channels  2.19 GB/s     2.19x. This is the default.
  4 C2H channels  3.04 GB/s     saturates the link; available via --channels

A segment is split evenly across the channels and read concurrently, so the
per-segment latency -- not the aggregate rate -- is what has to fit inside
the segment period.

NOTHING IS WRITTEN TO DISK
--------------------------
Segments land in a fixed-size RAM ring and the oldest are evicted first, so a
run of any length keeps the most recent --ram-gb worth of data. This is not
just a convenience: writing to the NVMe was measured to hold 3.8 GB/s for
about 57 GB and then collapse to 0.26 GB/s once the DRAM-less NM790's dynamic
SLC cache fills -- below even single-channel rate -- so a disk-backed stream
could not have run "indefinitely" the way the spec assumes anyway.

The ring is ZERO-COPY: the helpers DMA straight into the shared mapping that
IS the ring, so a segment is never copied after it arrives. /dev/shm caps the
size at 31 GB on this host, which is ~61 s single-channel or ~30 s in mode 3.
--dump writes the resident window out afterwards; being bounded by the ring
size it stays inside the SLC cache and runs at full speed.
"""
import argparse, mmap, os, select, subprocess, sys, threading, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "native", "xdma_shm_reader")
C2H_DEVS = [f"/dev/xdma0_c2h_{i}" for i in range(4)]


class Helper:
    """One resident xdma_shm_reader with a rotating set of shm slots."""

    def __init__(self, dev, slot_bytes, nslots, tag):
        self.dev, self.slot_bytes, self.nslots = dev, slot_bytes, nslots
        self.cap = slot_bytes * nslots
        self.shm_path = f"/dev/shm/adc_stream_{os.getpid()}_{tag}.buf"
        # bounce buffer sized to ONE slice, not to the whole ring: with
        # --ram-gb 30 the mapping is 15 GB per helper and a bounce buffer that
        # size would be grotesque even lazily faulted.
        self.p = subprocess.Popen([HELPER, dev, self.shm_path, str(self.cap),
                                   str(slot_bytes)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        r, _, _ = select.select([self.p.stdout], [], [], 5.0)
        banner = self.p.stdout.readline() if r else ""
        if not banner.startswith("READY"):
            raise RuntimeError(f"{dev}: helper did not start ({banner.strip()!r})")
        self._f = open(self.shm_path, "r+b")
        self.mm = mmap.mmap(self._f.fileno(), self.cap)

    def issue(self, addr, nbytes, slot):
        self.p.stdin.write(f"R {addr} {nbytes} {slot * self.slot_bytes}\n")
        self.p.stdin.flush()

    def wait(self, timeout):
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not r:
            raise A.DmaTimeout(f"{self.dev}: no reply in {timeout:.1f}s")
        rep = self.p.stdout.readline().strip()
        if not rep.startswith("OK"):
            raise A.DmaTimeout(f"{self.dev}: {rep}")
        return int(rep.split()[1])

    def view(self, slot, nbytes):
        off = slot * self.slot_bytes
        return memoryview(self.mm)[off:off + nbytes]

    def close(self):
        try:
            self.p.stdin.write("Q\n"); self.p.stdin.flush(); self.p.wait(timeout=3)
        except Exception:
            try: self.p.kill()
            except Exception: pass
        for o in ("mm", "_f"):
            try: getattr(self, o).close()
            except Exception: pass
        try: os.unlink(self.shm_path)
        except OSError: pass


class Stream:
    """Streaming session: start, read segments, ack, stop.

    `sink` is called as sink(list_of_memoryviews, seg_no) for each valid
    segment, in order, from the consumer thread. The views are only valid
    until the slot is recycled (nslots segments later), so a sink that keeps
    the data must copy it.
    """

    def __init__(self, adc, channel, nchan=2, nslots=4, seg_timeout=20.0,
                 ram_ring=False, envelope=False):
        if A.SEG_BYTES % (nchan * 4096):
            raise ValueError(f"segment does not split {nchan} ways on a 4 KB "
                             f"boundary")
        self.adc, self.channel, self.nchan = adc, channel, nchan
        self.slice_bytes = A.SEG_BYTES // nchan
        self.nslots, self.seg_timeout = nslots, seg_timeout
        self.helpers = [Helper(C2H_DEVS[i], self.slice_bytes, nslots, i)
                        for i in range(nchan)]
        self.ring = (RamRing(self.helpers, self.slice_bytes, nslots, channel)
                     if ram_ring else None)
        self.env = (RingEnvelope(nslots, 2 if channel == A.CH_BOTH else 1)
                    if envelope else None)
        if self.env is not None:
            per = 4 if channel == A.CH_BOTH else 2
            self.env.seconds_per_seg = A.SEG_BYTES / (A.BASE_CLOCK_HZ * per)
        # stats
        self.n_read = self.n_lost = self.n_late = 0
        self.first_overrun_at = None
        self.flags_seen = 0
        self.bytes_read = 0
        self.t_start = self.t_end = None
        self._stop = threading.Event()
        self._err = None
        self.sink_seconds = 0.0

    # ------------------------------------------------------------ registers
    def _rd(self, off): return self.adc.rd(off)
    def _wr(self, off, v): self.adc.wr(off, v)

    def start(self):
        """Spec 6.2. The start edge clears reg5, reg6 bit0/bit1 and reg7."""
        a = self.adc
        a.wr(A.REG_START, 0x0)                      # ensure stopped
        a.wr(A.REG_SPEED, 0)                        # spec 8.1
        a.wr(A.REG_CHANNEL, self.channel)           # only while stopped, 8.2
        a.wr(A.REG_NSAMPLES, A.seg_samples(self.channel))
        a.wr(A.REG_START, A.STREAM_BIT)             # stream mode, start low
        a.wr(A.REG_START, A.STREAM_BIT | A.START_BIT)
        self.t_start = time.monotonic()

    def stop(self):
        """Spec 6.5. The FPGA finishes the current lap, so up to one segment
        period. Poll reg5 until it stops changing, then drain."""
        self.adc.wr(A.REG_START, 0x0)
        prev = -1
        for _ in range(40):                          # <= 4 s
            time.sleep(0.10)
            cur = self._rd(A.REG_SEGCNT)
            if cur == prev:
                return cur
            prev = cur
        return prev

    # --------------------------------------------------------- reader loop
    def _read_segment(self, n, slot):
        """One segment, split across the C2H channels. Returns bytes read."""
        base = A.SEG_ADDR(n)
        for i, h in enumerate(self.helpers):
            h.issue(base + i * self.slice_bytes, self.slice_bytes, slot)
        got = 0
        for h in self.helpers:
            got += h.wait(self.seg_timeout)
        return got

    def _reader(self, want_segments, duration, sink):
        """Single thread: poll reg5, read, validate, ack, hand to `sink`.

        The sink runs INLINE rather than on a consumer thread. It used to be
        threaded, with the reader blocking until a slot was released -- but
        both sinks are now cheap relative to a segment period (the RAM ring is
        zero-copy and just moves a pointer; the counter check is 10.2 ms
        against 65.5 ms), and inline removes the slot-release handshake, which
        was the only way this loop could deadlock.
        """
        nxt = 0
        try:
            while not self._stop.is_set():
                if want_segments and nxt >= want_segments:
                    break
                if duration and time.monotonic() - self.t_start >= duration:
                    break
                done = self._rd(A.REG_SEGCNT)
                if nxt >= done:
                    time.sleep(0.001)                # spec 6.6: no spin loops
                    continue
                if done - nxt > A.MAX_LAG:
                    # Already unreadable before we started: everything from
                    # nxt up to done-MAX_LAG has been overwritten. Skip to the
                    # newest safe segment rather than reading known garbage.
                    skip = (done - A.MAX_LAG) - nxt
                    self.n_lost += skip
                    if self.first_overrun_at is None:
                        self.first_overrun_at = nxt
                    nxt = done - A.MAX_LAG
                    self._wr(A.REG_SEGACK, nxt)
                    continue

                slot = nxt % self.nslots
                got = self._read_segment(nxt, slot)
                self.bytes_read += got

                # SPEC 6.4: authoritative per-segment validity. reg5 re-read
                # AFTER the DMA completes must be <= n + MAX_LAG, because the
                # writer runs up to 8 bursts ahead of the last completed
                # segment and starts overwriting slot n%16 before reg5 reaches
                # n+16. reg6 bit0 is only a summary of the same condition.
                if self._rd(A.REG_SEGCNT) - nxt > A.MAX_LAG:
                    self.n_lost += 1
                    if self.first_overrun_at is None:
                        self.first_overrun_at = nxt
                    self._wr(A.REG_SEGACK, nxt + 1)
                    nxt += 1
                    continue

                self._wr(A.REG_SEGACK, nxt + 1)      # ack: 0..nxt consumed
                if self.ring is not None:
                    self.ring.note(nxt)
                if sink is not None:
                    t0 = time.monotonic()
                    sink([h.view(slot, self.slice_bytes) for h in self.helpers],
                         nxt)
                    self.sink_seconds += time.monotonic() - t0
                self.n_read += 1
                nxt += 1
                self.flags_seen |= self._rd(A.REG_FLAGS)
        except Exception as e:
            self._err = e
        finally:
            self._stop.set()

    def run(self, sink=None, segments=0, duration=0.0, progress=None):
        self.start()
        rt = threading.Thread(target=self._reader,
                              args=(segments, duration, sink))
        rt.start()
        try:
            while rt.is_alive():
                rt.join(timeout=2.0)
                if progress and rt.is_alive():
                    progress(self)
        except KeyboardInterrupt:
            print("  interrupted -- stopping cleanly", file=sys.stderr)
            self._stop.set()
            rt.join()
        self.final_segcnt = self.stop()
        self.t_end = time.monotonic()
        self.flags_seen |= self._rd(A.REG_FLAGS)
        if self._err:
            raise self._err
        return self

    # --------------------------------------------------- background lifecycle
    # The web server needs the ring filling continuously while a DISPLAY loop
    # samples it at its own constant cadence. run() blocks for a fixed
    # duration, which is right for a CLI capture and wrong for a live server,
    # so these start/stop the same reader thread without joining it.
    def _env_loop(self):
        """Summarise segments for the ring envelope, OFF the reader thread.

        Run inline in the reader this cost ~7 ms against a 32.8 ms mode-3
        segment period on top of ~15 ms of DMA, and the reader started losing
        segments (82, then 203, measured). The ring is 122 segments deep, so a
        summariser lagging a few segments behind is never close to eviction;
        if it does fall behind it skips forward and `seg_of` keeps the skipped
        slots out of the drawn span rather than showing stale data.
        """
        nxt = 0
        while not self._stop.is_set():
            r = self.ring
            if r is None or r.last_seg < 1:
                time.sleep(0.005); continue
            newest = r.last_seg - 1          # leave the in-flight slot alone
            if nxt > newest:
                time.sleep(0.005); continue
            if nxt < r.first_seg + 2:        # fell behind: jump forward
                self.env.skipped += (r.first_seg + 2) - nxt
                nxt = r.first_seg + 2
            slot = nxt % self.nslots
            try:
                self.env([h.view(slot, self.slice_bytes) for h in self.helpers],
                         nxt)
            except Exception:
                pass
            nxt += 1

    def start_background(self, sink=None):
        self.start()
        self._rt = threading.Thread(target=self._reader,
                                    args=(0, 0.0, sink), daemon=True)
        self._rt.start()
        if self.env is not None:
            self._et = threading.Thread(target=self._env_loop, daemon=True)
            self._et.start()
        return self

    def stop_background(self, timeout=10.0):
        self._stop.set()
        rt = getattr(self, "_rt", None)
        if rt is not None:
            rt.join(timeout=timeout)
        self.final_segcnt = self.stop()
        self.t_end = time.monotonic()
        self.flags_seen |= self._rd(A.REG_FLAGS)
        return self

    @property
    def alive(self):
        rt = getattr(self, "_rt", None)
        return rt is not None and rt.is_alive()

    def newest(self, back=1):
        """Index of a segment safe to READ while the reader keeps writing.

        `back=1` (the default) is the newest fully-written segment minus one:
        the reader may still be DMA-ing into `last_seg + 1`'s slot, and with a
        ring of hundreds of slots one segment of margin costs ~33-66 ms of
        latency and removes any chance of reading a slot mid-write.
        """
        if self.ring is None or self.ring.last_seg < 0:
            return None
        k = self.ring.last_seg - back
        return k if k >= self.ring.first_seg else None

    def close(self):
        for h in self.helpers:
            h.close()

    def clear_overrun(self):
        """Spec 3: any write to 0x18 clears reg6 bit0. bit1 survives until the
        next start."""
        self.adc.wr(A.REG_FLAGS, 0)
        return self._rd(A.REG_FLAGS)


# --------------------------------------------------------------------- sinks
class RamRing:
    """Fixed-size RAM ring: the newest data is kept, the oldest is dropped.

    ZERO-COPY. The helpers already DMA into their shared mapping at a
    caller-chosen offset, so the ring IS that mapping: slot `seg % nslots`
    receives segment `seg` directly and eviction is just the write pointer
    coming round again. Nothing is copied, nothing is written to disk, and
    there is no consumer to fall behind -- which also removes the only place
    the old file path could stall.

    With `nchan` C2H channels the ring is physically `nchan` parallel rings,
    one per helper, each holding that channel's slice of every segment.
    `read_segment()` stitches a segment back together in sample order.

    Capacity is what /dev/shm allows (31 GB on this host), so at mode-3 rate
    a 30 GB ring retains the most recent ~30 s, and ~61 s single-channel.
    """

    def __init__(self, helpers, slice_bytes, nslots, channel):
        self.helpers, self.slice_bytes = helpers, slice_bytes
        self.nslots, self.channel = nslots, channel
        self.seg_bytes = slice_bytes * len(helpers)
        self.first_seg = 0        # oldest segment still resident
        self.last_seg = -1        # newest segment written
        self.evicted = 0

    def note(self, seg):
        """Called by the reader once segment `seg` has landed in its slot."""
        self.last_seg = seg
        if seg - self.first_seg >= self.nslots:
            self.first_seg = seg - self.nslots + 1
            self.evicted += 1

    @property
    def resident(self):
        if self.last_seg < 0:
            return (0, 0)
        return (self.first_seg, self.last_seg)

    @property
    def n_resident(self):
        return 0 if self.last_seg < 0 else self.last_seg - self.first_seg + 1

    def read_segment(self, seg, out=None, nwords=None):
        """One segment as uint16, in sample order. Raises if evicted.

        `out` copies straight into a caller buffer, one memcpy per channel
        slice. Without it a multi-channel ring has to np.concatenate, which
        allocates and copies the whole segment and then the caller copies it
        AGAIN into its staging buffer -- measured at 173 ms per 32.8 MB
        segment on the display path, against 13 ms of GPU work, i.e. the
        copies were the entire reason the stream frame rate still sagged with
        record length. `nwords` takes only the first N words, which is all the
        display needs when the FFT consumes 524,288 samples of an 8.2 M-sample
        segment.
        """
        if not (self.first_seg <= seg <= self.last_seg):
            raise IndexError(f"segment {seg} no longer resident "
                             f"(have {self.first_seg}..{self.last_seg})")
        slot = seg % self.nslots
        per = self.slice_bytes // 2                  # uint16 per slice
        total = per * len(self.helpers)
        want = total if nwords is None else min(int(nwords), total)
        if out is None:
            if len(self.helpers) == 1:
                return np.frombuffer(self.helpers[0].view(slot, self.slice_bytes),
                                     dtype='<u2')[:want]
            out = np.empty(want, np.uint16)
        done = 0
        for h in self.helpers:
            if done >= want:
                break
            n = min(per, want - done)
            src = np.frombuffer(h.view(slot, n * 2), dtype='<u2')
            np.copyto(out[done:done + n], src)
            done += n
        return out[:done]

    @property
    def seg_words(self):
        """uint16 words in one whole segment, across all channel slices."""
        return (self.slice_bytes // 2) * len(self.helpers)

    def read_span(self, last_seg, nwords, out, margin=4):
        """Up to `nwords` uint16 ending at segment `last_seg`, oldest first.

        A record longer than one segment has to be stitched from consecutive
        ring slots -- without this the display was silently capped at ONE
        segment (32.768 ms in mode 3, 65.5 ms single-channel) no matter what
        record length was asked for. The ring holds hundreds of consecutive
        segments precisely so longer records can be served from it.

        `margin` segments are held back from the oldest end: the reader keeps
        writing while this copies, and a record that reaches all the way to
        `first_seg` could have its head evicted mid-copy.
        """
        sw = self.seg_words
        avail_segs = max(0, last_seg - (self.first_seg + margin) + 1)
        if avail_segs <= 0:
            return out[:0], last_seg
        nsegs = max(1, min(avail_segs, -(-int(nwords) // sw)))
        first = last_seg - nsegs + 1
        done = 0
        for seg in range(first, last_seg + 1):          # oldest -> newest
            if done >= nwords:
                break
            take = min(sw, int(nwords) - done)
            got = self.read_segment(seg, out=out[done:done + take], nwords=take)
            done += got.size
        return out[:done], first

    def iter_segments(self):
        for k in range(self.first_seg, self.last_seg + 1):
            yield k, self.read_segment(k)

    def dump(self, path, direct=False):
        """Write the resident window out, oldest first. Bounded by the ring
        size, so it cannot hit the NVMe SLC cliff the way live capture did."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (os.O_DIRECT if direct else 0)
        fd = os.open(path, flags, 0o644)
        n = 0
        try:
            for seg in range(self.first_seg, self.last_seg + 1):
                slot = seg % self.nslots
                for h in self.helpers:
                    v = h.view(slot, self.slice_bytes)
                    off = 0
                    while off < len(v):
                        off += os.write(fd, v[off:])
                    n += len(v)
            os.fsync(fd)
        finally:
            os.close(fd)
        return n


class RingEnvelope:
    """Min/max/mean summary of every segment in the ring, built as it arrives.

    WHY INCREMENTAL. The display time axis should be able to span the whole
    RAM ring -- seconds of signal -- not just the record. Pulling the ring
    through the CPU every frame is not an option: 4 GB per frame is ~1 s of
    memory traffic. But a segment only has to be summarised ONCE, when it
    lands, and the display then just stitches the stored summaries. Cost moves
    from O(ring) per frame to O(segment) per segment, which the reader can
    absorb.

    `stride` subsamples within a segment. Measured per segment (32.8 MB, two
    channels, 8 columns): stride 1 = 27.9 ms, stride 64 = 7.2 ms, against a
    32.8 ms mode-3 segment period of which the DMA already uses ~15 ms. So the
    default subsamples, and the min/max are therefore a subsampled envelope:
    good for seeing signal shape and drift across seconds, NOT a guarantee of
    catching a single-sample glitch. The per-record trace still sees every
    sample and is the right tool for that.
    """

    def __init__(self, nslots, nchan_data, cols=8, stride=64):
        self.nslots, self.cols, self.stride = nslots, cols, stride
        self.nch = nchan_data                     # 1, or 2 in A+B mode
        shape = (nslots, nchan_data, cols)
        self.mn = np.zeros(shape, np.float32)
        self.mx = np.zeros(shape, np.float32)
        self.mean = np.zeros(shape, np.float32)
        self.valid = np.zeros(nslots, bool)
        # Which segment each slot's summary is FOR. The envelope worker may
        # skip segments if it falls behind; without this, span() would happily
        # draw a stale summary left in the slot by an earlier lap of the ring.
        self.seg_of = np.full(nslots, -1, np.int64)
        self.skipped = 0
        self.seconds_per_seg = 0.0

    def __call__(self, views, n):
        slot = n % self.nslots
        C, st = self.cols, self.stride
        for ch in range(self.nch):
            parts = []
            for v in views:
                a = np.frombuffer(v, dtype='<u2')
                parts.append(a[ch::self.nch * st] if self.nch > 1 else a[::st])
            x = np.concatenate(parts) if len(parts) > 1 else parts[0]
            k = (x.size // C) * C
            if k == 0:
                continue
            g = A.to_signed_i16(x[:k]).reshape(C, -1)   # honours 0x14 inversion
            self.mn[slot, ch] = g.min(axis=1)
            self.mx[slot, ch] = g.max(axis=1)
            self.mean[slot, ch] = g.mean(axis=1)
        self.valid[slot] = True
        self.seg_of[slot] = n

    def span(self, first_seg, last_seg, ch, width):
        """Stitch segments first..last into `width` columns, oldest first."""
        if last_seg < first_seg:
            return None
        segs = [k for k in range(first_seg, last_seg + 1)
                if self.seg_of[k % self.nslots] == k]
        if not segs:
            return None
        mn = np.concatenate([self.mn[k % self.nslots, ch] for k in segs])
        mx = np.concatenate([self.mx[k % self.nslots, ch] for k in segs])
        me = np.concatenate([self.mean[k % self.nslots, ch] for k in segs])
        # Resample to the display width: max of maxima, min of minima, so
        # downsampling never hides an excursion the summary did capture.
        # reduceat, not a Python loop over `width` columns -- that loop ran
        # 1024 iterations per channel per frame and held the ring axis to
        # 10 fps where the rest of the pipeline was good for 20.
        n = mn.size
        if n >= width:
            starts = np.linspace(0, n, width + 1).astype(np.int64)[:-1]
            # reduceat needs non-decreasing starts and treats a repeated start
            # as "just that element", which is the behaviour we want when the
            # summary is coarser than the display
            out_mn = np.minimum.reduceat(mn, starts)
            out_mx = np.maximum.reduceat(mx, starts)
            cnt = np.diff(np.append(starts, n)).clip(1)
            out_me = np.add.reduceat(me, starts) / cnt
        else:
            # fewer summary columns than display columns: stretch, no invention
            src = np.linspace(0, n - 1, width) if n > 1 else np.zeros(width)
            i = src.astype(np.int64)
            out_mn, out_mx, out_me = mn[i], mx[i], me[i]
        return (out_mn.astype(np.float32), out_mx.astype(np.float32),
                out_me.astype(np.float32), len(segs))


class CounterCheckSink:
    """Spec 7.2: the ChannelSel=0 ramp must stay continuous ACROSS segment
    boundaries, so the last sample of segment n is carried into segment n+1.

    Kept entirely in uint16 with preallocated scratch. The obvious version --
    `a[1:].astype(np.int32) - a[:-1].astype(np.int32)` -- allocates two 64 MB
    int32 temporaries per slice and measured 71.3 ms per segment against a
    65.5 ms segment period, i.e. the CHECKER became the bottleneck and the
    stream fell behind by 10 segments over 20 s. uint16 subtraction wraps mod
    65536, and the only non-unit step a correct ramp produces is the 14-bit
    wrap 16383 -> 0, which is 49153 in uint16 and 1 after & 0x3FFF -- so the
    masked uint16 difference is exactly the quantity wanted, with no promotion.
    """

    def __init__(self):
        self.last = None          # last code of the previous segment
        self.violations = 0
        self.seg_breaks = 0       # violations that land exactly on a boundary
        self.checked = 0
        self.first_bad = None
        self.high_bits_set = 0    # bits 15:14 should never be set
        self._d = None            # preallocated difference scratch

    def __call__(self, views, n):
        for v in views:
            a = np.frombuffer(v, dtype='<u2')       # no copy
            if self._d is None or self._d.size < a.size - 1:
                self._d = np.empty(a.size - 1, np.uint16)
            d = self._d[:a.size - 1]
            np.subtract(a[1:], a[:-1], out=d)       # uint16, wraps
            np.bitwise_and(d, 0x3FFF, out=d)
            bad = int(np.count_nonzero(d != 1))

            if self.last is not None:
                if (int(a[0]) - self.last) & 0x3FFF != 1:
                    self.violations += 1; self.seg_breaks += 1
                    if self.first_bad is None:
                        self.first_bad = (n, "segment boundary",
                                          self.last, int(a[0]))
            if bad and self.first_bad is None:
                i = int(np.flatnonzero(d != 1)[0])
                self.first_bad = (n, int(i), int(a[i]), int(a[i + 1]))
            self.violations += bad
            self.checked += a.size
            self.last = int(a[-1])
            # one cheap pass: the 14-bit codes must be zero-padded (spec 4)
            if a[::4096].max() > 0x3FFF:
                self.high_bits_set += 1


def summarise(st, sink=None, label=""):
    dt = (st.t_end or time.monotonic()) - st.t_start
    rate = st.bytes_read / dt / 1e9 if dt > 0 else 0.0
    per_sample = 4 if st.channel == A.CH_BOTH else 2
    seg_period = A.SEG_BYTES / (A.BASE_CLOCK_HZ * per_sample)
    need = A.SEG_BYTES / seg_period / 1e9
    print(f"\n--- {label or 'stream'} ---")
    print(f"  elapsed            {dt:8.2f} s")
    print(f"  segments read      {st.n_read:8d}   ({st.bytes_read/1e9:.2f} GB)")
    print(f"  segments LOST      {st.n_lost:8d}"
          + (f"   first at segment {st.first_overrun_at}" if st.n_lost else ""))
    print(f"  final reg5         {getattr(st,'final_segcnt','?'):>8}"
          f"   (segments produced by the FPGA)")
    print(f"  host read rate     {rate:8.2f} GB/s   (capture produces {need:.2f})")
    print(f"  segment period     {seg_period*1e3:8.1f} ms")
    if st.sink_seconds and st.n_read:
        print(f"  sink cost          {st.sink_seconds/st.n_read*1e3:8.1f} ms/segment"
              f"   ({100*st.sink_seconds/dt:.0f}% of wall time)")
    f = st.flags_seen
    print(f"  reg6 flags seen    {f:#010x}"
          f"   overrun={bool(f & A.FLAG_OVERRUN)} "
          f"fifo_overflow={bool(f & A.FLAG_FIFO_OVF)}")
    if st.ring is not None:
        r = st.ring
        lo, hi = r.resident
        print(f"  RAM ring           {r.nslots * r.seg_bytes/1e9:8.2f} GB "
              f"({r.nslots} segments = {r.nslots*seg_period:.1f} s)")
        print(f"  resident segments  {r.n_resident:8d}   [{lo}..{hi}]")
        print(f"  evicted (oldest)   {r.evicted:8d}"
              + ("   <- ring wrapped, oldest data dropped as intended"
                 if r.evicted else "   (ring never filled)"))
    if isinstance(sink, CounterCheckSink):
        print(f"  ramp samples       {sink.checked:8d}")
        print(f"  ramp violations    {sink.violations:8d}"
              f"   (of which at segment boundaries: {sink.seg_breaks})")
        print(f"  bits 15:14 set     {sink.high_bits_set:8d} segments (must be 0)")
        if sink.first_bad:
            print(f"  first violation    {sink.first_bad}")
    ok = st.n_lost == 0 and not (f & A.FLAG_FIFO_OVF)
    if isinstance(sink, CounterCheckSink):
        ok = ok and sink.violations == 0 and sink.high_bits_set == 0
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--channel", type=int, default=0,
                   help="ChannelSel: 0=counter (default), 1=A, 2=B, 3=A+B")
    p.add_argument("-t", "--duration", type=float, default=10.0,
                   help="seconds to stream (default 10)")
    p.add_argument("-n", "--segments", type=int, default=0,
                   help="stop after this many segments instead of --duration")
    p.add_argument("--ram-gb", type=float, default=8.0,
                   help="RAM ring size in GB (default 8; /dev/shm caps it, "
                        "31 GB on this host). Oldest segments are evicted "
                        "first.")
    p.add_argument("--discard", action="store_true",
                   help="read and throw away; no ring (pure throughput test)")
    p.add_argument("--check-counter", action="store_true",
                   help="verify the ChannelSel=0 ramp across segment boundaries")
    p.add_argument("--dump", metavar="FILE",
                   help="after stopping, write the resident ring to FILE "
                        "(bounded by --ram-gb, so it cannot hit the NVMe "
                        "SLC cliff)")
    p.add_argument("--tap", type=int, default=None,
                   help="IDELAY tap to load first (default: stored config)")
    p.add_argument("--channels", type=int, default=2, choices=(1, 2, 4),
                   help="C2H channels to split each segment across (default 2)")
    a = p.parse_args()

    if a.channel not in A.VALID_CHANNELS:
        p.error(f"channel must be one of {A.VALID_CHANNELS}")
    if a.check_counter and a.channel != 0:
        p.error("--check-counter only makes sense with ChannelSel=0")
    if a.discard and a.dump:
        p.error("--discard keeps nothing to --dump")

    slice_bytes = A.SEG_BYTES // a.channels
    use_ring = not a.discard
    if use_ring:
        nslots = max(3, int(a.ram_gb * 1000**3) // A.SEG_BYTES)
        shm_need = nslots * A.SEG_BYTES
        free_shm = os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize
        if shm_need > free_shm:
            p.error(f"--ram-gb {a.ram_gb} needs {shm_need/1e9:.1f} GB of "
                    f"/dev/shm but only {free_shm/1e9:.1f} GB is free")
    else:
        nslots = 4

    sink = CounterCheckSink() if a.check_counter else None
    per_sample = 4 if a.channel == A.CH_BOTH else 2
    seg_period = A.SEG_BYTES / (A.BASE_CLOCK_HZ * per_sample)

    def progress(st):
        held = st.ring.n_resident if st.ring else 0
        print(f"  ... {st.n_read:5d} segments  {st.bytes_read/1e9:6.2f} GB  "
              f"lost {st.n_lost}  ring holds {held} "
              f"({held*seg_period:.1f} s)", flush=True)

    with A.Adc(tap=a.tap) as adc:
        st = Stream(adc, a.channel, nchan=a.channels, nslots=nslots,
                    ram_ring=use_ring)
        try:
            print(f"streaming ch={a.channel} via {a.channels} C2H channel(s), "
                  f"{A.seg_samples(a.channel):,} samples/segment, "
                  f"{seg_period*1e3:.1f} ms/segment")
            if use_ring:
                print(f"RAM ring {nslots * A.SEG_BYTES/1e9:.2f} GB = {nslots} "
                      f"segments = {nslots*seg_period:.1f} s; oldest evicted first")
            st.run(sink=sink, segments=a.segments, duration=a.duration,
                   progress=progress)
            ok = summarise(st, sink, label=f"ChannelSel={a.channel}")
            if a.dump and st.ring is not None:
                t0 = time.monotonic()
                n = st.ring.dump(a.dump)
                print(f"  dumped {n/1e9:.2f} GB to {a.dump} in "
                      f"{time.monotonic()-t0:.1f} s")
        finally:
            st.close()
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
