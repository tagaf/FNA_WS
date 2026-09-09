# AD9643 capture — resolved unknowns

The HDL sources are **not present on this machine** (no `.v`, `.bd`, or `.xci`
anywhere on the filesystem — they are still on the Vivado build host). Every
value below was therefore determined *empirically* against the live hardware,
not read from the sources. Method and evidence are given so each can be
re-checked against the HDL when it becomes available.

## 1. `Speed_Set` (0x04) — rate divider, not a decimator

    fs = 250 MHz / (Speed_Set + 1)          Speed_Set = 0  =>  full rate, 250 Msps

| Speed_Set | measured | implied fs | 250/(n+1) |
|---|---|---|---|
| 0 | 0.263 ms / 65536 | 249.06 Msps | 250.0 |
| 1 | 0.527 ms | 124.30 | 125.0 |
| 2 | 0.790 ms | 82.93 | 83.3 |
| 4 | 1.313 ms | 49.90 | 50.0 |
| 8 | 2.363 ms | 27.73 | 27.8 |
| 16 | 4.461 ms | 14.69 | 14.7 |
| 32 | 8.655 ms | 7.57 | 7.58 |

Confirmed at scale: 16777216 samples at Speed_Set=0 took 67.110 ms, exactly
16777216 / 250 MHz. **Use 0.** As the handoff notes, this drops samples with no
anti-alias filtering, so any non-zero value is metrologically wrong.

## 2. `Channel_Set` (0x08)

| Value | Meaning |
|---|---|
| 0 | **Internal test ramp — not ADC data.** Perfect 0..16383 sawtooth. |
| 1 | ADC channel A (real noise: mean ~186, std ~7.4) |
| 2 | ADC channel B (real noise: mean ~142, std ~4.2) |
| 3 | **Invalid — hangs the capture FSM.** Recoverable, see below. |

Channels 1 and 2 return data that differs between identical back-to-back
captures (real noise); channel 0 is bit-identical every run (a counter).
The ramp counter increments once per *captured* sample and is unaffected by
`Speed_Set`, which places it downstream of the decimation point — i.e. it is
generated inside the FPGA, so **channel 0 verifies the data path but says
nothing about the ADC link itself.**

## 3. `set_sample_num` (0x0C) — units and granularity

**1 count = one 16-bit sample = 2 bytes.** Measured by pre-filling DDR4 with a
known pattern and diffing after capture: N=65536 wrote exactly 131072 bytes,
N=1048576 wrote 2097152 bytes, etc. — 2.000 bytes/count throughout.

**N must be a multiple of 256 samples** (512 bytes = 32 AXI beats × 16 bytes,
consistent with `AXI_BURST_LEN = 31` meaning AWLEN=31 → 32 beats), and ≥ 256.
Other values misbehave silently:

| N | finish? | bytes written | expected 2N |
|---|---|---|---|
| 128, 192 | no | 0 | hangs, below one burst |
| 255 | yes | 512 | 510 — rounded up |
| 257 | yes | 512 | 514 — truncated down |
| 384 | **no** | 512 | 768 — wrote a partial buffer, then hung |
| 1000 | **no** | 1536 | 2000 — partial, then hung |
| 256/512/768/1024/1280/2048 | yes | exact | ✔ |

The driver rejects non-conforming values rather than letting them hang or
return a quietly wrong length. Max N = 262144000 (the 500 MB window).

## 4. Sample packing in DDR4

14-bit **unsigned**, right-aligned in 16-bit little-endian words; bits 15:14 are
always 0 (verified over 16.7 M samples). Samples are strictly sequential in
increasing address order — 8 per 128-bit AXI beat. **There is no channel
interleaving**: the channel is chosen per capture by `Channel_Set`, so a buffer
holds one channel only. Confirmed two ways: the ch=0 ramp increments by exactly
1 across the whole buffer with zero discontinuities, and for real ADC data the
even- and odd-index means agree (186.77 vs 186.16), which they would not if two
channels were interleaved.

Parse with `numpy.fromfile(path, dtype='<u2')` — no masking needed.

## 5. server.py hard-freeze — root cause found, fixed 2026-08-26

`gpu.FastC2H.read_into()` (raw `os.readv()` against `/dev/xdma0_c2h_0` into a
preallocated numpy buffer) hard-freezes the Orin — SSH, display, and keyboard
all die simultaneously, nothing reaches `dmesg`/`journalctl` even on boots
with persistent journald, and only a power cycle recovers it. `last -x reboot`
showed five such freezes in ~30 minutes while this was being chased.

An earlier fix attempt assumed the trigger was DMA'ing into CUDA-pinned
(`cudaHostAlloc`) memory specifically (nvmap/SMMU mapping conflict with the
PCIe DMA client) and routed the read through a plain malloc'd staging buffer
instead. `bisect_dma.py` (writes an fsync'd marker before each stage so the
culprit survives the freeze/reboot) disproved this: it died at **stage S4**,
`FastC2H.read_into()` into a *plain* array — it never even reached S6, the
CUDA-pinned case the fix targeted. `bisect_marker.txt` still reads S4; no rerun
has gotten further. So the raw XDMA character-device read is unsafe
regardless of destination buffer type — the pinned-vs-plain theory in
`gpu.py`'s old docstring was incidental, not the real cause.

Stage S3 — reading back via the vendor `dma_from_device` CLI tool as a
subprocess into a tempfile (`ad9643.ddr_read_samples()`) — completes reliably.
**`server.py` now uses that path exclusively** and no longer imports/uses
`FastC2H` at all. `gpu.FastC2H` is kept only for future bisection, clearly
marked unsafe.

Cost: readback now pays subprocess + tempfile overhead per frame instead of a
zero-copy DMA read, so frame rate is lower than the original design intended.
This has not been benchmarked yet. If raw-DMA readback speed is wanted later,
the open question is *why the vendor tool's own DMA read survives where an
equivalent-looking Python `read()`/`readv()` does not* (buffer alignment?
ioctl setup? a flag the CLI passes that the Python path doesn't?) — that
needs to be understood before re-attempting a fast path, not another guess.

## 6. Intermittent stall at very large (near-MAX_SAMPLES) transfers — open

While raising the UI's sample-size ceiling to the true `MAX_SAMPLES`
(262,144,000, the 500 MB window) on 2026-08-26, single-shot captures at that
size were tried repeatedly against the live `server.py`:

- Standalone diagnostics (`diag_big.py`: raw `Adc` capture + `ddr_read_samples`
  only, no server/threading; `diag_gpu.py`: `gpu.Spectrum` alloc + `process()`
  only, dummy data, no FPGA) both ran the full size ladder up to 262,144,000
  samples cleanly and fast every time — capture+DMA ~1.8 s, GPU alloc+process
  ~0.4 s combined, nothing hung.
- Through the actual running server, the same size+config **hung twice**
  (once on a long-lived instance, once fresh) with no error, no timeout, no
  child `dma_from_device` process, and near-zero CPU across all threads —
  `/status` just stopped advancing indefinitely (90+ s observed, manually
  killed both times). SSH/the OS stayed fully responsive throughout — this is
  not the SMMU-wedge hard freeze from §5, just this one capture never
  returning.
- The **same exact request** (single-shot trigger, N=262,144,000, channel 0,
  after the engine had already been free-running) also **succeeded** in a
  third attempt, completing `_one()` in 2.739 s per debug instrumentation.

So it's real but not reliably reproducible, and the standalone tests point
away from the obvious suspects (raw capture/DMA, GPU alloc/process each work
fine alone at this size). Not yet understood: what specifically differs
between the hung and successful runs — GC pause, scheduler contention from
concurrent HTTP threads, something in the vendor `dma_from_device` tool at
this scale under load. **Mitigation shipped**: `ad9643._tool()` now passes an
explicit `timeout=` to `subprocess.run()` (floor 5 s, else `nbytes/50MBps + 5s`
— generous relative to the ~0.3-0.7 GB/s measured), raising `DmaTimeout` on
expiry instead of hanging forever; `server.py` catches it like
`CaptureTimeout` (counted, recoverable, engine keeps going). This bounds the
damage but does not explain the cause — if it recurs, `self.err` /
`/status` will now show `DmaTimeout: ...` instead of silence, which is the
next debugging foothold (note whether it fires at all, and how close to the
timeout bound it lands).

**Likely explanation found 2026-08-26, later same day**: almost certainly
**not** a bug in the single-client code path above. `ad9643.Adc` opens
`/dev/xdma0_user` and writes registers directly with no locking or
arbitration — it assumes exactly one process talks to the board. Every hang
during this investigation was produced by a standalone test script or a
second `server.py` instance run *while the normal live session was also
running continuously*, i.e. two independent clients racing writes to
`REG_START`/`REG_CHANNEL`/`REG_NSAMPLES` against the same hardware. Confirmed
directly afterward: running one more isolated instance alongside the live
session (this time deliberately, to extend the FFT-size range below) reliably
reproduced `CaptureTimeout` with `regs={...}` showing **a different channel,
speed, and nsamples than the instance's own config** — i.e. it was reading
back the *other* process's register writes. Standalone diagnostics
(`diag_big.py`, `diag_gpu.py`, `diag_nfft.py`) never hung because they were
always run with no second capturing client active.

**Conclusion: `server.py` (or anything using `ad9643.Adc`) must not be run as
more than one concurrent instance against the same board.** If a second
instance is needed for testing, either stop the first one first, or extend
`Adc` with a lock file / single-instance guard — not attempted here since the
finding came late in this session. The `DmaTimeout` mitigation above stays
regardless (defends against a different, real failure mode: the DMA
subprocess itself stalling), but it was not the explanation for the original
hangs.

**`DmaTimeout` confirmed firing for real in normal single-instance use,
2026-08-26 (later)**: user reported hitting it "often" at the *default*
config (1,048,576 samples / 2,097,152 B — nowhere near MAX_SAMPLES). Checked
live: exactly one `server.py` instance running (concurrent-access explanation
above ruled out for this case), no PCIe/XDMA errors in `dmesg`, load average
normal, 3 timeouts in 3,014 frames. Resumed continuous capture at the same
config and ran 4,500 more frames (~100s at ~46fps) immediately after with
**zero** recurrences — genuinely rare and not reproducible on demand, but
real and size-independent (contradicts the earlier "near-MAX_SAMPLES only"
framing — see updated `DmaTimeout` docstring in `ad9643.py`). Root cause
still unknown.

Practical consequence: the timeout bound had a flat `+5.0s` margin sized for
worst-case (500 MB) transfers, so a stall on a typical few-MB capture made
the ~50 fps continuous loop stutter for a full 5s before recovering.
Tightened to `+1.5s` (1.0s absolute floor) — still >100x the normal small-
capture time, but recovers far faster when it does fire. If `DmaTimeout`
starts firing noticeably *more* often after this, that's a real signal (the
margin is now tighter); if the rate looks the same, the margin was pure
overhead being cut for no cost.

## Operational notes

- **A hang is recoverable.** Re-arming with a valid channel and re-triggering
  walks the FSM out of `ADC_SAMPLE`; no power cycle needed. `Adc.recover()`
  does this, and `capture()` calls it automatically before raising.
- `Adc_Finish` **idles high** and stays high from the previous run, so polling
  it immediately after a trigger can produce a false "complete". Confirm the
  bit drops first, or measure bytes written, when testing new parameters.
- Data path integrity: 16777216-sample ramp capture, **0 discontinuities**.
- Throughput measured here: 32 MB capture + readback in ~0.44 s wall clock.

## Not verified

**No signal generator is connected to this machine** (no USBTMC, no serial
instrument, no pyvisa) and there is no analogue front end on the board. The
amplitude/frequency check called for in the handoff — apply a known tone,
confirm the FFT peak lands where it should — **has not been done**. Until it is,
the following remain open:

- whether channel 1/2 really map to ADC A/B in that order;
- the code-to-volts scaling and whether the format is offset-binary or
  two's-complement (the observed means of ~186 and ~142 sit near the bottom of
  the 14-bit range, consistent with unterminated floating inputs, but a real
  tone is needed to confirm);
- absolute frequency accuracy of the 250 MHz encode clock.

The current FFT peaks (107.17 MHz on A, 50.00 MHz on B) are noise-floor
artefacts at −62 and −67 dB, not signals.

## 8. Efficiency + fluency pass, 2026-09-03

**"avg=1 still averages" — root cause.** Two averaging stages exist: the CUDA
`k_pow` kernel averages |X|² across `max_frames` Welch segments *inside every
capture*, and the Python EMA averages *across* captures. The UI only exposed
the EMA ("Avg depth"); `max_frames` was hard-wired to 64, so avg=1 still
showed a 64-segment average. Fixed by exposing it ("FFT frames"); validated
end-to-end in mock mode: Welch=1 gives 4.9 dB frame-to-frame noise-floor
wobble (a true single FFT), Welch=64 gives 0.7 dB — matching χ² theory.
avg=1 in the EMA now also hard-copies instead of blending.

**Web stutter — three independent causes fixed:**
1. DMA tempfiles lived on /tmp (disk-backed): megabytes/frame through the
   page cache; periodic writeback = multi-hundred-ms stalls. Moved to
   /dev/shm, one persistent file, `readinto` a preallocated buffer
   (`ad9643.ddr_read_into`) — no per-frame allocation.
2. Nagle + delayed-ACK on the HTTP socket: up to ~40 ms per small reply,
   and the UI polls /status every animation frame.
   `disable_nagle_algorithm=True`; /status now ~1.3 ms.
3. The client reallocated both canvas backing stores every frame
   (`cv.width=` resets the whole canvas); now only on genuine resize.

**Resident DMA helper (`native/xdma_shm_reader`), opt-in `--fast-dma`.**
Replicates the vendor tool's device access *exactly* — same
`open(O_RDWR|O_TRUNC)` (the Python freeze path used O_RDONLY; the vendor
comment says O_TRUNC tells the driver to flush — a real candidate for the §5
divide), same posix_memalign(4096) bounce buffer, same chunked lseek+read
loop and RW_MAX_SIZE — but stays resident and memcpys into a /dev/shm
mapping instead of spawning a process + writing a file per frame. One vendor
bug fixed on the way: `read_to_buffer` only seeks `if (offset)`, which is
wrong once the fd persists across reads (addr=0 after a prior read starts
from the wrong position); the helper always seeks. Protocol validated
against a pattern file (incl. that regression). **NOT yet validated against
the real device** — run `validate_fast_dma.py` (server stopped) once; only
on PASS use `--fast-dma`. Any runtime helper failure permanently falls back
to the vendor CLI and is reported in the `dma_path` metric.

**Other:** CUDA mid-pipeline `cudaStreamSynchronize` removed (mean consumed
on-device); EMA accumulator float32 (halves memory, matters at 128M-point
FFTs); noise-floor median strided (full median over 67M bins was ~0.5 s per
frame); stale `engine.err` now clears on the next successful capture;
`--mock` runs the whole server on a synthetic 25 MHz tone without opening
/dev/* (65 fps measured; also how this pass was tested — per standing
instruction the live server is only ever started by the user).

## 9. Stutter root-caused to the WiFi link; long-poll delivery, 2026-09-03

Probe of the live service (35 s, localhost): engine inter-frame p99 within
8% of median, zero skipped frames — production and local delivery are
smooth. The link is not: wlP1p1s0 at **-75 dBm, tx bitrate fallen to
6 Mbit/s, power save ON** — periodic PS wakeups/retry bursts are the
every-few-seconds stutter. All viewing traffic rides this uplink (no
Ethernet configured; Tailscale rides the same RF).

App-side hardening shipped anyway: `/frame?wait=1&since=<n>` long-poll
(server parks the request on a Condition until a newer frame exists, 204
after 25 s) replaces the status-poll-per-animation-frame client — ~10x
fewer round-trips, and an RF hiccup now delays one response instead of a
burst. Mock-verified: 1 request/frame, parks while paused, wakes <100 ms
after resume.

Same pass: `trace_samples` decouples the time trace from Welch depth (new
trace_n arg through adc_process/k_env; changing FFT frames no longer
changes trace span or shading — mock-verified identical envelopes at
Welch 1 vs 256); readback = max(FFT need, trace, explicit), rounded UP to
granularity, clamped to N; time-axis labels auto-scale ns/µs/ms/s; sample
and trace selector labels show real duration at the current divider; the
Samples control is now honestly "capture depth" (metrics show
acquired/FFT/trace splits — at N=67M with Welch=1×4096 only 0.006% of the
record was ever used, which is why the control "did nothing").

## 10. Control-matrix test; EMA cross-config bug; samples→FFT link, 2026-09-03

Mock-mode control matrix (11 configs × invariants: fs, nbins, bin_hz,
display-span == fs/2, displayed-argmax vs expected tone incl. alias at
divided rates, trace/read/fft splits, nframes, zoom slice mapping,
single-shot, ch=3 reject/recover): ALL PASS. MockDma now emulates the
FPGA's sample-dropping decimator (reads REG_SPEED from MockAdc), so
divided-rate cases assert physically correct alias frequencies.

The matrix caught a REAL bug: after a config change, frames produced under
the OLD config could re-seed the display EMA after configure()'s reset;
spectra from different fs/nfft share bin indices, so a stale 6.25 MHz peak
(speed=7) displayed as 50 MHz at speed=0 for ~avg frames. Fix: the
accumulator carries a (fs, nfft, channel) tag and reseeds on mismatch —
"peak at wrong frequency right after changing speed/nfft" is gone.

"Changing samples doesn't change the FFT axis": correct observation, by
design frequency span depends only on fs — but the user wants record
length to drive RESOLUTION. New default nfft = "auto — follow samples":
nfft = pow2floor(min(N, 2^24)), sent by the UI whenever samples change, so
more samples → finer bins → log axis reaches lower (a real "longer
analysis"). Manual sizes still selectable.

Header now shows the Orin's own WiFi link (RSSI bars + tx bitrate via
/proc/net/wireless + iw, 5 s cache, in sys.net) plus a client-measured
"lag" (frame arrival delay beyond engine loop time, EWMA). On frame-rate
adaptation: the long-poll transport is already self-pacing — the client
pulls and the server hands the NEWEST frame, so a slow link yields fewer,
current frames rather than a growing backlog; the badge makes that state
visible instead of mysterious.

## 11. Resolution follows the record, 2026-09-03 (cont.)

User expectation formalised: Δf = 1/T_analysed. NFFT_AUTO_CAP raised
2^24 → 2^27 after GPU validation (134,217,728-pt R2C: 116 ms process,
~1 GB, on 61 GB total) — auto nfft = pow2floor(N) now over the ENTIRE
sample range, so a longer record genuinely yields finer bins all the way
to the 500 MB window (N=262M → 134M-pt FFT → Δf 1.86 Hz). `need` is now
capped by the frames the record can supply (min(max_frames, N//nfft)·nfft)
so nfft≈N no longer forces a full-window DMA it cannot use. Spectrum
header states the law explicitly: "N-pt (T → Δf) · analysed X of Y".
Mock proof: 16.7M-pt full-record FFT resolves the 25 MHz tone to 0.3 bins
(±4.5 Hz) at Δf=14.9 Hz. Semantics recap — Samples: acquisition depth
(DDR record, sets available T). FFT auto: analysis length (resolution).
Trace length: time-plot display window only. Welch frames: how many
nfft-segments of the record are averaged (variance ↓, resolution
unchanged). Decimating a record into a small FFT would instead shrink
span fs/2 and alias (no AA filter) — deliberately not offered.

## 12. The 31 kHz floor; trace knob removed, 2026-09-03 (cont.)

User: "minimum of the FFT spectrum is always 31 kHz". Correct — a display
artifact: the overview sent to the browser was 4096 points spaced LINEARLY
over 0..fs/2, so its first point sat at 125 MHz/4096 = 30.5 kHz no matter
how long the FFT. The added resolution existed in the data (zoom proved
it) but never on the overview axis. Fix: `log_display()` — the overview is
now 4096 points uniform in log(f) from the TRUE first bin (Δf) to Nyquist
(max-pooled per point, so peaks survive), and the client maps them
linearly onto its log axis. The axis now genuinely starts at Δf: 14.9 Hz
at a 16.7M-pt FFT, 1.86 Hz at 2^27. Metadata disp_bin_hz replaced by
disp_log/disp_f0/disp_f1; matrix test asserts disp_f0==bin_hz and that the
displayed argmax lands at the correct frequency (incl. coarse-FFT cases
where one bin spans several log points).

Trace length control removed (user: samples+rate already define the
record). The time plot now always shows the WHOLE record: trace_samples
default -1 = follow N (explicit counts still accepted over /control).
The knob was compensating for auto-readback reading less than the record;
with nfft following the record that readback happens anyway.

## 13. Peak detection + noise classification (branch), 2026-09-04

New `noise.py` (pure NumPy/SciPy, hardware-free, tested against synthetic
spectra in `tests/test_noise.py`) and `tools/emi_sweep.py` (drives the live
server across conditions and cross-references). Windows became selectable in
`cuda/adcfft.cu` first, because Hann's -31.5 dB sidelobes make leakage skirts
indistinguishable from spurs.

**Windows** (harris 1978 coefficients; measured vs theory, all exact):
ENBW 1.50/2.00/3.77/1.00 and scalloping 1.42/0.83/0.01/3.92 dB for
Hann/Blackman-Harris/flat-top/rect. All four give identical on-bin amplitude,
confirming per-window coherent-gain normalisation (`norm = 2/sum(w)`, was
hard-coded 4/nfft for Hann). Near-sidelobe measurement at a half-bin offset:
at +25 bins Hann is -91.5 dB while BH4 is -133.5 dB -- 42 dB less leakage,
which is the difference between detecting spurs and detecting a skirt.

**Detection** is CFAR, not a fixed dB margin. Welch-averaged power is
Gamma(K, mean/K), so the threshold for a target Pfa is
gammainccinv(K,Pfa)/K; verified K=1 -> 11.40 dB = -ln(Pfa), and measured
false-alarm counts track the requested Pfa over 2^20 bins. The floor
estimator corrects median->mean for Gamma(K,1/K) (~1.6 dB at K=1) -- without
it the achieved Pfa is not the requested one. Sub-bin refinement is
parabolic-on-dB (magnitude only: Welch averaging discards phase, so Candan/
Jacobsen/Quinn are unavailable); measured error <=0.016 bin.

**The harmonic sieve needed three defences against overfitting**, all found
by testing rather than reasoning:
1. Scoring every candidate BEFORE claiming peaks. Greedy largest-first let
   2*f0 steal every even harmonic and split one comb into three families.
2. Density measured over harmonics 1..h_max(observed), not to the band edge
   -- otherwise 50 Hz mains scores 7/64 and is rejected, while a
   half-fundamental impostor is still correctly penalised at ~0.5.
3. Chance correction: with P peaks over bandwidth B a predicted harmonic
   matches by luck with p ~ 2tP/B, so a denser predicted grid collects
   accidental members. Score on excess-over-chance in sigma. Plus a two-pass
   LS refit of f0 with tightened tolerance. On live data this cut 15
   families to 6 and removed all near-duplicates.
4. Family merging must bound the integer ratio (<=8): 497.6 kHz is exactly
   9952 x 50 Hz, so an unbounded test folds the switcher into mains.

**Live result**: the pipeline independently recovers the 497.6 kHz comb
found by hand in section 12's investigation (density 0.90, 36 harmonics,
-86 dBFS) AND a second switcher at **532.85 kHz** (density 0.91, 59
harmonics) that the manual pass missed, common to both channels.

**Open / known limitation**: cross-fs inference is ambiguous for wideband
combs. At speed=1 the fitted fundamentals are exactly half those at speed=0.
Two readings -- (a) harmonics above the new Nyquist fold back, adding lines
so the sieve legitimately locks to f0/2, or (b) the disturbance is periodic
in sample index rather than time. Not resolved; the tool now reports this as
AMBIGUOUS rather than claiming either. The termination experiment is the
unambiguous discriminator and is the next measurement to run.

## 14. Classification wired into the live plot, 2026-09-04

`analyse()` costs 103 / 325 / 557 ms at 2^20 / 2^22 / 2^24 bins, so it cannot
run in a ~50 fps capture loop. It runs in a dedicated worker thread that
takes a snapshot only when idle (so work can never queue faster than it
retires) and at most every `classify_period` (1 s default). EMI sources
drift slowly; a ~1 Hz classification under a live-rate display is the right
trade. Measured cost with an interleaved A/B/A/B (needed -- a naive
before/after showed a bogus 57% because the first sample was warm-up):
**0.0% of frame rate**.

Frames carry an `analysis` block (families with member frequencies/levels,
unmatched spurs, floor and CFAR threshold) trimmed to 8 families x 96
members + 32 spurs, so it never dwarfs the spectrum payload. It is tagged
with the fs/nfft/channel it was computed from; the UI refuses to draw
markers when those disagree with the current frame, since an analysis of a
different spectrum would put ticks at meaningless places.

UI: coloured ticks at the top of the spectrum plus a dot on the trace for
every family member, one hue per family, grey for unmatched lines, and a
"Noise sources" card listing each family (f0, label, harmonic count,
density, significance, peak level). Enabled by the `Classify` control
(default off).


## 15. Marker sync + sieve made O(n log n), 2026-09-04

Two problems, reported as "markers do not update every FFT refresh".

**1. find_families was quadratic-ish.** `collect()` ran a Python loop doing
`argmin` over the whole peak array per harmonic: O(candidates x harmonics x
peaks). Fine for the tens of peaks in the unit tests, hopeless on a real
spectrum -- a live capture yields ~1500 peaks and ~18k candidates, i.e.
billions of comparisons and minutes per pass, which is why the first live
analysis took so long to appear. Replaced with `searchsorted` over the
already-sorted frequencies plus fully vectorised tolerance tests, and
candidates are now deduplicated onto a resolution grid (raw 1 mHz rounding
kept thousands of candidates differing by far less than the match
tolerance) with pairwise differences seeded only from the strongest 400
peaks. Measured 1600 peaks: minutes -> 546 ms. Full analyse() end to end:
0.17 / 0.51 / 1.28 s at 2^20 / 2^24 / 2^26 bins.

**2. Marker levels must not wait on the structural pass.** Family
*frequencies* drift slowly, but their *levels* change every frame, so
markers drawn from a 1 Hz analysis visibly lag the trace. The server now
re-samples the stored family/spur frequencies against the CURRENT spectrum
on every frame (`_live_analysis`, a few hundred lookups) while the worker
keeps refreshing the structure in the background. Frequencies are physical,
so this also survives an nfft change -- only the bin mapping moves. fs or
channel changing does invalidate the structure (aliasing differs; different
source), and that now returns an explicit `stale` flag the UI renders as
"reanalysing" instead of drawing ticks from the wrong spectrum.

Verified in mock: marker level changes frame-to-frame, structure survives
2^20 -> 2^18, channel change flags stale.

## 16. Marker amplitudes vs the drawn curve, 2026-09-04

Reported as "peak amplitudes often do not match the plot", suspected
averaging. Measured on live data instead: the discrepancy is strongly
frequency-dependent and it is not averaging.

    0.5 - 7.5 MHz    marker - plot = +0.00 dB   (exact)
    118 - 124 MHz    marker - plot = -1.0 to -4.5 dB

Cause: the overview is 4096 points spaced in log(f) and MAX-pooled, and the
client max-pools again per pixel. Below ~10 MHz a display point covers about
one bin so marker and curve coincide exactly. Above ~100 MHz one point pools
thousands of bins and the curve draws their maximum, while the marker drew a
single bin at the harmonic frequency -- arithmetically correct but visually
floating several dB under the envelope it annotates.

Fixes:
* Dots are now drawn at the value the polyline actually has at that pixel
  (`plotAt()`), so they sit on the curve by construction at every zoom.
  Zoomed in, pixel ~ bin and this degenerates to the true per-bin level.
* `levels()` takes a local max over +-2 bins rather than one rounded bin:
  the fundamental is refitted only once per structural pass, so a drifting
  switcher moves between passes and rounding can land on a shoulder.
* `peak_db` (the number in the Noise sources card) was carried from the
  analysis snapshot and never refreshed -- up to 2.9 dB adrift from the
  plot. It is now recomputed from the live levels each frame.
* The card states that levels are true per-bin peaks while the zoomed-out
  curve is a max envelope, so the two are read correctly.

## 17. Why most peaks were never marked, 2026-09-04

Reported as "not all the peaks are found, many are not noted". Measured on
live data: detection was finding **2454 peaks** while only 32 spurs plus
capped family members reached the browser. Two independent causes.

**1. Ranking by absolute level, not prominence.** `_shape_analysis` sorted
spurs by dB and kept the top 32 -- and every one of those 32 landed in
103-112 MHz, because that band's broad hump sits highest in absolute terms.
Genuinely isolated lines everywhere else were dropped. Ranking is now by
prominence above the local CFAR threshold, throughout: `detect_peaks`
trimming, spur ordering, and family-member selection. A modest line standing
clear of its neighbourhood is the more notable feature, and prominence is
also what the eye responds to.

**2. Wire caps sized for a per-frame payload.** Members were capped at 96
(one live family had 283) and spurs at 32, because the frequency arrays rode
in every frame. Since markers are now drawn at the polyline's own value
(section 16), the client needs no per-frame levels at all -- only the card's
`peak_db`. So the structure moved to a separate `GET /noise`, tagged with a
structure id (`sid`) that the frame carries; the client refetches only when
`sid` changes. Caps are now 12 families x 512 members and 600 spurs, and the
per-frame header actually *shrank* (frequency arrays no longer ride it).
`why` is a whole sentence that repeated across hundreds of spurs -- sent once
per label as `why_by_label`.

Also: near-duplicate detections within 3 bins (one broad peak with a dip)
are collapsed, keeping the most prominent of each cluster. Spur markers are
now drawn with size and opacity scaled by prominence, so density does not
flatten into undifferentiated clutter.

## 18. Marking every detected peak, 2026-09-04

`MAX_WIRE_SPURS` raised 600 -> 20000 and members 512 -> 4096, i.e. every
detected peak is now marked. Three changes made that affordable:

* **Detection and sieving decoupled.** `analyse(max_peaks=...)` bounds what
  is detected (and markable); `family_peaks=1200` bounds what is fed to the
  harmonic sieve, whose cost grows with peak count. Marking wants
  completeness, sieving wants the prominent lines that define a comb. The
  sieve input is a slice of the same list, so member identity (used by
  `classify`) is preserved.
* **Compact wire format.** `/noise` sends parallel arrays (`spur_f`,
  `spur_p`, `spur_l`) with a label table instead of per-peak objects with a
  repeated `why` sentence: measured 14.1 B/marker vs 51 B/marker, 3.6x
  smaller. 362 markers = 5.1 kB = ~7 ms on the 6 Mbit/s link, fetched only
  when `sid` changes.
* **Per-pixel reduction when drawing.** Thousands of peaks against ~1200
  plot pixels would overdraw and cost canvas calls for nothing. The client
  keeps at most one marker per pixel (the most prominent); zooming reveals
  the rest because the full list is already client-side.

**Known limitation (unresolved).** On a synthetic comb with steeply decaying
harmonics, the sieve split one 497.6 kHz source into families at 995.2 kHz
(2f0) and 1492.8 kHz (3f0): once the upper harmonics fall below threshold
the surviving subset is sparse and irregular, and the multiples score better
per member. `_merge_families` only folds INTEGER ratios, so 995.2 and
1492.8 (ratio 3/2) never merged. A rational-ratio consolidation (find small
p/q, test whether f0_a/p explains the union better) would fix it. Live
hardware data does NOT show this -- there the 497.6 kHz family is recovered
whole with 283 members -- so it is a sparse-comb failure mode, not a general
one.

## 19. Acquisition cost and detection coverage, 2026-09-04

**"Classification slows acquisition."** Measured on the live server with the
current build, medians over 14 frames, config restored afterwards:

    nfft=2^20   loop 18.6 -> 18.4 ms   (-1%)
    nfft=2^24   loop 168.1 -> 168.9 ms (+0%)

So the current build costs nothing; the slowdown was the PREVIOUS one, which
serialised thousands of member frequencies into every frame (fixed by the
/noise split, section 18). Two latent hazards hardened anyway:
* `levels()` was a Python loop over every family member, run per frame.
  Harmless at 96 members, but the cap is now 4096 x 12 families = ~49k
  iterations/frame. Replaced with a single gather over an (n x 5) index
  matrix.
* The worker snapshot is a full-spectrum copy (134 MB at 2^27) taken in the
  capture loop; its period now scales with spectrum size (~1% of a frame's
  budget) instead of a fixed 1 Hz.

**"Not all peaks are marked."** Measured: 100% of DETECTED peaks now reach
the browser (1524 sent of 1524 found). But only 35% of *visible bumps*
(>=4 dB over a rolling local median in the drawn overview) were marked, and
the unmarked ones stood 4-10 dB over that median against a CFAR threshold of
+11.40 dB. That threshold is not arbitrary: at avg=1 / welch=1 the power is
exponentially distributed, so -ln(1e-6) = 11.40 dB is exactly the bar for
one false alarm in a million bins. A 10 dB bump in a single periodogram
genuinely IS plausible noise.

Rather than quietly loosening it:
* The CFAR threshold is now DRAWN over the spectrum (dashed amber, 512-point
  curve shipped on /noise), so "why is that peak unmarked?" is answerable by
  looking.
* A **Sensitivity** control exposes Pfa (1e-2 .. 1e-9). Verified 1e-6 -> 1e-2
  drops the threshold 11.40 -> 6.63 dB and takes peaks 2 -> 4331.
* The card states that averaging is the physical fix. Verified: welch 1 ->
  32 drops the threshold 11.40 -> 3.17 dB, matching cfar_alpha_db theory
  exactly, because real peaks survive averaging while noise does not.

Measurement note: two intermediate readings here were artifacts of the test,
not the code -- a naive fps A/B again showed a bogus 55% (the control run
afterwards came back lower than the treatment), and a Welch threshold looked
stuck at 11.40 dB because the frame still carried the PREVIOUS analysis.
Both needed waiting for a fresh `sid` / interleaved sampling to see straight.

## 20. Log axis for wide zooms; control sync; UI syntax gate, 2026-09-04

**"Few peaks at 10-100 kHz in the overview, lots when zoomed -- is it
correct?"** The overview is correct. Measured marker distribution:

    band            bins available   markers
    1-10   kHz                  37         3
    10-100 kHz                 377         2
    100 kHz-1 MHz            3,774        28
    1-10   MHz              37,748       286
    10-125 MHz             482,344     6,098

33 of 6417 markers sit below 1 MHz -- there really is almost nothing there,
and at 238.4 Hz bins a 4.19 ms record only reaches bin 42 by 10 kHz. What
looked like "lots of peaks" on zoom was the 1-14 MHz content: the zoom
spanned 10 kHz-14 MHz but was drawn on a LINEAR axis, giving the first
decade 0.6% of the width. `useLog` now depends on the SPAN rather than on
being zoomed (log whenever hi/lo >= 10 and lo > 0), which also fixes the
"10k -> 2.7M" tick jump reported as a weird x scale.

**Control desync.** The screenshot showed "Classify: off" beside a populated
Noise sources card: on load the selects showed their own hardcoded defaults
while the server kept whatever configuration it was already running. The
panel now adopts the server's values from the first frame.

**tests/test_ui.py added.** The UI is one 700-line inline <script> edited by
string replacement; a syntax error takes the page down with no server-side
symptom, and references to payload fields removed during the /noise split
fail silently as `undefined`. The test node --check's the extracted script,
greps for known-dead field references, verifies every `$('id')` exists in
the markup, and checks each server-side control is wired.

## 21. "GUI switches between live and disconnected", 2026-09-04

Not a crash -- the journal showed no unhandled server exits (the restart
counter was my own `systemctl restart`s). The cause is bandwidth:

    frame payload  26,548 B
    engine rate    66 fps
    demand         14.0 Mbit/s
    Orin WiFi link ~7 Mbit/s (RSSI -76 dBm)

The client asked for every frame the engine produced, i.e. twice what the
link can carry, so TCP queued until fetches failed -- and a single failed
fetch flipped the badge straight to "disconnected". Four fixes:

* **Client pacing.** Requests are now spaced by an EWMA of how long a frame
  actually takes to arrive (floor 33 ms, ceiling 1 s). The long-poll always
  returns the NEWEST frame, so backing off simply skips frames, which is the
  correct behaviour on a slow link. The badge reports "N fps shown" when
  pacing is active.
* **Payload halved (wire format 2).** Spectrum as int16 hundredths of a dB
  (0.01 dB steps against a display resolving ~0.1 dB); envelope as uint16,
  which is EXACT since those are 14-bit ADC codes. 26,548 -> 14,288 B, so
  30 fps costs 3.4 Mbit/s instead of 6.4.
* **Badge hysteresis.** Three consecutive failures before "disconnected";
  one hiccup shows "reconnecting" and retries with backoff.
* **Quiet client disconnects + shorter park.** `_send` swallows
  BrokenPipe/ConnectionReset (a client vanishing mid-response is normal and
  was logging a traceback each time), and the long-poll park dropped 25 s ->
  10 s, since a 25 s idle connection is prime NAT/AP reaping material.

Also moved `SysMon` off the capture loop: it read /proc/net/wireless
(~1.5 ms) every frame and shelled out to `iw` (~4 ms) every 5 s, all inline.
It now samples on its own 1 Hz timer and the loop reads a cached dict.

## 22. Envelope shading, and what a shorted input should read, 2026-09-04

**Shading.** The time-domain plot decimates the whole record into 1024
columns; the shaded band is min..max of every sample in that column and the
line through it was (min+max)/2 -- the envelope MIDPOINT, which for
asymmetric data sits where no sample is. Measured on a deliberately skewed
test signal, that line was 434 codes away from the true mean. `k_env` now
also reduces a per-column mean (validated against numpy: max error 0.0000
codes) and the line draws that. Wire format 3 carries it as uint16 in
quarter-code units (0.25 code quantisation). The header now states
"shaded = min..max per column, line = mean".

**"Shorting the input gives much larger values -- is the conversion wrong?"**
The conversion is almost certainly right, and the observation is evidence
FOR that rather than against it. The bit extraction is proven by the ramp
test (section 4: a perfect 0..16383 sawtooth with bits 15:14 always clear).
What was never established is the code-to-input mapping.

For 14-bit OFFSET BINARY, mid-scale 8192 = zero differential input. Shorting
the inputs together IS zero differential, so it should read ~8192. A
floating differential input has no defined common mode and drifts to a rail,
which is exactly what is seen: ch2 currently reads mean 139.9, i.e. 98.3%
below mid-scale, hammered against the bottom code. So floating ~140 ->
shorted ~8192 is the correct behaviour of a correctly decoded offset-binary
converter, and the jump is ~58x.

If the shorted reading is near 8192 this RESOLVES the open format question
from section 4 (offset binary vs two's complement) -- record the number. If
it is near 0 or full scale instead, that would be the anomaly. A new
"vs mid-scale (8192)" row in the Signal panel makes the comparison direct.

## 23. Shorted input reads ~12000, not 8192 -- open, 2026-09-04

Reported: shorting the channel gives ~12k codes. Floating reads 186 (ch1) /
140 (ch2). Neither coding hypothesis explains 12000 as "zero input":

    offset binary   zero differential -> 8192   (12000 is +3808 off)
    two's complement zero differential -> ~0    (12000 is -4384 signed)

Three candidate explanations, in the order I would rank them:

1. **The input is not biased into the converter's operating range.** There
   is no analogue front end: no balun/transformer, no VCM bias network. The
   AD9643 has differential inputs that need a common mode near mid-supply
   (~0.9 V for a 1.8 V AVDD -- confirm against the datasheet). Shorting the
   connector to ground drives the common mode to 0 V, outside that range, so
   the output is not a meaningful "zero" at all. Floating is equally
   undefined and rails. On this reading nothing is wrong with the decode --
   the measurement itself is invalid.
2. **ADC -> FPGA LVDS capture misalignment.** Important: the internal ramp
   (channel 0) is generated INSIDE the FPGA, so every bit-packing fact
   verified from it (section 4) says nothing about the ADC link. A wrong DDR
   edge, swapped lanes or a shifted bit window would leave the ramp perfect
   while scrambling real samples.
3. A genuine decode error -- least likely, since bits 15:14 are always clear
   on real data, which is what right-aligned 14-bit should look like.

`diag_input.py` added to separate (1) from (2) without a signal generator:
code histogram (regular gaps = stuck/mis-ordered bit, IEEE 1241 histogram
test), per-bit toggle rates, even/odd split and lag-1/2/4 autocorrelation
(DDR edge/interleave errors alternate in a way converter noise does not).
Validated against injected faults: bit 3 forced low -> 42.9% missing codes
and a stuck-low flag; a corrupted DDR edge -> even/odd delta -700 and
acf lag1 -1.000 against -0.003 healthy.

Still unresolved; needs a run with the input actually shorted, and
ultimately a properly biased known signal.

## 24. Persistence / density display, 2026-09-04

Real-time-analyser style persistence added to the spectrum: every trace is
accumulated into a frequency x amplitude histogram and coloured by hit rate,
so a rare transient stays on screen beside the steady noise floor instead of
being averaged into it. Control in the spectrum header: off / 0.5 s / 2 s /
10 s / hold.

Accumulated entirely in the browser from the spectra already being sent, so
it costs no extra bandwidth -- which matters on a 7 Mbit/s link.

Three details that mattered:
* **Decay is time-based** (exp(-dt/tau)), not per-frame, so the look does
  not change when the frame rate does.
* **Cells saturate at 64 hits.** Without a cap a steady floor counts up for
  as long as the display runs while a one-off transient stays at 1, so the
  very events persistence exists to reveal fade to invisible -- worst in
  hold mode, where nothing decays. Verified: after 100 s of running, a
  single transient still renders at 15% brightness.
* **The vertical axis is held** while persistence accumulates (re-snapping
  only on a >6 dB change). The histogram is tied to an amplitude mapping; a
  drifting autoscale would smear and continually reset it. This is also how
  an analyser behaves with a set reference level.

Cost measured in node at full plot size (1450x280 = 406k cells):
0.49 ms accumulate+decay, 1.27 ms render with a precomputed colour LUT, so
~1.8 ms/frame, about 5% of one core at 30 fps.

## 25. Correction: the test-pattern origin is NOT settled, 2026-09-04

User: "I think the test pattern is generated inside the AD9643." Section 2
claimed the opposite, and that claim was weaker than it was written.

The argument was: the ramp steps by 1 at every `Speed_Set`, so it must be
generated after the decimation point, i.e. inside the FPGA. That only
follows **if the divider drops samples**. If it divides the encode clock
instead, the ADC converts more slowly and its own ramp also steps by 1 --
the observation cannot distinguish the two.

Aliasing test to decide which the divider does (ch1, 5-20 MHz median floor,
avg depth >=12, nfft 2^20):

    speed=0  fs=250.0 MHz   -117.89 dBFS
    speed=1  fs=125.0 MHz   -116.94 dBFS   (+0.95 dB)
    speed=3  fs= 62.5 MHz   -114.40 dBFS   (+3.49 dB)

Naive sample-dropping folds everything above the new Nyquist into the band
unfiltered. The 60-125 MHz region sits ~25 dB above the low-band floor here,
so folding would lift 5-20 MHz by tens of dB. It moves 1-3.5 dB. So the
divider is **not** a naive full-band decimator, which removes the basis for
the section 2 inference. (It does not cleanly prove clock division either --
for white noise at fixed nfft the dBFS floor should not move at all with fs,
and it moves a little.)

Remaining evidence for FPGA-internal: channel 0 is bit-identical run to run
and always starts at code 0, i.e. it is reset by the capture trigger. A
free-running ADC pattern would start at an arbitrary phase -- though an FPGA
could reset the ADC's generator too.

**Why this matters:** if the ramp really comes from the AD9643, it traverses
the LVDS/DDR capture, and a perfect 16.7M-sample ramp then PROVES that link
is bit-exact -- which would rule out capture misalignment as the explanation
for the ~12000 shorted reading (section 23) and point squarely at input
biasing. Decisive tests: read the AD9643's SPI registers if the FPGA exposes
them, or power down, remove the mezzanine, and see whether a channel-0
capture still completes.

## 26. HDL received 2026-09-08 -- three findings, one a real bug

Sources at ~/pcie_fpga_project. These settle several things that were
previously inferred, and one inference was WRONG.

### 26.1 `Speed_Set` does not decimate. fs is ALWAYS 250 Msps.

`speed_ctrl.v` divides `adc_data_en`, but that signal only gates the sample
COUNTER in `wr_ddr_ctrl.v`:

    if(ad_out_valid && adc_data_en) adc_sample_cnt <= adc_sample_cnt + 1;

The FIFO write enable is a different signal entirely
(`fifo_to_axi4_ctrl.v`):

    assign wr_fifo = dvalid && (ch_sel != 2'b11);

and `dvalid` = `ad_out_valid` = registered `ad_sample_en`, which is high on
EVERY adc_clk during ADC_SAMPLE. So DDR always receives full-rate samples.
The capture takes (Speed_Set+1)x longer only because
`write_ddr_done = (burst_cnt >= burst_num) && fifo_empty` -- the FIFO keeps
being fed until the sampling FSM stops, so completion waits on the counter.

**Section 1 was wrong**: fs = 250 MHz/(Speed_Set+1) is false, and every
frequency axis at Speed_Set>0 was out by exactly that factor. This also
explains, retrospectively, the two anomalies it caused: the EMI sweep's
fundamentals appearing to halve at speed=1 (section 25 -- same bin, wrong
assumed fs), and the aliasing test showing no folding (there is no
decimation to fold). `sample_rate()` now returns 250 MHz unconditionally and
the control is relabelled "Capture stretch".

### 26.2 The test ramp IS FPGA-internal -- confirmed, user's guess was wrong

`ad9643_14bit_to_16bit.v`:

    reg [13:0] adc_test_data;
    always@(posedge clk) adc_test_data <= ad_sample_en ? adc_test_data+1 : 0;
    ...
    2'b00: ad_out <= {2'd0, adc_test_data};

A counter in the FPGA, cleared when sampling stops -- which is exactly why
channel 0 always starts at 0 and is bit-identical run to run. So section 2's
conclusion stands and section 25's doubt is resolved: **the ramp never
traverses the ADC link, and proves nothing about it.** Capture misalignment
therefore remains a live hypothesis for the railed/glitching ADC data.

### 26.3 Dual-channel mode EXISTS: ch_sel = 2'b11

    2'b11: ad_out_comb <= {2'd0, s_ad_in1, 2'd0, s_ad_in2};

routed to a separate 32-bit `fifo_comb`, muxed into the same AXI writer
(`wr_fifo_comb = dvalid && (ch_sel==2'b11)`, and dout/empty/full/rd_count
all switch on ch_sel==3). Both channels are captured over the SAME time
window. Channel A occupies bits [31:16], channel B [15:0], so little-endian
uint16 order is B,A,B,A,...

Now exposed as "Channel 3 -- A + B simultaneously" with a "Show" selector
for which one to plot; both channels' statistics are reported. Marked
EXPERIMENTAL: an earlier ch_sel=3 attempt timed out, and the HDL's byte
accounting for this mode is ambiguous (`burst_num` is computed identically
to 16-bit mode while each write carries two samples), so the de-interleave
order and record length need confirming against hardware.

### 26.4 The LVDS capture is uncalibrated

`ad9643_md.v` samples the data with the raw DCO: `IDELAYE3` is
`DELAY_TYPE("FIXED")` with `DELAY_VALUE(0)`, `adc_clk` is a plain BUFG off
the incoming clock with no MMCM phase shift, and there is no training or
bitslip. `IDDRE1` takes channel B on the rising edge and channel A on the
falling. Nothing establishes a sampling point in the middle of the data eye,
so the capture margin is whatever the board layout happens to give.

Also worth noting: `wr_en` on both FIFOs is never gated by `wr_rst_busy` or
by `full`, so writes during reset-recovery or overflow are silently lost.

### 26.5 Measured with 50 ohm terminations fitted

    ch1(A): mean 16126.5  std 1972.6  min 0      max 16383
    ch2(B): mean 16321.6  std   16.3  min 6      max 16382

Both channels are railed at POSITIVE FULL SCALE, and channel A additionally
takes excursions all the way to 0 (a zero appears in 99% of 4096-sample
envelope columns; channel B: none). Railing is consistent with the input
being outside the converter's common-mode range -- a 50 ohm SMA termination
to ground gives 0 V common mode with no front end to bias it -- and rules
the current data unusable as a measurement either way.

## 27. THE DATA IS SIGNED. Vendor client received 2026-09-08

`~/pcie_client_sw` (Puzhi's Qt client, Windows-only) settles the format
question that has been open since the first handoff -- and shows that every
statistic this tool has reported was misinterpreted.

    #define ADC_FS_VOLTAGE   1.75
    #define ADC_MAX_CODE     8192.0
    #define VOLT_SCALE       (ADC_FS_VOLTAGE / ADC_MAX_CODE)
    int16_t a = ((int16_t)(raw[ri*2]   << 2)) >> 2;
    int16_t b = ((int16_t)(raw[ri*2+1] << 2)) >> 2;

`((int16_t)(x<<2))>>2` is a 14-bit **two's complement** sign extension. So:

* codes are SIGNED, -8192..+8191, not unsigned 0..16383
* full scale is +-1.75 V, i.e. **213.6 uV per code**
* in dual mode `raw[2i]` is channel **A**, `raw[2i+1]` is channel **B**
  (the opposite of what section 26.3 guessed from bit order)

Re-reading the terminated measurements with the correct sign:

    ch1(A) mean 16126.5 unsigned  ->   -258 codes  =  -55.1 mV
    ch2(B) mean 16321.6 unsigned  ->    -63 codes  =  -13.5 mV
    floating           186        ->   +186 codes  =  +39.7 mV

Nothing was railed. Both terminated channels sit tens of millivolts from
zero, which is what a terminated input should do. And the "spikes to 0" that
prompted this: unsigned 16383 is signed -1 and unsigned 0 is 0, so a signal
dithering either side of zero renders as violent jumps between the two rails
when read as unsigned. Channel A showed it far more than B simply because A
sat closer to zero (-55 mV with 1973 codes of apparent spread vs B's tight
16). It was a display artifact of the wrong sign convention, not hardware.

This also invalidates spectra taken since the terminations went on: the
wraparound injects huge discontinuities. Floating-input spectra (section 22
and earlier) were unaffected -- those codes were all small and positive, so
they never wrapped.

Fixed everywhere: `s14()` in the CUDA stats/window/envelope kernels
(validated -- data that read mean 7097/std 8116 as unsigned now recovers
mean +0.007/std 3.014, matching the truth), `ad9643.to_signed()/to_volts()`,
wire format 4 with a signed int16 envelope, and the time-domain axis now
reads millivolts rather than raw codes.

## 28. Can the vendor client run on the AGX? No.

`pcie_client_sw` is Windows-only:

* `xdma_public.h` includes `<Windows.h>`
* `pcie_xdma.cpp` uses SetupAPI (7 `SetupDi*` calls) to enumerate the driver
  by interface GUID, then `CreateFile`/`HANDLE`/`CloseHandle` (12 uses)
* the .pro links `-lopengl32 -lglu32`, plus `RC_ICONS`, a `.rc` resource and
  a C# launcher

Porting is nevertheless small: **only `pcie_xdma.cpp` (157 lines) is
OS-specific**, and its whole interface is four functions
(`xdma_node_open/read/write/close`) which map onto Linux `open`/`pread`/
`pwrite`/`close` on `/dev/xdma0_*` -- exactly what `ad9643.py` already does.
Qt and QCustomPlot are portable; Qt is not installed here
(`apt install qtbase5-dev`, 5.15 available). Not worth doing for its own
sake -- this repo's client already does more -- but the source is the
authority on register semantics and data format, as section 27 shows.

## 29. Dual-channel (ch_sel=3) does not work on this bitstream -- FPGA-side

Symptom: selecting A+B stalls, then `Adc_Finish low after 0.517s
regs={'start':1,'speed':0,'channel':3,'nsamples':1048576,'finish':0}`.

Established:

* The bitstream matches these sources. The .bit header reads build date
  **2026/07/31 19:53:37**, part xcku040-ffva1156-2-i, Vivado 2021.1 --
  the same build the original handoff recorded as loaded. Unmodified vendor
  code.
* `fifo_comb` IS present in the implemented design (it appears in the routed
  DRC, control-sets and timing reports), so the comb path was built in.
* The failure is **deterministic at every record length** -- 256, 1024,
  4096, 16384, 65536, 262144, 1048576 all time out. Not a threshold or
  fill-level effect.
* The vendor's client writes **double** the sample count in this mode
  (`fpgaDataNum = depth*2` when ch==3, `bps = 4`), because DDR accounting is
  in 16-bit words while each sample clock emits two. We now do the same --
  it is the correct accounting -- but it does **not** fix the hang: all
  sizes still time out with the doubled count.

**The design fails timing, on exactly the logic that generates
`Adc_Finish`.** From the routed timing summary: WNS **-2.533 ns**, TNS
-131.434 ns, **178 failing endpoints**. The violated paths are

    burst_cnt_reg[4]        -> xdma_0 AXI-MM bridge        -2.533 ns
    AXI_CMD slv_reg3_reg[11]-> burst_cnt_reg[*]/D          -1.279 ns

`slv_reg3` is `set_sample_num`, and `burst_num = wr_ddr_num/256` is compared
against `burst_cnt` to produce
`write_ddr_done = (burst_cnt >= burst_num) && fifo_empty` -- which is
`Adc_Finish`. A 32-bit compare on the 300 MHz DDR UI clock missing by 2.5 ns
means this signal is not reliably generated by construction; it happens to
work in the paths the vendor exercised and not in the comb path.

Other latent problems found while reading (not proven to cause this):

* `wrfifo_rd_cnt` is truncated from the FIFO's 9/10-bit read count to an
  8-bit `FIFO_ADDR_WIDTH` (`clogb2(255)`), while the FIFO output depth is
  512. Counts above 255 wrap, which can make `wr_ddr3_req`'s
  `fifo_rd_cnt >= 32` test fail and stall the reader. Comb mode reaches a
  given word count in half the writes (4 per 128-bit word vs 8).
* Neither FIFO's `wr_en` is gated by `full` or `wr_rst_busy`, so overflow and
  reset-window writes are silently dropped.
* Both FIFOs share one `rd_en`, so the idle one is popped while empty.

**Conclusion: this is an FPGA-side limitation, not a client bug.** It needs
the vendor, or a rebuild that closes timing (and preferably widens
`FIFO_ADDR_WIDTH`). The definitive next step is available on the board: the
design instantiates `ila_0` and `/dev/xdma0_xvc` exists, so Vivado Hardware
Manager can attach over XVC and watch `state`, `wrfifo_full`, `empty_comb`
and `burst_cnt` live during a ch_sel=3 capture.

Client side: selecting channel 3 now falls back to channel A after two
consecutive timeouts with an explanatory message, instead of stalling the
display half a second per frame indefinitely.

## 30. Dual channel DOES work -- Adc_Finish is what is broken, 2026-09-08

Section 29 concluded ch_sel=3 was unusable. That was wrong in an important
way, found by porting the vendor client to Linux and running its exact
sequence (`~/pcie_client_sw/linux`).

**`Adc_Finish` never asserts in dual-channel mode -- at any depth from 4096
to 262144 -- but the capture itself is completely fine.** Blind-waiting the
expected time and reading gives data that matches single-channel captures of
the same inputs, and differs run to run (so it is fresh, not stale):

    ch A alone            -15.54 codes, std 6.87
    ch B alone            -62.13 codes, std 4.12
    ch3 blind, de-interleaved:
      A                   -15.40 codes, std 6.87
      B                   -62.11 codes, std 4.15

This also confirms the interleave order from section 27: `raw[2i]` is A,
`raw[2i+1]` is B. The vendor works around the same defect -- their client
skips the Adc_Finish poll in this mode and sleeps instead.

`server.py` now blind-waits for ch_sel=3 (`max(0.02, expect*4 + 0.02)`) and
writes the doubled sample count. Verified through the server at N = 65536,
262144 and 1048576: correct per-channel statistics, zero timeouts, 36/26/11
fps respectively.

So the earlier "FPGA-side limitation" framing was half right: the *completion
flag* is broken in this mode (and the design does miss timing on exactly that
logic, section 29), but the data path is sound and the mode is usable.

## 31. Porting the vendor client -- three Linux-specific defects

`~/pcie_client_sw/linux` replaces only the Windows driver layer; every vendor
source stays byte-identical (the Linux `pcie_xdma.h` shadows theirs via
include order). Found by running it:

1. `lseek` on `/dev/xdma0_user` fails with **ESPIPE** -- the register node is
   not seekable, so BAR access needs `pread`/`pwrite`. The DMA nodes are
   seekable and keep the chunked loop.
2. A **12-byte register write lands only the first word**. The vendor writes
   Speed/Channel/SampleNum in one call; on Linux the driver services one
   32-bit register per call, so Channel_Set and set_sample_num silently kept
   their old values -- every channel returned channel A's data.
   `xdma_node_write` now splits into word accesses.
3. `start_sample` is edge triggered and the vendor only writes 1, relying on
   a trailing 0 from the previous capture. With it already high there is no
   edge, `Adc_Finish` idles high so the poll returns instantly, and stale DDR
   reads look like a successful capture.

The C2H path deliberately replicates `dma_from_device` (O_RDWR|O_TRUNC,
posix_memalign(4096), chunked lseek+read) because a naive `read()` there has
hard-frozen this machine before (section 5).

## 32. Dual channel draws BOTH traces, 2026-09-08

The "Show" selector is gone. With Channel = 3 the server now runs the CUDA
pass once per channel and ships both, and the UI draws them together in both
plots, one colour per channel (A green, B blue) with a legend chip in each
card header.

Wire format 5: the payload is one array group per trace --
`disp, [zoom on trace 0 only], tmin, tmax, tmean` -- with `n_traces` and
`traces` in the header. The zoom slice is a crop of the primary spectrum so
it is sent once, not per trace. Layout verified byte-exact for single, dual,
and dual+zoom.

Costs: two GPU passes per frame instead of one (a few ms each), and roughly
double the payload, but only in dual mode. Measured on hardware at
N=1,048,576 / nfft 8192 / 16 Welch frames: 10.5 fps, zero timeouts, trace A
mean -15.43 codes and trace B -62.93, matching single-channel captures of
the same inputs.

The EMA accumulator is now per trace. Peaks, the noise classifier and the
signal panel still read trace 0 (channel A) -- the spectrum analysis is
single-source by design; the second trace is drawn, not analysed.

## 33. Are the two channels sampled simultaneously? Yes -- 2026-09-08

Asked whether dual mode captures A and B at the same instant or takes a
block of one then a block of the other.

**From the HDL it is sample-by-sample interleaved, not blocked.** In
`ad9643_md.v` a single `IDDRE1` recovers both channels from the same LVDS
lane on opposite edges of the ADC clock (`Q1` -> B, `Q2` -> A), and they are
registered together on one `posedge adc_clk`. `ad9643_14bit_to_16bit.v` then
emits ONE 32-bit word per sample clock holding both:
`ad_out_comb <= {2'd0, s_ad_in1, 2'd0, s_ad_in2}`. There is no buffering that
could serialise a millisecond of one channel ahead of the other.

**Confirmed on hardware** (`tools/channel_skew.py`). Splitting the record
even/odd gives two streams whose DC levels stay 47 codes apart with only
0.42 codes of drift across 8 chunks of the record. Were it blocked, each
array would contain half of each block and both would step by ~47 codes
partway through. They do not.

Residual skew is NOT resolved: the delay search over coherent cross-spectrum
bins returns -13.5 ns (-3.4 samples) but with the score at zero skew still
0.74 of the peak -- a shallow optimum, because the only common-mode signal
on a bare board is narrowband switching noise, which makes delay ambiguous.
It does rule out anything remotely like a block offset (1 ms = 250,000
samples). For a real skew number, split one generator output to both inputs
with equal-length cables and re-run the tool.

Method note: the first attempt unwrapped phase across the sparse set of
coherent bins and reported a confident -87 ns. That is wrong -- gaps between
retained bins exceed pi and unwrap incorrectly. The tool now searches for the
delay that best aligns the phase instead, with no unwrapping.
