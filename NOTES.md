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
