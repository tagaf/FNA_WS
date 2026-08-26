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
