# AD9643 capture

Live scope / spectrum analyzer for an AD9643 ADC mezzanine on a Puzhi
xcku040 FPGA board (XDMA over PCIe Gen3 x4), running on a Jetson AGX Orin.

```
AD9643 (dual 14-bit, 250 Msps) ── LVDS DDR ──► FPGA block capture ──► DDR4 (≤500 MB)
                                                                        │ XDMA C2H DMA
                 browser ◄── HTTP long-poll ── server.py ── CUDA cuFFT ◄┘
```

The FPGA design is **one-shot block capture, not streaming**: each frame is
arm → capture N samples into DDR4 → DMA to host → GPU FFT → publish. The
loop re-arms as fast as it can; the UI's *coverage* metric reports honestly
what fraction of wall-clock time is actually digitised.

The HDL sources are not on this machine — every register/timing/format fact
used here was determined **empirically against the live hardware**. This
README is the map; **`NOTES.md` is the lab notebook** (numbered sections,
referenced below as §N). Read it before changing the capture path.

## Quick start

```bash
cuda/build.sh                      # one-time: build the CUDA spectrum engine
make -C native                     # optional: fast-DMA helper (see below)
python3 server.py                  # → http://<jetson-ip>:8090/
python3 server.py --mock           # no hardware: synthetic 25 MHz tone
```

Port 8090 by default (8080 is taken locally by `openshell-gateway`). If the
port is busy the server walks forward and prints what it chose; a second
instance refuses to start (single-instance lock — two clients racing the
capture registers corrupts both, §6) — use `--replace` to take over.

### Hardware prerequisites

- Bitstream loaded on the FPGA **before the Orin enumerates PCIe**. If
  `lspci -d 10ee:` is empty there is no endpoint: power the FPGA (flash
  boot), then **warm** `sudo reboot` the Orin. A JTAG-loaded bitstream is
  volatile — lost on cold power cycle.
- Xilinx XDMA driver loaded, nodes at `/dev/xdma0_*` mode 0666.
- Vendor tools built at `~/dma_ip_drivers/XDMA/linux-kernel/tools/`
  (`dma_from_device` is the readback engine — see DMA paths below).

### Running as a service (recommended)

A manually-launched `python3 server.py` dies with its SSH session.
`deploy/adc-capture.service` runs it under systemd: starts on boot,
restarts on crash, and makes the UI's **⟲ Restart server** button work
(the button exits the process on purpose; `Restart=always` brings it back).

```bash
sudo systemctl link "$(pwd)/deploy/adc-capture.service"
sudo systemctl daemon-reload
sudo systemctl enable --now adc-capture.service
journalctl -u adc-capture -f          # logs
```

## Using the UI — the measurement model

Three orthogonal ideas (§10–§12):

- **Samples × rate = the record.** `Samples` is acquisition depth into
  DDR4; `Rate divider` sets fs = 250 MHz/(n+1) (a naive sample-dropper —
  no anti-alias filter, so divided rates fold high frequencies in; full
  rate is the honest setting, §1).
- **The FFT analyses the record.** `FFT size` defaults to *auto — follow
  samples*: the whole record in one FFT, so resolution obeys
  **Δf = 1/T_record** (16.7 M samples = 67 ms → 14.9 Hz bins; the 2^27
  ceiling was validated on this GPU at 116 ms/transform). Pick a smaller
  manual FFT and the record is chopped into **FFT frames** (Welch
  segments) whose *averaged* spectra smooth the noise floor — variance
  down, resolution unchanged. `Trace avg` is a second, display-only
  exponential average across captures. Welch=1 + avg=1 = one raw FFT.
- **The time plot shows the record.** Always all N samples, min/max
  enveloped to 1024 columns. (There is deliberately no separate trace
  length — samples + rate already define it, §12.)

The spectrum overview is 4096 points spaced **uniformly in log(f)** from
the true first bin (= Δf) to Nyquist, max-pooled so narrow peaks survive
(§12 — a linear overview pinned the axis floor at 31 kHz forever).
Drag-select a band for a full-resolution linear zoom of just that band;
double-click to reset.

**Status badges**: capture state, engine fps, source (warns when the
FPGA-internal test ramp is selected — it is *not* ADC data, §2), coverage,
and the network badge (Orin's own WiFi RSSI/rate + measured frame lag —
the long-poll transport self-adapts to a slow link by delivering fewer,
*newest* frames rather than building a backlog, §9/§11).

### Known artifacts with a floating input (§ baselines/)

- A **harmonic comb of a ~497.6 kHz switching regulator** (strongest at the
  3rd harmonic, ~−85 dBFS): field pickup into the unterminated high-Z
  input. `baselines/comb_floating.json` holds the full measured line list
  for before/after comparisons once the input is terminated.
- A spur pinned **exactly at fs/2** (~−82 dBFS ≈ 0.6 LSB) that tracks
  Nyquist when fs changes: a sampling-chain artifact, not an input signal.

## Architecture & files

| Path | What |
|---|---|
| `ad9643.py` | Register map + measured semantics (§1–§4), arm/trigger/poll/recover, DDR4 readback via the vendor CLI (tmpfs transfer file, `readinto`) |
| `gpu.py` | ctypes bridge to the CUDA engine; pinned + staging buffers |
| `cuda/adcfft.cu` | CUDA: stats, Hann window, batched cuFFT R2C, power→dBFS, min/max envelope (single stream, no mid-pipeline sync) |
| `cuda/build.sh` | builds `cuda/libadcfft.so` |
| `server.py` | acquisition engine (continuous / single-shot), DmaReader with helper→vendor fallback, mock mode, HTTP server (long-poll wire format) |
| `web/index.html` | browser UI (no external deps) |
| `native/xdma_shm_reader.c` | resident DMA helper — vendor-exact device access minus per-frame spawn (opt-in, below) |
| `validate_fast_dma.py` | hardware qualification for the helper — run once before `--fast-dma` |
| `noise.py` | peak detection + noise classification: CFAR, sub-bin refinement, harmonic sieve, rule-based labels |
| `tools/emi_sweep.py` | drives the server across channel/fs conditions, classifies, cross-references, saves JSON |
| `tests/test_controls.py` | full control-matrix regression against `--mock` (no hardware needed) |
| `tests/test_noise.py` | `noise.py` unit tests against synthetic spectra (no hardware) |
| `deploy/adc-capture.service` | systemd unit |
| `adc_capture.py` | standalone CLI capture → `.npy`/`.bin` |
| `baselines/` | measured EMI baselines (JSON) |
| `experiments/`, `diag_*.py`, `bisect_dma.py` | the scripts that established §1–§7; not part of the live path |
| `NOTES.md` | the lab notebook — evidence for everything above |

### HTTP API

| Endpoint | Meaning |
|---|---|
| `GET /` | the UI |
| `GET /frame` | latest frame: `u32 len + JSON header + f32 spectrum + [f32 zoom] + f32 tmin + f32 tmax` |
| `GET /frame?wait=1&since=N` | long-poll: parks until a frame newer than N (204 after ~25 s) |
| `GET /status` | tiny JSON: running/frames/timeouts/err |
| `GET /limits` | hardware constants (granularity, max samples, base clock) |
| `POST /control` | any subset of `{nsamples, channel, speed, nfft, max_frames, avg, trace_samples, readback, running, trigger, zoom}` |
| `POST /restart` | clean exit; systemd restarts it |

## DMA readback paths — read §5 before touching this

- **Default (proven)**: the vendor `dma_from_device` CLI per frame, output
  through a `/dev/shm` transfer file. A raw Python `read()`/`readv()` on
  the C2H node **hard-freezes the entire SoC** (bisected, §5) — the unsafe
  code is kept only in `gpu.FastC2H` for future bisection, clearly marked.
- **Opt-in fast path** (`--fast-dma`): `native/xdma_shm_reader` replicates
  the vendor tool's device access *exactly* (same `O_RDWR|O_TRUNC` open,
  posix_memalign(4096) bounce, same chunked read loop) but stays resident
  — no per-frame process spawn or file round-trip. **Qualify it on your
  hardware first**: stop the server, run `python3 validate_fast_dma.py`,
  and only use the flag on PASS. Any runtime helper failure falls back to
  the vendor path automatically (visible as `dma_path` in the UI).

## Testing

```bash
python3 tests/test_controls.py     # spins up --mock on localhost; asserts
                                   # frequency mapping, alias positions,
                                   # display metadata, welch/trace splits,
                                   # zoom, single-shot, error recovery
```

Mock mode is a faithful stand-in (timing semantics, rate-divider aliasing,
a 25 MHz tone) that never opens `/dev/*` — safe on machines without the
FPGA, and how every change in §8–§12 was verified before touching hardware.

## Noise classification

`noise.py` turns a spectrum into labelled sources; `tools/emi_sweep.py`
collects the evidence that makes labelling possible:

```bash
python3 tools/emi_sweep.py --band 3e5 2e7 --channels 1 2 --speeds 0 1 \
    --label "input floating"
```

Detection is CFAR (threshold from a stated false-alarm probability given the
Welch depth, not a fixed dB margin); peaks are refined to ~0.02 bin; a
harmonic sieve groups them into families with chance-correction and a
two-pass fundamental fit; rules label them (`switching_regulator`,
`mains_harmonics`, `harmonic_distortion`, `sampling_artifact_fs_2`, ...).
Results are saved to `baselines/` so runs can be diffed across
interventions -- **the interventions are what actually classify**: changing
fs separates aliases and sampling artifacts, comparing channels separates
common board-level sources from per-channel pickup, and terminating the
input separates radiated from conducted coupling. See NOTES.md §13 for the
measured behaviour and the one open ambiguity.

Use a low-sidelobe window (`blackman-harris`) for spur hunting; Hann's
-31.5 dB sidelobes let a strong tone's skirt be detected as spurs.

The time-domain plot autoscales; the **lock Y** checkbox in its header
freezes the vertical range at its current value, so a signal that shrinks
actually looks smaller instead of the axis shrinking with it.

In the UI, the **Classify** control turns on live labelling: each detected
family gets a colour, with ticks above the spectrum and dots on the trace at
its harmonics, and a *Noise sources* card lists the families (fundamental,
label, harmonic count, density, significance, peak level). Unmatched lines
are grey. It runs ~1x/s in a worker thread -- measured cost to the capture
loop is 0% -- and markers are hidden whenever the analysis and the displayed
frame disagree on fs/nfft/channel.

## Known constraints (evidence in NOTES.md)

- Max capture 262,144,000 samples (500 MiB window, linear, no wrap) ≈
  1.05 s at 250 Msps; `nsamples` must be a multiple of 256 (§3 — other
  values hang or silently truncate).
- `Channel_Set`: 0 = FPGA test ramp (not ADC data), 1/2 = ADC A/B
  (mapping to physical inputs still unverified — no signal generator has
  been connected yet), 3 = hangs the FSM (recoverable, §2).
- `Adc_Finish` idles high from the previous capture — poll it only after
  confirming it dropped (§ Operational notes).
- **One capture client at a time.** The register interface has no
  arbitration; `server.py` enforces this with a lock file (§6).
- No analogue front end: no anti-alias filter, no termination, no
  calibration to volts. Fine for the data path; not yet a measurement.

**Remote access over the NetBird VPN**: this host has a local `iptables`
allowlist on `wt0`, independent of dashboard ACLs. Exposing a new port:

```bash
sudo iptables -I INPUT <n> -i wt0 -p tcp --dport <port> -s 100.82.0.0/16 \
  -j ACCEPT -m comment --comment '<service> from mesh peers'
sudo netfilter-persistent save
```
