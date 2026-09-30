# Working on this repo

Host software for an AD9643 dual 14-bit / 250 MSPS ADC on a Puzhi PZ-KU040
(Kintex UltraScale) PCIe card, read over AMD XDMA, driven from a Jetson Orin
(aarch64). Block capture, continuous streaming, a CUDA spectrum engine, laser
phase-noise analysis, and a web UI.

**The FPGA is not ours.** `docs/HOST_INTERFACE.md` in the FPGA repo is the
contract. If the hardware disagrees with it, report that — do not work around
it silently. Several entries below are exactly such disagreements.

## First: is the board usable?

Two things bite every session.

**1. Reprogramming the FPGA wipes the PCIe BARs.** Every register then reads
`0xFFFFFFFF` and `/sys/bus/pci/devices/0005:01:00.0/enable` reads 0. It needs
sudo, which an agent session does not have — give the user this and wait:

```bash
sudo systemctl stop adc-capture
sudo rmmod xdma
echo 1 | sudo tee /sys/bus/pci/devices/0005:01:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
sudo modprobe xdma
cat /sys/bus/pci/devices/0005:01:00.0/enable   # must print 1
```

**2. `adc-capture.service` holds the single-instance lock** (`~/.adc_capture.lock`)
and the device nodes. Stop it before any hardware work, and `Restart=always`
means `kill` is not enough — use `systemctl`. Start it again when you finish.

Quick health check (read-only, safe while the service runs):

```bash
python3 -c "
import mmap,os,numpy as np
fd=os.open('/dev/xdma0_user',os.O_RDWR|os.O_SYNC)
r=np.frombuffer(mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE),dtype='<u4')
print('design 0x%08X  reg10 0x%X  reg13 %d/%d'%(r[15],r[10],r[13]&0x1FF,(r[13]>>16)&0x1FF))"
```

`design` must be `0xAD964302`. `reg10` should be `0x9` or `0xF` — bit0 capture
clock locked, bit3 IDELAYCTRL ready. **If bit3 is 0 the ADC capture path is
dead** and every sample will be a constant; nothing else is worth running.

## Hardware facts that cost real time to establish

Measured, not read from a datasheet. `NOTES.md` has the evidence, 39 numbered
sections.

- **The ADC output is INVERTED.** `0x14 = 0x05` is the power-up default and
  bit 2 means invert on this part (the datasheet table documents the polarity
  backwards). Decode real ADC data with `ad9643.adc_signed()`, never
  `to_signed()`. Proven against the converter's own reference patterns:
  midscale → 0, +FS → +8191, −FS → −8192 only under `-x-1`.
  **`to_signed()` is deliberately NOT inverted** — the FPGA's internal counter
  (ChannelSel 0) is generated in fabric and never passes through the inverter.
  The CUDA kernel has a `c_invert` constant the server sets per capture.
- **`DataNum` (reg3) counts sample CLOCKS in every mode**, including ch 3.
  Writing `N*2` for dual mode (an old convention) makes the capture run twice
  as long and, past 65,536,000 pairs, wrap the 500 MB window onto its own
  record. Use `ad9643.datanum_for()`.
- **`Adc_Finish` (reg4) idles HIGH** and holds the previous capture's state for
  under a microsecond. Sleep ~10 µs after the start edge before polling, or you
  read a stale "finished" and return a buffer that was never filled.
- **`Speed_Set` (reg1) must be 0.** It does not decimate — fs is always
  250 MSPS — and non-zero corrupts block lengths. The driver rejects it.
- **DataNum granularity**: multiple of 256 samples in modes 0/1/2, **128** in
  mode 3.
- **IDELAY tap ≈ 6.25 ps**, not the 2.2 ps first documented. Measured eye is
  ~248–252 taps ≈ 1.57 ns, centre ~122. The FPGA loses the tap on every
  reconfiguration, so the host re-applies a stored value from
  `~/.adc_capture.conf` at open (`ad9643.load_stored_tap`).
- **The AXI-Lite file decodes 6 address bits**: reg16..31 alias reg0..15. It
  was 5 bits before reg8 existed.
- **The ramp checker (reg11/reg12)** counts a sample bad when
  `x[n]-x[n-2] != x[n-2]-x[n-4]`. Valid only with the ramp or checkerboard
  pattern; PN and real signals count errors meaninglessly. One corrupted
  sample counts 3. `ad9643.ramp_deviations()` is an independent host-side
  cross-check — keep using it, the two have disagreed before.

## The DMA rule

**Raw `read`/`readv`/`pread` on `/dev/xdma0_c2h_*` hard-freezes the whole
SoC.** Reproduced at `bisect_dma.py` stage S4; `bisect_marker.txt` still says
so. Only the vendor CLI's exact access pattern is safe: `O_RDWR|O_TRUNC`,
`posix_memalign(4096)`, chunked `lseek`+`read`. That is what
`native/xdma_shm_reader.c` replicates and what `stream.py` uses. Do not
"simplify" it to `pread` because a spec snippet shows `pread`.

## Layout

| file | what |
|---|---|
| `ad9643.py` | register map, `Adc` driver, SPI, decode, stored tap |
| `diag.py` | 9 diagnostics as library functions (progress + cancel + JSON/CSV) |
| `stream.py` | continuous streaming, RAM ring, per-segment envelope |
| `server.py` | HTTP server, acquisition engine, `DiagSession` |
| `web/index.html` | the whole UI, one file, no build step |
| `gpu.py` + `cuda/adcfft.cu` | CUDA spectrum engine (`bash cuda/build.sh`) |
| `phasenoise.py`, `noise.py` | phase-noise demodulation, peak classification |
| `tools/` | `adc_spi.py`, `ramp_check.py`, `eye_scan.py`, `sine_check.py` |
| `NOTES.md` | the evidence log — read before re-deriving anything |

CLI tools delegate to `diag.py`; do not let a pass criterion drift between a
tool and the server.

## Running and testing

```bash
python3 -m pytest tests/ -q          # 36 tests, ~21 s, no hardware needed
python3 tests/test_ui.py             # markup/JS self-checks (also in pytest)
python3 server.py --mock             # synthetic tone, never opens /dev/*
python3 server.py --mock-interferometer   # synthetic Michelson, phase-noise tab
```

`tests/test_ui.py` checks that every `$('id')` resolves **and that every called
function is defined** — a block-replacing edit once deleted three helpers and
left the call sites, and the whole tab came up blank with no other symptom.

The service binds `0.0.0.0:8090`. Tests spin up their own server on **8098** —
do not leave a manual one there or they fail confusingly.

## Conventions

- Comments explain **why**, especially where the code looks odd because the
  hardware is odd. Match that density.
- Diagnostics must **always restore the ADC** — `0x0D = 0x00` + transfer, and
  re-apply the stored tap — on success, failure, cancel and client
  disappearance. `diag.restore_adc()` in a `finally`.
- **Never** clear `0x09` bit 0 (duty-cycle stabiliser) or write `0x14` (output
  format). `Adc.adc_wr` refuses both; the guard is in the driver, not left to
  call sites.
- Measure before prescribing. Most wrong turns here came from a plausible
  story that one benchmark would have killed.

## Open / unresolved

- A **~99.9992 MHz interferer** mixes with the analog input, putting spurs at
  `100 MHz ± f_in` at −38 dBc and wrecking SFDR (38 dBc against a spec of 88).
  Independent of the IDELAY tap, so it is analog, not capture timing. Prime
  suspect is the PCIe 100 MHz reference clock. Channel-to-channel crosstalk
  −39 dBc may be the AWG rather than the board — terminating one input settles
  it.
- Front-end noise is ~7.2/4.1 codes rms where the ADC alone would give ~1.7.
- `README.md` is user-facing and predates streaming, `diag.py` and the decode
  fix. Treat this file as current where they disagree.
