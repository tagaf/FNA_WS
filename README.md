# AD9643 capture

Live scope/spectrum tool for an AD9643 ADC mezzanine on a Puzhi xcku040
FPGA board (XDMA over PCIe), running on a Jetson AGX Orin. One-shot block
capture -> DDR4 -> DMA readback -> CUDA FFT -> browser, over HTTP.

The HDL sources for the FPGA design are not available on this machine --
every register/timing/format fact used here was determined empirically
against the live hardware. **See `NOTES.md` for that evidence and for the
operational gotchas** (recoverable hangs, capture-size granularity, the
DMA path that looks safe but freezes the SoC, etc.) before changing the
capture path. This README is the map; `NOTES.md` is the lab notebook.

## Quick start

```bash
# one-time: build the CUDA spectrum engine
cuda/build.sh

# run the live server
python3 server.py
# -> http://<jetson-ip>:8090/
```

Default port is 8090 (8080 is taken by an unrelated local service,
`openshell-gateway`).

### Running as a service (recommended)

A manually-launched `python3 server.py` dies with its SSH session -- lost
this way once already (see NOTES.md, the NetBird incident). `deploy/adc-capture.service`
runs it under systemd instead: starts on boot, restarts on crash, survives
SSH/VPN drops entirely.

```bash
sudo systemctl link "$(pwd)/deploy/adc-capture.service"   # registers it from this repo path, no copy needed
sudo systemctl daemon-reload
sudo systemctl enable --now adc-capture.service
```

The web UI's **"Restart server"** button (Control panel) calls `POST
/restart`, which releases the ADC/GPU cleanly and calls `os._exit(0)` --
`Restart=always` in the unit is what actually brings it back. Without a
supervisor configured that way, that button just kills the server for
good, so don't wire it up without the service running.

Useful commands: `sudo systemctl status adc-capture`,
`journalctl -u adc-capture -f` (logs), `sudo systemctl restart adc-capture`
(same effect as the UI button, from the shell).

**Remote access over the NetBird VPN**: this host has a local `iptables`
allowlist on the `wt0` (NetBird) interface, independent of whatever the
NetBird dashboard ACLs say (`sudo iptables -L INPUT -n -v --line-numbers`
to see it). Only explicitly listed ports get through; everything else on
that interface is dropped by design. Exposing a new port means adding a
rule there too, e.g.:

```bash
sudo iptables -I INPUT <n> -i wt0 -p tcp --dport <port> -s 100.82.0.0/16 \
  -j ACCEPT -m comment --comment '<service> from mesh peers'
sudo netfilter-persistent save   # or it's gone on reboot
```

(`<n>` = the line number *before* the trailing default-drop rule from
`iptables -L INPUT --line-numbers`.)

## Architecture

```
FPGA (block capture, DDR4)
  -> ad9643.py       register map + capture/recover FSM handling, DDR4 readback
       via the vendor dma_from_device CLI (see NOTES.md #5 for why not a
       raw Python read() -- that path hangs the SoC)
  -> gpu.py + cuda/adcfft.cu   CUDA: window, cuFFT, power/dB, min/max envelope
  -> server.py        capture loop (continuous or single-shot-triggered),
                       HTTP server, downsampled + zoom-crop wire format
  -> web/index.html   browser UI: time/freq plots, control, live status
```

## Files

| Path | What |
|---|---|
| `ad9643.py` | Register map, capture/arm/poll/recover, DDR4 readback |
| `gpu.py` | ctypes bridge to `cuda/adcfft.cu`, pinned-buffer handling |
| `cuda/adcfft.cu` | CUDA: Hann window, batched cuFFT R2C, power->dB, envelope |
| `cuda/build.sh` | Builds `cuda/libadcfft.so` from `adcfft.cu` |
| `deploy/adc-capture.service` | systemd unit -- see "Running as a service" above |
| `server.py` | Acquisition loop + HTTP server (the live scope backend) |
| `web/index.html` | Browser UI |
| `adc_capture.py` | Standalone CLI: capture N samples, save/summarize |
| `experiments/` | One-off scripts used to empirically derive the register
map and timing in `NOTES.md` (DDR4 pattern write/verify, speed-divider
sweep) -- not part of the live path, kept for reference |
| `bisect_dma.py` | Bisection tool for the DMA-hang investigation (NOTES.md #5) |
| `diag_big.py`, `diag_gpu.py` | Standalone diagnostics: capture+DMA and GPU-only
paths at large sizes, isolated from the threaded server (used to separate
"is this the driver, the GPU, or the server loop" during debugging) |
| `NOTES.md` | Empirical hardware findings, resolved bugs, open questions |

## Known constraints (see NOTES.md for detail)

- Max single capture: 262,144,000 samples (500 MiB DDR4 window, linear,
  no wrap) -- ~1.05 s of real time at full 250 Msps.
- `nsamples` must be a multiple of 256.
- `Channel_Set=3` hangs the capture FSM (recoverable, see `Adc.recover()`).
- Run only **one** `server.py` (or anything using `ad9643.Adc`) against the
  board at a time -- the register interface has no locking, and two
  concurrent clients racing writes produces hard-to-diagnose capture
  failures that look like a hang.
