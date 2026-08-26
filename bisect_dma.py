#!/usr/bin/env python3
"""Find the exact operation that hard-freezes the Orin.

A hard hang flushes nothing to disk, so this writes an fsync'd marker BEFORE
each step. After the freeze and reboot:

    cat ~/adc_capture/bisect_marker.txt

names the operation that killed it.

Uses the internal test ramp (channel 0) and a tiny capture, so it does not
depend on an ADC mezzanine being fitted.

    python3 bisect_dma.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MARKER = os.path.join(HERE, "bisect_marker.txt")

N = 4096                 # small: 8 KB, multiple of 256
CH = 0                   # internal test ramp — no mezzanine needed
SPEED = 0


def mark(stage, text):
    line = f"{stage}: {text}"
    with open(MARKER, "w") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.sync()
    print(f"[{stage}] {text}", flush=True)
    time.sleep(0.3)          # give the console time to render before a hang


def ok(stage):
    print(f"[{stage}] ok", flush=True)


def main():
    prev = None
    if os.path.exists(MARKER):
        prev = open(MARKER).read().strip()
        print(f"--- previous run died at: {prev}\n")

    mark("S0", "import ad9643 + gpu")
    import ad9643 as A
    import gpu
    ok("S0")

    mark("S1", "open /dev/xdma0_user, read registers (MMIO only)")
    adc = A.Adc()
    print("     regs:", adc.regs(), flush=True)
    ok("S1")

    mark("S2", f"arm + poll a {N}-sample test-ramp capture (no DMA)")
    A.Adc._validate(N, CH, SPEED)
    adc.wr(A.REG_SPEED, SPEED)
    adc.wr(A.REG_CHANNEL, CH)
    adc.wr(A.REG_NSAMPLES, N)
    adc.wr(A.REG_START, 0)
    adc.wr(A.REG_START, 1)
    t0 = time.monotonic()
    while not adc.finished:
        if time.monotonic() - t0 > 2.0:
            print("     TIMEOUT — Adc_Finish never asserted", flush=True)
            adc.recover()
            break
        time.sleep(1e-4)
    else:
        print(f"     finished in {(time.monotonic()-t0)*1e3:.3f} ms", flush=True)
    ok("S2")

    mark("S3", "read back via subprocess dma_from_device (known-good path)")
    d = A.ddr_read_samples(N)
    print(f"     got {len(d)} samples, min {d.min()} max {d.max()}", flush=True)
    ok("S3")

    mark("S4", "FastC2H.read_into a PLAIN numpy array  <-- key test")
    c2h = gpu.FastC2H()
    plain = np.empty(N, np.uint16)
    got = c2h.read_into(plain, N * 2)
    print(f"     got {got} bytes, min {plain.min()} max {plain.max()}", flush=True)
    ok("S4")

    mark("S5", "create gpu.Spectrum (CUDA init + pinned alloc, no DMA)")
    sp = gpu.Spectrum(8192, max_samples=1 << 22, trace_width=1024)
    print(f"     nbins={sp.nbins} max_frames={sp.max_frames} "
          f"host.nbytes={sp.host.nbytes}", flush=True)
    ok("S5")

    mark("S6", "FastC2H.read_into sp.host (CUDA PINNED)  <-- suspected killer")
    got = c2h.read_into(sp.host, N * 2)
    print(f"     got {got} bytes", flush=True)
    ok("S6")

    mark("S7", "sp.process() — cuFFT")
    sp.process(N, max_frames=8)
    print(f"     stats={sp.stats}", flush=True)
    ok("S7")

    mark("DONE", "all stages survived")
    print("\nAll stages completed. The freeze is not in this path.")
    sp.close(); c2h.close(); adc.close()


if __name__ == "__main__":
    main()
