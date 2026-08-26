#!/usr/bin/env python3
"""Isolate the GPU side for large captures: Spectrum() alloc + process().
No FPGA/DMA involved -- fills the stage buffer with dummy data directly."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu

SIZES = [4 << 20, 16 << 20, 64 << 20, 128 << 20, 262144000]

def main():
    for N in SIZES:
        print(f"\n=== N={N:,} samples ===", flush=True)
        t0 = time.monotonic()
        sp = gpu.Spectrum(8192, max_samples=max(1 << 22, N), trace_width=1024)
        print(f"    Spectrum() alloc: {time.monotonic()-t0:.3f}s "
              f"(host {sp.host.nbytes/1e6:.1f} MB, stage {sp.stage.nbytes/1e6:.1f} MB)",
              flush=True)

        t0 = time.monotonic()
        sp.stage[:N] = 100  # dummy data, avoids needing the FPGA
        print(f"    fill stage: {time.monotonic()-t0:.3f}s", flush=True)

        t0 = time.monotonic()
        sp.load(N * 2)
        print(f"    load (copy to pinned): {time.monotonic()-t0:.3f}s", flush=True)

        t0 = time.monotonic()
        sp.process(N, max_frames=64)
        print(f"    process() [cuFFT etc]: {time.monotonic()-t0:.3f}s "
              f"stats={sp.stats}", flush=True)
        sp.close()
    print("\nDONE", flush=True)

if __name__ == "__main__":
    main()
