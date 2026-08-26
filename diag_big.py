#!/usr/bin/env python3
"""Bisect where large captures stall: FPGA arm/poll vs. the vendor
dma_from_device readback. Uses channel 0 (test ramp) so no ADC mezzanine is
needed. Every subprocess call gets an explicit timeout so a real hang can't
block this script forever -- unlike ad9643.ddr_read_samples(), which has none.
"""
import os, sys, time, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad9643 as A

SIZES = [4 << 20, 16 << 20, 64 << 20, 128 << 20, A.MAX_SAMPLES]
# 4M, 16M, 64M, 128M, true max (262,144,000) samples

def dma_read_bounded(nsamples, addr=0, dev=A.C2H_DEV, timeout=30):
    nbytes = nsamples * A.BYTES_PER_SAMPLE
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        t0 = time.monotonic()
        try:
            subprocess.run(
                [f"{A.TOOLS}/dma_from_device", "-d", dev, "-a", str(addr),
                 "-s", str(nbytes), "-f", f.name],
                check=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"    dma_from_device TIMED OUT after {timeout}s", flush=True)
            return None, time.monotonic() - t0
        except subprocess.CalledProcessError as e:
            print(f"    dma_from_device FAILED rc={e.returncode} "
                  f"stderr={e.stderr[:300]!r}", flush=True)
            return None, time.monotonic() - t0
        dt = time.monotonic() - t0
        sz = os.path.getsize(f.name)
        return sz, dt

def main():
    adc = A.Adc()
    for N in SIZES:
        print(f"\n=== N={N:,} samples ({N*2/1e6:.1f} MB) ===", flush=True)
        A.Adc._validate(N, A.CH_TEST_RAMP, 0)
        expect = N / A.BASE_CLOCK_HZ
        timeout = max(0.5, expect * 4 + 0.5)
        adc.wr(A.REG_SPEED, 0)
        adc.wr(A.REG_CHANNEL, A.CH_TEST_RAMP)
        adc.wr(A.REG_NSAMPLES, N)
        t0 = time.monotonic()
        adc.wr(A.REG_START, 0)
        adc.wr(A.REG_START, 1)
        finished = False
        while time.monotonic() - t0 < timeout:
            if adc.finished:
                finished = True
                break
            time.sleep(1e-4)
        t_cap = time.monotonic() - t0
        if not finished:
            print(f"    CAPTURE TIMEOUT after {t_cap:.3f}s (expected {expect*1e3:.1f} ms) "
                  f"regs={adc.regs()}", flush=True)
            adc.recover()
            continue
        print(f"    capture ok in {t_cap*1e3:.3f} ms (expected {expect*1e3:.1f} ms)", flush=True)

        sz, t_dma = dma_read_bounded(N, timeout=30)
        if sz is None:
            print(f"    DMA readback did not complete within 30s bound", flush=True)
            continue
        ok = sz == N * 2
        gbps = (sz / t_dma) / 1e9 if t_dma > 0 else 0
        print(f"    DMA read {sz:,} B in {t_dma:.3f}s ({gbps:.3f} GB/s) "
              f"expected {N*2:,} B -> {'OK' if ok else 'MISMATCH'}", flush=True)
    adc.close()
    print("\nDONE", flush=True)

if __name__ == "__main__":
    main()
