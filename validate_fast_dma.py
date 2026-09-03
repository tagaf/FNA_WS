#!/usr/bin/env python3
"""Qualify the resident DMA helper against the proven vendor CLI, on hardware.

Run this ONCE (with server.py STOPPED — it takes the same single-instance
lock) before ever using `server.py --fast-dma`:

    python3 validate_fast_dma.py

It reads the same DDR regions through both paths and compares byte-for-byte,
then soaks the helper for 30 s. It only READS DDR over C2H — it never touches
the AXI_CMD capture registers, so whatever is in DDR simply gets read twice.
If this script hangs or the machine freezes, the helper is NOT safe on this
kernel/driver combo: keep the default vendor path and say so in NOTES.md.
"""
import fcntl, os, select, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ad9643 as A

LOCK_PATH = os.path.expanduser("~/.adc_capture.lock")


def main():
    lk = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lk.seek(0)
        print(f"server.py is running ({lk.read().strip()}). Stop it first — "
              f"two DMA clients at once is exactly what NOTES.md §6 warns "
              f"about.", file=sys.stderr)
        return 1

    exe = os.path.join(HERE, "native", "xdma_shm_reader")
    shm = f"/dev/shm/adc_validate_{os.getpid()}.buf"
    MAX = 64 << 20
    p = subprocess.Popen([exe, A.C2H_DEV, shm, str(MAX)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    assert p.stdout.readline().startswith("READY"), "helper failed to start"
    import mmap
    f = open(shm, "r+b"); mm = mmap.mmap(f.fileno(), MAX)
    view = np.frombuffer(mm, dtype=np.uint16)

    def helper_read(addr, nbytes, deadline=10.0):
        p.stdin.write(f"R {addr} {nbytes}\n"); p.stdin.flush()
        r, _, _ = select.select([p.stdout], [], [], deadline)
        if not r:
            raise TimeoutError(f"helper gave no reply in {deadline}s")
        rep = p.stdout.readline().strip()
        if not rep.startswith("OK"):
            raise RuntimeError(f"helper: {rep}")
        return view[:int(rep.split()[1]) // 2].copy()

    fail = 0
    print("A/B compare (helper vs vendor CLI), byte-for-byte:")
    for addr, nbytes in [(0, 4096), (0, 2 << 20), (512, 1 << 20),
                         (8 << 20, 8 << 20), (0, 64 << 20)]:
        h = helper_read(addr, nbytes)
        v = A.ddr_read_samples(nbytes // 2, addr=addr)
        okp = np.array_equal(h, v)
        fail += not okp
        print(f"  addr={addr:>9} n={nbytes:>9}: {'MATCH' if okp else 'MISMATCH'}")

    print("30 s soak of repeated 2 MB helper reads:")
    t0 = time.monotonic(); reps = 0
    while time.monotonic() - t0 < 30:
        helper_read(0, 2 << 20); reps += 1
    dt = time.monotonic() - t0
    print(f"  {reps} reads, {reps/dt:.1f}/s, {(reps*(2<<20))/dt/1e9:.2f} GB/s "
          f"— machine still alive")

    p.stdin.write("Q\n"); p.stdin.flush(); p.wait(timeout=5)
    os.unlink(shm)
    if fail:
        print(f"\nVERDICT: FAIL ({fail} mismatches) — do NOT use --fast-dma")
        return 1
    print("\nVERDICT: PASS — safe to run:  python3 server.py --fast-dma")
    return 0


if __name__ == "__main__":
    sys.exit(main())
