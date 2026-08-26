import sys, time, numpy as np; sys.path.insert(0,'/home/btiagx/adc_capture')
import ad9643 as A
N=65536

def busy(a, timeout=5.0):
    t0=time.monotonic()
    while time.monotonic()-t0 < timeout:
        if a.finished: return time.monotonic()-t0
    raise A.CaptureTimeout("to")

with A.Adc() as a:
    print("=== Speed_Set law (busy-poll, N=65536) ===")
    print(f"{'speed':>6} {'ms':>8} {'implied Msps':>13}")
    for sp in [0,1,2,4,8,16,32]:
        a.arm(N,speed=sp); a.trigger(); el=busy(a)
        print(f"{sp:>6} {el*1e3:>8.3f} {N/el/1e6:>13.2f}")

    print("\n=== Channel_Set sweep (N=65536, speed=0) ===")
    res={}
    for ch in range(4):
        a.arm(N,speed=0,channel=ch); a.trigger(); el=busy(a)
        d=A.ddr_read(N*2); res[ch]=d
        print(f" ch={ch} {el*1e3:7.3f} ms  first12={list(d[:12])}  max={d.max()}")
    for ch in range(1,4):
        print(f"  ch{ch} identical to ch0: {np.array_equal(res[ch],res[0])}")

    print("\n=== repeatability (same settings twice) ===")
    a.arm(N,speed=0,channel=0); a.trigger(); busy(a); d1=A.ddr_read(N*2).copy()
    a.arm(N,speed=0,channel=0); a.trigger(); busy(a); d2=A.ddr_read(N*2).copy()
    print("  bit-identical across runs:", np.array_equal(d1,d2))
