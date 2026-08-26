import sys, numpy as np; sys.path.insert(0,'/home/btiagx/adc_capture')
import ad9643 as A
N=4096
print(f"{'Speed':>5} {'ms':>7} {'step(mode)':>11} {'uniq steps':>28}")
with A.Adc() as a:
    for sp in list(range(0,9))+[15,16,255,256,1000]:
        try:
            a.arm(N, speed=sp, channel=0); a.trigger(); el=a.wait(timeout=3.0)
        except A.CaptureTimeout:
            print(f"{sp:>5} {'TIMEOUT':>7}"); continue
        d = A.ddr_read(N*2).astype(np.int32)
        s = np.diff(d) % 16384
        vals, cnt = np.unique(s, return_counts=True)
        mode = vals[np.argmax(cnt)]
        top = ", ".join(f"{v}x{c}" for v,c in zip(vals[np.argsort(-cnt)][:4], np.sort(cnt)[::-1][:4]))
        print(f"{sp:>5} {el*1e3:>7.2f} {mode:>11} {top:>28}")
