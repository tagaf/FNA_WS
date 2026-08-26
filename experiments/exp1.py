import sys, numpy as np; sys.path.insert(0,'/home/btiagx/adc_capture')
import ad9643 as A

N = 65536
pat = (np.arange(4*1024*1024//4, dtype='<u4') | 0xA5000000)
A.ddr_write(pat)
print("pattern loaded into DDR4")

with A.Adc() as a:
    print("before:", a.regs())
    a.arm(N, speed=0, channel=0)
    print("armed :", a.regs())
    a.trigger()
    try:
        r = a.capture_wait = a.wait(timeout=2.0)
        print(f"CAPTURE COMPLETE in {r*1e3:.2f} ms")
    except A.CaptureTimeout as e:
        print("TIMEOUT:", e)
    print("after :", a.regs())

rb = A.ddr_read(4*1024*1024).view('<u4')
diff = np.nonzero(rb != pat)[0]
if len(diff)==0:
    print("DDR4 UNCHANGED - nothing was written")
else:
    print(f"DDR4 changed: first word {diff[0]}, last word {diff[-1]}, "
          f"count {len(diff)} -> {(diff[-1]+1)*4} bytes touched")
    print("first 8 captured words:", [hex(x) for x in rb[:8]])
