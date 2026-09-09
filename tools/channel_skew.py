#!/usr/bin/env python3
"""Check that dual-channel capture really is simultaneous.

Two questions, two tests:

  1. Is the record sample-INTERLEAVED (A,B,A,B...) or is it a block of one
     channel followed by a block of the other? Decisive without any signal
     source: if it were blocks, splitting even/odd indices would put half of
     each block into BOTH arrays, so each would show a step partway through.
     Interleaved data instead keeps two stable, distinct levels throughout.

  2. What is the residual skew between the channels? Estimated from the
     cross-spectrum over bins where the two channels are coherent, by
     searching for the delay that best aligns their phase. NOTE: do not
     unwrap phase over a sparse set of coherent bins -- gaps larger than pi
     unwrap wrongly and produce a confident, meaningless number. The delay
     search below avoids unwrapping entirely.

Test 2 is only well conditioned with a BROADBAND common signal. On a bare
board the only shared content is narrowband switching-regulator noise, which
makes the delay ambiguous; the `score at 0` figure reports how shallow the
estimate is. For a real number, split one generator output to both inputs
with equal-length cables and run this again.

    python3 tools/channel_skew.py [--url http://127.0.0.1:8090] [--n 1048576]
"""
import argparse, json, sys, time, urllib.request
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--seg", type=int, default=8192)
    ap.add_argument("--coh", type=float, default=0.5)
    a = ap.parse_args()
    U = a.url.rstrip("/")

    def get(p, t=60):
        return urllib.request.urlopen(U + p, timeout=t)

    def post(o):
        rq = urllib.request.Request(U + "/control", data=json.dumps(o).encode(),
                                    headers={"Content-Type": "application/json"})
        urllib.request.urlopen(rq, timeout=5).read()

    m0 = None
    try:
        import struct
        b = get("/frame").read()
        hl = struct.unpack("<I", b[:4])[0]
        m0 = json.loads(b[4:4 + hl])
    except Exception:
        pass
    saved = {k: m0["cfg"][k] for k in ("nsamples", "nfft", "channel", "speed",
                                       "max_frames", "avg", "classify")} if m0 else None

    post({"classify": 0, "channel": 3, "nsamples": a.n, "nfft": 8192,
          "avg": 1, "max_frames": 8})
    time.sleep(3)
    try:
        get("/raw")            # first call arms the snapshot
    except Exception:
        pass
    time.sleep(2)
    raw = np.frombuffer(get("/raw").read(), dtype="<u2")
    if raw.size < 4096:
        print("not enough raw data; is channel 3 capturing?", file=sys.stderr)
        return 1

    sgn = lambda u: ((u.astype(np.int32) ^ 0x2000) - 0x2000).astype(np.float64)
    A, B = sgn(raw[0::2]), sgn(raw[1::2])
    n = min(A.size, B.size)
    A, B = A[:n], B[:n]
    fs = m0["fs_hz"] if m0 else 250e6

    print(f"{n:,} sample pairs, fs = {fs/1e6:.1f} MHz "
          f"(1 sample = {1e9/fs:.1f} ns)\n")

    # ---- 1. interleaved or blocked --------------------------------------
    k, ch = 8, n // 8
    mA = np.array([A[i*ch:(i+1)*ch].mean() for i in range(k)])
    mB = np.array([B[i*ch:(i+1)*ch].mean() for i in range(k)])
    gap = abs(mA.mean() - mB.mean())
    drift = max(mA.max() - mA.min(), mB.max() - mB.min())
    print("1. structure")
    for i in range(k):
        print(f"     chunk {i}: even {mA[i]:+8.2f}   odd {mB[i]:+8.2f} codes")
    print(f"   level gap between streams : {gap:8.2f} codes")
    print(f"   drift within a stream     : {drift:8.2f} codes")
    ok = drift < gap / 4
    print(f"   -> {'SAMPLE-INTERLEAVED (two stable distinct streams)' if ok else 'STEP FOUND: looks block-sequential'}\n")

    # ---- 2. skew ---------------------------------------------------------
    A -= A.mean(); B -= B.mean()
    seg = min(a.seg, n)
    nseg = max(1, n // seg)
    w = np.hanning(seg)
    Sab = np.zeros(seg // 2 + 1, complex)
    Saa = np.zeros(seg // 2 + 1); Sbb = np.zeros(seg // 2 + 1)
    for i in range(nseg):
        x = np.fft.rfft(A[i*seg:(i+1)*seg] * w)
        y = np.fft.rfft(B[i*seg:(i+1)*seg] * w)
        Sab += x * np.conj(y); Saa += abs(x)**2; Sbb += abs(y)**2
    coh = np.abs(Sab)**2 / (Saa * Sbb + 1e-30)
    f = np.arange(seg // 2 + 1) * fs / seg
    good = (coh > a.coh) & (f > 1e5) & (f < fs * 0.12)
    print(f"2. skew  ({nseg} segments, {int(good.sum())} bins with "
          f"coherence > {a.coh})")
    if good.sum() < 8:
        print("   too little common signal to estimate a delay.")
        print("   Split one generator output to both inputs and re-run.")
    else:
        taus = np.linspace(-5, 5, 20001) / fs
        S, F = Sab[good], f[good]
        score = np.abs(np.exp(-2j * np.pi * np.outer(taus, F)) @ S)
        best = taus[int(np.argmax(score))]
        flat = float(np.abs(S.sum()) / score.max())
        print(f"   mean coherence     : {coh[good].mean():.3f}")
        print(f"   best-fit skew      : {best*1e9:+.2f} ns "
              f"({best*fs:+.3f} samples)")
        print(f"   score at zero skew : {flat:.3f} of peak")
        if flat > 0.6:
            print("   -> shallow optimum: the shared signal is too narrowband")
            print("      to resolve sub-sample skew. It does rule out any")
            print(f"      block offset ({1e-3*fs:,.0f} samples for 1 ms).")
        else:
            print("   -> well-conditioned estimate")
    if saved:
        post(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
