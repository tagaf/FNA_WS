#!/usr/bin/env python3
"""Evidence collector: sweep acquisition conditions, detect and classify peaks.

Classifying electrical noise is driven far more by *interventions* than by any
single spectrum -- changing fs separates sampling artifacts and aliases from
real in-band signals, and comparing channels separates a common board-level
source from per-channel pickup. This drives the live server through its HTTP
API across those conditions, runs noise.analyse() on a full-resolution slice
at each, and cross-references the results.

    python3 tools/emi_sweep.py --band 3e5 2e7 --channels 1 2 --speeds 0 1

Saves per-condition peak lists plus derived flags to JSON (default
baselines/sweep_<timestamp>.json) so runs can be diffed later -- e.g. before
and after terminating the input, which separates radiated pickup from
conducted supply coupling.

The server's current config is saved and restored; the live display will
flicker through the swept settings while this runs.
"""
import argparse, json, os, struct, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import noise


class Client:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def get(self, path, timeout=40):
        return urllib.request.urlopen(self.url + path, timeout=timeout)

    def post(self, obj, timeout=10):
        rq = urllib.request.Request(self.url + "/control",
                                    data=json.dumps(obj).encode(),
                                    headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(rq, timeout=timeout).read()

    def frame(self, since=0, timeout=40):
        r = self.get(f"/frame?wait=1&since={since}", timeout)
        if r.status != 200:
            return None, None, None
        b = r.read()
        hl = struct.unpack("<I", b[:4])[0]
        m = json.loads(b[4:4 + hl])
        off = 4 + hl
        disp = np.frombuffer(b[off:off + m["disp_bins"] * 4], np.float32)
        off += m["disp_bins"] * 4
        zoom = None
        if m["zoom"]["active"] and m["zoom"]["bins"] > 0:
            zoom = np.frombuffer(b[off:off + m["zoom"]["bins"] * 4], np.float32)
        return m, disp, zoom

    def settle(self, want, since, tries=40, need_zoom=False, min_avg=1):
        last = since
        for _ in range(tries):
            m, d, z = self.frame(last)
            if m is None:
                continue
            last = m["totals"]["frames"]
            if not all(m["cfg"].get(k) == v for k, v in want.items()):
                continue
            if need_zoom and z is None:
                continue
            if m["totals"]["avg_depth"] < min_avg:
                continue
            return m, d, z, last
        raise TimeoutError(f"condition never settled: {want}")


def infer(conds):
    """Derive fs-independence and channel-commonality from the sweep."""
    summary, findings = [], []
    by_ch = {}
    for c in conds:
        by_ch.setdefault(c["channel"], []).append(c)

    for ch, cs in sorted(by_ch.items()):
        if len(cs) < 2:
            continue
        ref = cs[0]
        for fam in ref["families"]:
            f0 = fam["f0_hz"]
            hits, ratios = [], []
            for other in cs[1:]:
                tol = max(3 * other["bin_hz"], 2e-3 * f0)
                exact = any(abs(g["f0_hz"] - f0) <= tol for g in other["families"])
                # A wideband comb aliases when fs drops: harmonics above the
                # new Nyquist fold back and fill in extra lines, so the sieve
                # can legitimately settle on f0/2. Treat a small-integer
                # ratio as "related", not as a disappearance.
                rel = None
                for g in other["families"]:
                    for num, den in ((1, 2), (2, 1), (1, 3), (3, 1)):
                        if abs(g["f0_hz"] * num / den - f0) <= max(tol, 2e-3 * f0):
                            rel = f"{num}/{den}"
                            break
                    if rel:
                        break
                hits.append(exact)
                ratios.append(rel)
            if hits and all(hits):
                findings.append({"channel": ch, "f0_hz": f0,
                                 "fs_independent": True, "label": fam["label"]})
                summary.append(
                    f"ch{ch}: {fam['label']} at {f0/1e3:.3f} kHz is fs-INDEPENDENT"
                    f" -> real in-band source (not an alias, not sampling)")
            elif any(ratios):
                findings.append({"channel": ch, "f0_hz": f0,
                                 "fs_independent": "related", "label": fam["label"]})
                summary.append(
                    f"ch{ch}: {f0/1e3:.3f} kHz appears at a {ratios[0]} ratio at the"
                    f" other fs -- AMBIGUOUS. A wideband comb aliases when fs drops"
                    f" (harmonics fold back and the sieve may lock to f0/2), so this"
                    f" neither confirms nor refutes an external source. Terminate the"
                    f" input instead: that test is unambiguous.")
            else:
                summary.append(
                    f"ch{ch}: family at {f0/1e3:.3f} kHz did NOT persist across"
                    f" fs -> suspect alias or sampling artifact")

    chans = sorted(by_ch)
    if len(chans) >= 2:
        a_f = by_ch[chans[0]][0]["families"]
        b_c = by_ch[chans[1]][0]
        for fa in a_f:
            tol = max(3 * b_c["bin_hz"], 2e-3 * fa["f0_hz"])
            m = [fb for fb in b_c["families"] if abs(fb["f0_hz"] - fa["f0_hz"]) <= tol]
            if m:
                d = fa["peak_db"] - m[0]["peak_db"]
                summary.append(
                    f"{fa['label']} at {fa['f0_hz']/1e3:.3f} kHz on BOTH ch"
                    f"{chans[0]} and ch{chans[1]} (delta={d:+.1f} dB) -> common"
                    f" board-level source, not per-channel pickup")
    if not summary:
        summary.append("no cross-condition structure found (single condition?)")
    return {"findings": findings, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--band", nargs=2, type=float, default=[3e5, 2e7],
                    metavar=("LO_HZ", "HI_HZ"))
    ap.add_argument("--channels", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--speeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--nsamples", type=int, default=1 << 20)
    ap.add_argument("--nfft", type=int, default=1 << 20)
    ap.add_argument("--avg", type=int, default=8)
    ap.add_argument("--pfa", type=float, default=1e-6)
    ap.add_argument("--label", default="", help="note, e.g. 'terminated 50R'")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    c = Client(a.url)
    m0, _, _ = c.frame()
    if m0 is None:
        print("no frame from the server", file=sys.stderr)
        return 1
    saved = {k: m0["cfg"][k] for k in ("nsamples", "nfft", "channel", "speed",
                                       "max_frames", "avg")}
    last = m0["totals"]["frames"]
    out = {"ts": time.time(), "label": a.label, "band_hz": a.band,
           "pfa": a.pfa, "url": a.url, "conditions": []}
    try:
        for ch in a.channels:
            for sp in a.speeds:
                cfg = dict(nsamples=a.nsamples, nfft=a.nfft, max_frames=1,
                           avg=a.avg, channel=ch, speed=sp)
                c.post(cfg)
                c.post({"zoom": list(a.band)})
                m, _, z, last = c.settle(cfg, last, need_zoom=True,
                                         min_avg=min(a.avg, 6))
                fs = m["fs_hz"]
                # Welch frames and the EMA both narrow the power distribution;
                # the EMA is exponential so effective DoF ~ its depth.
                K = max(1, m["nframes"] * max(1, m["totals"]["avg_depth"]))
                r = noise.analyse(np.asarray(z, dtype=np.float64),
                                  bin_hz=m["zoom"]["bin_hz"], fs_hz=fs,
                                  nframes=K, pfa=a.pfa,
                                  f_offset=m["zoom"]["lo_hz"])
                out["conditions"].append({
                    "channel": ch, "speed": sp, "fs_hz": fs,
                    "bin_hz": m["zoom"]["bin_hz"], "K": K,
                    "floor_db": r["floor_db"],
                    "thr_db": r["threshold_db_over_floor"],
                    "n_peaks": r["n_peaks"],
                    "peaks": [{k: p[k] for k in ("freq_hz", "db", "snr_db", "label")}
                              for p in r["peaks"]],
                    "families": [{k: f[k] for k in
                                  ("f0_hz", "n_members", "density", "label",
                                   "peak_db", "harmonics", "why")}
                                 for f in r["families"]],
                })
                print(f"ch{ch} speed={sp} fs={fs/1e6:6.2f}M  bin={m['zoom']['bin_hz']:7.1f} Hz"
                      f"  K={K:<5} floor={r['floor_db']:7.1f}  thr=+{r['threshold_db_over_floor']:.1f} dB"
                      f"  {r['n_peaks']:3d} peaks  {len(r['families'])} families")
                for f in r["families"][:4]:
                    print(f"      {f['label']:<22} f0={f['f0_hz']/1e3:10.3f} kHz"
                          f"  n={f['n_members']:2d}  density={f['density']:.2f}"
                          f"  peak={f['peak_db']:.1f} dBFS")
    finally:
        c.post({"zoom": None})
        c.post(saved)

    out["inference"] = infer(out["conditions"])
    path = a.out or os.path.join(HERE, "baselines",
                                 f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved -> {path}")
    for line in out["inference"]["summary"]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
