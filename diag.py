"""ADC/FPGA diagnostics, shared by the CLI tools and the web server.

Every test here is a plain function taking (adc, ..., progress=None,
cancel=None) and returning a DiagResult. The CLI tools and server.py both call
these, so a fix or a changed pass criterion lands in one place instead of
drifting between the two.

  progress(frac, message, **extra)   0..1, called between steps. `extra`
                                     carries partial data for live plots.
  cancel()                           truthy -> raise Cancelled between steps.

RESTORING THE ADC IS THE CALLER'S JOB, via restore_adc() in a finally. These
functions deliberately do not swallow Cancelled, so a cancelled sweep unwinds
to the caller's cleanup rather than quietly returning half a result.
"""
from __future__ import annotations

import io
import csv
import time

import numpy as np

import ad9643 as A


class Cancelled(Exception):
    """Raised between steps when the caller's cancel() went truthy."""


class DiagResult:
    """One test's outcome. JSON-serialisable; CSV when it has rows."""

    def __init__(self, name, params=None, ok=None, summary="", data=None,
                 columns=None, rows=None):
        self.name = name
        self.params = dict(params or {})
        self.ok = ok                      # True/False, or None when N/A
        self.summary = summary
        self.data = dict(data or {})
        self.columns = list(columns or [])
        self.rows = list(rows or [])
        self.ts = time.time()

    def to_json(self):
        return {"name": self.name, "params": self.params, "ok": self.ok,
                "summary": self.summary, "data": self.data,
                "columns": self.columns, "rows": self.rows, "ts": self.ts}

    def to_csv(self):
        if not self.columns:
            return ""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.columns)
        w.writerows(self.rows)
        return buf.getvalue()

    def __repr__(self):
        return f"<DiagResult {self.name} ok={self.ok} {self.summary!r}>"


def _ck(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def _tick(progress, frac, msg, **extra):
    if progress is not None:
        progress(max(0.0, min(1.0, float(frac))), msg, **extra)


def restore_adc(adc, tap=None):
    """Normal output + stored data delay. Safe to call from a finally, and
    safe to call twice. Never raises -- it runs on the error path."""
    out = {}
    try:
        adc.adc_wr_transfer(A.ADC_TEST_MODE, A.ADC_TEST_NORMAL)
        out["test_mode"] = adc.adc_rd(A.ADC_TEST_MODE)
    except Exception as e:
        out["test_mode_error"] = f"{type(e).__name__}: {e}"
    try:
        t = A.load_stored_tap() if tap is None else int(tap)
        if t is not None:
            adc.set_tap(t)
            out["tap"] = t
    except Exception as e:
        out["tap_error"] = f"{type(e).__name__}: {e}"
    return out


def set_pattern(adc, code):
    """Select an ADC test pattern (0x0D) and latch it."""
    adc.adc_wr_transfer(A.ADC_TEST_MODE, int(code) & 0xFF)
    return adc.adc_rd(A.ADC_TEST_MODE)


PATTERNS = [(0x00, "normal ADC data"), (0x01, "midscale short"),
            (0x02, "+full scale short"), (0x03, "-full scale short"),
            (0x04, "alternating checkerboard"), (0x05, "PN long sequence"),
            (0x06, "PN short sequence"), (0x07, "one/zero word toggle"),
            (0x0F, "ramp")]
PATTERN_NAME = dict(PATTERNS)

# Expected constants for the patterns that have one, as the ANALOG INPUT would
# have produced them (i.e. after the 0x14 output inversion is undone).
PATTERN_CONST = {0x01: 0, 0x02: 8191, 0x03: -8192}

# Registers worth naming in the dump. The rest are printed raw.
REG_NAMES = {
    0x00: "chip port config", 0x01: "chip ID", 0x02: "chip grade",
    0x05: "channel index", 0x08: "power mode", 0x09: "clock / DCS",
    0x0B: "clock divide", 0x0D: "test mode", 0x0E: "BIST",
    0x10: "offset adjust", 0x14: "output mode", 0x15: "output adjust",
    0x16: "output phase", 0x17: "DCO output delay", 0x18: "input span",
    0x19: "user pattern 1 LSB", 0x1A: "user pattern 1 MSB",
    0x1B: "user pattern 2 LSB", 0x1C: "user pattern 2 MSB",
    0x1D: "user pattern 3 LSB", 0x1E: "user pattern 3 MSB",
    0x1F: "user pattern 4 LSB", 0x20: "user pattern 4 MSB",
    0x21: "serial control", 0x22: "serial ch status", 0x3A: "sync control",
    0xFF: "transfer",
}


# ------------------------------------------------------------ 1. SPI self-test
def spi_selftest(adc, progress=None, cancel=None):
    """Chip ID and speed grade. Confirms the SPI wiring end to end."""
    _tick(progress, 0.1, "reading chip ID")
    st = adc.adc_selftest()
    _ck(cancel)
    _tick(progress, 0.6, "reading output mode")
    om = adc.adc_rd(A.ADC_OUTPUT_MODE)
    fmt = {0: "offset binary", 1: "two's complement", 2: "gray code"}.get(om & 3,
                                                                          "reserved")
    inv = bool(om & A.ADC_INVERT_BIT)
    ok = st["chip_id_ok"] and st["grade_ok"]
    _tick(progress, 1.0, "done")
    return DiagResult(
        "spi_selftest", {}, ok,
        (f"chip ID 0x{st['chip_id']:02X} "
         f"{'OK' if st['chip_id_ok'] else 'WRONG (expect 0x82)'}, "
         f"grade bits {st['grade_bits']:02b} "
         f"{'= 250 MSPS' if st['grade_ok'] else 'NOT 250 MSPS'}"),
        {"chip_id": st["chip_id"], "chip_id_ok": st["chip_id_ok"],
         "grade_raw": st["grade_raw"], "grade_bits": st["grade_bits"],
         "grade_ok": st["grade_ok"], "output_mode": om,
         "output_format": fmt, "output_invert": inv,
         "design_id": adc.design_id})


# --------------------------------------------------------- 2. Register dump
def register_dump(adc, lo=0x00, hi=0x3A, progress=None, cancel=None):
    rows = []
    n = max(1, hi - lo + 1)
    for i, a in enumerate(range(lo, hi + 1)):
        _ck(cancel)
        v = adc.adc_rd(a)
        rows.append([f"0x{a:02X}", f"0x{v:02X}", f"{v:08b}",
                     REG_NAMES.get(a, ""), a in A.ADC_SHADOWED])
        if i % 8 == 0:
            _tick(progress, i / n, f"0x{a:02X}")
    _tick(progress, 1.0, "done")
    return DiagResult(
        "register_dump", {"lo": lo, "hi": hi}, None,
        f"read {len(rows)} registers 0x{lo:02X}..0x{hi:02X}",
        {}, ["addr", "hex", "binary", "name", "shadowed"], rows)


# ------------------------------------------------------- 3. FPGA ramp checker
def ramp_check(adc, seconds=60.0, pattern=A.ADC_TEST_RAMP, interval=0.5,
               progress=None, cancel=None):
    """Accumulate reg11/reg12 with a test pattern applied, sampling as we go
    so the caller can plot the counters against time.

    Only meaningful with the RAMP or CHECKERBOARD pattern. The checker tests
    that the second difference is constant, which those two satisfy and the PN
    sequences and real signals do not -- pointing it at anything else counts
    errors that mean nothing. One corrupted sample shows up as 3 errors,
    because it breaks three consecutive second differences.
    """
    rows = []
    with adc.ramp_mode() if pattern == A.ADC_TEST_RAMP else _pattern_ctx(adc, pattern):
        adc.ramp_clear()
        t0 = time.monotonic()
        while True:
            _ck(cancel)
            el = time.monotonic() - t0
            if el >= seconds:
                break
            time.sleep(min(interval, max(0.0, seconds - el)))
            el = time.monotonic() - t0
            a, b = adc.ramp_errors()
            rows.append([round(el, 3), int(a), int(b)])
            _tick(progress, el / seconds, f"t={el:.1f}s  A={a:,} B={b:,}",
                  point=[round(el, 3), int(a), int(b)])
        a, b = adc.ramp_errors()
        st = adc.adc_status()
        # One capture as an independent opinion. The checker and the converter
        # have disagreed before -- the whole reason ramp_deviations exists --
        # and a single block costs milliseconds against a run measured in
        # seconds, so there is no reason to take reg11/reg12 on trust.
        ha = hb = None
        if pattern == A.ADC_TEST_RAMP:
            try:
                d = adc.capture(1 << 18, channel=A.CH_BOTH)
                ha = A.ramp_deviations(d[0::2])["errors"]
                hb = A.ramp_deviations(d[1::2])["errors"]
            except Exception:
                pass
    ok = (a == 0 and b == 0)
    agree = None if ha is None else ((a == 0) == (ha == 0) and (b == 0) == (hb == 0))
    n = seconds * A.BASE_CLOCK_HZ
    return DiagResult(
        "ramp_check", {"seconds": seconds, "pattern": pattern}, ok,
        (f"A={a:,} B={b:,} over {n:,.0f} samples/channel"
         + ("" if ok else f"  ({a/n:.3f}, {b/n:.3f} per sample)")
         + ("" if agree is None else
            f"; host cross-check {ha:,}/{hb:,}"
            + ("" if agree else " -- DISAGREES with the FPGA counters"))),
        {"errors_a": int(a), "errors_b": int(b), "samples": int(n),
         "host_errors_a": ha, "host_errors_b": hb, "agree": agree,
         "saturated": bool(a == 0xFFFFFFFF or b == 0xFFFFFFFF), "status": st},
        ["t_s", "errors_a", "errors_b"], rows)


class _pattern_ctx:
    """ramp_mode() for an arbitrary pattern, with the same restore guarantee."""

    def __init__(self, adc, code):
        self.adc, self.code = adc, code

    def __enter__(self):
        set_pattern(self.adc, self.code)
        return self.adc

    def __exit__(self, *a):
        try:
            set_pattern(self.adc, A.ADC_TEST_NORMAL)
        except Exception:
            pass
        return False


# ---------------------------------------------------------- 4/5. Eye scans
def _eye_errors_fpga(adc, dwell):
    adc.ramp_clear()
    time.sleep(dwell)
    a, b = adc.ramp_errors()
    return int(a), int(b)


def _eye_errors_host(adc, nsamp):
    d = adc.capture(nsamp, channel=A.CH_BOTH)
    return (A.ramp_deviations(d[0::2])["errors"],
            A.ramp_deviations(d[1::2])["errors"])


def eye_scan(adc, method="fpga", lo=0, hi=A.DELAY_TAPS, step=4, dwell=0.010,
             samples=65536, ps_per_tap=None, progress=None, cancel=None):
    """Sweep the FPGA data delay and count errors per tap on both channels.

    method='fpga'  reg11/reg12. The default: it needs no capture, so it is
                   faster, and the FPGA sees every sample rather than a block.
    method='host'  the same measurement from captured data.

    Both are kept because they HAVE disagreed. The checker used to assert a
    +1-per-conversion ramp that this converter does not produce (it decrements
    and holds each value for two conversions), so it read errors at every tap
    and could never find an eye, and only the host method worked. The
    2026-09-29 bitstream rewrote the checker to test the second difference,
    x[n]-x[n-2] == x[n-2]-x[n-4], which accepts that pattern. Measured after
    the change, step 4: fpga centre 122 / width 248 taps, host centre 124 /
    width 252, disagreeing on 2 of 128 taps -- both at the transition edge,
    where a tap is marginal by definition. Keeping both means the next time
    the checker and the converter disagree, one line of CLI proves which.
    """
    ps = A.PS_PER_TAP if ps_per_tap is None else float(ps_per_tap)
    step = max(1, int(step))
    samples = max(256, int(samples)) - (int(samples) % A.SAMPLE_GRANULARITY_DUAL)
    taps = list(range(int(lo), int(hi), step))
    rows = []
    tap0 = adc.get_tap()["requested"]
    with adc.ramp_mode():
        for i, tap in enumerate(taps):
            _ck(cancel)
            adc.wr(A.REG_DELAY, tap)
            time.sleep(50e-6)
            if method == "fpga":
                ea, eb = _eye_errors_fpga(adc, dwell)
            else:
                ea, eb = _eye_errors_host(adc, samples)
            rows.append([tap, ea, eb])
            _tick(progress, (i + 1) / len(taps), f"tap {tap}",
                  point=[tap, ea, eb])
        adc.wr(A.REG_DELAY, tap0)

    best = _widest_zero(rows, step)
    d = {"method": method, "ps_per_tap": ps, "ui_taps": 2000.0 / ps,
         "tap0": tap0, "taps": len(rows)}
    if best:
        wlo, whi, width = best
        centre = (wlo + whi) // 2
        edge = wlo <= lo or whi >= hi - step
        d.update({"found": True, "lo": wlo, "hi": whi, "width_taps": width,
                  "width_ps": width * ps, "width_ns": width * ps / 1000.0,
                  "centre": centre, "touches_edge": edge})
        summary = (f"eye taps {wlo}..{whi}, {width} taps = "
                   f"{width*ps/1000.0:.2f} ns, centre {centre}"
                   + ("  (touches the range end -- true centre is outside "
                      "the IDELAY range; shift with the ADC DCO delay)" if edge else ""))
        ok = True
    else:
        d["found"] = False
        summary = "no tap with zero errors on both channels"
        ok = False
    return DiagResult("eye_scan",
                      {"method": method, "lo": lo, "hi": hi, "step": step,
                       "dwell": dwell, "samples": samples, "ps_per_tap": ps},
                      ok, summary, d, ["tap", "errors_a", "errors_b"], rows)


def _widest_zero(rows, step):
    best = cur = None
    for tap, a, b in rows:
        if a == 0 and b == 0:
            cur = (tap, tap) if cur is None else (cur[0], tap)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    if best is None:
        return None
    return best[0], best[1], best[1] - best[0] + step


# ------------------------------------------------------------ 6. DCO sweep
def dco_sweep(adc, dco_values=None, step=16, method="host", dwell=0.010,
              samples=16384, progress=None, cancel=None):
    """eye_scan repeated for each ADC DCO output-delay setting (0x17).

    Produces a (DCO x tap) error grid for a heatmap, plus the eye width at
    each setting. Also the only way to calibrate tap size: shifting the DCO by
    a known number of picoseconds and watching how far the transition moves
    gives ps-per-tap directly.
    """
    vals = list(range(0, 32, 4)) if dco_values is None else list(dco_values)
    grid, widths, rows = [], [], []
    total = len(vals) + 1
    try:
        for k, dco in enumerate([None] + vals):
            _ck(cancel)
            if dco is None:
                adc.adc_wr_transfer(A.ADC_DCO_DELAY, 0x00)
                label, ps = "off", None
            else:
                adc.adc_wr_transfer(A.ADC_DCO_DELAY, 0x80 | (int(dco) & 0x1F))
                label, ps = f"0x{0x80 | (int(dco) & 0x1F):02X}", (int(dco) + 1) * 100
            time.sleep(0.02)
            r = eye_scan(adc, method=method, step=step, dwell=dwell,
                         samples=samples,
                         progress=lambda f, m, **kw: _tick(
                             progress, (k + f) / total, f"DCO {label}: {m}"),
                         cancel=cancel)
            errs = [row[1] + row[2] for row in r.rows]
            taps = [row[0] for row in r.rows]
            grid.append(errs)
            w = r.data.get("width_taps", 0) if r.data.get("found") else 0
            widths.append({"dco": dco, "label": label, "ps": ps,
                           "width_taps": w, "found": r.data.get("found", False),
                           "centre": r.data.get("centre")})
            for t, e in zip(taps, errs):
                rows.append([label, ps if ps is not None else "", t, e])
    finally:
        try:
            adc.adc_wr_transfer(A.ADC_DCO_DELAY, 0x00)
        except Exception:
            pass
    return DiagResult(
        "dco_sweep",
        {"values": vals, "step": step, "method": method, "samples": samples},
        None,
        f"{len(grid)} DCO settings x {len(grid[0]) if grid else 0} taps",
        {"taps": list(range(0, A.DELAY_TAPS, step)), "grid": grid,
         "widths": widths, "labels": [w["label"] for w in widths]},
        ["dco", "dco_ps", "tap", "errors"], rows)


# ------------------------------------------------------ 7. Datapath counter
def counter_test(adc, nsamples=1048576, progress=None, cancel=None):
    """ChannelSel 0: the FPGA's own counter must be a clean ramp.

    Exercises FIFO, DDR writer, XDMA and the host decode WITHOUT the ADC, so a
    failure here separates a broken datapath from a broken converter capture.
    The counter is generated in fabric and does NOT pass through the ADC's
    output inverter, so it is checked on the raw codes.
    """
    n = int(nsamples)
    if n < 256 or n % A.SAMPLE_GRANULARITY:
        raise ValueError(f"nsamples must be a multiple of "
                         f"{A.SAMPLE_GRANULARITY} and >= 256 (got {n})")
    _tick(progress, 0.2, f"capturing {n:,} samples")
    d = adc.capture(n, channel=A.CH_TEST_RAMP)
    _ck(cancel)
    _tick(progress, 0.7, "checking ramp continuity")
    c = (d & 0x3FFF).astype(np.int32)
    diff = (c[1:] - c[:-1]) & 0x3FFF
    bad = int(np.count_nonzero(diff != 1))
    first_bad = int(np.argmax(diff != 1)) if bad else -1
    hi = bool((d & 0xC000).any())
    _tick(progress, 1.0, "done")
    ok = (bad == 0 and not hi)
    return DiagResult(
        "counter_test", {"nsamples": n}, ok,
        (f"{c.size:,} samples, {bad} violations"
         + ("" if not hi else ", bits 15:14 SET (should always be 0)")),
        {"samples": int(c.size), "violations": bad, "first_bad": first_bad,
         "high_bits_set": hi, "first": c[:16].tolist()})


# ----------------------------------------------------- 8. Streaming self-test
def stream_selftest(adc, channel=A.CH_TEST_RAMP, seconds=10.0, nchan=2,
                    ram_gb=1.0, progress=None, cancel=None):
    """Stream for N seconds and check continuity ACROSS segment boundaries.

    Carries the last sample of each segment into the next segment's check,
    which is the part a per-segment check misses and the only way a dropped or
    duplicated segment shows up.
    """
    import stream as ST
    nslots = max(3, int(ram_gb * 1000**3) // A.SEG_BYTES)
    st = ST.Stream(adc, channel, nchan=int(nchan), nslots=nslots, ram_ring=True)
    viol = [0]
    seen = [0]
    last = [None]

    def sink(views, n):
        # Counter mode only: check the ramp, carrying across the boundary.
        if channel != A.CH_TEST_RAMP:
            seen[0] += 1
            return
        for v in views:
            a = np.frombuffer(v, dtype='<u2') & 0x3FFF
            if a.size < 2:
                continue
            d = (a[1:].astype(np.int32) - a[:-1].astype(np.int32)) & 0x3FFF
            viol[0] += int(np.count_nonzero(d != 1))
            if last[0] is not None:
                step = (int(a[0]) - last[0]) & 0x3FFF
                if step != 1:
                    viol[0] += 1
            last[0] = int(a[-1])
        seen[0] += 1

    try:
        st.start_background(sink=sink)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            _ck(cancel)
            time.sleep(0.25)
            el = time.monotonic() - t0
            _tick(progress, el / seconds,
                  f"t={el:.1f}s  {st.n_read} segments, {st.n_lost} lost",
                  point=[round(el, 2), st.n_read, st.n_lost, viol[0]])
        st.stop_background()
    finally:
        try:
            st.close()
        except Exception:
            pass
    dt = max(1e-9, (st.t_end or time.monotonic()) - st.t_start)
    rate = st.bytes_read / dt / 1e9
    flags = int(st.flags_seen)
    ok = (viol[0] == 0 and st.n_lost == 0 and flags == 0)
    return DiagResult(
        "stream_selftest",
        {"channel": channel, "seconds": seconds, "nchan": nchan,
         "ram_gb": ram_gb}, ok,
        (f"{st.n_read} segments, {st.n_lost} lost, {viol[0]} ramp violations, "
         f"{rate:.2f} GB/s, reg6=0x{flags:X}"),
        {"segments": st.n_read, "lost": st.n_lost, "violations": viol[0],
         "gb_read": st.bytes_read / 1e9, "rate_gbps": rate, "flags": flags,
         "overrun": bool(flags & A.FLAG_OVERRUN),
         "fifo_overflow": bool(flags & A.FLAG_FIFO_OVF),
         "checked_continuity": channel == A.CH_TEST_RAMP})


# -------------------------------------------- 9. Channel identity / quality
def _sine_fit(x, fs):
    """Least-squares fit at the dominant FFT bin. Returns freq, amplitude,
    offset, residual rms, and how many residuals exceed 6 sigma."""
    x = np.asarray(x, np.float64)
    n = x.size
    w = np.hanning(n)
    S = np.abs(np.fft.rfft((x - x.mean()) * w))
    k = int(np.argmax(S[1:]) + 1)
    # refine by parabolic interpolation on the log magnitude
    if 1 <= k < S.size - 1:
        a, b, c = np.log(S[k-1] + 1e-30), np.log(S[k] + 1e-30), np.log(S[k+1] + 1e-30)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    f = k * fs / n
    t = np.arange(n) / fs
    M = np.column_stack([np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t), np.ones(n)])
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    fit = M @ coef
    r = x - fit
    rms = float(np.sqrt(np.mean(r**2)))
    out = int(np.count_nonzero(np.abs(r) > 6 * (rms if rms > 0 else 1)))
    amp = float(np.hypot(coef[0], coef[1]))
    return {"freq_hz": float(f), "amplitude_codes": amp,
            "offset_codes": float(coef[2]), "residual_rms_lsb": rms,
            "outliers_6sigma": out}


def _snr_sfdr(x, fs):
    x = np.asarray(x, np.float64)
    n = x.size
    w = np.hanning(n)
    P = np.abs(np.fft.rfft((x - x.mean()) * w))**2
    if P.size < 8:
        return {}
    k = int(np.argmax(P[1:]) + 1)
    sig = P[max(1, k-3):k+4].sum()
    rest = P[1:].sum() - sig
    spur = P[1:].copy()
    spur[max(0, k-4):k+4] = 0
    ks = int(np.argmax(spur) + 1)
    return {"snr_db": float(10*np.log10(sig / max(rest, 1e-30))),
            "sfdr_db": float(10*np.log10(P[k] / max(spur.max(), 1e-30))),
            "signal_bin_hz": float(k * fs / n),
            "worst_spur_hz": float(ks * fs / n)}


def channel_quality(adc, nsamples=262144, fs=None, progress=None, cancel=None):
    """ChannelSel 3 capture with an external sine: which channel carries it,
    sine-fit quality, SNR/SFDR. Needs a generator on one input."""
    fs = A.BASE_CLOCK_HZ if fs is None else float(fs)
    _tick(progress, 0.2, f"capturing {nsamples:,} pairs")
    d = adc.capture(int(nsamples), channel=A.CH_BOTH)
    _ck(cancel)
    out = {}
    for i, nm in ((0, "A"), (1, "B")):
        _tick(progress, 0.4 + 0.3 * i, f"analysing channel {nm}")
        x = A.adc_signed(d[i::2]).astype(np.float64)
        r = _sine_fit(x, fs)
        r.update(_snr_sfdr(x, fs))
        r["std_codes"] = float(x.std())
        r["min"] = int(x.min()); r["max"] = int(x.max())
        out[nm] = r
    amp_a, amp_b = out["A"]["amplitude_codes"], out["B"]["amplitude_codes"]
    carrier = "A" if amp_a > amp_b * 3 else ("B" if amp_b > amp_a * 3 else "both/neither")
    _tick(progress, 1.0, "done")
    return DiagResult(
        "channel_quality", {"nsamples": int(nsamples), "fs": fs}, None,
        (f"signal on {carrier}; A {amp_a:.1f} codes rms-fit "
         f"{out['A']['residual_rms_lsb']:.2f} LSB, "
         f"B {amp_b:.1f} codes rms-fit {out['B']['residual_rms_lsb']:.2f} LSB"),
        {"carrier": carrier, "A": out["A"], "B": out["B"],
         "output_invert": bool(A.OUTPUT_INVERT)})


# ------------------------------------- 10. Pattern capture / bit activity
def _expected_law(codes, pattern):
    """Deviations from the selected pattern's law, where one is defined.
    Returns (deviations, description) or (None, why-not)."""
    x = (np.asarray(codes) & 0x3FFF).astype(np.int64)
    if pattern in PATTERN_CONST:
        want = PATTERN_CONST[pattern] & 0x3FFF
        inv = A.OUTPUT_INVERT
        w = (want ^ 0x3FFF) if inv else want
        return int(np.count_nonzero(x != w)), \
            f"constant 0x{w:04X} ({PATTERN_NAME[pattern]})"
    if pattern in (0x04, 0x07):
        # checkerboard and one/zero toggle alternate between two words
        if x.size < 4:
            return None, "too few samples"
        bad = int(np.count_nonzero(x[2:] != x[:-2]))
        return bad, "alternation: x[n] == x[n-2]"
    if pattern == 0x0F:
        r = A.ramp_deviations(x)
        return r["errors"], (f"ramp law: each value held {r['hold']}, "
                             f"step {r['dir']:+d}")
    return None, f"no closed-form law for {PATTERN_NAME.get(pattern, hex(pattern))}"


def pattern_capture(adc, npairs=4096, pattern=None, progress=None, cancel=None):
    """Short ChannelSel 3 capture for the pattern graph.

    Returns the per-channel traces (raw codes and decoded), the first 64 words
    as hex/binary, per-bit one-fraction for both channels, and a check against
    the selected pattern's expected law.
    """
    n = int(npairs)
    n -= n % A.SAMPLE_GRANULARITY_DUAL
    n = max(A.SAMPLE_GRANULARITY_DUAL, n)
    _tick(progress, 0.2, f"capturing {n:,} pairs")
    if pattern is not None:
        set_pattern(adc, pattern)
        time.sleep(0.02)
    cur = adc.adc_rd(A.ADC_TEST_MODE)
    d = adc.capture(n, channel=A.CH_BOTH)
    _ck(cancel)
    _tick(progress, 0.7, "analysing")
    out = {"pattern": cur, "pattern_name": PATTERN_NAME.get(cur, f"0x{cur:02X}"),
           "output_mode": adc.adc_rd(A.ADC_OUTPUT_MODE),
           "output_invert": bool(A.OUTPUT_INVERT),
           "npairs": int(n), "channels": {}}
    for i, nm in ((0, "A"), (1, "B")):
        raw = (d[i::2] & 0x3FFF).astype(np.int32)
        dec = A.adc_signed(d[i::2]).astype(np.int32)
        bits = [float(np.mean((raw >> b) & 1)) for b in range(14)]
        dev, law = _expected_law(raw, cur)
        out["channels"][nm] = {
            "raw": raw.tolist(), "decoded": dec.tolist(),
            "hex64": [f"{v:04X}" for v in raw[:64].tolist()],
            "bin64": [f"{v:014b}" for v in raw[:64].tolist()],
            "bit_ones": bits,
            "min": int(raw.min()), "max": int(raw.max()),
            "unique": int(np.unique(raw).size),
            "decoded_mean": float(dec.mean()), "decoded_std": float(dec.std()),
            "deviations": dev, "law": law,
        }
    a = out["channels"]["A"]; b = out["channels"]["B"]
    devs = [c["deviations"] for c in (a, b) if c["deviations"] is not None]
    ok = (sum(devs) == 0) if devs else None
    _tick(progress, 1.0, "done")
    return DiagResult(
        "pattern_capture",
        {"npairs": int(n), "pattern": cur}, ok,
        (f"{out['pattern_name']}: A {a['unique']} unique / B {b['unique']} unique"
         + (f", {sum(devs)} deviations from {a['law']}" if devs else
            f"  ({a['law']})")),
        out)


ALL_TESTS = {
    "spi_selftest": spi_selftest,
    "register_dump": register_dump,
    "ramp_check": ramp_check,
    "eye_scan": eye_scan,
    "dco_sweep": dco_sweep,
    "counter_test": counter_test,
    "stream_selftest": stream_selftest,
    "channel_quality": channel_quality,
    "pattern_capture": pattern_capture,
}
