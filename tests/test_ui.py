#!/usr/bin/env python3
"""Static checks on web/index.html.

The UI is one large inline <script>, edited surgically. A syntax error takes
the whole page down with no server-side symptom, and stale references to
payload fields that were removed fail silently as `undefined`. This gates
both. Requires node for the syntax check; skips it with a warning if absent.

Run: python3 tests/test_ui.py
"""
import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(os.path.dirname(HERE), "web", "index.html")

fails = []
def chk(name, cond, info=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{info}]" if info else ""))
    if not cond:
        fails.append(f"{name}: {info}")

src = open(HTML).read()
m = re.search(r"<script>(.*)</script>", src, re.S)
chk("script block present", bool(m))
js = m.group(1) if m else ""

node = shutil.which("node") or shutil.which("nodejs")
if node:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js); path = f.name
    try:
        r = subprocess.run([node, "--check", path], capture_output=True, text=True)
        chk("javascript parses", r.returncode == 0,
            (r.stderr or "").strip().splitlines()[-1] if r.returncode else "")
    finally:
        os.unlink(path)
else:
    print("  SKIP  javascript parses (node not found)")

# fields removed from the wire during the /noise split: any surviving
# reference silently reads undefined at runtime
for dead in ("an.spurs", "fam.dbs", "stc.spurs", "st.spurs", "disp_bin_hz"):
    chk(f"no stale reference to {dead}", dead not in js)

# every element the script addresses by id must exist in the markup
ids_used = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", js))
ids_decl = set(re.findall(r'id="([A-Za-z0-9_]+)"', src))
missing = sorted(ids_used - ids_decl)
chk("every $('id') exists in the markup", not missing, ", ".join(missing))

# controls the server accepts must all be wired
for cid in ("channel", "nsamples", "speed", "nfft", "max_frames", "avg",
            "classify", "pfa_exp"):
    chk(f"control '{cid}' is in CTL_IDS or handled",
        f"'{cid}'" in js, "")

# phase-noise controls, same rule. Two wiring styles are in use -- a direct
# $('id').addEventListener and a ['a','b'].forEach(id=>$(id).addEventListener)
# for the group that all send the same payload -- so accept either.
bulk = set()
for lst in re.findall(r"\[([^\]]*)\]\.forEach\(id=>\s*\n?\s*\$\(id\)\.addEventListener", js):
    bulk |= set(re.findall(r"'([A-Za-z0-9_]+)'", lst))
for cid in ("pnEnabled", "pnLength", "pnGeom", "pnNg", "pnTau", "pnCalMode",
            "pnRequire", "pnPsi", "pnDecim", "pnNperseg", "pnWindow",
            "pnFmax", "pnPre", "pnRecal"):
    chk(f"phase-noise control '{cid}' is wired",
        f"$('{cid}').addEventListener" in js or cid in bulk, "")

# The static checks above cannot see a runtime error inside a draw function,
# and no browser launches on this host (headless Chromium finds no usable
# sandbox under the Jetson's AppArmor policy). test_pn_render.js runs the
# real render path against a real /pn payload under a DOM stub instead.
RENDER = os.path.join(HERE, "test_pn_render.js")
if node and os.path.exists(RENDER):
    import json, signal, subprocess as sp, time, urllib.request
    print("\n  running the phase-noise render check (spins up --mock-interferometer)")
    root = os.path.dirname(HERE)
    proc = sp.Popen([sys.executable, os.path.join(root, "server.py"),
                     "--mock-interferometer", "-n", "4194304", "-p", "8098"],
                    stdout=sp.DEVNULL, stderr=sp.DEVNULL, cwd=root)
    try:
        d = None
        for _ in range(90):
            time.sleep(1)
            try:
                d = json.load(urllib.request.urlopen(
                    "http://localhost:8098/pn", timeout=5))
            except Exception:
                continue
            if d.get("f") and not d.get("error"):
                break
        chk("mock interferometer produced a phase-noise result",
            bool(d and d.get("f")), (d or {}).get("error") or "")
        if d and d.get("f"):
            # the mock is built to a 50 kHz Lorentzian; the whole chain is
            # only working if that number comes back out of it
            got = d.get("lw_white_hz") or 0
            chk("recovered linewidth matches the mock's 50 kHz",
                0.8 < got / 50e3 < 1.25, f"{got/1e3:.1f} kHz")
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as fh:
                json.dump(d, fh)
                pnjson = fh.name
            r = sp.run([node, RENDER, HTML, pnjson], capture_output=True,
                       text=True)
            print("\n".join("  " + l for l in r.stdout.strip().splitlines()))
            chk("phase-noise render checks", r.returncode == 0,
                r.stderr.strip()[:200])
            os.unlink(pnjson)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=25)
        except Exception:
            proc.kill()

print()
if fails:
    print("FAILURES:")
    for f_ in fails: print("  -", f_)
    sys.exit(1)
print("ALL UI CHECKS PASSED")
