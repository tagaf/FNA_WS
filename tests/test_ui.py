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

print()
if fails:
    print("FAILURES:")
    for f_ in fails: print("  -", f_)
    sys.exit(1)
print("ALL UI CHECKS PASSED")
