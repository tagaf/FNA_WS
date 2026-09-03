"""Full control-matrix regression against `server.py --mock` (no hardware).

Asserts, per config: fs / nbins / bin_hz metadata, log-display metadata
(disp_f0 == first bin, disp_f1 == Nyquist), the DISPLAYED peak lands at the
physically expected frequency (incl. alias positions at divided rates),
trace/read/fft sample splits, Welch frame counts, zoom slice mapping,
single-shot semantics, and invalid-channel reject + recover.

Run:  python3 tests/test_controls.py    (exits nonzero on failure)
"""
import os, subprocess, sys, time, json, struct, urllib.request, numpy as np
PORT=18096; U=f"http://127.0.0.1:{PORT}"
srv=subprocess.Popen([sys.executable,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"server.py"),"--mock",
                      "--bind","127.0.0.1","-p",str(PORT)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
def get(p,t=10): return urllib.request.urlopen(U+p,timeout=t)
def post(o):
    rq=urllib.request.Request(U+"/control",data=json.dumps(o).encode(),
                              headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(rq,timeout=5).read()
def frame(since,t=12):
    r=get(f"/frame?wait=1&since={since}",t)
    b=r.read()
    if r.status!=200: return r.status,None,None
    hl=struct.unpack("<I",b[:4])[0]; m=json.loads(b[4:4+hl]); off=4+hl
    disp=np.frombuffer(b[off:off+m["disp_bins"]*4],np.float32)
    return 200,m,disp

fails=[]
def chk(name,cond,info=""):
    if not cond: fails.append(f"{name}: {info}")
    return cond

def applied(want, last, tmo=12):
    t0=time.monotonic()
    while time.monotonic()-t0<tmo:
        st,m,d=frame(last)
        if st!=200: continue
        last=m["totals"]["frames"]
        if all(m["cfg"].get(k)==v for k,v in want.items()):
            return m,d,last
    raise TimeoutError(f"cfg never applied: {want}")

def alias(f0,fs):
    fa=f0%fs
    return min(fa,fs-fa)

try:
    for _ in range(80):
        time.sleep(0.1)
        try: get("/status"); break
        except Exception: pass
    last=0
    F0=25e6; BASE=250e6

    print(f"{'case':<46}{'fs/2 MHz':>9}{'peak MHz':>10}{'disp MHz':>10}{'ok':>4}")
    cases=[]
    for N,nfft in [(4096,1024),(4096,4096),(262144,4096),(1048576,8192),
                   (1048576,65536),(16777216,65536)]:
        cases.append(dict(nsamples=N,nfft=nfft,speed=0,max_frames=1,trace_samples=4096))
    for sp in (1,7):
        cases.append(dict(nsamples=1048576,nfft=8192,speed=sp,max_frames=1,trace_samples=4096))
    for mf in (64,256):
        cases.append(dict(nsamples=1048576,nfft=8192,speed=0,max_frames=mf,trace_samples=4096))
    cases.append(dict(nsamples=1048576,nfft=8192,speed=0,max_frames=64,trace_samples=262144))

    for cfg in cases:
        post(cfg); m,disp,last=applied(cfg,last)
        # settle one extra frame (EMA reset etc.)
        m,disp,last=applied(cfg,last)
        fs=BASE/(cfg["speed"]+1); nfft=cfg["nfft"]; N=cfg["nsamples"]
        fa=alias(F0,fs)
        a=m["acq"]; ok=True
        ok&=chk("fs",abs(m["fs_hz"]-fs)<1,f"{m['fs_hz']}")
        ok&=chk("nbins",m["nbins"]==nfft//2+1,f"{m['nbins']}")
        ok&=chk("bin_hz",abs(m["bin_hz"]-fs/nfft)<0.01,f"{m['bin_hz']}")
        ok&=chk("disp_bins",m["disp_bins"]==4096,f"{m['disp_bins']}")
        ok&=chk("disp_f0==bin",abs(m["disp_f0"]-m["bin_hz"])<0.01,f"{m['disp_f0']}")
        ok&=chk("disp_f1==fs/2",abs(m["disp_f1"]-fs/2)<1.0,f"{m['disp_f1']}")
        need=cfg["max_frames"]*nfft
        exp_trace=min(cfg["trace_samples"],N)
        ok&=chk("trace",a["trace_samples"]==exp_trace,f"{a['trace_samples']}")
        ok&=chk("fft_samples",a["fft_samples"]==min(need,a["read_samples"]),
                f"{a['fft_samples']}")
        ok&=chk("read>=uses",a["read_samples"]>=max(min(need,N)-255,exp_trace),
                f"read={a['read_samples']}")
        ok&=chk("nframes",m["nframes"]==min(cfg["max_frames"],a["read_samples"]//nfft),
                f"{m['nframes']}")
        ok&=chk("bytes_ok",a["bytes_ok"]); ok&=chk("err",m["err"] is None,str(m["err"]))
        tolm=2*m["bin_hz"]
        ok&=chk("peak_hz",abs(m["sig"]["peak_hz"]-fa)<=tolm,
                f"{m['sig']['peak_hz']/1e6:.3f} vs {fa/1e6:.3f}")
        # displayed argmax must land at the log-spaced point covering fa
        nb=m["nbins"]; K=m["disp_bins"]
        idx=np.power(10.0,np.linspace(0.0,np.log10(nb-1),K+1))
        starts=np.maximum.accumulate(np.clip(np.floor(idx[:-1]).astype(np.int64),1,nb-2))-1
        b=int(round(fa/m["bin_hz"]))-1              # index into shown[1:]
        kexp=int(np.searchsorted(starts,b,side="right"))-1
        di=int(np.argmax(disp))
        dfreq=(starts[di]+1)*m["bin_hz"]
        # coarse FFTs duplicate one bin across several log points; argmax
        # returns the first copy -- judge by FREQUENCY, which is what the
        # axis draws, with the point index as a secondary sanity bound
        ok&=chk("disp_axis",abs(dfreq-fa)<=1.01*m["bin_hz"] or abs(di-kexp)<=2,
                f"{dfreq/1e6:.4f} vs {fa/1e6:.4f} MHz (pt {di}/{kexp})")
        tag=f"N={N} nfft={nfft} sp={cfg['speed']} mf={cfg['max_frames']} tr={cfg['trace_samples']}"
        print(f"{tag:<46}{fs/2/1e6:>9.2f}{m['sig']['peak_hz']/1e6:>10.3f}{dfreq/1e6:>10.3f}{'OK' if ok else 'FAIL':>4}")

    # trace follows the record by default
    post({"trace_samples":-1,"nsamples":1048576,"nfft":8192,"max_frames":4})
    m,disp,last=applied({"nsamples":1048576,"trace_samples":-1},last)
    m,disp,last=applied({"nsamples":1048576},last)
    chk("trace-follows-N",m["acq"]["trace_samples"]==1048576,
        f"{m['acq']['trace_samples']}")
    print(f"trace default follows record: trace={m['acq']['trace_samples']:,} of N=1,048,576 "
          + ("OK" if m["acq"]["trace_samples"]==1048576 else "FAIL"))

    # zoom slice mapping
    post({"zoom":[20e6,30e6],"nsamples":1048576,"nfft":65536,"speed":0,"max_frames":8})
    m,disp,last=applied({"nfft":65536,"max_frames":8},last)
    m,disp,last=applied({"nfft":65536},last)
    z=m["zoom"]; okz=z["active"] and z["bins"]>0
    r=get(f"/frame?wait=1&since={last-1}"); b=r.read()
    hl=struct.unpack("<I",b[:4])[0]; mm=json.loads(b[4:4+hl]); off=4+hl+mm["disp_bins"]*4
    zs=np.frombuffer(b[off:off+mm["zoom"]["bins"]*4],np.float32)
    zi=int(np.argmax(zs)); zf=mm["zoom"]["lo_hz"]+zi*mm["zoom"]["bin_hz"]
    okz&=abs(zf-25e6)<=2*mm["zoom"]["bin_hz"]
    chk("zoom",okz,f"zoom peak {zf/1e6:.4f} MHz bins={mm['zoom']['bins']}")
    print(f"zoom 20-30 MHz: bins={mm['zoom']['bins']} peak at {zf/1e6:.4f} MHz {'OK' if okz else 'FAIL'}")
    post({"zoom":None})

    # single-shot: exactly one frame
    post({"running":False}); time.sleep(0.4)
    st0=json.loads(get("/status").read())["frames"]
    post({"trigger":True}); time.sleep(1.2)
    st1=json.loads(get("/status").read())["frames"]; time.sleep(0.6)
    st2=json.loads(get("/status").read())["frames"]
    chk("single",st1==st0+1 and st2==st1,f"{st0}->{st1}->{st2}")
    print(f"single-shot: {st0} -> {st1} -> {st2} {'OK' if st1==st0+1 and st2==st1 else 'FAIL'}")
    post({"running":True})

    # invalid channel: error surfaces, then clears
    post({"channel":3}); time.sleep(0.8)
    e1=json.loads(get("/status").read())["err"] or ""
    post({"channel":1}); time.sleep(0.8)
    e2=json.loads(get("/status").read())["err"]
    chk("ch3-err","Channel_Set=3" in e1 and e2 is None,f"err1={e1[:60]!r} err2={e2!r}")
    print(f"channel=3 rejected then recovers: {'OK' if 'Channel_Set=3' in e1 and e2 is None else 'FAIL'}")

    st=json.loads(get("/status").read())
    print(f"\ntimeouts={st['timeouts']}  residual err={st['err']}")
    print("FAILURES:" if fails else "ALL CONTROL CHECKS PASSED")
    for f_ in fails: print("  -",f_)
    if fails: sys.exit(1)
finally:
    srv.terminate()
    try: srv.wait(timeout=10)
    except subprocess.TimeoutExpired: srv.kill()
