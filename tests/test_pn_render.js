// Runs web/index.html's phase-noise render path against a REAL /pn payload,
// under a minimal DOM stub.
//
// The alternative -- a browser -- does not launch on this host (headless
// Chromium has no usable sandbox under the Jetson's AppArmor policy), and a
// static syntax check cannot catch the failures that actually happen here:
// a field renamed on the server, a null slipping into an arithmetic path, a
// canvas call on an element that does not exist. Those are runtime errors in
// draw code that only ever runs on one tab, so nothing else would notice.
//
// Usage: node tests/test_pn_render.js <index.html> <pn-payload.json>
const fs = require('fs');

const [htmlPath, jsonPath] = process.argv.slice(2);
const src = fs.readFileSync(htmlPath, 'utf8');
const payload = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
const js = src.match(/<script>([\s\S]*)<\/script>/)[1];

const fails = [];
const chk = (name, cond, info) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}` + (info ? `  [${info}]` : ''));
  if (!cond) fails.push(name + (info ? ': ' + info : ''));
};

// ---- canvas stub that COUNTS work, so "drew nothing" is a detectable state
function makeCtx(el) {
  const calls = { stroke: 0, fill: 0, fillText: 0, fillRect: 0, lineTo: 0 };
  const noop = () => {};
  const ctx = {
    _calls: calls, _el: el,
    setTransform: noop, clearRect: noop, beginPath: noop, moveTo: noop,
    closePath: noop, strokeRect: noop, save: noop, restore: noop,
    translate: noop, rotate: noop, setLineDash: noop, putImageData: noop,
    drawImage: noop, scale: noop, arc: noop, rect: noop, clip: noop,
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    stroke: () => calls.stroke++,
    fill: () => calls.fill++,
    fillText: (t) => { calls.fillText++; calls.lastText = String(t); },
    fillRect: () => calls.fillRect++,
    lineTo: () => calls.lineTo++,
  };
  return ctx;
}

const ids = new Map();
function makeEl(id) {
  const el = {
    id, innerHTML: '', textContent: '', value: '', checked: false,
    disabled: false, width: 0, height: 0, dataset: {}, style: {},
    children: [], _ctx: null,
    classList: { _s: new Set(), add(x) { this._s.add(x) }, remove(x) { this._s.delete(x) },
                 contains(x) { return this._s.has(x) }, toggle(x) { this._s.add(x) } },
    _listeners: {},
    addEventListener(ev, fn) { (this._listeners[ev] ||= []).push(fn) },
    dispatch(ev, e) { (this._listeners[ev] || []).forEach(fn => fn(e || { target: this })) },
    getBoundingClientRect: () => ({ width: 800, height: 300, left: 0, top: 0 }),
    getContext() { return this._ctx ||= makeCtx(this) },
    querySelectorAll: () => [],
    appendChild: noopEl, removeChild: noopEl, remove: noopEl, focus: noopEl,
    setAttribute: noopEl, getAttribute: () => null,
  };
  return el;
}
function noopEl() {}
for (const m of src.matchAll(/id="([A-Za-z0-9_]+)"/g)) ids.set(m[1], makeEl(m[1]));

const tabbtns = [...src.matchAll(/data-tab="([a-z]+)"/g)].map(m => {
  const el = makeEl('tab_' + m[1]); el.dataset.tab = m[1]; return el;
});

global.document = {
  getElementById: id => ids.get(id) || null,
  querySelectorAll: sel => sel.includes('tabbtn') ? tabbtns : [],
  createElement: () => makeEl('_tmp'),
  addEventListener: noopEl,
  activeElement: null,
};
global.window = {
  devicePixelRatio: 1, addEventListener: noopEl,
  requestAnimationFrame: () => {},        // never fires: parks poll() harmlessly
};
global.requestAnimationFrame = window.requestAnimationFrame;
global.performance = { now: () => Date.now() };
global.location = { reload: noopEl };
global.setTimeout = () => 0;              // no timers: one render pass, no loop
global.setInterval = () => 0;
global.clearTimeout = noopEl;
global.confirm = () => false;

const fetched = [];
global.fetch = async (url) => {
  fetched.push(url);
  if (url.startsWith('/pn') && !url.includes('control'))
    return { ok: true, status: 200, json: async () => payload };
  if (url.startsWith('/limits'))
    return { ok: true, status: 200, json: async () => ({ max_samples: 262144000 }) };
  return { ok: true, status: 204, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(0) };
};

// ---- run the page script, then reach in
let api;
try {
  api = new Function(js + '\n;return {pnRender,pnDrawSpectra,pnDrawLissajous,' +
                     'pnDrawLinewidth,pnDrawTrace,drawLogLog,pnSend};')();
  chk('page script evaluates under the DOM stub', true);
} catch (e) {
  chk('page script evaluates under the DOM stub', false, e.message);
  process.exit(1);
}

// ---- the real payload renders
try {
  api.pnRender(payload);
  chk('pnRender(real /pn payload) does not throw', true);
} catch (e) {
  chk('pnRender(real /pn payload) does not throw', false, e.stack.split('\n')[0] + ' | ' + e.stack.split('\n')[1]);
}

const strokes = id => (ids.get(id)._ctx ? ids.get(id)._ctx._calls.stroke : 0);
chk('frequency-noise plot drew traces', strokes('cPnNu') >= 2, 'strokes=' + strokes('cPnNu'));
chk('phase-noise plot drew traces', strokes('cPnPhi') >= 2, 'strokes=' + strokes('cPnPhi'));
chk('linewidth plot drew', strokes('cPnLw') >= 1, 'strokes=' + strokes('cPnLw'));
chk('Lissajous drew the point cloud',
    (ids.get('cPnLiss')._ctx || { _calls: {} })._calls.fillRect > 100,
    'points=' + (ids.get('cPnLiss')._ctx || { _calls: {} })._calls.fillRect);
chk('phase trace drew an envelope', strokes('cPnTrace') >= 1);

const res = ids.get('tPnResult').innerHTML;
const acq = ids.get('tPnAcq').innerHTML;
chk('result table is populated', res.length > 200, res.length + ' chars');
chk('result table has no undefined/NaN', !/undefined|NaN/.test(res),
    (res.match(/undefined|NaN/g) || []).join(','));
chk('acquisition table has no undefined/NaN', !/undefined|NaN/.test(acq),
    (acq.match(/undefined|NaN/g) || []).join(','));
chk('linewidth is reported', /linewidth/.test(res));
chk('banner says something', ids.get('pnBanner').innerHTML.length > 20);

// ---- the states that are NOT a happy result must render too, not throw
for (const [name, d] of [
  ['off',        { enabled: false, cfg: { enabled: 0, length_m: 10 }, tau_s: 9.8e-8 }],
  ['wrong channel', { enabled: true, channel: 1, cfg: { enabled: 1, length_m: 10 }, tau_s: 9.8e-8 }],
  ['pending',    { enabled: true, channel: 3, pending: true, cfg: { enabled: 1, length_m: 10 }, tau_s: 9.8e-8 }],
  ['error',      { enabled: true, channel: 3, error: 'calibration not trustworthy: x',
                   cal: { a1: 10, a2: 10, dc1: 0, dc2: 0, psi_deg: 90, span: 0.1, resid: 0.5,
                          source: 'ellipse', trustworthy: false },
                   cfg: { enabled: 1, length_m: 10 }, tau_s: 9.8e-8 }],
]) {
  try { api.pnRender(d); chk(`renders the "${name}" state`, true) }
  catch (e) { chk(`renders the "${name}" state`, false, e.message) }
}

// ---- a payload full of nulls (every FSR bin masked) must not crash
try {
  const nulled = Object.assign({}, payload);
  for (const k of ['S_nu', 'S_dnu', 'S_phi', 'S_dphi', 'L'])
    if (nulled[k]) nulled[k] = nulled[k].map(() => null);
  api.pnRender(nulled);
  chk('all-masked spectrum renders as an empty plot', true);
} catch (e) {
  chk('all-masked spectrum renders as an empty plot', false, e.message);
}

console.log(fails.length ? `\n${fails.length} FAILED` : '\nALL PHASE-NOISE RENDER CHECKS PASSED');
process.exit(fails.length ? 1 : 0);
