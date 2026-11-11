/* rounderr_ui.js   the window
   ==========================
   Nothing waits for a button. Every edit of the formula, a number or the precision reruns the
   analysis, and every analysis is followed by the experiments (precision sweep, random trials,
   input landscape). A newer edit makes older answers stale and they are dropped.

   The page talks to the engine through api(route, body). Two engines answer the same routes
       http     rounderr_app.py running on this computer (native Python, Bean, model readers)
       worker   engine-worker.js, the same Python files inside Pyodide, no server at all
*/
'use strict';
const $ = id => document.getElementById(id);
const S = {payload: null, sel: 0, tab: 'Overview', trials: null, prec: null, land: null, vals: {}, auto: new Set(), vars: [],
  status: {}, examples: [], qa: [], dev: null, open: new Set(['stage:read']), seq: 0, bits: 53, autorun: true, discrete: [],
  tn: 2000, ts: 'x10', landX: null, landY: undefined, landDec: 2, expStale: false, nums: {}, graphSig: null, times: {}, ready: false};
const TABS = ['Overview', 'Tape', 'Trials', 'Matrix', 'Bean', 'Data', 'Developer'];
const EXP = new Set(['scatter', 'prec', 'land']);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const words = s => esc(s).replace(/(\d(?:\.\d+)?)e([+-]?\d+)/g, (_, m, e) => m + '×10<sup>' + Number(e) + '</sup>').replace(/lambda_i/g, 'λᵢ').replace(/\blambda\b/g, 'λ').replace(/\bkappa\b/g, 'κ');
const pretty = s => esc(s).replace(/\s*\*\*\s*2\b/g, '²').replace(/\s*\*\*\s*3\b/g, '³').replace(/ \* /g, ' × ');
const numStr = x => String(Number(Number(x).toPrecision(12)));
const ms = x => x === undefined || x === null ? '' : (x < 1 ? '<1 ms' : x < 1000 ? Math.round(x) + ' ms' : (x / 1000).toFixed(1) + ' s');

/* ------------------------------------------------------------------ the two engines */
// the same tag index.html puts on this script, passed on so one deploy is never served half from cache
const VERSION = (document.currentScript && new URL(document.currentScript.src).search) || '';
let backend = null, worker = null, nextId = 0;
const waiting = new Map();
function api(route, body) {
  if (backend === 'http') {
    const get = route === 'status' || route === 'examples' || route === 'dev';
    return fetch('api/' + route, get ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})}).then(r => r.json());
  }
  return new Promise(done => { const id = ++nextId; waiting.set(id, done); worker.postMessage({id, route, body}); });
}
function engineChip(state, text) {
  $('engine').innerHTML = `<i class="led ${state}"></i><span>${esc(text)}</span>`;
}
function live(state, text) {
  $('liveled').className = 'led ' + state;
  if (text !== undefined) $('livetext').textContent = text;
}
function say(text) { const m = $('msg'); m.hidden = !text; m.textContent = text || ''; }

async function boot() {
  const local = /^(127\.0\.0\.1|localhost|\[::1\])$/.test(location.hostname);
  if (local) {
    try { const r = await fetch('api/status'); if (r.ok) { S.status = await r.json(); backend = 'http'; } } catch (e) { /* no server, use the browser build */ }
  }
  if (backend === 'http') {
    S.ready = true;
    engineChip('on', 'native Python ' + S.status.python + ' on this computer');
    S.examples = await api('examples');
    chips(); statusLine();
    const ex = S.examples[0];
    $('text').value = '-b + sqrt(b^2 - c)'; S.vals = Object.assign({}, ex.values);
    run();
    return;
  }
  backend = 'worker';
  $('bootbar').hidden = false;
  engineChip('busy', 'starting Python in this page');
  for (const o of $('mode').options) if (o.value === 'anthropic' || o.value === 'openai') { o.disabled = true; o.textContent += ' (run the app locally)'; }
  worker = new Worker('engine-worker.js' + VERSION);
  worker.onmessage = onWorker;
  worker.onerror = e => { engineChip('bad', 'engine failed to start'); say('The Python engine could not start in this browser. ' + (e.message || '')); $('bootbar').hidden = true; };
  try {                                             // a saved run of the default example paints at once
    const snap = await fetch('snapshot.json' + VERSION).then(r => r.json());
    if (!S.ready && !S.payload) {
      S.status = snap.status; S.examples = snap.examples; chips();
      $('text').value = snap.text; S.vals = Object.assign({}, snap.values);
      S.trials = snap.trials.trials; S.prec = snap.precision; S.land = snap.landscape;
      live('busy', 'saved run shown while Python starts');
      show(snap.analyze, true);
    }
  } catch (e) { /* no snapshot, the live run paints when the engine is up */ }
}
async function onWorker(e) {
  const m = e.data;
  if (m.type === 'reply') { const done = waiting.get(m.id); waiting.delete(m.id); if (done) done(JSON.parse(m.json)); return; }
  if (m.type === 'boot') { engineChip('busy', {runtime: 'downloading Python', packages: 'loading mpmath', engine: 'starting the engine'}[m.stage] || 'starting'); return; }
  if (m.type === 'fail') { engineChip('bad', 'engine failed'); say('The Python engine failed. ' + m.error); $('bootbar').hidden = true; return; }
  if (m.type === 'numpy') { S.numpy = true; return; }
  if (m.type === 'ready') {
    S.ready = true; $('bootbar').hidden = true;
    engineChip('on', 'Python running in this page');
    S.status = await api('status');
    if (!S.examples.length) { S.examples = await api('examples'); chips(); }
    statusLine();
    if (!$('text').value.trim()) { $('text').value = '-b + sqrt(b^2 - c)'; S.vals = {b: '1000', c: '1'}; }
    if (S.seq === 0) run();                         // if the user already typed, that run is on its way
  }
}

/* ------------------------------------------------------------------ the left side */
function chips() {
  $('examples').innerHTML = S.examples.map((x, i) => `<button data-ex="${i}" title="${esc(x.text)}">${esc(x.name.replace(/\s{2,}.*/, '').replace(/ at x = .*/, ''))}</button>`).join('');
}
function fields(vars, shown, missing = []) {
  const box = $('fields'), same = vars.length === S.vars.length && vars.every((v, i) => v === S.vars[i]);
  S.vars = vars;
  $('valwrap').hidden = !vars.length;
  if (!same) {
    const act = document.activeElement, focus = act && act.dataset ? act.dataset.v : null;
    box.innerHTML = vars.map(v => `<div class="field" data-f="${esc(v)}"><span>${esc(v)}</span><input data-v="${esc(v)}" aria-label="value of ${esc(v)}" inputmode="decimal" autocomplete="off"><button data-step="${esc(v)}" data-by="0.1" title="divide by 10" aria-label="divide ${esc(v)} by 10">÷10</button><button data-step="${esc(v)}" data-by="10" title="multiply by 10" aria-label="multiply ${esc(v)} by 10">×10</button></div>`).join('');
    if (focus) { const again = box.querySelector(`input[data-v="${CSS.escape(focus)}"]`); if (again) again.focus(); }
  }
  box.querySelectorAll('.field').forEach(f => {
    const v = f.dataset.f, inp = f.querySelector('input');
    if (inp !== document.activeElement || !same) { const val = (shown && shown[v] !== undefined) ? shown[v] : (S.vals[v] ?? ''); if (inp.value !== String(val)) inp.value = val; }
    f.classList.toggle('miss', missing.includes(v));
    f.classList.toggle('auto', S.auto.has(v));
  });
  $('autohint').hidden = !vars.some(v => S.auto.has(v));
}
function readback(d, soft) {
  const b = $('readback');
  if (!d) { b.hidden = true; return; }
  b.hidden = false; b.classList.toggle('soft', !!soft);
  if (soft) { b.innerHTML = `Not a formula I can run yet. <span class="dim">${esc(d)}</span>`; return; }
  const typeset = d.results && d.results[0].math && d.results[0].math.every(Boolean) ? d.results[0].math.join('<br>') : null;
  b.innerHTML = `Read as<span class="rd">${typeset || ($('mode').value === 'rule' ? '<span class="num">y = ' + pretty(d.reading) + '</span>' : esc(d.reading))}</span>` + (d.notes && d.notes.length ? `<ul>${d.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : '');
}
function setBits(b, wait) {
  S.bits = b; $('bits').value = b; $('bitslabel').textContent = b + ' bits, u = 2' + sup(-b);
  $('seg').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(Number(x.dataset.b) === b)));
  schedule(wait || 0);
}
function setVals(o) {
  for (const [n, x] of Object.entries(o)) { S.vals[n] = typeof x === 'number' ? numStr(x) : String(x); S.auto.delete(n); }
  fields(S.vars, S.vals); schedule(0);
}
window.setBits = setBits; window.setVals = setVals;
function statusLine() {
  const s = S.status, m = $('mode').value, bits = [];
  if (backend === 'worker') bits.push('Running fully in this page. Bean and the Claude reader need the local app, see the README on GitHub.');
  else {
    if (m === 'anthropic' && !s.anthropic_key) bits.push('The Claude reader needs ANTHROPIC_API_KEY set before the app starts.');
    if (!s.bean) bits.push('Bean is not installed, so the Bean tab only shows the translation.');
    if (!s.numpy) bits.push('Install numpy to make random trials fast.');
  }
  $('statusline').textContent = bits.join(' ');
}

/* ------------------------------------------------------------------ running */
let timer = null, extraTimer = null;
const autoOK = () => S.autorun && ($('mode').value === 'rule' || $('mode').value === 'mock');
function schedule(wait) {
  clearTimeout(timer);
  $('go').hidden = autoOK();
  if (!autoOK()) { live('', 'edited, press Analyze or ⌘ Enter'); return; }
  live('busy', 'running');
  timer = setTimeout(run, wait);
}
async function run(extra) {
  clearTimeout(timer); clearTimeout(extraTimer);
  const text = $('text').value.trim(), seq = ++S.seq;
  if (!text) { live('', 'type a formula'); readback(null); return; }
  say(''); live('busy', S.ready ? 'analyzing' : 'waiting for the engine');
  const values = {};
  for (const [k, v] of Object.entries(S.vals)) if (String(v).trim() !== '') values[k] = String(v);
  const t0 = performance.now();
  let d;
  try { d = await api('analyze', Object.assign({text, mode: $('mode').value, bits: S.bits, values, discrete: S.discrete, autofill: true}, extra || {})); }
  catch (e) { d = {error: 'The machine is not answering. Is rounderr_app.py still running?'}; }
  if (seq !== S.seq) return;
  if (d.error) { readback(d.error, true); $('pane').classList.add('old'); live('', 'waiting for a formula that reads'); return; }
  if (d.need_values) {
    readback(d); fields(d.vars, d.values, d.need_values);
    live('', 'give a number for ' + d.need_values.join(', ')); return;
  }
  S.times = {analyze: performance.now() - t0};
  show(d);
  extraTimer = setTimeout(() => experiments(seq), 80);
}
function show(d, saved) {
  const tv = new Set(d.text_values || []);
  S.payload = d;
  if (S.sel >= d.results.length) S.sel = 0;
  (d.autofilled || []).forEach(v => S.auto.add(v));
  for (const v of d.vars) if (!tv.has(v) && d.values[v] !== undefined) S.vals[v] = d.values[v];
  for (const v of [...S.auto]) if (!d.vars.includes(v)) S.auto.delete(v);
  S.expStale = !saved;
  $('pane').classList.remove('old');
  readback(d); fields(d.vars, d.values);
  if (!saved) live('busy', 'analysis ' + ms(S.times.analyze) + ' · experiments running');
  render(true);
}
const landReq = rid => ({run_id: rid, x: S.landX, y: S.landY === undefined ? null : (S.landY || ''), kind: S.landY === '' ? 'curve' : 'auto', decades: S.landDec});
async function experiments(seq) {
  const rid = S.payload.run_id, t0 = performance.now();
  const step = async (route, body, keep) => {
    const d = await api(route, body);
    if (seq !== S.seq) return false;
    if (!d.error) keep(d); else if (d.error !== 'stale') S.times.note = d.error;
    return true;
  };
  if (!await step('precision', {run_id: rid}, d => { S.prec = d; paint('prec'); })) return;
  if (!await step('trials', {run_id: rid, n: S.tn, spread: S.ts}, d => { S.trials = d.trials; paint('scatter'); })) return;
  if (!await step('landscape', landReq(rid), d => { S.land = d; })) return;
  S.expStale = false; S.times.exp = performance.now() - t0;
  paint('scatter', 'prec', 'land');
  if (S.tab === 'Trials') render();
  const n = S.trials && S.trials[0] && !S.trials[0].error ? S.trials[0].n : 0;
  live('on', `analysis ${ms(S.times.analyze)} · ${n} trials + sweeps ${ms(S.times.exp)}`);
}
async function rerunPart(which) {                     // a control on one experiment card changed
  if (!S.payload || !S.ready) return;
  const seq = S.seq, rid = S.payload.run_id;
  live('busy', 'running');
  if (which === 'trials') { const d = await api('trials', {run_id: rid, n: S.tn, spread: S.ts}); if (seq === S.seq && !d.error) S.trials = d.trials; }
  if (which === 'land') { const d = await api('landscape', landReq(rid)); if (seq === S.seq && !d.error) S.land = d; }
  if (seq !== S.seq) return;
  live('on', 'up to date');
  if (S.tab === 'Overview') paint(which === 'trials' ? 'scatter' : 'land'); else render();
}

/* ------------------------------------------------------------------ the pipeline strip */
function chain(pulse) {
  const p = S.payload, r = p && p.results[S.sel], t = p ? p.timings : {};
  const st = [
    ['Read', p ? (p.mode === 'rule' ? 'rules' : 'model') + (p.read_cached ? ', from table' : '') + ' ' + ms(t.read) : 'rules or a model'],
    ['Tape', r ? r.steps + ' steps' : 'one line per step'],
    ['Hash', r ? r.hash.key + (r.hash.seen ? ', seen ' + r.hash.hits + ' times' : ', new') : 'structure only'],
    ['Sweep', p ? 'Lᵀλ = w ' + ms(t.sweep) : 'reverse mode'],
    ['Bean', r ? (r.bean.ok ? 'typed' + (r.bean.cached ? ', from table' : '') : (r.bean.stage === 'missing' ? 'not installed' : 'cannot type it')) : 'all inputs'],
    ['Data', p ? p.log_rows + ' rows logged' : 'log and trials']
  ];
  $('chain').innerHTML = st.map(([a, b], i) => `<li class="${p && i === 4 && r && !r.bean.ok && r.bean.stage !== 'missing' ? 'flag' : ''}"><b>${a.toLowerCase()}</b><span>${esc(b)}</span></li>`).join('');
}

/* ------------------------------------------------------------------ the right side */
function render(pulse) {
  chain(pulse);
  const p = S.payload;
  $('tabs').innerHTML = p ? TABS.map(t => `<button role="tab" aria-selected="${t === S.tab}" data-t="${t}">${t}</button>`).join('') : '';
  if (!p) return;
  const c = $('cands'); c.hidden = p.results.length < 2 || S.tab === 'Overview';
  c.innerHTML = p.results.map((r, i) => `<button class="cand" aria-pressed="${i === S.sel}" data-i="${i}"><i class="sw" style="--c:${SERIES[i % 4]}"></i>${esc(r.label)} <span class="pill ${r.word[0]}">${r.word.toLowerCase()}</span></button>`).join('');
  const pane = $('pane'), r = p.results[S.sel];
  if (S.tab === 'Overview') {
    if (!$('c-hero')) {
      pane.className = 'pane';
      pane.innerHTML = `<div class="grid"><section class="card" id="c-hero"></section><section class="card" id="c-ladder"></section><section class="card" id="c-graph"></section>
        <section class="card half" id="c-blame"></section><section class="card half" id="c-why"></section>
        <section class="card half" id="c-prec"></section><section class="card half" id="c-scatter"></section><section class="card" id="c-land"></section></div>`;
      S.graphSig = null;
    }
    paint();
    return;
  }
  pane.className = 'pane sheet';
  pane.innerHTML = ({Tape: vTape, Trials: vTrials, Matrix: vMatrix, Bean: vBean, Data: vData, Developer: vDev})[S.tab](r, p);
}
function paint(...ids) {
  if (S.tab !== 'Overview' || !S.payload) return;
  const p = S.payload, r = p.results[S.sel];
  (ids.length ? ids : Object.keys(CARDS)).forEach(id => {
    const el = $('c-' + id);
    if (!el) return;
    el.classList.toggle('stale', EXP.has(id) && S.expStale);
    if (id === 'graph') {                           // do not cut a playing animation short for an identical redraw
      const sig = r.hash.key + '|' + S.sel + '|' + r.lines.map(l => l.value + ':' + l.lam).join(',');
      if (sig === S.graphSig) return;
      const structure = r.hash.key + '|' + S.sel, animate = structure !== S.graphKey;
      S.graphSig = sig; S.graphKey = structure;
      el.innerHTML = cGraph(r, p, animate);
      return;
    }
    const before = {};
    el.querySelectorAll('[data-k]').forEach(e => { before[e.dataset.k] = [e.style.left, e.style.width]; });
    el.innerHTML = CARDS[id](r, p);
    if (!REDUCED) el.querySelectorAll('[data-k]').forEach(e => {      // glide from where the mark was
      const b = before[e.dataset.k];
      if (!b) return;
      const a = [e.style.left, e.style.width];
      e.style.transition = 'none'; e.style.left = b[0]; e.style.width = b[1];
      void e.offsetWidth;
      e.style.transition = ''; e.style.left = a[0]; e.style.width = a[1];
    });
    el.querySelectorAll('.tw').forEach(tween);
    if (id === 'land') el.querySelectorAll('canvas[data-map]').forEach(cv => drawMap(cv, S.land, S.land.cands[Number(cv.dataset.map)]));
  });
}
function tween(el) {
  const k = el.dataset.n, to = Number(el.dataset.v), d = Number(el.dataset.d || 3), from = S.nums[k];
  S.nums[k] = to;
  if (REDUCED || from === undefined || from === to || !(from > 0) || !(to > 0) || !Number.isFinite(from) || !Number.isFinite(to)) return;
  const a = Math.log(from), b = Math.log(to), t0 = performance.now();
  const stepf = now => {
    if (!el.isConnected) return;
    const t = Math.min(1, (now - t0) / 380), e = 1 - Math.pow(1 - t, 3);
    el.innerHTML = fmt(Math.exp(a + (b - a) * e), t < 1 ? Math.min(d, 3) : d);
    if (t < 1) requestAnimationFrame(stepf);
  };
  requestAnimationFrame(stepf);
}
const tw = (key, v, d = 3) => (v === null || v === undefined || !Number.isFinite(v)) ? `<b>${fmt(v, d)}</b>` : `<b class="tw" data-n="${key}" data-v="${v}" data-d="${d}">${fmt(v, d)}</b>`;
const formula = r => r.multi ? r.script.map(([n, py]) => esc((n || 'y') + ' = ') + pretty(py)).join('<br>') : 'y = ' + pretty(r.src);
const head = (title, sub, extra = '') => `<div class="ch"><h2>${title}</h2><p>${sub}</p>${extra ? `<span class="ctl">${extra}</span>` : ''}</div>`;
const opt = (v, t, cur) => `<option value="${esc(v)}"${String(v) === String(cur) ? ' selected' : ''}>${esc(t)}</option>`;

/* ---------- overview cards ---------- */
const CARDS = {
  hero(r, p) {
    const w = r.word[0], name = {S: 'Stable', U: 'Unstable', N: 'Undefined'}[w];
    let why = r.verdict.replace(/^\w+\s+\(?/, '').replace(/\)$/, '');
    why = why.charAt(0).toUpperCase() + why.slice(1);
    const total = r.bits * Math.log10(2), rel = r.rel === null ? null : Math.abs(r.rel);
    const good = rel === null ? 0 : rel === 0 ? total : clamp(-lg(rel), 0, total);
    const sure = r.A > 0 && Number.isFinite(r.A) ? clamp(-lg(r.A * r.u), 0, total) : (r.A === 0 ? total : 0);
    const ratio = r.A === null ? null : r.A / Math.max(1, r.kappa ?? 1);
    const eq = r.math && r.math.every(Boolean) ? `<div class="lines">${r.math.join('')}</div>` : `<span class="fx">${formula(r)}</span>`;
    return `<p class="vline ${w}"><b>${name}.</b> ${w === 'N' ? words(r.verdict) : words(why) + '.'}</p>
      <div class="eq">${eq}<span class="val">= ${esc(r.y)}${p.results.length > 1 ? ` &nbsp; (${esc(r.label)})` : ''}</span></div>
      <div class="facts">
        <div data-tip="The error this run really made, computed value against a 60 digit reference."><small>measured error</small>${tw('act', r.rel_u === null ? null : Math.abs(r.rel_u))} u<em>relative ${fmt(rel, 3)}</em></div>
        <div data-tip="A, the worst case this algorithm allows: u times the sum of |λ d t| over the tape."><small>ceiling A</small>${tw('A', r.A)} u<em>absolute ${fmt(r.ceiling, 3)}</em></div>
        <div data-tip="κ, the condition number. The error the problem itself forces on any algorithm."><small>condition κ</small>${tw('kap', r.kappa)}<em>the problem's own</em></div>
        <div data-tip="Stable when A stays within the limit times max(1, κ)."><small>A / κ</small>${tw('ratio', ratio)}<em>limit ${fmt(r.allow)}</em></div>
        <div data-tip="Black: digits the ceiling guarantees. Grey: digits this run actually got right."><small>correct digits</small><b>${good.toFixed(1)}</b> of ${total.toFixed(1)}
          <div class="dbar"><i class="m" data-k="dm" style="width:${100 * good / total}%"></i><i class="g" data-k="dg" style="width:${100 * sure / total}%"></i></div><em>${sure.toFixed(1)} guaranteed</em></div>
      </div>`;
  },
  ladder(r, p) {
    return head(p.results.length > 1 ? 'Candidates' : 'Problem against algorithm',
      'Stable when the ceiling A stays left of the limit.' + (p.results.length > 1 ? ' Click a row to inspect that form.' : '')) + ladderHTML(p.results, S.sel);
  },
  blame(r) {
    const rows = r.lines.filter(l => l.op !== 'input').sort((a, b) => b.share - a.share).slice(0, 8);
    const any = rows.some(l => l.share > 0);
    return head('Blame', 'Share of the ceiling carried by each rounded step, |λ d t|.') + (any ? rows.map(l =>
      `<div class="brow" data-tip="${esc(l.text + '\n|λ d t| = ' + plain(l.weight) + '\nd = ' + l.d + (l.flags.length ? '  ' + l.flags.join(', ') : ''))}"><span class="c">${esc(l.text)}</span><div class="bar"><i data-k="b${l.idx}" style="width:${Math.max(l.share > 0 ? 1.5 : 0, 100 * l.share)}%"></i></div><span class="p">${l.share > 0 ? pct(l.share) : (l.d === 0 ? 'exact' : '0%')}</span></div>`).join('')
      : none('Every step of this run is exact, so there is nothing to blame.'));
  },
  why(r, p) {
    let h = head('Diagnosis', '');
    h += `<ul class="why">${r.diagnosis.map(s => `<li>${words(s)}</li>`).join('')}</ul>`;
    if (r.warnings.length) h += r.warnings.map(w => `<div class="warn">${words(w)}</div>`).join('');
    const alt = p.results.filter((q, i) => i > 0 && q.why);
    if (alt.length) h += `<h3>Rewrites found</h3><ul class="why">${alt.map(q => `<li><b>${esc(q.label)}</b> <span class="num">${pretty(q.src)}</span><br><span class="dim">${esc(q.why)}</span></li>`).join('')}</ul>`;
    if (p.dropped && p.dropped.length) h += `<h3>Tapes the model proposed that failed the 60 digit check</h3><ul class="why">${p.dropped.map(d => `<li><b>${esc(d.name)}</b> ${esc(d.note)}</li>`).join('')}</ul>`;
    if (p.narrative) h += `<h3>The model explains</h3><div class="qa">${words(p.narrative)}</div><p class="hint">${p.loose.length ? 'Check these numbers, they are not in the report. ' + p.loose.map(x => fmt(x)).join(', ') : 'Every number in this explanation was found in the report.'}</p>`;
    if (p.mode !== 'rule') h += `<div class="ask"><input id="q" placeholder="Ask a follow up, for example what if b = 1e6"><button class="quiet" data-act="ask">Ask</button></div>` + S.qa.map(([q, a]) => `<div class="qa"><b>${esc(q)}</b>\n${esc(a)}</div>`).join('');
    return h;
  },
  prec() {
    return head('Precision sweep', 'Same numbers, 8 to 53 bits. A stable form tracks u.') + precisionSVG(S.prec, S.sel);
  },
  scatter() {
    const ctl = `<select data-set="tn" aria-label="How many random inputs">${[[500, '500'], [2000, '2000'], [5000, '5000'], [20000, '20000']].map(([v, t]) => opt(v, t + ' inputs', S.tn)).join('')}</select>
      <select data-set="ts" aria-label="How far the inputs spread">${opt('pct10', 'within 10%', S.ts)}${opt('x10', 'up to 10 times', S.ts)}${opt('x1000', 'up to 1000 times', S.ts)}</select>`;
    return head('Random trials', 'Inputs drawn around yours.', ctl) + scatterSVG(S.trials, S.sel);
  },
  land() {
    const L = S.land;
    if (!L || L.kind === 'none') return head('Input landscape', '') + none(L ? 'This formula has no inputs to vary.' : 'Waiting for the sweep.');
    const ctl = `<select data-set="landX" aria-label="Input on the horizontal axis">${L.names.map(n => opt(n, n + ' across', L.x)).join('')}</select>` +
      (L.names.length > 1 ? `<select data-set="landY" aria-label="Input on the vertical axis">${opt('', 'curve, one input', L.y || '')}${L.names.filter(n => n !== L.x).map(n => opt(n, n + ' up', L.y || '')).join('')}</select>` : '') +
      `<select data-set="landDec" aria-label="Range">${[[1, '÷10 to ×10'], [2, '÷100 to ×100'], [4, '÷10⁴ to ×10⁴']].map(([v, t]) => opt(v, t, L.decades)).join('')}</select>`;
    return head('Input landscape', L.kind === 'map' ? 'A/κ over two inputs, the rest held fixed.' : 'Ceiling and κ along one input, the rest held fixed.', ctl) +
      (L.kind === 'map' ? mapHTML(L) : curveSVG(L, S.sel));
  }
};
function cGraph(r, p, animate) {
  return head('Tape', 'Values run forward to y, λ runs back. Yellow marks blame.', '<button class="quiet" data-act="replay">Replay</button>') + graphSVG(r, animate) ;
}
CARDS.graph = cGraph;

/* ---------- detail tabs ---------- */
function vTape(r) {
  const rows = r.lines.map((l, i) => {
    const isIn = l.op === 'input', w = Math.round(100 * l.share);
    return `<div class="tl${isIn ? ' in' : ''}" style="--i:${i}"><span class="c" title="${esc(l.text)}">${esc(l.text)}</span><span class="c">${fmt(l.value, 6)}</span><span class="c">${isIn ? '' : l.d}</span><span class="c">${fmt(l.lam)}</span>` +
      (isIn ? `<span class="note">gradient of y</span>` : `<span class="bar" title="${plain(l.weight)}"><i style="width:${Math.max(l.share > 0 ? 2 : 0, 100 * l.share)}%"></i><span>${l.share > 0 ? (w < 1 ? '<1%' : w + '%') : ''}</span></span>`) +
      `<span class="note" title="${esc(l.flags.join(', '))}">${esc(l.flags.join(', '))}</span></div>`;
  }).join('');
  return `<p class="hint" style="margin:0 0 10px">One line per rounded step. λ is the weight of that line on y. The bar is that line's share of the error ceiling, |λ d t|.</p>
  <div class="scroll"><div class="tape"><div class="tl hd"><span>Line</span><span>Value t</span><span>d</span><span>λ</span><span>Share of the ceiling</span><span>Notes</span></div>${rows}</div></div>
  <h3>What actually happened</h3><div class="two"><div class="box"><dl>
    <dt>Computed y</dt><dd>${esc(r.y)}</dd><dt>True y</dt><dd>${esc(r.y_true)} <small>60 digits</small></dd>
    <dt>Actual error</dt><dd>${fmt(r.actual)} <small>which is</small> ${fmt(r.rel_u)} u <small>relative</small></dd>
    <dt>Predicted</dt><dd>${fmt(r.predicted)} <small>sum of λᵢρᵢ</small></dd>
    ${r.input_effect ? `<dt>Stored inputs</dt><dd>${fmt(r.input_effect)} <small>moved y before any step ran</small></dd>` : ''}</dl></div>
  <div class="box"><dl><dt>Ceiling</dt><dd>${fmt(r.ceiling)} <small>which is</small> ${fmt(r.A)} u <small>relative</small></dd><dt>Typical</dt><dd>${fmt(r.typical)}</dd>
    <dt>Problem forces</dt><dd>κ = ${fmt(r.kappa)}</dd><dt>Algorithm allows</dt><dd>A = ${fmt(r.A)} <small>so A/κ is</small> ${fmt(r.A / Math.max(1, r.kappa ?? 1))} <small>against a limit of</small> ${fmt(r.allow)}</dd>
    <dt>Backward error</dt><dd>β₁ = ${fmt(r.beta1)}, β₂ = ${fmt(r.beta2)} <small>in units of u</small></dd></dl></div></div>`;
}
function grid(M, rowLab, colLab, solve) {
  const cols = M[0].length, big = Math.max(...M.flat().map(x => Math.abs(x)).filter(x => x > 0 && x !== 1), 1e-300), small = Math.min(...M.flat().map(x => Math.abs(x)).filter(x => x > 0), big);
  const heat = x => { const a = Math.abs(x); return big > small ? Math.round(10 + 45 * (lg(a) - lg(small)) / (lg(big) - lg(small))) : 30; };
  let h = `<div class="mat${solve ? ' solve' : ''}" style="grid-template-columns:auto repeat(${cols},auto)"><div class="lab"></div>` + colLab.map(c => `<div class="lab">${esc(c)}</div>`).join('');
  M.forEach((row, i) => { h += `<div class="lab">${esc(rowLab[i])}</div>` + row.map(x => `<div class="${x === 0 ? 'z' : (x === 1 ? 'one' : 'nz')}" style="--o:${M.length - 1 - i};--heat:${x === 0 || x === 1 ? 0 : heat(x)}%">${x === 0 ? '0' : fmt(x, 3)}</div>`).join(''); });
  return h + '</div>';
}
function vMatrix(r) {
  const m = r.matrix;
  let h = `<p class="hint" style="margin:0 0 12px">The tape as a system. F(t, x) = t − G(t, x) = 0, with L = ∂F/∂t and B = ∂G/∂x. Solving Lᵀλ = w from the bottom up is the backward sweep, which is the order λ fills in below. The gradient is Bᵀλ.</p>`;
  if (m.big) return h + `<p>This tape has ${m.n} lines, so L is ${m.n} by ${m.n} with ${m.nnz} nonzeros. Too big to print. The table stores its pattern under hash <span class="num">${r.hash.key}</span>.</p>`;
  h += `<div class="scroll"><div><b>L</b><br>${grid(m.L, m.tlabels, m.tlabels)}</div><div style="display:flex;flex-wrap:wrap"><div><b>B</b><br>${grid(m.B, m.tlabels, m.xlabels)}</div><div><b>w</b><br>${grid(m.w.map(x => [x]), m.tlabels, [''])}</div><div><b>λ</b><br>${grid(m.lam.map(x => [x]), m.tlabels, [''], true)}</div></div></div>`;
  h += `<p>Gradient Bᵀλ <span class="num">(${m.xlabels.map((n, i) => esc(n) + ' ' + fmt(m.grad[i])).join(', ')})</span></p>`;
  h += `<p class="hint">Stored in the table under hash <span class="num">${r.hash.key}</span>, ${r.hash.nnz} nonzeros, first seen ${esc(r.hash.first_seen)}, used ${r.hash.hits} times.</p>`;
  return h;
}
function vBean(r) {
  const b = r.bean;
  let h = `<p class="hint" style="margin:0 0 12px">Bean checks the program once for every possible input. This machine checks it at your numbers. Bean only knows add, sub, mul and div, and a value that can carry error may be used once.</p>`;
  if (b.stage === 'missing') return h + `<div class="warn">${backend === 'worker' ? 'Bean is a native program, so it cannot run inside a web page. Run the app on your own computer to use it:' : 'Bean is not installed on this computer.'} build it from github.com/Athena-Types/Bean with opam and dune, then start <span class="num">python rounderr_app.py --bean path/to/bean.exe</span>.</div>` + (b.src ? `<h3>The Bean program this formula becomes</h3><pre>${esc(b.src)}</pre>` : '');
  if (!b.ok) {
    h += `<div class="verdict N"><b>Bean cannot type it</b><span>${esc(b.msg.replace(/^not expressible\. |^Bean rejects it\. /, ''))}</span></div>`;
    h += `<p>That is not a verdict of unstable. The engine says <span class="pill ${r.word[0]}">${r.word.toLowerCase()}</span> at your numbers.</p>`;
    if (b.src) h += `<h3>The Bean program that was tried</h3><pre>${esc(b.src)}</pre>`;
  } else {
    h += `<div class="verdict S"><b>Typed</b><span>backward stable for every input value${b.cached ? ', answer taken from the table' : ''}</span></div>`;
    h += `<div class="two"><div><h3 style="margin-top:0">Bean's bound, all inputs</h3><table><tr><th>Input</th><th>May move by at most</th></tr>${Object.entries(b.grades).map(([n, g]) => `<tr><td class="n">${esc(n)}</td><td class="n">${fmt(g)} ε</td></tr>`).join('')}</table></div>
      <div><h3 style="margin-top:0">This machine, your numbers</h3><dl><dt>Adjoint ceiling</dt><dd>${fmt(b.eta_ceil)} u</dd><dt>Measured</dt><dd>${fmt(b.eta_meas)} u</dd><dt>Share of Bean's bound used</dt><dd>${pct(b.used)}</dd></dl></div></div>`;
    h += `<h3>Generated Bean program</h3><pre>${esc(b.src)}</pre>`;
  }
  if (b.notes && b.notes.length) h += `<ul class="why">${b.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>`;
  h += `<div class="row" style="margin-top:12px"><label for="disc">Inputs Bean must not change</label><input id="disc" class="num" style="padding:5px 8px;border:1px solid var(--rule);border-radius:2px;background:var(--bg)" placeholder="x, z" value="${esc((S.discrete || []).join(', '))}"><button class="quiet" data-act="disc">Check again</button></div>`;
  return h;
}
function vTrials(r, p) {
  let h = `<p class="hint" style="margin:0 0 12px">Random inputs around your numbers, every candidate, one compiled sweep for all of them at once. They rerun by themselves after every analysis.</p>
  <div class="row" style="flex-wrap:wrap;margin-bottom:14px"><select data-set="tn" style="width:auto">${[[500, '500'], [2000, '2000'], [5000, '5000'], [20000, '20000']].map(([v, t]) => opt(v, t + ' inputs', S.tn)).join('')}</select>
  <select data-set="ts" style="width:auto">${opt('pct10', 'within 10% of my numbers', S.ts)}${opt('x10', 'up to 10 times bigger or smaller', S.ts)}${opt('x1000', 'up to 1000 times bigger or smaller', S.ts)}</select>
  ${S.trials ? `<button class="quiet" data-act="export" data-kind="trials">Download trials CSV</button>` : ''}</div>`;
  if (!S.trials) return h + `<div class="empty" style="padding:24px 0">The trials are on their way.</div>`;
  h += `<div class="scroll"><table><tr><th>Candidate</th><th>Inputs used</th><th>Unstable</th><th>Median A/κ</th><th>Largest A/κ</th><th>Largest β₁</th><th>Time</th></tr>` +
    S.trials.map((t, i) => t.error ? `<tr><td>${esc(t.label)}</td><td colspan="6">${esc(t.error)}</td></tr>` :
      `<tr class="pick" data-i="${i}"><td><i class="sw" style="--c:${SERIES[i % 4]}"></i>${esc(t.label)}</td><td class="n">${t.valid}${t.dropped ? ' <span class="note">(' + t.dropped + ' dropped)</span>' : ''}</td><td class="n"><span class="pill ${t.unstable ? 'U' : 'S'}">${pct(t.unstable_share)}</span></td><td class="n">${fmt(t.med_ratio)}</td><td class="n">${fmt(t.max_ratio)}</td><td class="n">${fmt(t.max_beta)}</td><td class="n">${fmt(1000 * t.seconds, 3)} ms${t.fast ? '' : ' <span class="note">slow path</span>'}</td></tr>`).join('') + `</table></div>`;
  const ti = S.trials[S.sel] && !S.trials[S.sel].error ? S.sel : S.trials.findIndex(x => !x.error), t = S.trials[ti];
  if (t) {
    h += `<h3>${esc(t.label)}, how far A is above κ</h3>${histSVG(t, SERIES[ti % 4])}<p class="hint">Each bar is a quarter of a power of ten of A/κ. Red bars are past the limit of ${fmt(t.allow)}.</p>`;
    h += `<h3>The cloud</h3>${scatterSVG(S.trials, S.sel)}`;
    h += `<h3>Worst inputs found</h3><div class="scroll"><table><tr><th>Inputs</th><th>y</th><th>A/κ</th><th>β₁</th><th></th></tr>` + t.worst.map((w, i) =>
      `<tr><td class="n">${Object.entries(w.inputs).map(([n, x]) => esc(n) + ' = ' + fmt(x, 6)).join(', ')}</td><td class="n">${fmt(w.y)}</td><td class="n">${fmt(w.ratio)}</td><td class="n">${fmt(w.beta1)}</td><td><button class="quiet" data-act="load" data-w="${i}" data-t="${ti}">Load these</button></td></tr>`).join('') + `</table></div>`;
    if (t.worst[0].actual_rel_u !== undefined) h += `<p class="hint">Checked the worst one with the full engine. Actual error ${fmt(t.worst[0].actual_rel_u)} u against a ceiling of ${fmt(t.worst[0].ceiling_rel_u)} u.</p>`;
  }
  return h;
}
function vData(r, p) {
  return `<div class="dl"><button class="quiet" data-act="export" data-kind="log">Download run log CSV</button><button class="quiet" data-act="export" data-kind="tex">Download LaTeX report</button>${backend === 'http' && S.status.pdflatex ? `<a class="quiet" href="api/report.pdf">Download PDF report</a>` : ''}</div>
  <p class="hint" style="margin:0 0 10px">${p.log_rows} rows logged this session, one per distinct run. The hash table holds ${p.structures} structures${backend === 'http' ? ' and lives in rounderr_cache.json next to the app' : ', kept in memory for this visit'}.</p>
  <div class="scroll" id="logbox"><table><tr><th>This run</th><th>Hash</th><th>Seen before</th><th>Steps</th><th>Actual</th><th>Ceiling</th><th>κ</th><th>β₁</th><th>Verdict</th><th>Bean</th></tr>` +
  p.results.map(q => `<tr><td>${esc(q.label)}</td><td class="n">${q.hash.key}</td><td>${q.hash.seen ? 'yes, ' + q.hash.hits + ' uses' : 'no'}</td><td class="n">${q.steps}</td><td class="n">${fmt(q.rel_u)} u</td><td class="n">${fmt(q.A)} u</td><td class="n">${fmt(q.kappa)}</td><td class="n">${fmt(q.beta1)}</td><td><span class="pill ${q.word[0]}">${q.word.toLowerCase()}</span></td><td>${q.bean.ok ? fmt(q.bean.max) + ' ε' : (q.bean.stage === 'missing' ? 'not installed' : 'cannot type it')}</td></tr>`).join('') + `</table></div>
  <h3>Time per stage, inside the engine</h3><dl>${Object.entries(p.timings).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmt(v, 3)} ms</dd>`).join('')}</dl>
  <h3>Time as you felt it</h3><dl><dt>analysis, round trip</dt><dd>${ms(S.times.analyze) || 'saved run'}</dd><dt>experiments</dt><dd>${ms(S.times.exp) || 'running'}</dd><dt>engine</dt><dd>${backend === 'http' ? 'native Python on this computer' : 'Python ' + esc(S.status.python || '') + ' as WebAssembly in this page'}</dd></dl>`;
}
/* developer tab: design, trace of this run, source code */
function srcBlock(f) {
  const first = f.line || 1;
  const body = f.source.split('\n').map((l, i) => {
    let t = esc(l);
    if (f.kind === 'code') t = t.replace(/^(\s*)(def|class)(\s+)([A-Za-z_]\w*)/, '$1$2$3<span class="df">$4</span>').replace(/(^|\s)(#.*)$/, '$1<span class="cm">$2</span>');
    return `<span class="ln">${f.kind === 'code' ? first + i : ''}</span>${t}`;
  }).join('\n');
  const id = 'code:' + f.file + ':' + f.name;
  return `<details class="code" data-id="${esc(id)}"${S.open.has(id) ? ' open' : ''}><summary><span class="num">${esc(f.file)}</span>${f.line ? ', line ' + f.line : ''}, <b>${esc(f.name)}</b>${f.kind === 'prompt' ? ', prompt text sent to the model' : ''}</summary><pre class="src">${body}</pre></details>`;
}
const lite = lines => `<pre class="lite">${lines}</pre>`;
function runBlock(id, r, p) {
  const t = r.trace;
  if (id === 'read') {
    return `<div class="scroll"><table><tr><th>Typed</th><th>After normalize</th><th>Python source handed to the tape builder</th></tr>${t.read.rows.map(x => `<tr><td class="n">${esc(x[0])}</td><td class="n">${esc(x[1])}</td><td class="n">${esc(x[2])}</td></tr>`).join('')}</table></div>` +
      (t.read.notes.length ? `<ul class="why">${t.read.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : '') +
      lite(esc(t.read.values.join('\n') || 'no inputs')) +
      `<details class="code" data-id="ast"${S.open.has('ast') ? ' open' : ''}><summary>Python syntax tree of the output line</summary><pre class="src">${esc(t.read.ast)}</pre></details>`;
  }
  if (id === 'tape') return lite(esc('idx  op     operands   line\n' + t.tape.join('\n')));
  if (id === 'hash') return `<dl><dt>String that is hashed</dt><dd>${esc(t.hash.structure)}</dd><dt>SHA 256</dt><dd>${esc(t.hash.sha)}</dd><dt>Key</dt><dd>${esc(t.hash.key)} <small>${t.hash.seen ? 'found in the table' : 'new, stored now'}</small></dd></dl>` + lite(esc(JSON.stringify(t.hash.entry, null, 1)));
  if (id === 'forward') return `<div class="scroll"><table><tr><th>Line</th><th>Operands</th><th>Computed</th><th>Exact, same operands</th><th>Slip ρ</th><th>In u</th><th>d</th><th>Why that d</th><th>Partials kept</th></tr>${t.forward.map(f => `<tr><td class="n">${esc(f.line)}</td><td style="font-family:var(--num);min-width:16ch">${esc(f.operands)}</td><td class="n">${esc(f.computed)}</td><td class="n">${esc(f.exact)}</td><td class="n">${fmt(f.slip)}</td><td class="n">${fmt(f.slip_u, 3)}</td><td class="n">${f.d}</td><td>${esc(f.rule)}</td><td class="n">${words(f.partials)}</td></tr>`).join('')}</table></div>`;
  if (id === 'sweep') return lite(words(t.sweep.join('\n')));
  if (id === 'numbers') return lite(words(t.numbers.join('\n')));
  if (id === 'bean') return `<dl><dt>Discrete, may not move</dt><dd>${esc(t.bean.discrete.join(', ') || 'none')}</dd><dt>Linear, may move</dt><dd>${esc(t.bean.linear.join(', ') || 'none')}</dd><dt>Outcome</dt><dd>${esc(t.bean.stage)} <small>${t.bean.cached ? 'answer from the table' : ''}</small></dd></dl>` +
      (r.bean.src ? `<pre>${esc(r.bean.src)}</pre>` : '') + lite(esc(t.bean.raw || 'Bean did not run'));
  if (id === 'data') return `<dl>${Object.entries(p.timings).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmt(v, 3)} ms</dd>`).join('')}</dl>` + lite(esc(JSON.stringify(p.log_row[S.sel] || {}, null, 1)));
  if (id === 'model') {
    const m = p.model_trace;
    if (!m) return `<p class="hint">This run used the built in rules, so the model layer did not run. Its code is below.</p>`;
    return `<dl><dt>Provider</dt><dd>${esc(m.provider)} <small>${m.cached ? 'proposal taken from the table' : 'called now'}</small></dd><dt>Sent as user text</dt><dd>${esc(m.user)}</dd><dt>Grounding check</dt><dd>${esc(m.grounding)}</dd></dl>` +
      `<table><tr><th>Tape from the model</th><th>60 digit check</th></tr>${m.checks.map(c => `<tr><td>${esc(c.name)}</td><td>${c.kept ? 'kept' : 'dropped'}, ${esc(c.note)}</td></tr>`).join('')}</table>` + lite(esc(JSON.stringify(m.proposal, null, 1)));
  }
  return '';
}
function stageLine(id, r, p) {
  const t = r.trace;
  return ({read: () => t.read.rows.length + ' line' + (t.read.rows.length > 1 ? 's' : '') + ' read, ' + t.read.notes.length + ' guesses noted',
    tape: () => r.lines.length + ' lines, ' + r.steps + ' rounded steps',
    hash: () => t.hash.key + (t.hash.seen ? ', found in the table' : ', new'),
    forward: () => t.forward.filter(f => f.slip !== 0).length + ' lines slipped, ' + t.forward.filter(f => f.d === 0 && f.line.indexOf('input') < 0).length + ' exact',
    sweep: () => (t.sweep.length - 2) + ' updates of λ',
    numbers: () => r.word.toLowerCase() + ', A/κ is ' + plain(r.A / Math.max(1, r.kappa ?? 1)) + ' against a limit of ' + plain(r.allow),
    bean: () => t.bean.stage === 'typed' ? 'typed' : (t.bean.stage === 'missing' ? 'not installed' : 'cannot type it'),
    data: () => p.log_rows + ' rows in the log',
    model: () => p.model_trace ? p.model_trace.checks.length + ' tapes proposed' : 'not used in this run'})[id]();
}
async function loadDev() { S.dev = await api('dev'); if (S.tab === 'Developer') render(); }
function vDev(r, p) {
  if (!S.ready) return `<p class="hint">The source view opens as soon as the Python engine is up.</p>`;
  if (!S.dev) { loadDev(); return `<p class="hint">Reading the source files</p>`; }
  if (!r.trace) return `<p class="hint">No trace for this run yet.</p>`;
  let h = `<p class="hint" style="margin:0 0 10px">How one run moves through the machine. Open a stage to see what it does, what it received and produced on this run, and the code that did it. The code is read live from the files ${backend === 'http' ? 'next to the app' : 'loaded into this page'}.</p>
  <div class="flow">typed text → read → tape → hash and table → forward pass → backward sweep → numbers and verdict → Bean → data</div>`;
  h += S.dev.stages.map((st, i) => {
    const id = 'stage:' + st.id;
    return `<details class="stage" data-id="${id}"${S.open.has(id) ? ' open' : ''}><summary><span class="k">${i + 1}</span><b>${esc(st.title)}</b><span class="note">${esc(stageLine(st.id, r, p))}</span></summary><div class="in">
      <p>${esc(st.what)}</p><h3>This run</h3>${runBlock(st.id, r, p)}<h3>Code</h3>${st.funcs.map(srcBlock).join('')}</div></details>`;
  }).join('');
  h += `<h3>Files</h3><table><tr><th>File</th><th>Lines</th><th>Role</th></tr>${S.dev.files.map(f => `<tr><td class="n">${esc(f.name)}</td><td class="n">${f.lines || ''}</td><td>${esc(f.role)}</td></tr>`).join('')}</table>`;
  h += `<h3>Routes, what the window asks the engine</h3><table><tr><th>Route</th><th>What it does</th></tr>${S.dev.routes.map(x => `<tr><td class="n">${esc(x.route)}</td><td>${esc(x.what)}</td></tr>`).join('')}</table>`;
  h += `<h3>Raw data</h3><details class="code" data-id="raw"${S.open.has('raw') ? ' open' : ''}><summary>Everything the engine returned for this candidate, as JSON</summary><pre class="src">${S.open.has('raw') ? esc(JSON.stringify(r, null, 1)) : ''}</pre></details>
  <p class="hint">The same trace prints in a terminal with <span class="num">python rounderr_app.py --trace "your formula with x = 1"</span></p>`;
  return h;
}

/* ------------------------------------------------------------------ events, all delegated */
const tip = $('tip');
function tipShow(html, x, y) {
  tip.innerHTML = html; tip.classList.add('on');
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = clamp(x + 14, 6, innerWidth - w - 6) + 'px';
  tip.style.top = (y + 18 + h > innerHeight ? y - h - 12 : y + 18) + 'px';
}
const tipHide = () => tip.classList.remove('on');
document.addEventListener('mousemove', e => {
  const t = e.target;
  if (t.classList && t.classList.contains('hit')) {                 // crosshair on a line chart
    const ch = CH[t.dataset.ch], s = t.closest('svg'), box = s.getBoundingClientRect();
    const at = ch.at((e.clientX - box.left) * ch.W / box.width), xh = s.querySelector('.xh');
    xh.setAttribute('x1', at.x); xh.setAttribute('x2', at.x); xh.style.opacity = .7;
    return tipShow(ch.tip(at.key), e.clientX, e.clientY);
  }
  if (t.dataset && t.dataset.map !== undefined && S.land && S.land.kind === 'map') {
    const o = mapAt(t, e), L = S.land, c = L.cands[Number(t.dataset.map)], z = c.z[o.j * L.m + o.i];
    return tipShow(`<b>${esc(L.x)} = ${fmt(L.xs[o.i], 3)}, ${esc(L.y)} = ${fmt(L.ys[o.j], 3)}</b><br>` + (z === null ? 'no finite result here' : `A/κ ${fmt(Math.pow(10, z), 2)}, ${Math.pow(10, z) > c.allow ? 'unstable' : 'stable'}`) + '<br><span class="dim">click to move there</span>', e.clientX, e.clientY);
  }
  const d = t.closest && t.closest('[data-tip]');
  if (d) return tipShow(esc(d.dataset.tip), e.clientX, e.clientY);
  document.querySelectorAll('.xh').forEach(x => { x.style.opacity = 0; });
  tipHide();
});
function mapAt(cv, e) {
  const b = cv.getBoundingClientRect(), m = S.land.m;
  return {i: clamp(Math.floor((e.clientX - b.left) / b.width * m), 0, m - 1), j: clamp(m - 1 - Math.floor((e.clientY - b.top) / b.height * m), 0, m - 1)};
}
async function download(kind) {
  const d = await api('export', {kind});
  if (d.error) { say(d.error); return; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([d.text], {type: d.mime})); a.download = d.name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
async function ask() {
  const q = $('q').value.trim();
  if (!q) return;
  live('busy', 'asking the model');
  const d = await api('ask', {question: q});
  if (d.error) { say(d.error); live('', 'the model did not answer'); return; }
  S.qa.unshift([q, d.answer]);
  if (d.payload && !d.payload.error) { setBitsQuiet(d.payload.bits); S.vals = Object.assign({}, d.payload.values); show(d.payload); experiments(S.seq); } else paint('why');
  live('on', 'up to date');
}
function setBitsQuiet(b) { S.bits = b; $('bits').value = b; $('bitslabel').textContent = b + ' bits, u = 2' + sup(-b); $('seg').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(Number(x.dataset.b) === b))); }

document.addEventListener('click', e => {
  const t = e.target;
  const tab = t.closest('[data-t]:not([data-act])');
  if (tab && tab.closest('#tabs')) { S.tab = tab.dataset.t; S.graphSig = null; S.graphKey = null; $('pane').innerHTML = ''; render(); return; }
  const ex = t.closest('[data-ex]');
  if (ex) { const x = S.examples[Number(ex.dataset.ex)]; $('text').value = x.text; S.vals = Object.assign({}, x.values); S.auto.clear(); $('mode').value = 'rule'; S.sel = 0; S.qa = []; run(); return; }
  const stepb = t.closest('[data-step]');
  if (stepb) {
    const n = stepb.dataset.step, r = S.payload && S.payload.results[0];
    let x = Number(S.vals[n] !== undefined && S.vals[n] !== '' ? S.vals[n] : NaN);
    if (!Number.isFinite(x) && r && r.inputs[n] !== undefined) x = r.inputs[n];
    if (Number.isFinite(x)) setVals({[n]: x * Number(stepb.dataset.by)});
    return;
  }
  const seg = t.closest('#seg button');
  if (seg) return setBits(Number(seg.dataset.b));
  const act = t.closest('[data-act]');
  if (act) {
    const a = act.dataset.act;
    if (a === 'replay') { S.graphSig = null; S.graphKey = null; paint('graph'); }
    else if (a === 'export') download(act.dataset.kind);
    else if (a === 'ask') ask();
    else if (a === 'disc') { S.discrete = $('disc').value.split(',').map(s => s.trim()).filter(Boolean); run(); }
    else if (a === 'load') {
      const w = S.trials[Number(act.dataset.t)].worst[Number(act.dataset.w)], o = {};
      for (const [n, x] of Object.entries(w.inputs)) o[n] = x;
      S.tab = 'Overview'; $('pane').innerHTML = ''; setVals(o);
    }
    return;
  }
  if (t.classList && t.classList.contains('hit')) {
    const ch = CH[t.dataset.ch], box = t.closest('svg').getBoundingClientRect();
    return ch.click(ch.at((e.clientX - box.left) * ch.W / box.width).key);
  }
  if (t.dataset && t.dataset.map !== undefined && S.land && S.land.kind === 'map') { const o = mapAt(t, e); return setVals({[S.land.x]: S.land.xs[o.i], [S.land.y]: S.land.ys[o.j]}); }
  const pick = t.closest('[data-i]');
  if (pick && (pick.closest('#pane') || pick.closest('#cands'))) { S.sel = Number(pick.dataset.i); render(); }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); return run(); }
  if (e.key === 'Enter' && e.target.id === 'q') return ask();
  if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('lrow')) { e.preventDefault(); S.sel = Number(e.target.dataset.i); render(); return; }
  if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && e.target.dataset && e.target.dataset.v) {   // nudge a number by 10%, with shift by 10 times
    const x = Number(e.target.value);
    if (!Number.isFinite(x) || e.target.value.trim() === '') return;
    e.preventDefault();
    const f = e.shiftKey ? 10 : 1.1, v = numStr(e.key === 'ArrowUp' ? x * f : x / f);
    e.target.value = v; S.vals[e.target.dataset.v] = v; S.auto.delete(e.target.dataset.v);
    e.target.closest('.field').classList.remove('auto'); schedule(0);
  }
});
document.addEventListener('input', e => {
  const t = e.target;
  if (t.id === 'text') {
    if (S.payload) (S.payload.text_values || []).forEach(v => { delete S.vals[v]; });   // numbers written in the text win again
    return schedule(140);
  }
  if (t.dataset && t.dataset.v) { S.vals[t.dataset.v] = t.value; S.auto.delete(t.dataset.v); t.closest('.field').classList.remove('auto', 'miss'); return schedule(90); }
  if (t.id === 'bits') return setBits(Number(t.value), 40);
});
document.addEventListener('change', e => {
  const t = e.target;
  if (t.id === 'auto') { S.autorun = t.checked; $('go').hidden = autoOK(); if (autoOK()) run(); else live('', 'auto run is off'); return; }
  if (t.id === 'mode') { S.qa = []; statusLine(); $('go').hidden = autoOK(); if (autoOK()) run(); else live('', 'press Analyze or ⌘ Enter, a model reader is not run on every keystroke'); return; }
  if (t.dataset && t.dataset.set) {
    const k = t.dataset.set;
    S[k] = k === 'tn' || k === 'landDec' ? Number(t.value) : t.value;
    if (k === 'landX' && S.landY === S.landX) S.landY = undefined;
    rerunPart(k === 'tn' || k === 'ts' ? 'trials' : 'land');
  }
});
document.addEventListener('toggle', e => {
  const d = e.target;
  if (!d.dataset || !d.dataset.id) return;
  const had = S.open.has(d.dataset.id);
  if (d.open) S.open.add(d.dataset.id); else S.open.delete(d.dataset.id);
  if (d.dataset.id === 'raw' && d.open && !had) { const y = scrollY; render(); scrollTo(0, y); }
}, true);
$('go').onclick = () => { S.qa = []; run(); };
$('theme').onclick = () => {
  const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('rounderr-theme', t); } catch (e) { /* private mode */ }
  $('theme').textContent = t === 'dark' ? 'light' : 'dark';
  paint('land');
};

$('theme').textContent = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
setBitsQuiet(53);
chain();
boot();
