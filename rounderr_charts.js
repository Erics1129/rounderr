/* rounderr_charts.js   every picture in the window, drawn by hand as SVG, HTML or canvas
   ====================================================================================
   No chart library. Each function takes plain data from the engine and returns a string of markup
   (or paints a canvas). Colours come from CSS variables so light and dark share one code path.

     graphSVG      the tape as a computation graph, with the forward pass, the backward sweep
                   and the blame played as an animation
     ladderHTML    one log axis in units of u. What the problem forces (kappa), what the algorithm
                   allows (A), the limit between them, and the error that was actually measured
     scatterSVG    random trials as a cloud, kappa against A, with the stability boundary
     histSVG       how far A sits above kappa over the trials
     precisionSVG  the same run at every precision from 8 to 53 bits
     curveSVG      A and kappa along one input
     drawMap       A / kappa across two inputs, a diverging map around the limit
*/
'use strict';
const SUP = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return 'n/a';
  if (x === 0) return '0';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e6) return String(Number(x.toPrecision(d)));
  const [m, e] = x.toExponential(d - 1).split('e');
  return Number(m) + '×10<sup>' + Number(e) + '</sup>';
}
const sup = e => String(e).split('').map(c => SUP[c] || c).join('');
const plain = (x, d) => fmt(x, d).replace(/<sup>(-?\d+)<\/sup>/, (_, e) => sup(e));
const p10 = e => e === 0 ? '1' : e === 1 ? '10' : '10' + sup(e);
const pct = x => (x === null || x === undefined) ? 'n/a' : (100 * x < 1 && x > 0 ? '<1%' : Math.round(100 * x) + '%');
const lg = Math.log10;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
const CH = {};                                   // crosshair data of the charts on screen, by chart id

function logTicks(lo, hi, most = 7) {
  const a = Math.ceil(lo - 1e-9), b = Math.floor(hi + 1e-9);
  const step = Math.max(1, Math.ceil((b - a + 1) / most));
  const t = [];
  for (let e = Math.ceil(a / step) * step; e <= b; e += step) t.push(e);
  return t;
}
const legend = items => `<div class="legend">${items.map(([c, t, kind]) => `<span><i class="${kind || 'sw'}" style="--c:${c}"></i>${t}</span>`).join('')}</div>`;
const none = text => `<div class="nochart">${text}</div>`;

/* ------------------------------------------------------------------ the tape as a graph */
function graphSVG(r, animate) {
  const L = r.lines, n = L.length;
  if (n > 90) return none(`This tape has ${n} lines, too many to draw as a graph. The Tape tab lists every line.`);
  const depth = new Array(n).fill(0), cons = L.map(() => []);
  L.forEach((l, i) => l.args.forEach(a => cons[a].push(i)));
  L.forEach((l, i) => { if (l.args.length) depth[i] = 1 + Math.max(...l.args.map(a => depth[a])); });
  L.forEach((l, i) => { if (l.op === 'const' && cons[i].length) depth[i] = Math.min(...cons[i].map(c => depth[c])) - 1; });
  const maxD = Math.max(...depth);
  const layers = Array.from({length: maxD + 1}, () => []);
  L.forEach((l, i) => layers[depth[i]].push(i));
  const pos = new Array(n).fill(0);
  layers.forEach(ly => ly.forEach((i, k) => { pos[i] = k - (ly.length - 1) / 2; }));
  const bary = (i, fwd) => { const nb = fwd ? L[i].args : cons[i]; return nb.length ? nb.reduce((a, j) => a + pos[j], 0) / nb.length : pos[i]; };
  for (let s = 0; s < 6; s++) {                          // order every layer by where its neighbours sit
    const fwd = s % 2 === 0;
    (fwd ? layers : layers.slice().reverse()).forEach(ly => {
      const want = {}; ly.forEach(i => { want[i] = bary(i, fwd); });
      ly.sort((a, b) => want[a] - want[b] || a - b);
      let last = -Infinity;
      ly.forEach(i => { pos[i] = Math.max(want[i], last + 1); last = pos[i]; });
      const shift = ly.reduce((a, i) => a + want[i] - pos[i], 0) / ly.length;
      ly.forEach(i => { pos[i] += shift; });
    });
  }
  const NW = 136, NH = 44, GX = 50, GY = 30, PAD = 16;
  const minP = Math.min(...pos), maxP = Math.max(...pos);
  const X = i => PAD + depth[i] * (NW + GX), Y = i => PAD + (pos[i] - minP) * (NH + GY);
  const W = 2 * PAD + (maxD + 1) * NW + maxD * GX, H = 2 * PAD + (maxP - minP) * (NH + GY) + NH + 16;
  const step = maxD > 10 ? 70 : 130, t1 = (maxD + 1) * step + 250, t2 = t1 + (maxD + 1) * step + 450;
  const out = r.out === undefined ? n - 1 : r.out;
  const top = L.reduce((b, l) => (l.share > (b ? b.share : 0) ? l : b), null);
  let edges = '', back = '', nodes = '';
  L.forEach((l, i) => l.args.forEach(a => {
    const x1 = X(a) + NW, y1 = Y(a) + NH / 2, x2 = X(i), y2 = Y(i) + NH / 2, dx = Math.max(18, (x2 - x1) / 2);
    const d = `M${x1} ${y1}C${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`;
    edges += `<path class="e" pathLength="1" d="${d}" style="--d:${depth[i]}"/>`;
    back += `<path class="b" pathLength="1" d="${d}" style="--r:${maxD - depth[i]}"/>`;
  }));
  L.forEach((l, i) => {
    const isIn = l.op === 'input', isC = l.op === 'const';
    let head = isIn ? l.name : isC ? l.name : l.text;
    if (head.length > 19) head = head.slice(0, 18) + '…';
    const share = l.share || 0;
    const fill = share > 0 ? `color-mix(in srgb, var(--blame) ${Math.round(14 + 76 * share)}%, var(--node))` : 'var(--node)';
    const cls = ['nd', isIn ? 'in' : isC ? 'const' : (l.d === 0 ? 'exact' : 'round'), i === out ? 'out' : '', top && l === top && share > 0.2 ? 'hot' : '', share > 0.3 ? 'amber' : ''].join(' ');
    const lost = l.digits_lost && l.digits_lost >= 1;
    const tip = [l.text, 'value  ' + plain(l.value, 10), 'λ  ' + plain(l.lam, 5) + (isIn ? '   (gradient of y)' : ''),
      isIn ? '' : 'd  ' + l.d + (l.flags.length ? '   ' + l.flags.join(', ') : (l.d === 1 ? '   rounded step' : l.d === 2 ? '   library function' : '')),
      isIn ? '' : 'slip  ' + plain(l.slip_u, 3) + ' u', isIn ? '' : 'share of the ceiling  ' + pct(share),
      lost ? 'cancellation throws away ' + l.digits_lost.toFixed(1) + ' digits' : ''].filter(Boolean).join('\n');
    nodes += `<g transform="translate(${X(i)} ${Y(i)})"><g class="${cls}" style="--d:${depth[i]};--r:${maxD - depth[i]}" data-tip="${esc(tip)}">
      ${share > 0.2 && top === l ? `<rect class="ring" x="-4" y="-4" width="${NW + 8}" height="${NH + 8}" rx="11"/>` : ''}
      <rect class="box" width="${NW}" height="${NH}" rx="8" style="fill:${fill}"/>
      <text class="t1" x="10" y="18">${esc(head)}</text>
      <text class="t2" x="10" y="35">${esc(plain(l.value, 7))}</text>
      ${share >= 0.005 ? `<text class="sh" x="${NW - 8}" y="35" text-anchor="end">${pct(share)}</text>` : ''}
      ${lost ? `<g class="lost"><rect x="${NW - 70}" y="-10" width="76" height="18" rx="9"/><text x="${NW - 32}" y="3" text-anchor="middle">−${l.digits_lost.toFixed(1)} digits</text></g>` : ''}
      <text class="lam" x="${NW / 2}" y="${NH + 14}" text-anchor="middle">λ ${esc(plain(l.lam, 3))}</text>
    </g></g>`;
  });
  return `<div class="gwrap"><svg class="g${animate ? '' : ' still'}" viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;min-width:${Math.min(W, 640)}px;--stp:${step}ms;--t1:${t1}ms;--t2:${t2}ms" role="img" aria-label="Computation graph of the tape, ${n} lines">
    ${edges}${back}${nodes}</svg></div>
    <div class="phases${animate ? '' : ' still'}" style="--t1:${t1}ms;--t2:${t2}ms"><span class="ph p1"><b>1</b> forward pass, values flow to y</span><span class="ph p2"><b>2</b> backward sweep, λ flows back from y</span><span class="ph p3"><b>3</b> blame, |λ d t| on every rounded step</span></div>`;
}

/* ------------------------------------------------------------------ the error ladder */
function ladderHTML(results, sel) {
  const rows = results.map(r => {
    const k = r.kappa, A = r.A, ay = Math.abs(Number(r.y));
    return {r, act: r.rel_u === null ? null : Math.abs(r.rel_u), typ: ay ? r.typical / (ay * r.u) : null, kap: k, A, lim: r.allow * Math.max(1, k || 1)};
  });
  const vals = [];
  rows.forEach(o => ['act', 'typ', 'kap', 'A', 'lim'].forEach(f => { if (o[f] > 0 && Number.isFinite(o[f])) vals.push(lg(o[f])); }));
  if (!vals.length) return none('The output is 0, so relative error has no meaning here.');
  const lo = Math.min(-1, Math.floor(Math.min(...vals) - 0.3)), hi = Math.ceil(Math.max(...vals) + 0.4);
  const P = v => (!(v > 0) || !Number.isFinite(v)) ? 0 : clamp(100 * (lg(v) - lo) / (hi - lo), 0, 100);
  const body = rows.map((o, i) => {
    const a = Math.min(P(o.kap), P(o.A)), b = Math.max(P(o.kap), P(o.A)), w = o.r.word[0];
    const tipA = `ceiling A = ${plain(o.A)} u\nwhat this algorithm allows in the worst case`;
    return `<div class="lrow${i === sel ? ' sel' : ''}" data-i="${i}" role="button" tabindex="0" aria-label="${esc(o.r.label)}, ${o.r.word.toLowerCase()}">
      <div class="lname"><i class="sw" style="--c:${SERIES[i % 4]}"></i><span>${esc(o.r.label)}</span><span class="pill ${w}">${w === 'S' ? '✓ stable' : w === 'U' ? '✕ unstable' : '? undefined'}</span></div>
      <div class="ltrack">
        <div class="lzone" data-k="z${i}" style="width:${P(o.lim)}%"></div>
        <div class="lgap ${w}" data-k="g${i}" style="left:${a}%;width:${b - a}%"></div>
        <div class="lm lim" data-k="l${i}" style="left:${P(o.lim)}%" data-tip="${esc(`limit = ${plain(o.r.allow)} × max(1, κ) = ${plain(o.lim)} u\nstable when the ceiling stays left of this line`)}"></div>
        <div class="lm kap" data-k="k${i}" style="left:${P(o.kap)}%" data-tip="${esc(`κ = ${plain(o.kap)}\nwhat the problem itself forces, no algorithm can do better`)}"></div>
        <div class="lm cei ${w}" data-k="c${i}" style="left:${P(o.A)}%" data-tip="${esc(tipA)}"></div>
        ${o.act === null ? '' : `<div class="lm act${o.act > 0 ? '' : ' zero'}" data-k="a${i}" style="left:${P(o.act)}%" data-tip="${esc(o.act > 0 ? `measured error = ${plain(o.act)} u\nthe error this run really made, against 60 digits` : 'measured error = 0\nthis run happened to be exact')}"></div>`}
      </div></div>`;
  }).join('');
  const ticks = logTicks(lo, hi, 9).map(e => `<span style="left:${100 * (e - lo) / (hi - lo)}%">${e === 0 ? '1 u' : p10(e)}</span>`).join('');
  return `<div class="ladder">${body}<div class="lrow axis"><div class="lname">relative error, in units of u</div><div class="ltrack ticks">${ticks}</div></div></div>` +
    legend([['var(--ink)', 'measured error', 'dot'], ['var(--ink-soft)', 'κ, what the problem forces', 'dia'], ['var(--blame)', 'A, what the algorithm allows', 'bar'], ['var(--ink-soft)', 'limit', 'dash'], ['var(--ok-soft)', 'stable zone', 'zone']]);
}

/* ------------------------------------------------------------------ shared axes for the SVG charts */
function frame(W, H, m, x, y, o = {}) {
  const X = v => m.l + (v - x[0]) / (x[1] - x[0]) * (W - m.l - m.r), Y = v => H - m.b - (v - y[0]) / (y[1] - y[0]) * (H - m.t - m.b);
  let s = '';
  (o.yt || logTicks(y[0], y[1], 6)).forEach(e => {
    s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(e)}" y2="${Y(e)}"/><text class="tick" x="${m.l - 6}" y="${Y(e) + 3.5}" text-anchor="end">${o.yf ? o.yf(e) : p10(e)}</text>`;
  });
  (o.xt || logTicks(x[0], x[1], 7)).forEach(e => {
    s += `<line class="grid v" x1="${X(e)}" x2="${X(e)}" y1="${m.t}" y2="${H - m.b}"/><text class="tick" x="${X(e)}" y="${H - m.b + 14}" text-anchor="middle">${o.xf ? o.xf(e) : p10(e)}</text>`;
  });
  s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
  if (o.xl) s += `<text class="alab" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${o.xl}</text>`;
  if (o.yl) s += `<text class="alab" x="${m.l - 40}" y="${m.t - 9}" text-anchor="start">${o.yl}</text>`;
  return {X, Y, s};
}
const svg = (id, W, H, inner, label) => `<svg class="chart" id="${id}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${inner}</svg>`;
const hit = (id, W, H, m) => `<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}"/><rect class="hit" data-ch="${id}" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
function path(pts) {                              // pts: [x, y] or null, a null breaks the line
  let d = '', pen = false;
  pts.forEach(p => { if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) { pen = false; return; } d += (pen ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); pen = true; });
  return d;
}

/* ------------------------------------------------------------------ trials, kappa against A */
function scatterSVG(trials, sel) {
  const T = (trials || []).map((t, i) => Object.assign({i}, t)).filter(t => !t.error && t.points && t.points.length);
  if (!T.length) return none(trials && trials[0] && trials[0].error ? esc(trials[0].error) : 'Random trials appear here by themselves.');
  const W = 560, H = 300, m = {l: 50, r: 14, t: 24, b: 36};
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  T.forEach(t => t.points.forEach(([k, a]) => { if (k > 0 && a > 0) { x0 = Math.min(x0, lg(k)); x1 = Math.max(x1, lg(k)); y0 = Math.min(y0, lg(a)); y1 = Math.max(y1, lg(a)); } }));
  if (!Number.isFinite(x0)) return none('No finite trial points to draw.');
  const padr = (a, b) => { const p = Math.max(0.35, (b - a) * 0.08); return [Math.floor((a - p) * 2) / 2, Math.ceil((b + p) * 2) / 2]; };
  [x0, x1] = padr(x0, x1); [y0, y1] = padr(Math.min(y0, 0), y1);
  const f = frame(W, H, m, [x0, x1], [y0, y1], {xl: 'κ   what the problem forces', yl: 'A   what the algorithm allows, in u'});
  const cur = T.find(t => t.i === sel) || T[0], la = lg(cur.allow);
  const bx = [x0, Math.max(x0, Math.min(0, x1)), x1], by = bx.map(e => la + Math.max(0, e));
  const cy = v => clamp(f.Y(v), m.t, H - m.b);
  let s = f.s + `<path class="unzone" d="M${f.X(bx[0])} ${cy(by[0])}L${f.X(bx[1])} ${cy(by[1])}L${f.X(bx[2])} ${cy(by[2])}L${f.X(x1)} ${m.t}L${f.X(x0)} ${m.t}Z"/>
    <path class="limline" d="M${f.X(bx[0])} ${cy(by[0])}L${f.X(bx[1])} ${cy(by[1])}L${f.X(bx[2])} ${cy(by[2])}"/>
    <text class="note bad" x="${m.l + 6}" y="${m.t + 11}">unstable above the line</text><text class="note" x="${W - m.r - 4}" y="${H - m.b - 6}" text-anchor="end">stable</text>`;
  T.forEach(t => {
    s += `<g style="--c:${SERIES[t.i % 4]}">` + t.points.map(([k, a]) => (k > 0 && a > 0) ?
      `<circle class="pt" cx="${f.X(lg(k)).toFixed(1)}" cy="${f.Y(lg(a)).toFixed(1)}" r="2.6" data-tip="${esc(t.label)}&#10;κ ${plain(k, 3)}   A ${plain(a, 3)} u&#10;A/κ ${plain(a / Math.max(1, k), 3)}"/>` : '').join('') + '</g>';
  });
  return legend(T.map(t => [SERIES[t.i % 4], `${esc(t.label)}, ${pct(t.unstable_share)} unstable`])) + svg('ch-scatter', W, H, s, 'Random trials, condition number against amplification');
}

function histSVG(t, color) {
  const B = t.bins || [];
  if (!B.length) return '';
  const W = 560, H = 170, m = {l: 40, r: 12, t: 14, b: 34};
  const x0 = Math.min(B[0][0], 0) - 0.25, x1 = Math.max(B[B.length - 1][0] + 0.5, lg(t.allow) + 0.5), top = Math.max(...B.map(b => b[1]));
  const f = frame(W, H, m, [x0, x1], [0, top * 1.08], {yt: [0, top], yf: e => String(Math.round(e)), xl: 'A / κ', xt: logTicks(x0, x1, 9)});
  const la = lg(t.allow), bw = Math.max(2, f.X(0.25) - f.X(0) - 2);
  let s = f.s;
  B.forEach(([e, c]) => {
    if (!c) return;
    const h = Math.max(1.5, H - m.b - f.Y(c)), un = e + 0.125 >= la;
    s += `<rect class="hbar${un ? ' un' : ''}" x="${(f.X(e) + 1).toFixed(1)}" y="${(H - m.b - h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="2" style="--c:${color}" data-tip="A/κ from ${plain(Math.pow(10, e), 2)} to ${plain(Math.pow(10, e + 0.25), 2)}&#10;${c} of ${t.valid} inputs${un ? '&#10;past the limit' : ''}"/>`;
  });
  s += `<line class="limline" x1="${f.X(la)}" x2="${f.X(la)}" y1="${m.t}" y2="${H - m.b}"/><text class="note" x="${f.X(la) + 4}" y="${m.t + 9}">limit ${plain(t.allow)}</text>`;
  return svg('ch-hist', W, H, s, 'Histogram of A over kappa across the random trials');
}

/* ------------------------------------------------------------------ every precision */
function precisionSVG(prec, sel) {
  const P = (prec && prec.precision || []).map((c, i) => Object.assign({i}, c)).filter(c => c.rows.length);
  if (!P.length) return none('The precision sweep appears here by itself.');
  const W = 560, H = 300, m = {l: 50, r: 14, t: 24, b: 36};
  let y0 = 0, y1 = -Infinity;
  P.forEach(c => c.rows.forEach(([b, act, cei]) => { [act, cei].forEach(v => { if (v > 0) { y0 = Math.min(y0, lg(v)); y1 = Math.max(y1, lg(v)); } }); }));
  y0 = Math.max(-18, Math.floor(Math.min(y0, -53 * lg(2)) - 0.3)); y1 = Math.min(3, Math.ceil(y1 + 0.2));
  const xt = [8, 11, 16, 24, 32, 40, 48, 53];
  const f = frame(W, H, m, [8, 53], [y0, y1], {xt, xf: b => String(b), xl: 'precision, bits', yl: 'relative error of y', yt: logTicks(y0, y1, 7)});
  const cy = v => v > 0 ? clamp(f.Y(lg(v)), m.t, H - m.b) : null;
  let s = f.s;
  [[11, 'half'], [24, 'single'], [53, 'double']].forEach(([b, name]) => { s += `<text class="note" x="${f.X(b) + (b === 53 ? -4 : 4)}" y="${m.t + 10}" text-anchor="${b === 53 ? 'end' : 'start'}">${name}</text>`; });
  s += `<path class="uline" d="M${f.X(8)} ${f.Y(-8 * lg(2))}L${f.X(53)} ${f.Y(-53 * lg(2))}"/>`;
  P.forEach(c => {
    s += `<g style="--c:${SERIES[c.i % 4]}"><path class="ln" d="${path(c.rows.map(([b, act, cei]) => cei > 0 ? [f.X(b), cy(cei)] : null))}"/>` +
      c.rows.map(([b, act]) => act > 0 ? `<circle class="dot" cx="${f.X(b).toFixed(1)}" cy="${cy(act).toFixed(1)}" r="2.6"/>` : '').join('') + '</g>';
  });
  s += `<line class="now" x1="${f.X(prec.bits)}" x2="${f.X(prec.bits)}" y1="${m.t}" y2="${H - m.b}"/>`;
  const bitsOf = px => clamp(Math.round(8 + (px - m.l) / (W - m.l - m.r) * 45), 8, 53);
  CH['ch-prec'] = {W, at: px => { const b = bitsOf(px); return {x: f.X(b), key: b}; },
    tip: b => `<b>${b} bits</b>, u = ${fmt(Math.pow(2, -b), 2)}` + P.map(c => { const row = c.rows.find(q => q[0] === b); return row ? `<br><i class="sw" style="--c:${SERIES[c.i % 4]}"></i>${esc(c.label)}: measured ${fmt(row[1], 2)}, ceiling ${fmt(row[2], 2)}` : ''; }).join('') + '<br><span class="dim">click to run at this precision</span>',
    click: b => window.setBits && window.setBits(b)};
  return legend(P.map(c => [SERIES[c.i % 4], esc(c.label)]).concat([['var(--ink-soft)', 'line: ceiling', 'bar'], ['var(--ink-soft)', 'dots: measured', 'dot'], ['var(--muted)', 'u, one rounding', 'dash']])) +
    svg('ch-prec', W, H, s + hit('ch-prec', W, H, m), 'Relative error against precision from 8 to 53 bits');
}

/* ------------------------------------------------------------------ one input: a curve */
const axisVal = (base, e, log) => log ? base * Math.pow(10, e) : e;
function curveSVG(land, sel) {
  const C = land.cands, W = 560, H = 300, m = {l: 50, r: 14, t: 24, b: 36}, e = land.e, D = land.decades;
  let y0 = 0, y1 = 1;
  C.forEach(c => { c.A.concat(c.kappa).forEach(v => { if (v > 0) { y0 = Math.min(y0, lg(v)); y1 = Math.max(y1, lg(v)); } }); c.dots.forEach(d => { if (d[1] > 0) y0 = Math.min(y0, lg(d[1])); }); });
  y0 = Math.max(-4, Math.floor(y0 - 0.2)); y1 = Math.min(20, Math.ceil(y1 + 0.3));
  const x0v = land.base[land.x], xf = t => land.log ? plain(x0v * Math.pow(10, t), 2) : plain(t, 2);
  const xt = [-D, -D / 2, 0, D / 2, D];
  const f = frame(W, H, m, [-D, D], [y0, y1], {xt, xf, xl: `${esc(land.x)}, everything else held at your numbers`, yl: 'relative error, in u'});
  const cy = v => v > 0 ? clamp(f.Y(lg(v)), m.t, H - m.b) : null;
  let s = f.s;
  const k0 = C[0].kappa, lim = k0.map(k => k === null ? null : C[sel < C.length ? sel : 0].allow * Math.max(1, k));
  s += `<path class="limline" d="${path(lim.map((v, i) => v === null ? null : [f.X(e[i]), cy(v)]))}"/>`;
  s += `<path class="kline" d="${path(k0.map((v, i) => v === null || !(v > 0) ? null : [f.X(e[i]), cy(v)]))}"/>`;
  C.forEach((c, i) => {
    s += `<g style="--c:${SERIES[i % 4]}"><path class="ln" d="${path(c.A.map((v, j) => v === null || !(v > 0) ? null : [f.X(e[j]), cy(v)]))}"/>` +
      c.dots.map(d => d[1] > 0 ? `<circle class="dot" cx="${f.X(d[0]).toFixed(1)}" cy="${cy(d[1]).toFixed(1)}" r="2.6"/>` : '').join('') + '</g>';
  });
  s += `<line class="now" x1="${f.X(0)}" x2="${f.X(0)}" y1="${m.t}" y2="${H - m.b}"/><text class="note" x="${f.X(0) + 4}" y="${m.t + 9}">your ${esc(land.x)}</text>`;
  const idx = px => clamp(Math.round((px - m.l) / (W - m.l - m.r) * (e.length - 1)), 0, e.length - 1);
  CH['ch-land'] = {W, at: px => { const i = idx(px); return {x: f.X(e[i]), key: i}; },
    tip: i => `<b>${esc(land.x)} = ${fmt(land.xs[i], 4)}</b><br>κ ${fmt(k0[i], 3)}` + C.map((c, j) => `<br><i class="sw" style="--c:${SERIES[j % 4]}"></i>${esc(c.label)}: A ${fmt(c.A[i], 3)} u`).join('') + '<br><span class="dim">click to move there</span>',
    click: i => window.setVals && window.setVals({[land.x]: land.xs[i]})};
  return legend(C.map((c, i) => [SERIES[i % 4], esc(c.label) + ', ceiling A']).concat([['var(--ink-soft)', 'κ, the problem', 'dash'], ['var(--bad)', 'limit', 'dash'], ['var(--ink-soft)', 'dots: measured', 'dot']])) +
    svg('ch-land', W, H, s + hit('ch-land', W, H, m), 'Error ceiling and condition number along one input');
}

/* ------------------------------------------------------------------ two inputs: a map */
function mapColor(z, la, dark) {
  const N = dark ? [56, 58, 66] : [236, 237, 240], B = dark ? [57, 135, 229] : [28, 92, 171], R = dark ? [240, 96, 127] : [176, 28, 62];
  let t, to;
  if (z <= la) { t = 0.22 + 0.78 * clamp((la - z) / Math.max(la, 0.5), 0, 1); to = B; } else { t = 0.3 + 0.7 * Math.pow(clamp((z - la) / 7, 0, 1), 0.6); to = R; }
  return [0, 1, 2].map(i => Math.round(N[i] + (to[i] - N[i]) * t));
}
function drawMap(canvas, land, c) {
  const m = land.m, dark = document.documentElement.dataset.theme === 'dark', la = lg(c.allow);
  const off = document.createElement('canvas'); off.width = m; off.height = m;
  const ctx = off.getContext('2d'), img = ctx.createImageData(m, m), void_ = dark ? [30, 36, 54] : [250, 251, 253];
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
    const z = c.z[j * m + i], rgb = z === null ? void_ : mapColor(z, la, dark), o = 4 * ((m - 1 - j) * m + i);
    img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const g = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(off, 0, 0, w, h);
  g.strokeStyle = dark ? 'rgba(255,255,255,.9)' : 'rgba(20,33,61,.9)'; g.lineWidth = 2;
  g.beginPath(); g.arc(w / 2, h / 2, 7, 0, 2 * Math.PI); g.stroke();
  g.beginPath(); g.moveTo(w / 2 - 13, h / 2); g.lineTo(w / 2 - 3, h / 2); g.moveTo(w / 2 + 3, h / 2); g.lineTo(w / 2 + 13, h / 2);
  g.moveTo(w / 2, h / 2 - 13); g.lineTo(w / 2, h / 2 - 3); g.moveTo(w / 2, h / 2 + 3); g.lineTo(w / 2, h / 2 + 13); g.stroke();
}
function mapHTML(land) {
  const D = land.decades, m = land.m;
  const tx = [0, (m - 1) / 2, m - 1].map(i => plain(land.xs[Math.round(i)], 2)), ty = [m - 1, (m - 1) / 2, 0].map(i => plain(land.ys[Math.round(i)], 2));
  const dark = document.documentElement.dataset.theme === 'dark';
  const stops = [-1, -0.5, -0.01, 0.01, 2, 4, 7].map(d => { const [r, g, b] = mapColor(1 + d, 1, dark); return `rgb(${r},${g},${b})`; });
  return `<div class="maps">${land.cands.map((c, i) => `<figure class="map"><figcaption><i class="sw" style="--c:${SERIES[i % 4]}"></i>${esc(c.label)}<span class="pill ${c.unstable_share > 0 ? 'U' : 'S'}">${c.unstable_share === null ? 'n/a' : pct(c.unstable_share) + ' of the map unstable'}</span></figcaption>
    <div class="mapbox"><div class="yax"><span>${ty[0]}</span><span>${ty[1]}</span><span>${ty[2]}</span></div><canvas width="366" height="366" data-map="${i}" aria-label="Map of A over kappa across ${esc(land.x)} and ${esc(land.y)}"></canvas>
    <div class="yname">${esc(land.y)}</div><div class="xax"><span>${tx[0]}</span><span>${tx[1]}</span><span>${tx[2]}</span></div><div class="xname">${esc(land.x)}</div></div></figure>`).join('')}</div>
    <div class="mapkey"><span>A/κ = 1</span><i style="background:linear-gradient(90deg,${stops.slice(0, 3).join(',')})"></i><span>limit</span><i style="background:linear-gradient(90deg,${stops.slice(3).join(',')})"></i><span>10⁷ past it</span><span class="dim">◎ your numbers. Click anywhere on a map to move there.</span></div>`;
}
