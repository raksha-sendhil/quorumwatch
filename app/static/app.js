'use strict';
/* QuorumWatch demo app: Explore (maps + timeline) and Results (simulated study). No dependencies. */

const $ = s => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
// Colour scale for the Isolation Forest score P = ln(0.5/p). Daily synthetic sites calibrate on 365 days,
// so P reaches about 4 (top 1%). Weekly districts calibrate on one year (52 weeks): P tops out at
// ln(0.5 x 53) = 3.3, meaning "higher than every week of the calibration year".
const scoreMax = () => (state.result && state.result.dataset === 'karnataka' ? 3.3 : 4);
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const RAMPS = {
  light: { score: ['#f3ece6', '#eb6834', '#6e2508'], signal: ['#e9eff7', '#2a78d6', '#0b3570'] },
  dark: { score: ['#2a2421', '#d95926', '#ffc7a3'], signal: ['#1e2533', '#3987e5', '#c4e2ff'] },
};
const state = { meta: null, ds: null, result: null, t: 0, site: null, hoverT: null, sources: [], results: null, view: 'explore' };

/* ---------------- helpers ---------------- */
function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function text(parent, x, y, content, attrs = {}) {
  const node = el('text', { x, y, ...attrs }, parent);
  node.textContent = content;
  return node;
}
async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (e) { /* not JSON */ }
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status})`);
  return body;
}
function toast(message) {
  const t = $('#toast');
  t.textContent = message;
  t.classList.add('on');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('on'), 5000);
}
function busy(on) { $('#busy').classList.toggle('hidden', !on); }
const isNum = v => v !== null && v !== undefined && !Number.isNaN(v);
const fmt = (v, d = 2) => isNum(v) ? Number(v).toFixed(d) : '–';
const fmtInt = v => isNum(v) ? Math.round(v).toLocaleString('en-IN') : '–';
function fmtDate(iso, withYear = true) {
  const d = new Date(iso + 'T00:00:00');
  return `${d.getDate()} ${MONTHS[d.getMonth()]}${withYear ? ' ' + d.getFullYear() : ''}`;
}
function theme() { return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark'; }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function ramp(kind, t) {
  const stops = RAMPS[theme()][kind].map(hexRgb);
  t = Math.min(1, Math.max(0, t));
  const seg = t < 0.5 ? 0 : 1, u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const c = stops[seg].map((v, k) => Math.round(v + (stops[seg + 1][k] - v) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function rampCss(kind) { return `linear-gradient(90deg, ${RAMPS[theme()][kind].join(', ')})`; }

const tip = {
  show(html, ev) {
    const t = $('#tooltip');
    t.innerHTML = html;
    t.classList.add('on');
    const w = t.offsetWidth, h = t.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    t.style.left = x + 'px';
    t.style.top = y + 'px';
  },
  hide() { $('#tooltip').classList.remove('on'); },
};
const row = (label, value) => `<div class="t-row"><span>${label}</span><b>${value}</b></div>`;

/* ---------------- theme & views ---------------- */
function setTheme(name) {
  document.documentElement.setAttribute('data-theme', name);
  try { localStorage.setItem('qw-theme-v2', name); } catch (e) { /* storage unavailable */ }
  if (state.result) { renderLegends(); renderMaps(); }
  if (state.results && state.view === 'results') renderResults();
}
function setView(view) {
  state.view = view;
  document.querySelectorAll('.view-btn').forEach(b => b.classList.toggle('on', b.dataset.view === view));
  $('#view-explore').classList.toggle('hidden', view !== 'explore');
  $('#view-results').classList.toggle('hidden', view !== 'results');
  if (view === 'results') loadResults();
  else if (state.result) renderTimeline();
  const url = new URL(location.href);
  if (view === 'results') url.searchParams.set('view', 'results'); else url.searchParams.delete('view');
  history.replaceState(null, '', url);
}

/* ================= EXPLORE ================= */
function sites() { return state.ds.facilities; }
function siteName(i) {
  const label = sites()[i].label;
  return state.ds.name === 'karnataka' ? label.toLowerCase().replace(/\b\w/g, c => c.toUpperCase()) : label;
}
function unit() { return state.ds.unit; }

function useResult(result, sourceLabel) {
  state.result = result;
  state.ds = state.meta.datasets[result.dataset];
  const s = result.series;
  const T = result.dates.length;
  // Start on the first alert (or the strongest signal), showing the first alerting site.
  let t = result.events.length ? result.events[0].day : 0;
  if (!result.events.length) {
    let best = -1;
    for (let k = 0; k < T; k++) {
      const total = s.signal[k].reduce((a, b) => a + (b || 0), 0);
      if (total > best) { best = total; t = k; }
    }
  }
  state.t = Math.min(Math.max(t, 0), T - 1);
  const alerting = s.alert[state.t].map((v, i) => v ? i : -1).filter(i => i >= 0);
  state.site = alerting.length ? alerting[0] : argmax(s.signal[state.t]);
  document.querySelectorAll('.u-site').forEach(n => { n.textContent = unit().site.toLowerCase(); });
  document.querySelectorAll('.u-step').forEach(n => { n.textContent = unit().step.toLowerCase(); });
  renderSummary();
  buildMaps();
  renderLegends();
  renderAll();
}
function argmax(values) {
  let best = 0;
  values.forEach((v, i) => { if ((v || 0) > (values[best] || 0)) best = i; });
  return best;
}
function renderAll() { renderMaps(); renderTimeline(); renderWeek(); renderSite(); }
function setT(t) {
  const T = state.result.dates.length;
  state.t = Math.min(Math.max(t, 0), T - 1);
  renderMaps(); moveCursor(); renderWeek(); renderSite();
}
function selectSite(i) { state.site = i; renderMaps(); renderSite(); }

function renderSummary() {
  const r = state.result;
  const box = $('#summary');
  if (!r.events.length) {
    box.innerHTML = `<i class="dot ok"></i>No regional alert in this period`;
    return;
  }
  const first = r.events[0];
  const names = first.sites.map(siteName).join(', ');
  const more = r.events.length > 1 ? ` · ${r.events.length - 1} later alert${r.events.length > 2 ? 's' : ''}` : '';
  box.innerHTML = `<i class="dot alert"></i>First alert <b>${fmtDate(r.dates[first.day])}</b> · ${names}${more}`;
}

/* ---- maps ---- */
function mapGeometry() {
  const f = sites();
  const outlines = state.ds.outlines || null;
  const flip = state.ds.name === 'synthetic';
  const pts = f.map(d => [d.x, flip ? 100 - d.y : d.y]);
  let xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), pad = 6;
  if (outlines) {
    const nums = outlines.join(' ').match(/-?\d+(\.\d+)?/g).map(Number);
    xs = nums.filter((_, k) => k % 2 === 0);
    ys = nums.filter((_, k) => k % 2 === 1);
    pad = 1.5;
  }
  const minX = Math.min(...xs) - pad, minY = Math.min(...ys) - pad;
  const viewBox = `${minX} ${minY} ${Math.max(...xs) - minX + pad} ${Math.max(...ys) - minY + pad}`;
  return { outlines, pts, viewBox };
}
function buildMaps() {
  const geo = mapGeometry();
  state.maps = {};
  for (const kind of ['score', 'signal']) {
    const svg = $(kind === 'score' ? '#map-score' : '#map-signal');
    svg.innerHTML = '';
    svg.setAttribute('viewBox', geo.viewBox);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const gFill = el('g', {}, svg), gEdge = el('g', {}, svg), gSel = el('g', {}, svg), gRing = el('g', {}, svg);
    if (kind === 'signal') {
      for (const e of state.ds.edges) {
        const [a, b] = [geo.pts[e.s], geo.pts[e.t]];
        el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 'edge', 'stroke-width': geo.outlines ? .25 : .3 }, gEdge);
      }
    }
    const shapes = [], rings = [], sels = [];
    sites().forEach((_, i) => {
      const make = (parent, cls) => geo.outlines
        ? el('path', { d: geo.outlines[i], class: cls }, parent)
        : el('circle', { cx: geo.pts[i][0], cy: geo.pts[i][1], r: cls === 'site' ? 2.6 : 3.6, class: cls }, parent);
      const shape = make(gFill, 'site');
      shape.addEventListener('mousemove', ev => siteTip(i, ev));
      shape.addEventListener('mouseleave', tip.hide);
      shape.addEventListener('click', () => selectSite(i));
      shapes.push(shape);
      rings.push(make(gRing, 'ring'));
      sels.push(make(gSel, 'ring'));
    });
    state.maps[kind] = { shapes, rings, sels, gRing };
  }
}
function renderMaps() {
  if (!state.maps) return;
  const s = state.result.series, t = state.t, scale = state.result.scale;
  const N = sites().length;
  for (let i = 0; i < N; i++) {
    const missing = !isNum(s.reported[t][i]);
    const score = state.maps.score, signal = state.maps.signal;
    score.shapes[i].style.fill = missing ? css('--none') : ramp('score', (s.production[t][i] || 0) / scoreMax());
    signal.shapes[i].style.fill = ramp('signal', (s.signal[t][i] || 0) / scale);
    const alert = s.alert[t][i], active = s.active[t][i];
    signal.rings[i].setAttribute('class', 'ring' + (alert ? ' alert' : active ? ' active' : ''));
    score.rings[i].setAttribute('class', 'ring' + (alert ? ' alert' : ''));
    for (const m of [score, signal]) m.sels[i].setAttribute('class', 'ring' + (i === state.site ? ' sel' : ''));
  }
  // Draw alert rings last so neighbouring outlines do not cover them.
  for (const m of [state.maps.score, state.maps.signal]) {
    m.rings.forEach((r, i) => { if (s.alert[t][i]) m.gRing.appendChild(r); });
  }
}
function siteTip(i, ev) {
  const s = state.result.series, t = state.t;
  const missing = !isNum(s.reported[t][i]);
  tip.show(`<div class="t-head">${siteName(i)}</div>` +
    row(unit().value, missing ? 'no report' : fmtInt(s.reported[t][i])) +
    row('Forest score', missing ? '–' : fmt(s.production[t][i])) +
    row('Signal', fmt(s.signal[t][i])) +
    row('State', s.alert[t][i] ? 'Alert' : s.active[t][i] ? 'Activated' : 'Quiet'), ev);
}
function renderLegends() {
  const p = state.meta.params;
  $('#legend-score').innerHTML =
    `<span class="scale">normal <span class="bar" style="background:${rampCss('score')}"></span> ${state.result.dataset === 'karnataka' ? 'above all normal weeks' : 'top 1%'}</span>` +
    `<span class="key"><i class="box none-box"></i>No report</span>` +
    `<span class="key"><i class="box alert-box"></i>Alert</span>`;
  const mark = Math.min(100, 100 * p.signal_on / state.result.scale);
  $('#legend-signal').innerHTML =
    `<span class="scale">0 <span class="bar" style="background:${rampCss('signal')};position:relative">` +
    `<i style="position:absolute;left:${mark}%;top:-3px;bottom:-3px;width:2px;background:var(--text)"></i></span> ${fmt(state.result.scale, 1)}</span>` +
    `<span class="key">│ activation level</span>` +
    `<span class="key"><i class="box active-box"></i>Activated</span>` +
    `<span class="key"><i class="box alert-box"></i>Alert (quorum)</span>`;
}

/* ---- timeline ---- */
function seriesTotals() {
  const s = state.result.series;
  return state.result.dates.map((_, t) => {
    let cases = 0, reported = 0, unusual = 0, active = 0, alert = 0;
    s.reported[t].forEach((v, i) => {
      if (isNum(v)) { cases += v; reported++; }
      if (s.flagged[t][i]) unusual++;
      if (s.active[t][i]) active++;
      if (s.alert[t][i]) alert++;
    });
    return { cases: reported ? cases : null, reported, unusual, active, alert };
  });
}
function renderTimeline() {
  const box = $('#timeline');
  if (!state.result || !box.clientWidth) return;
  const totals = state.totals = seriesTotals();
  const dates = state.result.dates, T = dates.length, N = sites().length;
  const W = box.clientWidth, left = 190, right = 16;
  const rows = [
    { key: 'cases', label: 'Reported cases', sub: `all ${unit().site.toLowerCase()}s`, h: 110 },
    { key: 'unusual', label: 'Unusual score', sub: `${unit().site.toLowerCase()}s (top 5%)`, h: 80 },
    { key: 'alert', label: 'Activated · alert', sub: `${unit().site.toLowerCase()}s`, h: 80 },
  ];
  const gap = 16, axisH = 30;
  const H = rows.reduce((a, r) => a + r.h + gap, 0) + axisH;
  box.innerHTML = '';
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const x = t => left + (W - left - right) * (t + 0.5) / T;
  const step = (W - left - right) / T;
  const bw = Math.max(1, step - (step > 4 ? 2 : 0.5));
  let y0 = 0;
  const textCol = css('--text-2');
  for (const r of rows) {
    text(svg, 0, y0 + 17, r.label, { class: 'tl-row-label', 'font-weight': 600 });
    text(svg, 0, y0 + 36, r.sub, { class: 'tl-row-label', fill: css('--text-3') });
    el('line', { x1: left, x2: W - right, y1: y0 + r.h, y2: y0 + r.h, class: 'gridline' }, svg);
    if (r.key === 'cases') {
      const max = Math.max(1, ...totals.map(d => d.cases || 0));
      let d = '', open = false;
      totals.forEach((v, t) => {
        if (v.cases === null) { open = false; return; }
        const yy = y0 + r.h - (r.h - 6) * v.cases / max;
        d += `${open ? 'L' : 'M'}${x(t)},${yy}`;
        open = true;
      });
      el('path', { d, fill: 'none', stroke: textCol, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
      text(svg, left - 10, y0 + 12, fmtInt(max), { class: 'tl-row-label', 'text-anchor': 'end', fill: css('--text-3') });
    } else {
      const max = Math.max(4, ...totals.map(d => r.key === 'alert' ? d.active : d.unusual));
      text(svg, left - 10, y0 + 12, String(max), { class: 'tl-row-label', 'text-anchor': 'end', fill: css('--text-3') });
      totals.forEach((v, t) => {
        const draw = (value, fill) => {
          if (!value) return;
          const h = Math.max(2, (r.h - 4) * value / max);
          el('rect', { x: x(t) - bw / 2, y: y0 + r.h - h, width: bw, height: h, rx: Math.min(2, bw / 2), fill }, svg);
        };
        if (r.key === 'unusual') draw(v.unusual, ramp('score', 0.62));
        else { draw(v.active, ramp('signal', 0.45)); draw(v.alert, css('--alert')); }
      });
    }
    y0 += r.h + gap;
  }
  // Month ticks
  const ax = el('g', { class: 'axis' }, svg);
  let lastMonth = null, lastX = -1e9;
  dates.forEach((iso, t) => {
    const d = new Date(iso + 'T00:00:00');
    const key = d.getFullYear() * 12 + d.getMonth();
    if (key === lastMonth) return;
    lastMonth = key;
    if (step * 4 < 22 && d.getMonth() % 2) return;
    if (x(t) - lastX < 70) return;
    lastX = x(t);
    el('line', { x1: x(t), x2: x(t), y1: y0 - gap, y2: y0 - gap + 4 }, ax);
    text(ax, x(t), y0 + 12, d.getMonth() === 0 || t === 0 ? `${MONTHS[d.getMonth()]} ${d.getFullYear()}` : MONTHS[d.getMonth()], { 'text-anchor': 'middle' });
  });
  state.tl = { svg, x, left, right, W, H, T, plotBottom: y0 - gap };
  state.tl.hover = el('line', { y1: 0, y2: y0 - gap, class: 'hover-line', visibility: 'hidden' }, svg);
  state.tl.cursor = el('line', { y1: 0, y2: y0 - gap, class: 'cursor' }, svg);
  const hit = el('rect', { x: left, y: 0, width: W - left - right, height: y0, fill: 'transparent' }, svg);
  const toT = ev => {
    const rect = svg.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * W / rect.width;
    return Math.min(T - 1, Math.max(0, Math.floor((px - left) / step)));
  };
  hit.addEventListener('mousemove', ev => {
    const t = toT(ev), v = totals[t];
    state.tl.hover.setAttribute('x1', x(t)); state.tl.hover.setAttribute('x2', x(t));
    state.tl.hover.setAttribute('visibility', 'visible');
    tip.show(`<div class="t-head">${stepLabel(t)}</div>` + row(`Reported ${unit().value}`, v.cases === null ? 'no reports' : fmtInt(v.cases)) +
      row('Unusual score', v.unusual) + row('Activated', v.active) + row('In alert', v.alert), ev);
  });
  hit.addEventListener('mouseleave', () => { state.tl.hover.setAttribute('visibility', 'hidden'); tip.hide(); });
  hit.addEventListener('click', ev => setT(toT(ev)));
  moveCursor();
}
function moveCursor() {
  if (!state.tl) return;
  const xx = state.tl.x(state.t);
  state.tl.cursor.setAttribute('x1', xx);
  state.tl.cursor.setAttribute('x2', xx);
}
function stepLabel(t) {
  const iso = state.result.dates[t];
  return state.result.dataset === 'karnataka' ? `Week of ${fmtDate(iso)}` : fmtDate(iso);
}

/* ---- side panel ---- */
function renderWeek() {
  const t = state.t, s = state.result.series;
  const v = (state.totals || seriesTotals())[t];
  const alerting = s.alert[t].map((a, i) => a ? i : -1).filter(i => i >= 0);
  const activeOnly = s.active[t].map((a, i) => a && !s.alert[t][i] ? i : -1).filter(i => i >= 0);
  const chips = (ids, cls) => ids.map(i => `<span class="chip ${cls}" data-site="${i}">${siteName(i)}</span>`).join('');
  $('#week-card').innerHTML = `
    <div class="card-title">${stepLabel(t)}</div>
    <div class="stats">
      <div class="stat"><b>${v.cases === null ? '–' : fmtInt(v.cases)}</b><span>reported ${unit().value} (${v.reported}/${sites().length} reporting)</span></div>
      <div class="stat"><b>${v.unusual}</b><span>${unit().site.toLowerCase()}s with an unusual forest score</span></div>
      <div class="stat"><b>${v.active}</b><span>${unit().site.toLowerCase()}s activated</span></div>
      <div class="stat ${v.alert ? 'alerting' : ''}"><b>${v.alert}</b><span>${unit().site.toLowerCase()}s in regional alert</span></div>
    </div>
    <div class="list">${alerting.length ? 'In alert: ' + chips(alerting, 'alert') : 'No regional alert this ' + unit().step.toLowerCase() + '.'}
      ${activeOnly.length ? '<div style="margin-top:6px">Activated: ' + chips(activeOnly, '') + '</div>' : ''}</div>`;
  $('#week-card').querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => selectSite(+c.dataset.site)));
}
function renderSite() {
  const card = $('#site-card');
  const i = state.site;
  if (i === null || i === undefined) { card.innerHTML = `<div class="card-title">${unit().site}</div><div class="empty">Click a ${unit().site.toLowerCase()} on either map.</div>`; return; }
  const s = state.result.series, t = state.t, p = state.meta.params;
  const missing = !isNum(s.reported[t][i]);
  const score = s.production[t][i] || 0;
  const tail = 0.5 * Math.exp(-score);
  const stateName = s.alert[t][i] ? 'Alert' : s.active[t][i] ? 'Activated' : missing ? 'No report' : 'Quiet';
  const dotColor = s.alert[t][i] ? 'var(--alert)' : s.active[t][i] ? ramp('signal', 0.5) : 'var(--text-3)';
  card.innerHTML = `
    <div class="card-title">${siteName(i)}</div>
    <div class="state"><i class="dot" style="background:${dotColor}"></i>${stateName}</div>
    <dl class="kv">
      <dt>${unit().value} this ${unit().step.toLowerCase()}</dt><dd>${missing ? 'no report' : fmtInt(s.reported[t][i])}</dd>
      <dt>Typical</dt><dd>${fmt(sites()[i].volume, 1)}</dd>
      <dt>Forest score</dt><dd>${missing ? '–' : fmt(score)}${!missing && score > 0 ? ` <span style="color:var(--text-3);font-weight:400">(top ${tail < 0.01 ? '<1' : Math.round(tail * 100)}%)</span>` : ''}</dd>
      <dt>Reliability</dt><dd>${fmt(s.reliability[t][i])}</dd>
      <dt>Signal</dt><dd>${fmt(s.signal[t][i])} <span style="color:var(--text-3);font-weight:400">/ ${fmt(p.signal_on, 1)} to activate</span></dd>
    </dl>
    <div id="sparks"></div>`;
  const box = card.querySelector('#sparks');
  spark(box, `${unit().value}`, s.reported.map(r => r[i]), { kind: 'bar', color: css('--text-2') });
  spark(box, 'Isolation Forest score', s.production.map(r => r[i]), { color: ramp('score', 0.62), max: scoreMax() });
  spark(box, 'Bacterial-layer signal', s.signal.map(r => r[i]), { color: ramp('signal', 0.55), threshold: p.signal_on });
}
function spark(parent, label, values, opts) {
  const wrap = document.createElement('div');
  wrap.className = 'spark';
  const s = state.result.series, i = state.site, T = values.length;
  const W = 300, H = 72;
  const max = Math.max(opts.max || 0, opts.threshold ? opts.threshold * 1.25 : 0, ...values.filter(isNum), 1e-6);
  wrap.innerHTML = `<div class="spark-label"><span>${label}</span><span>max ${fmt(Math.max(...values.filter(isNum), 0), opts.kind === 'bar' ? 0 : 1)}</span></div>`;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, preserveAspectRatio: 'none' }, wrap);
  const x = t => W * (t + 0.5) / T, y = v => H - 2 - (H - 6) * Math.min(v, max) / max;
  // Weeks this site was in alert
  s.alert.forEach((r, t) => { if (r[i]) el('rect', { x: W * t / T, y: 0, width: W / T, height: H, fill: css('--alert'), opacity: .14 }, svg); });
  if (opts.kind === 'bar') {
    values.forEach((v, t) => { if (isNum(v) && v > 0) el('rect', { x: x(t) - Math.max(.5, W / T * .35), y: y(v), width: Math.max(1, W / T * .7), height: H - 2 - y(v), fill: opts.color, opacity: .8 }, svg); });
  } else {
    let d = '', open = false;
    values.forEach((v, t) => { if (!isNum(v)) { open = false; return; } d += `${open ? 'L' : 'M'}${x(t)},${y(v)}`; open = true; });
    el('path', { d, fill: 'none', stroke: opts.color, 'stroke-width': 1.6, 'vector-effect': 'non-scaling-stroke' }, svg);
  }
  if (opts.threshold) el('line', { x1: 0, x2: W, y1: y(opts.threshold), y2: y(opts.threshold), stroke: css('--text-3'), 'stroke-dasharray': '3 3', 'vector-effect': 'non-scaling-stroke' }, svg);
  el('line', { x1: x(state.t), x2: x(state.t), y1: 0, y2: H, stroke: css('--text'), 'stroke-width': 1.2, 'vector-effect': 'non-scaling-stroke' }, svg);
  svg.style.cursor = 'pointer';
  svg.addEventListener('click', ev => {
    const rect = svg.getBoundingClientRect();
    setT(Math.floor((ev.clientX - rect.left) / rect.width * T));
  });
  parent.appendChild(wrap);
}

/* ---- data sources ---- */
function renderSources(selected) {
  const sel = $('#source');
  sel.innerHTML = '';
  const group = document.createElement('optgroup');
  group.label = 'Karnataka dengue · 30 districts · weekly';
  for (const d of state.meta.demos) {
    const o = document.createElement('option');
    o.value = 'demo:' + d.id;
    o.textContent = `${d.title} — ${d.note}`;
    group.appendChild(o);
  }
  sel.appendChild(group);
  if (state.uploadName) {
    const o = document.createElement('option');
    o.value = 'upload';
    o.textContent = `Uploaded: ${state.uploadName}`;
    sel.appendChild(o);
  }
  if (selected) sel.value = selected;
}
async function loadDemo(id) {
  busy(true);
  try {
    const result = await api(`/api/demo/${id}`, { method: 'POST' });
    useResult(result);
    renderSources('demo:' + id);
    const url = new URL(location.href);
    url.searchParams.set('demo', id);
    history.replaceState(null, '', url);
  } catch (e) { toast(e.message); } finally { busy(false); }
}
async function uploadFile(file) {
  if (!file) return;
  busy(true);
  try {
    const result = await api('/api/upload', { method: 'POST', body: await file.text(), headers: { 'Content-Type': 'text/csv' } });
    state.uploadName = file.name;
    useResult(result);
    renderSources('upload');
    setView('explore');
  } catch (e) { toast(e.message); } finally { busy(false); $('#file').value = ''; }
}

/* ================= RESULTS ================= */
async function loadResults() {
  if (!state.results) {
    try { state.results = await api('/api/results'); } catch (e) { toast(e.message); return; }
  }
  renderResults();
}
const pct = v => `${Math.round(v * 100)}%`;
const ci = (a, d = 2) => `${fmt(a[1], d)}–${fmt(a[2], d)}`;
function renderResults() {
  const R = state.results;
  const geo = R.geographies.find(g => g.key === 'primary');
  if (!geo) { $('#tiles').innerHTML = '<div class="empty">Run the experiment first.</div>'; return; }
  const all = geo.scenarios.find(s => s.scenario === 'all_scenarios');
  const f = all.full, a = all.forest_alone;
  $('#results-sub').textContent = `Simulated study · ${R.test_runs} paired held-out test runs · 30 facilities · mixed outbreak types and reporting quality`;
  const pd = geo.paired.attributable_detection_rate, pdl = geo.paired.attributable_restricted_delay_days;
  $('#tiles').innerHTML = [
    tile(`Detection at ${R.target} false alerts / year`, fmt(f.detection[0]), `vs ${fmt(a.detection[0])} alone`, `95% CI ${ci(f.detection)} vs ${ci(a.detection)}`),
    tile('Mean delay to detection', `${fmt(f.delay[0], 1)} d`, `vs ${fmt(a.delay[0], 1)} d alone`, `95% CI ${ci(f.delay, 1)} vs ${ci(a.delay, 1)} days`),
    tile('Paired gain from the bacterial layer', `+${fmt(pd[0])}`, `detection · ${fmt(pdl[0], 1)} days`, `95% CI +${fmt(pd[1])} to +${fmt(pd[2])}; ${fmt(pdl[1], 1)} to ${fmt(pdl[2], 1)} days`),
    tile('Communication cost', `${Math.round(geo.messages.full)}`, `vs ${Math.round(geo.messages.forest_alone)} messages / day`, 'One scalar per message, network-wide'),
  ].join('');
  chartTradeoff(R);
  chartReliability(geo, R.target);
  chartTypes(R.by_type);
  tableGeo(R);
  methodCard(R.params);
}
function tile(label, big, vs, sub) {
  return `<div class="card tile"><div class="t-label">${label}</div><div class="t-main"><span class="t-big">${big}</span><span class="t-vs">${vs}</span></div><div class="t-ci">${sub}</div></div>`;
}
function axes(svg, m, W, H, xs, ys, opts = {}) {
  const g = el('g', { class: 'axis' }, svg);
  for (const v of ys.ticks) {
    const yy = ys.f(v);
    el('line', { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: 'gridline' }, g);
    text(g, m.l - 10, yy + 5, ys.fmt(v), { 'text-anchor': 'end' });
  }
  if (xs) for (const v of xs.ticks) text(g, xs.f(v), H - m.b + 22, xs.fmt(v), { 'text-anchor': 'middle' });
  if (opts.xLabel) text(g, (m.l + W - m.r) / 2, H - 4, opts.xLabel, { 'text-anchor': 'middle' });
  if (opts.yLabel) text(g, 16, (m.t + H - m.b) / 2, opts.yLabel, { 'text-anchor': 'middle', transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });
}
function interp(points, x) {
  if (x < points[0][0] || x > points[points.length - 1][0]) return null;
  for (let k = 1; k < points.length; k++) {
    const [x0, y0] = points[k - 1], [x1, y1] = points[k];
    if (x <= x1) return x1 === x0 ? y1 : y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
  return null;
}
function chartTradeoff(R) {
  const box = $('#chart-tradeoff');
  box.innerHTML = '';
  const W = Math.max(320, box.clientWidth), H = 420, m = { l: 70, r: 200, t: 16, b: 52 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const series = { full: R.tradeoff.full, forest_alone: R.tradeoff.forest_alone };
  const xmax = 5;
  const X = v => m.l + (W - m.l - m.r) * Math.min(v, xmax) / xmax;
  const Y = v => m.t + (H - m.t - m.b) * (1 - v);
  axes(svg, m, W, H, { ticks: [0, 1, 2, 3, 4, 5], f: X, fmt: v => v }, { ticks: [0, .25, .5, .75, 1], f: Y, fmt: v => v.toFixed(2) },
    { xLabel: 'False alerts per year (outbreak-free runs)', yLabel: 'Share of outbreaks detected' });
  el('line', { x1: X(R.target), x2: X(R.target), y1: m.t, y2: H - m.b, stroke: css('--text-3'), 'stroke-dasharray': '4 4' }, svg);
  text(svg, X(R.target) + 6, m.t + 12, `${R.target} per year (operating point)`, { class: 'lab' });
  const colors = { full: css('--full'), forest_alone: css('--forest') };
  const names = { full: 'IF + bacterial layer', forest_alone: 'IF alone' };
  for (const [key, pts] of Object.entries(series)) {
    const shown = pts.filter(p => p[0] <= xmax);
    el('path', { d: shown.map((p, k) => `${k ? 'L' : 'M'}${X(p[0])},${Y(p[1])}`).join(''), fill: 'none', stroke: colors[key], 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
    for (const p of shown) el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 3.5, fill: colors[key] }, svg);
    const last = shown[shown.length - 1];
    text(svg, X(last[0]) + 8, Y(last[1]) + (key === 'full' ? -4 : 12), names[key], { class: 'direct' });
  }
  const hover = el('line', { y1: m.t, y2: H - m.b, class: 'hover-line', visibility: 'hidden' }, svg);
  const dots = Object.keys(series).map(k => el('circle', { r: 5, fill: colors[k], stroke: css('--surface'), 'stroke-width': 2, visibility: 'hidden' }, svg));
  const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
  hit.addEventListener('mousemove', ev => {
    const rect = svg.getBoundingClientRect();
    const xv = Math.max(0, Math.min(xmax, ((ev.clientX - rect.left) * W / rect.width - m.l) / (W - m.l - m.r) * xmax));
    hover.setAttribute('x1', X(xv)); hover.setAttribute('x2', X(xv)); hover.setAttribute('visibility', 'visible');
    let html = `<div class="t-head">${xv.toFixed(1)} false alerts / year</div>`;
    Object.entries(series).forEach(([k, pts], j) => {
      const yv = interp(pts, xv);
      dots[j].setAttribute('visibility', yv === null ? 'hidden' : 'visible');
      if (yv !== null) { dots[j].setAttribute('cx', X(xv)); dots[j].setAttribute('cy', Y(yv)); }
      html += row(names[k], yv === null ? 'not reached' : yv.toFixed(2));
    });
    tip.show(html, ev);
  });
  hit.addEventListener('mouseleave', () => { hover.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); tip.hide(); });
}
function chartReliability(geo, target) {
  const box = $('#chart-reliability');
  box.innerHTML = '';
  const W = Math.max(300, box.clientWidth), H = 440, m = { l: 60, r: 16, t: 20, b: 56 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const groups = [['high_reliability', '10%'], ['medium_reliability', '30%'], ['low_reliability', '50%']];
  const lo = 0.5;
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (1 - lo));
  const band = (W - m.l - m.r) / groups.length;
  axes(svg, m, W, H, null, { ticks: [.5, .6, .7, .8, .9, 1], f: Y, fmt: v => v.toFixed(1) }, { xLabel: 'Sites with unreliable reporting' });
  const colors = { full: css('--full'), forest_alone: css('--forest') };
  groups.forEach(([key, label], g) => {
    const sc = geo.scenarios.find(s => s.scenario === key);
    const cx = m.l + band * (g + 0.5);
    text(svg, cx, H - m.b + 24, label, { 'text-anchor': 'middle', class: 'lab' });
    ['full', 'forest_alone'].forEach((method, j) => {
      const d = sc[method].detection, xx = cx + (j ? 20 : -20);
      el('line', { x1: xx, x2: xx, y1: Y(d[2]), y2: Y(d[1]), stroke: colors[method], 'stroke-width': 2, 'stroke-linecap': 'round', opacity: .55 }, svg);
      const dot = el('circle', { cx: xx, cy: Y(d[0]), r: 7, fill: colors[method], stroke: css('--surface'), 'stroke-width': 2 }, svg);
      text(svg, xx + (j ? 12 : -12), Y(d[0]) + 5, d[0].toFixed(2), { class: 'lab', 'text-anchor': j ? 'start' : 'end' });
      const hitArea = el('rect', { x: xx - 10, y: m.t, width: 20, height: H - m.t - m.b, fill: 'transparent' }, svg);
      hitArea.addEventListener('mousemove', ev => tip.show(`<div class="t-head">${label} unreliable · ${method === 'full' ? 'IF + bacterial layer' : 'IF alone'}</div>` +
        row('Detection', d[0].toFixed(2)) + row('95% CI', ci(d)) + row('Delay', `${fmt(sc[method].delay[0], 1)} days`), ev));
      hitArea.addEventListener('mouseleave', tip.hide);
      void dot;
    });
  });
}
function typeLabel(dim, value) {
  if (dim === 'peak_band') {
    const nums = value.match(/[\d.]+/g).map(Number);
    return `${Math.round(nums[0] * 100 / 5) * 5}–${Math.round(nums[1] * 100)}% rise`;
  }
  if (dim === 'ramp_days') return value === '1' ? 'Sudden (1 day)' : `${value}-day ramp`;
  if (dim === 'duration_days') return `${value} days long`;
  if (dim === 'cluster_size') return `${value} sites`;
  return value;
}
function chartTypes(rows) {
  const box = $('#chart-types');
  box.innerHTML = '';
  const dims = [['peak_band', 'Peak rise'], ['ramp_days', 'Onset'], ['duration_days', 'Duration'], ['cluster_size', 'Sites affected']];
  const rowH = 28, headH = 30;
  const n = rows.length;
  const W = Math.max(300, box.clientWidth), m = { l: 160, r: 20, t: 4, b: 38 };
  const H = m.t + m.b + n * rowH + dims.length * headH;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, box);
  const lo = 0.5;
  const X = v => m.l + (W - m.l - m.r) * (Math.max(v, lo) - lo) / (1 - lo);
  const g = el('g', { class: 'axis' }, svg);
  for (const v of [.5, .6, .7, .8, .9, 1]) {
    el('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: 'gridline' }, g);
    text(g, X(v), H - m.b + 22, v.toFixed(1), { 'text-anchor': 'middle' });
  }
  const colors = { full: css('--full'), forest_alone: css('--forest') };
  let y = m.t;
  for (const [dim, title] of dims) {
    text(svg, 0, y + 20, title, { class: 'group' });
    y += headH;
    for (const r of rows.filter(r => r.dimension === dim)) {
      const cy = y + rowH / 2;
      text(svg, 10, cy + 5, typeLabel(dim, r.value), { class: 'lab' });
      el('line', { x1: X(r.forest_alone), x2: X(r.full), y1: cy, y2: cy, stroke: css('--text-3'), 'stroke-width': 2, opacity: .5 }, svg);
      el('circle', { cx: X(r.forest_alone), cy, r: 6, fill: colors.forest_alone, stroke: css('--surface'), 'stroke-width': 1.5 }, svg);
      el('circle', { cx: X(r.full), cy, r: 6, fill: colors.full, stroke: css('--surface'), 'stroke-width': 1.5 }, svg);
      const hit = el('rect', { x: 0, y, width: W, height: rowH, fill: 'transparent' }, svg);
      hit.addEventListener('mousemove', ev => tip.show(`<div class="t-head">${title}: ${typeLabel(dim, r.value)}</div>` +
        row('IF + bacterial layer', r.full.toFixed(2)) + row('IF alone', r.forest_alone.toFixed(2)) + row('Test runs', r.n), ev));
      hit.addEventListener('mouseleave', tip.hide);
      y += rowH;
    }
  }
}
function tableGeo(R) {
  const rows = R.geographies.map(g => {
    const all = g.scenarios.find(s => s.scenario === 'all_scenarios');
    const d = g.paired.attributable_detection_rate;
    return `<tr><td>${g.label}</td>
      <td>${fmt(all.full.detection[0])}<small>${ci(all.full.detection)}</small></td>
      <td>${fmt(all.forest_alone.detection[0])}<small>${ci(all.forest_alone.detection)}</small></td>
      <td>${fmt(all.full.delay[0], 1)}<small>${ci(all.full.delay, 1)}</small></td>
      <td>${fmt(all.forest_alone.delay[0], 1)}<small>${ci(all.forest_alone.delay, 1)}</small></td>
      <td>+${fmt(d[0])}<small>${fmt(d[1])} to ${fmt(d[2])}</small></td></tr>`;
  }).join('');
  $('#table-geo').innerHTML = `<table class="geo"><thead><tr><th></th><th>Detection<br>+ layer</th><th>Detection<br>alone</th><th>Delay (d)<br>+ layer</th><th>Delay (d)<br>alone</th><th>Paired<br>gain</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function methodCard(p) {
  $('#method').innerHTML = `
    <div class="eq">dA<sub>i</sub>/dt = P<sub>i</sub> + β·H(A<sub>i</sub>) + D·Σ<sub>j</sub> w<sub>ij</sub>(A<sub>j</sub> − A<sub>i</sub>) − δ·A<sub>i</sub>
      &nbsp;&nbsp;&nbsp; H(A) = A<sup>n</sup> / (K<sup>n</sup> + A<sup>n</sup>)</div>
    <div class="terms">
      <b>P<sub>i</sub></b><span>Signal production: the site's calibrated Isolation Forest score × its reporting reliability (quorum quenching)</span>
      <b>β = ${p.beta}</b><span>Positive feedback (autoinduction) through the Hill function</span>
      <b>D = ${p.D}</b><span>Diffusion: signal shared with neighbours along weighted links w<sub>ij</sub></span>
      <b>δ = ${p.delta}</b><span>Decay: signal fades without new evidence</span>
      <b>K = ${p.K}, n = ${p.n}</b><span>Hill function: switch-like, cooperative response</span>
      <b>θ = ${fmt(p.theta_on)}</b><span>A site activates when H(A) ≥ θ and stays on until H(A) &lt; 0.6θ (hysteresis)</span>
      <b>q = ${p.quorum}</b><span>Quorum: an activated site alerts when ≥ ${Math.round(p.quorum * 100)}% of its neighbourhood (and ≥ 2 sites) is activated</span>
    </div>`;
}

/* ================= start ================= */
async function start() {
  let saved = null;
  try { saved = localStorage.getItem('qw-theme-v2'); } catch (e) { /* storage unavailable */ }
  setTheme(saved || 'light');
  $('#theme').addEventListener('click', () => setTheme(theme() === 'dark' ? 'light' : 'dark'));
  document.querySelectorAll('.view-btn').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  $('#source').addEventListener('change', ev => { const v = ev.target.value; if (v.startsWith('demo:')) loadDemo(v.slice(5)); });
  $('#file').addEventListener('change', ev => uploadFile(ev.target.files[0]));
  document.addEventListener('dragover', ev => ev.preventDefault());
  document.addEventListener('drop', ev => { ev.preventDefault(); uploadFile(ev.dataTransfer.files[0]); });
  document.addEventListener('keydown', ev => {
    if (state.view !== 'explore' || !state.result || ['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
    if (ev.key === 'ArrowRight') { setT(state.t + 1); ev.preventDefault(); }
    if (ev.key === 'ArrowLeft') { setT(state.t - 1); ev.preventDefault(); }
  });
  let resizeTimer;
  addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.view === 'explore') renderTimeline(); else if (state.results) renderResults(); }, 120);
  });

  try { state.meta = await api('/api/meta'); } catch (e) { toast(e.message); return; }
  const params = new URLSearchParams(location.search);
  renderSources();
  if (params.get('view') === 'results') setView('results');
  const wanted = params.get('demo');
  const demo = state.meta.demos.find(d => d.id === wanted) || state.meta.demos[0];
  if (demo) await loadDemo(demo.id);
  else toast('No example data found. Upload a case file.');
}
start();
