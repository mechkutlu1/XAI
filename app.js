/* =====================================================================
   XAI Sandbox — client-side Shapley laboratory
   All computation runs in-tab. No network calls beyond CDN asset loads.
   ===================================================================== */
'use strict';

const S = {
  raw: null,          // {features:[names], target:name, X:[[...]], y:[...]}
  X: null, y: null, featNames: null,
  mu: null, sd: null, // standardisation
  Xtr: null, ytr: null, Xte: null, yte: null,
  model: null, history: null,
  background: null,   // standardised background rows (array of arrays)
  baseline: null,     // mean row (standardised) = zeros
  explain: { rows: [], idx: [], phi: [], ig: [], perm: [], fx: [], base: 0 },
  diag: null,
  task: 'binary',     // 'binary' | 'regression'
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function toast(msg, ms = 2600) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(t._t); t._t = setTimeout(() => (t.hidden = true), ms);
}
function markStage(stage) {
  const el = document.querySelector(`.pipeline li[data-stage="${stage}"]`);
  if (el) el.classList.add('done');
}

/* ---------------------------------------------------------------------
   Tab navigation
   ------------------------------------------------------------------- */
$$('#tabbar button').forEach((b) => {
  b.addEventListener('click', () => {
    $$('#tabbar button').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    $$('.tab-panel').forEach((p) => p.classList.remove('active'));
    $(`#panel-${b.dataset.tab}`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
function goTab(name) { $(`#tabbar button[data-tab="${name}"]`).click(); }

/* =====================================================================
   1. DATA
   ===================================================================== */
function seeded(seed) { // deterministic PRNG for reproducible benchmark
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeBenchmark() {
  const rng = seeded(42), n = 600;
  const names = ['glucose', 'bmi', 'insulin', 'age', 'pedigree', 'bloodpressure'];
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const glucose = 0.5 + rng();          // 0..1.5
    const bmi = 0.3 + rng() * 0.9;
    const insulin = rng();
    const age = 0.2 + rng() * 0.8;
    const pedigree = rng();
    const bp = rng();                     // irrelevant noise feature
    // known structural equation
    const logit = 1.8 * glucose + 1.1 * bmi * bmi - 1.4 * insulin * age
                + 0.6 * pedigree + 0.0 * bp - 2.4 + 0.35 * gauss(rng);
    const p = 1 / (1 + Math.exp(-logit));
    X.push([glucose, bmi, insulin, age, pedigree, bp]);
    y.push(rng() < p ? 1 : 0);
  }
  loadDataset(names, 'outcome', X, y, 'binary');
  toast('Benchmark generated · n=600, 6 features');
}

function loadDataset(featNames, targetName, X, y, task) {
  S.featNames = featNames; S.X = X; S.y = y; S.task = task;
  S.raw = { features: featNames, target: targetName };
  // standardise
  const m = featNames.length, n = X.length;
  const mu = new Array(m).fill(0), sd = new Array(m).fill(0);
  for (let j = 0; j < m; j++) { for (let i = 0; i < n; i++) mu[j] += X[i][j]; mu[j] /= n; }
  for (let j = 0; j < m; j++) { for (let i = 0; i < n; i++) sd[j] += (X[i][j]-mu[j])**2; sd[j] = Math.sqrt(sd[j]/n) || 1; }
  S.mu = mu; S.sd = sd;
  S.Xstd = X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j]));
  S.baseline = new Array(m).fill(0); // standardised mean

  // split 80/20 deterministically
  const order = [...Array(n).keys()];
  const rng = seeded(7);
  for (let i = n - 1; i > 0; i--) { const k = Math.floor(rng() * (i + 1)); [order[i], order[k]] = [order[k], order[i]]; }
  const cut = Math.floor(n * 0.8);
  const tr = order.slice(0, cut), te = order.slice(cut);
  S.Xtr = tr.map((i) => S.Xstd[i]); S.ytr = tr.map((i) => y[i]);
  S.Xte = te.map((i) => S.Xstd[i]); S.yte = te.map((i) => y[i]);
  S.teIdxRaw = te;

  renderDataSummary(); renderPreview();
  $('#btn-train').disabled = false;
  markStage('data');
  resetDownstream();
}

function resetDownstream() {
  S.model = null; S.explain = { rows: [], idx: [], phi: [], ig: [], perm: [], fx: [], base: 0 }; S.diag = null;
  $('#btn-explain').disabled = true; $('#btn-diagnose').disabled = true;
  ['beeswarm-card','local-card','dependence-card','global-card','curves-card','faith-card','loss-card','train-metrics']
    .forEach((id) => { const e = $('#'+id); if (e) e.hidden = true; });
}

function renderDataSummary() {
  const posRate = S.task === 'binary'
    ? (S.y.reduce((a, b) => a + b, 0) / S.y.length)
    : null;
  const el = $('#data-summary'); el.hidden = false;
  el.innerHTML = `
    <div class="stat"><b>${S.X.length}</b><span>rows</span></div>
    <div class="stat"><b>${S.featNames.length}</b><span>features</span></div>
    <div class="stat"><b>${S.Xtr.length}/${S.Xte.length}</b><span>train / test</span></div>
    <div class="stat"><b>${S.task==='binary' ? (posRate*100).toFixed(0)+'%' : '—'}</b><span>${S.task==='binary'?'positive rate':'regression'}</span></div>`;
}

function renderPreview() {
  const card = $('#preview-card'); card.hidden = false;
  $('#preview-shape').textContent = `${S.X.length} × ${S.featNames.length}`;
  const rows = Math.min(6, S.X.length);
  let html = '<thead><tr>' + S.featNames.map((f) => `<th>${f}</th>`).join('') + `<th>${S.raw.target}</th></tr></thead><tbody>`;
  for (let i = 0; i < rows; i++) {
    html += '<tr>' + S.X[i].map((v) => `<td>${(+v).toFixed(3)}</td>`).join('')
      + `<td>${S.task==='binary'? S.y[i] : (+S.y[i]).toFixed(3)}</td></tr>`;
  }
  html += '</tbody>';
  $('#preview-table').innerHTML = html;
}

/* CSV upload -------------------------------------------------------- */
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.length);
  const header = lines[0].split(',').map((s) => s.trim());
  const rows = lines.slice(1).map((l) => l.split(',').map((s) => s.trim()));
  return { header, rows };
}
let pendingCSV = null;
function handleCSV(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const { header, rows } = parseCSV(reader.result);
      if (header.length < 2 || rows.length < 20) { toast('Need ≥2 columns and ≥20 rows'); return; }
      pendingCSV = { header, rows };
      const sel = $('#target-select');
      sel.innerHTML = header.map((h, i) => `<option value="${i}"${i===header.length-1?' selected':''}>${h}</option>`).join('');
      $('#target-row').hidden = false;
      toast('CSV parsed · choose the target column');
    } catch (e) { toast('Could not parse CSV'); }
  };
  reader.readAsText(file);
}
function applyTarget() {
  if (!pendingCSV) return;
  const ti = +$('#target-select').value;
  const { header, rows } = pendingCSV;
  const featIdx = header.map((_, i) => i).filter((i) => i !== ti);
  if (featIdx.length > 16) { toast('Please use ≤16 feature columns'); return; }
  const X = [], y = [];
  for (const r of rows) {
    const fv = featIdx.map((i) => parseFloat(r[i]));
    const tv = parseFloat(r[ti]);
    if (fv.some((v) => !isFinite(v)) || !isFinite(tv)) continue; // skip non-numeric rows
    X.push(fv); y.push(tv);
  }
  if (X.length < 20) { toast('Too few fully-numeric rows'); return; }
  const uniq = new Set(y);
  const task = (uniq.size <= 2 && [...uniq].every((v) => v === 0 || v === 1)) ? 'binary' : 'regression';
  loadDataset(featIdx.map((i) => header[i]), header[ti], X, y, task);
  toast(`Loaded ${X.length} rows · ${task}`);
}

/* =====================================================================
   2. TRAIN (TensorFlow.js)
   ===================================================================== */
async function train() {
  if (!S.Xtr) return;
  const units = +$('#hidden-units').value, layers = +$('#hidden-layers').value;
  const epochs = +$('#epochs').value, lr = +$('#lr').value;
  const m = S.featNames.length;

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [m], units, activation: 'relu' }));
  for (let l = 1; l < layers; l++) model.add(tf.layers.dense({ units, activation: 'relu' }));
  const outAct = S.task === 'binary' ? 'sigmoid' : 'linear';
  model.add(tf.layers.dense({ units: 1, activation: outAct }));
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: S.task === 'binary' ? 'binaryCrossentropy' : 'meanSquaredError',
    metrics: S.task === 'binary' ? ['accuracy'] : ['mse'],
  });

  const xs = tf.tensor2d(S.Xtr), ys = tf.tensor2d(S.ytr.map((v) => [v]));
  const prog = $('#train-progress'); prog.hidden = false;
  const bar = prog.querySelector('.bar'), lbl = prog.querySelector('.label');
  $('#btn-train').disabled = true;
  const losses = [], vlosses = [];
  $('#loss-card').hidden = false; // visible before the live trace is drawn

  await model.fit(xs, ys, {
    epochs, batchSize: 32, validationSplit: 0.15, shuffle: true,
    callbacks: {
      onEpochEnd: (ep, logs) => {
        losses.push(logs.loss); vlosses.push(logs.val_loss ?? logs.loss);
        const pct = Math.round(((ep + 1) / epochs) * 100);
        bar.style.width = pct + '%';
        lbl.textContent = `epoch ${ep+1}/${epochs} · loss ${logs.loss.toFixed(4)}`;
        drawLoss(losses, vlosses);
        return tf.nextFrame();
      },
    },
  });
  xs.dispose(); ys.dispose();
  S.model = model;

  // evaluate on held-out test
  const metrics = evaluate();
  renderTrainMetrics(metrics);
  drawLoss(losses, vlosses);
  $('#btn-train').disabled = false;
  $('#btn-explain').disabled = false;
  markStage('model');
  toast('Model trained · open the Explain tab');
}

function predictBatch(rows) { // rows: array of standardised arrays -> Float array of f(x)
  return tf.tidy(() => {
    const t = tf.tensor2d(rows);
    const out = S.model.predict(t);
    return out.dataSync();
  });
}

function evaluate() {
  const p = predictBatch(S.Xte);
  if (S.task === 'binary') {
    let correct = 0, tp = 0, fp = 0, fn = 0;
    // AUC via rank statistic
    const pairs = p.map((v, i) => [v, S.yte[i]]);
    for (let i = 0; i < p.length; i++) {
      const pred = p[i] >= 0.5 ? 1 : 0;
      if (pred === S.yte[i]) correct++;
      if (pred === 1 && S.yte[i] === 1) tp++;
      if (pred === 1 && S.yte[i] === 0) fp++;
      if (pred === 0 && S.yte[i] === 1) fn++;
    }
    const pos = pairs.filter((x) => x[1] === 1).map((x) => x[0]);
    const neg = pairs.filter((x) => x[1] === 0).map((x) => x[0]);
    let auc = 0;
    for (const a of pos) for (const b of neg) auc += a > b ? 1 : a === b ? 0.5 : 0;
    auc = pos.length && neg.length ? auc / (pos.length * neg.length) : 0.5;
    const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
    const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    return { task: 'binary', acc: correct / p.length, auc, f1 };
  } else {
    let sse = 0, sst = 0; const mean = S.yte.reduce((a,b)=>a+b,0)/S.yte.length;
    for (let i = 0; i < p.length; i++) { sse += (p[i]-S.yte[i])**2; sst += (S.yte[i]-mean)**2; }
    const r2 = 1 - sse / (sst || 1), rmse = Math.sqrt(sse / p.length);
    return { task: 'regression', r2, rmse };
  }
}

function renderTrainMetrics(m) {
  const el = $('#train-metrics'); el.hidden = false;
  if (m.task === 'binary') {
    const cls = m.auc >= 0.8 ? 'good' : m.auc >= 0.65 ? '' : 'warn';
    el.innerHTML = `
      <div class="stat ${cls}"><b>${m.auc.toFixed(3)}</b><span>test AUC</span></div>
      <div class="stat"><b>${(m.acc*100).toFixed(1)}%</b><span>accuracy</span></div>
      <div class="stat"><b>${m.f1.toFixed(3)}</b><span>F1</span></div>`;
  } else {
    const cls = m.r2 >= 0.7 ? 'good' : m.r2 >= 0.4 ? '' : 'warn';
    el.innerHTML = `
      <div class="stat ${cls}"><b>${m.r2.toFixed(3)}</b><span>test R²</span></div>
      <div class="stat"><b>${m.rmse.toFixed(3)}</b><span>RMSE</span></div>`;
  }
}

/* =====================================================================
   3. EXPLAIN — Kernel SHAP + Integrated Gradients
   ===================================================================== */

/* combinations helper for exact coalition enumeration */
function shapKernelWeight(M, s) { // s = coalition size (1..M-1)
  if (s === 0 || s === M) return 1e6; // constrained rows handled separately
  const binom = combln(M, s);
  return (M - 1) / (Math.exp(binom) * s * (M - s));
}
function combln(n, k) { // log C(n,k)
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}
function lgamma(x) {
  const g = 7, c = [0.99999999999980993,676.5203681218851,-1259.1392167224028,
    771.32342877765313,-176.61502916214059,12.507343278686905,
    -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/* enumerate all coalitions (M<=11) or sample paired coalitions */
function buildCoalitions(M) {
  const masks = [], weights = [];
  if (M <= 11) {
    for (let z = 1; z < (1 << M) - 1; z++) {
      const bits = []; let s = 0;
      for (let j = 0; j < M; j++) { const b = (z >> j) & 1; bits.push(b); s += b; }
      masks.push(bits); weights.push(shapKernelWeight(M, s));
    }
  } else {
    // paired sampling by coalition size drawn from kernel-proportional distribution
    const nSamp = Math.min(2048, 40 * M);
    const sizes = [];
    for (let s = 1; s < M; s++) sizes.push(s);
    const sw = sizes.map((s) => shapKernelWeight(M, s) * Math.exp(combln(M, s)));
    const tot = sw.reduce((a, b) => a + b, 0);
    const rng = seeded(123);
    for (let k = 0; k < nSamp; k += 2) {
      let r = rng() * tot, s = sizes[0];
      for (let i = 0; i < sizes.length; i++) { r -= sw[i]; if (r <= 0) { s = sizes[i]; break; } }
      const idx = [...Array(M).keys()];
      for (let i = M - 1; i > 0; i--) { const t = Math.floor(rng() * (i + 1)); [idx[i], idx[t]] = [idx[t], idx[i]]; }
      const on = new Set(idx.slice(0, s));
      const bits = Array.from({ length: M }, (_, j) => (on.has(j) ? 1 : 0));
      masks.push(bits); weights.push(shapKernelWeight(M, s));
      masks.push(bits.map((b) => 1 - b)); weights.push(shapKernelWeight(M, M - s)); // paired complement
    }
  }
  return { masks, weights };
}

/* weighted least squares with efficiency constraint (Lundberg & Lee) */
function solveWLS(masks, weights, vz, v0, vAll) {
  const M = masks[0].length;
  // constrained: last feature phi_{M-1} = (vAll - v0) - sum_{j<M-1} phi_j
  // Build reduced design on first M-1 features
  const P = M - 1;
  const phiFull = new Array(M).fill(0);
  // A^T W A  (P x P), A^T W b (P)
  const AtA = Array.from({ length: P }, () => new Array(P).fill(0));
  const Atb = new Array(P).fill(0);
  for (let r = 0; r < masks.length; r++) {
    const z = masks[r], w = weights[r];
    // adjusted target: y = v(z) - v0 - z_{M-1}*(vAll - v0)
    const yAdj = (vz[r] - v0) - z[M - 1] * (vAll - v0);
    // design row: x_j = z_j - z_{M-1}, j=0..P-1
    const xr = new Array(P);
    for (let j = 0; j < P; j++) xr[j] = z[j] - z[M - 1];
    for (let a = 0; a < P; a++) {
      Atb[a] += w * xr[a] * yAdj;
      for (let b = 0; b < P; b++) AtA[a][b] += w * xr[a] * xr[b];
    }
  }
  // ridge for numerical stability
  for (let a = 0; a < P; a++) AtA[a][a] += 1e-8;
  const phiReduced = solveLinear(AtA, Atb);
  let sum = 0;
  for (let j = 0; j < P; j++) { phiFull[j] = phiReduced[j]; sum += phiReduced[j]; }
  phiFull[M - 1] = (vAll - v0) - sum;
  return phiFull;
}

function solveLinear(A, b) { // Gaussian elimination
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c] || 1e-12;
    for (let j = c; j <= n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map((r) => r[n]);
}

async function explain() {
  if (!S.model) return;
  const M = S.featNames.length;
  const nExp = Math.min(+$('#n-explain').value, S.Xte.length);
  const nBg = Math.min(+$('#n-background').value, S.Xtr.length);

  // background sample (standardised)
  const rngB = seeded(99);
  const bgIdx = [...Array(S.Xtr.length).keys()].sort(() => rngB() - 0.5).slice(0, nBg);
  S.background = bgIdx.map((i) => S.Xtr[i]);
  // v0 = E[f] over background
  const v0 = mean(predictBatch(S.background));

  // instances to explain
  const rows = S.Xte.slice(0, nExp);
  const { masks, weights } = buildCoalitions(M);

  const prog = $('#explain-progress'); prog.hidden = false;
  const bar = prog.querySelector('.bar'), lbl = prog.querySelector('.label');
  $('#btn-explain').disabled = true;

  const phiAll = [], fxAll = [];
  for (let e = 0; e < rows.length; e++) {
    const x = rows[e];
    // value function per coalition, averaged over background (interventional)
    const vz = coalitionValues(x, masks, S.background);
    const vAll = predictBatch([x])[0];
    const phi = solveWLS(masks, weights, vz, v0, vAll);
    phiAll.push(phi); fxAll.push(vAll);
    if (e % 3 === 0 || e === rows.length - 1) {
      const pct = Math.round(((e + 1) / rows.length) * 100);
      bar.style.width = pct + '%'; lbl.textContent = `SHAP ${e+1}/${rows.length}`;
      await tf.nextFrame();
    }
  }

  // Integrated gradients for the same rows
  lbl.textContent = 'integrated gradients…'; await tf.nextFrame();
  const igAll = integratedGradients(rows, S.baseline);

  // permutation importance (global)
  lbl.textContent = 'permutation importance…'; await tf.nextFrame();
  const perm = permutationImportance();

  S.explain = { rows, phi: phiAll, ig: igAll, perm, fx: fxAll, base: v0, idxEnd: nExp };
  bar.style.width = '100%'; lbl.textContent = 'done';

  // cards must be visible BEFORE drawing — a hidden (display:none) canvas
  // measures 0×0 via getBoundingClientRect and would render nothing.
  ['beeswarm-card','local-card','dependence-card','global-card'].forEach((id)=>$('#'+id).hidden=false);
  populateInstanceSelect(); populateDependence();
  renderBeeswarm(); renderLocal(0); renderDependence(); renderGlobal();
  $('#btn-diagnose').disabled = false;
  markStage('phi');
  toast('Attributions computed');
}

function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }

/* coalition value v(z) = mean over background of f(x where z=1 keeps x_i, z=0 takes bg_i) */
function coalitionValues(x, masks, background) {
  const M = x.length, B = background.length, K = masks.length;
  // Build one big batch: for each mask, for each bg row -> composed input
  const batch = new Array(K * B);
  let p = 0;
  for (let k = 0; k < K; k++) {
    const z = masks[k];
    for (let b = 0; b < B; b++) {
      const row = new Array(M);
      const bg = background[b];
      for (let j = 0; j < M; j++) row[j] = z[j] ? x[j] : bg[j];
      batch[p++] = row;
    }
  }
  const preds = predictBatch(batch);
  const vz = new Array(K);
  for (let k = 0; k < K; k++) {
    let s = 0; const off = k * B;
    for (let b = 0; b < B; b++) s += preds[off + b];
    vz[k] = s / B;
  }
  return vz;
}

/* Integrated gradients via tf.js autodiff, mean baseline, 64-step Riemann */
function integratedGradients(rows, baseline, steps = 64) {
  const M = baseline.length;
  const out = [];
  const gradFn = tf.grad((inp) => {
    const y = S.model.predict(inp);
    return y.sum();
  });
  for (const x of rows) {
    const attr = new Array(M).fill(0);
    // accumulate gradients along path in a single batch of `steps`
    const path = [];
    for (let s = 0; s < steps; s++) {
      const alpha = (s + 0.5) / steps;
      path.push(baseline.map((b, j) => b + alpha * (x[j] - b)));
    }
    const g = tf.tidy(() => {
      const t = tf.tensor2d(path);
      return gradFn(t).mean(0).dataSync();
    });
    for (let j = 0; j < M; j++) attr[j] = (x[j] - baseline[j]) * g[j];
    out.push(attr);
  }
  return out;
}

/* permutation importance on test set (drop in performance when feature shuffled) */
function permutationImportance() {
  const base = evaluate();
  const baseScore = S.task === 'binary' ? base.auc : base.r2;
  const M = S.featNames.length, imp = new Array(M).fill(0);
  const rng = seeded(555);
  for (let j = 0; j < M; j++) {
    const col = S.Xte.map((r) => r[j]);
    const shuffled = [...col];
    for (let i = shuffled.length - 1; i > 0; i--) { const k = Math.floor(rng()*(i+1)); [shuffled[i],shuffled[k]]=[shuffled[k],shuffled[i]]; }
    const Xp = S.Xte.map((r, i) => { const rr = [...r]; rr[j] = shuffled[i]; return rr; });
    const p = predictBatch(Xp);
    let score;
    if (S.task === 'binary') {
      const pos = [], neg = [];
      p.forEach((v, i) => (S.yte[i] === 1 ? pos : neg).push(v));
      let auc = 0; for (const a of pos) for (const b of neg) auc += a>b?1:a===b?0.5:0;
      score = pos.length&&neg.length ? auc/(pos.length*neg.length) : 0.5;
    } else {
      let sse=0,sst=0; const mn=mean(S.yte);
      p.forEach((v,i)=>{sse+=(v-S.yte[i])**2; sst+=(S.yte[i]-mn)**2;});
      score = 1 - sse/(sst||1);
    }
    imp[j] = Math.max(0, baseScore - score);
  }
  return imp;
}

/* =====================================================================
   PLOTTING (canvas, retina-aware)
   ===================================================================== */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const rect = cv.getBoundingClientRect();
  // Fall back to layout/parent width and the CSS height if the element is
  // momentarily unmeasurable (e.g. drawn before its container is shown).
  const cssH = cv.classList.contains('tall') ? 320 : 240;
  const w = rect.width || cv.offsetWidth || (cv.parentElement ? cv.parentElement.clientWidth - 32 : 320) || 320;
  const h = rect.height || cssH;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
const COL = { ink:'#E9EEF4', muted:'#8A94A6', amber:'#F2A93B', pos:'#FF4D6D', neg:'#4D9FFF', grid:'#242D39', ok:'#6fd3a3' };

function drawLoss(losses, vlosses) {
  const cv = $('#loss-canvas'); const { ctx, w, h } = setupCanvas(cv);
  const pad = { l: 44, r: 12, t: 12, b: 26 };
  const all = losses.concat(vlosses); const mx = Math.max(...all), mn = Math.min(...all);
  const X = (i, n) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - mn) / ((mx - mn) || 1)) * (h - pad.t - pad.b);
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const y = pad.t + g/4*(h-pad.t-pad.b); ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.font = '10px "IBM Plex Mono"'; ctx.textAlign='right';
    ctx.fillText((mx-(g/4)*(mx-mn)).toFixed(2), pad.l-6, y+3); }
  const line = (arr, color, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash||[]); ctx.beginPath();
    arr.forEach((v,i)=>{ const x=X(i,arr.length), y=Y(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); ctx.setLineDash([]);
  };
  line(losses, COL.amber); line(vlosses, COL.neg, [4,3]);
  ctx.fillStyle = COL.amber; ctx.textAlign='left'; ctx.fillText('train', w-90, pad.t+10);
  ctx.fillStyle = COL.neg; ctx.fillText('val', w-44, pad.t+10);
  ctx.fillStyle = COL.muted; ctx.textAlign='center'; ctx.fillText('epoch', w/2, h-6);
}

function featureOrderByImportance() {
  const M = S.featNames.length; const meanAbs = new Array(M).fill(0);
  for (const phi of S.explain.phi) for (let j=0;j<M;j++) meanAbs[j] += Math.abs(phi[j]);
  for (let j=0;j<M;j++) meanAbs[j] /= S.explain.phi.length;
  return [...Array(M).keys()].sort((a,b)=>meanAbs[b]-meanAbs[a]);
}

let beeswarmHit = [];
function renderBeeswarm() {
  const cv = $('#beeswarm-canvas'); const { ctx, w, h } = setupCanvas(cv);
  const order = featureOrderByImportance();
  const pad = { l: 96, r: 16, t: 8, b: 26 };
  const rowH = (h - pad.t - pad.b) / order.length;
  // global phi range
  let mx = 0; for (const phi of S.explain.phi) for (const v of phi) mx = Math.max(mx, Math.abs(v));
  mx = mx || 1;
  const X = (v) => pad.l + ((v + mx) / (2 * mx)) * (w - pad.l - pad.r);
  beeswarmHit = [];
  // zero line
  ctx.strokeStyle = COL.grid; ctx.beginPath(); ctx.moveTo(X(0),pad.t); ctx.lineTo(X(0),h-pad.b); ctx.stroke();
  ctx.fillStyle = COL.muted; ctx.font='9px "IBM Plex Mono"'; ctx.textAlign='center';
  ctx.fillText('0', X(0), h-pad.b+12); ctx.fillText(`+${mx.toFixed(2)}`, w-pad.r-14, h-pad.b+12); ctx.fillText(`−${mx.toFixed(2)}`, pad.l+14, h-pad.b+12);

  order.forEach((j, r) => {
    const cy = pad.t + r * rowH + rowH / 2;
    ctx.fillStyle = COL.ink; ctx.font = '11px "IBM Plex Mono"'; ctx.textAlign = 'right';
    ctx.fillText(truncate(S.featNames[j], 12), pad.l - 8, cy + 3);
    // feature value range for colour
    const fvals = S.explain.rows.map((row) => row[j]);
    const fmin = Math.min(...fvals), fmax = Math.max(...fvals);
    // jittered points
    S.explain.phi.forEach((phi, i) => {
      const x = X(phi[j]);
      const jit = (hash(i * 31 + j) - 0.5) * (rowH * 0.62);
      const y = cy + jit;
      const t = (fvals[i] - fmin) / ((fmax - fmin) || 1); // 0..1
      ctx.fillStyle = lerpColor(COL.neg, COL.pos, t);
      ctx.globalAlpha = 0.72;
      ctx.beginPath(); ctx.arc(x, y, 3.1, 0, 7); ctx.fill();
      beeswarmHit.push({ x, y, i });
    });
    ctx.globalAlpha = 1;
  });
  cv.onclick = (ev) => {
    const rect = cv.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    let best = -1, bd = 1e9;
    for (const p of beeswarmHit) { const d = (p.x-px)**2+(p.y-py)**2; if (d<bd){bd=d;best=p.i;} }
    if (best >= 0 && bd < 400) { $('#instance-select').value = best; renderLocal(best); $('#local-card').scrollIntoView({behavior:'smooth'}); }
  };
}

function populateInstanceSelect() {
  const sel = $('#instance-select');
  sel.innerHTML = S.explain.rows.map((_, i) => `<option value="${i}">#${i}</option>`).join('');
  sel.onchange = () => renderLocal(+sel.value);
}

function renderLocal(idx) {
  const phi = S.explain.phi[idx], fx = S.explain.fx[idx], base = S.explain.base;
  $('#local-title').textContent = `instance #${idx}`;
  const predTxt = S.task === 'binary'
    ? `f&#770;(x) = ${fx.toFixed(3)}  ·  E[f&#770;] = ${base.toFixed(3)}`
    : `f&#770;(x) = ${fx.toFixed(3)}  ·  E[f&#770;] = ${base.toFixed(3)}`;
  $('#local-pred').innerHTML = predTxt;
  const sum = phi.reduce((a,b)=>a+b,0);
  const gap = Math.abs((base + sum) - fx);
  $('#local-additivity').innerHTML =
    `Additivity check: E[f&#770;] + Σφ = ${(base+sum).toFixed(3)} vs f&#770;(x) = ${fx.toFixed(3)} · residual ${gap.toExponential(1)}`;
  drawWaterfall(idx);
}

function drawWaterfall(idx) {
  const cv = $('#waterfall-canvas'); const { ctx, w, h } = setupCanvas(cv);
  const phi = S.explain.phi[idx];
  const M = S.featNames.length;
  const order = [...Array(M).keys()].sort((a,b)=>Math.abs(phi[b])-Math.abs(phi[a]));
  const pad = { l: 108, r: 54, t: 8, b: 8 };
  const rowH = (h - pad.t - pad.b) / M;
  const base = S.explain.base;
  let cum = base;
  const total = base + phi.reduce((a,b)=>a+b,0);
  const lo = Math.min(base, total) - 0.05, hi = Math.max(base, total) + 0.05;
  const X = (v) => pad.l + ((v - lo) / ((hi - lo)||1)) * (w - pad.l - pad.r);
  order.forEach((j, r) => {
    const y = pad.t + r * rowH + 2, bh = rowH - 5;
    const from = cum, to = cum + phi[j];
    const x0 = X(Math.min(from,to)), x1 = X(Math.max(from,to));
    ctx.fillStyle = phi[j] >= 0 ? COL.pos : COL.neg;
    ctx.fillRect(x0, y, Math.max(2, x1-x0), bh);
    ctx.fillStyle = COL.ink; ctx.font='11px "IBM Plex Mono"'; ctx.textAlign='right';
    ctx.fillText(truncate(S.featNames[j],13), pad.l-8, y+bh/2+4);
    ctx.fillStyle = COL.muted; ctx.textAlign='left'; ctx.font='10px "IBM Plex Mono"';
    ctx.fillText((phi[j]>=0?'+':'')+phi[j].toFixed(3), x1+4, y+bh/2+4);
    cum = to;
  });
  // baseline & output markers
  ctx.strokeStyle = COL.muted; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(X(base),pad.t); ctx.lineTo(X(base),h-pad.b); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = COL.amber; ctx.textAlign='center'; ctx.font='9px "IBM Plex Mono"';
  ctx.fillText('E[f]', X(base), h-1);
}

function populateDependence() {
  const order = featureOrderByImportance();
  const fSel = $('#dep-feature'), cSel = $('#dep-color');
  fSel.innerHTML = order.map((j)=>`<option value="${j}">${S.featNames[j]}</option>`).join('');
  cSel.innerHTML = `<option value="auto">auto (max interaction)</option>` +
    order.map((j)=>`<option value="${j}">${S.featNames[j]}</option>`).join('');
  fSel.onchange = renderDependence; cSel.onchange = renderDependence;
}

function autoInteraction(j) {
  // pick feature k maximising |corr(feature_k value, phi_j)| excluding j
  const M = S.featNames.length; let best = (j+1)%M, ba = -1;
  const phij = S.explain.phi.map((p)=>p[j]);
  for (let k=0;k<M;k++) if (k!==j) {
    const fk = S.explain.rows.map((r)=>r[k]);
    const c = Math.abs(pearson(fk, phij));
    if (c>ba){ba=c;best=k;}
  }
  return best;
}

function renderDependence() {
  const j = +$('#dep-feature').value;
  const cRaw = $('#dep-color').value;
  const k = cRaw === 'auto' ? autoInteraction(j) : +cRaw;
  const cv = $('#dependence-canvas'); const { ctx, w, h } = setupCanvas(cv);
  const pad = { l: 52, r: 16, t: 14, b: 34 };
  const xs = S.explain.rows.map((r)=>r[j] * S.sd[j] + S.mu[j]); // de-standardise for readability
  const ys = S.explain.phi.map((p)=>p[j]);
  const cvals = S.explain.rows.map((r)=>r[k]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const cmin=Math.min(...cvals),cmax=Math.max(...cvals);
  const X=(v)=>pad.l+((v-xmin)/((xmax-xmin)||1))*(w-pad.l-pad.r);
  const Y=(v)=>pad.t+(1-(v-ymin)/((ymax-ymin)||1))*(h-pad.t-pad.b);
  // grid + zero
  ctx.strokeStyle=COL.grid;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,Y(0));ctx.lineTo(w-pad.r,Y(0));ctx.stroke();
  S.explain.rows.forEach((_,i)=>{
    const t=(cvals[i]-cmin)/((cmax-cmin)||1);
    ctx.fillStyle=lerpColor(COL.neg,COL.pos,t);ctx.globalAlpha=0.75;
    ctx.beginPath();ctx.arc(X(xs[i]),Y(ys[i]),3.4,0,7);ctx.fill();
  });
  ctx.globalAlpha=1;
  ctx.fillStyle=COL.muted;ctx.font='10px "IBM Plex Mono"';ctx.textAlign='center';
  ctx.fillText(S.featNames[j], w/2, h-6);
  ctx.save();ctx.translate(12,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('φ ('+S.featNames[j]+')',0,0);ctx.restore();
  ctx.textAlign='right';ctx.fillStyle=COL.amber;
  ctx.fillText('colour → '+S.featNames[k], w-pad.r, pad.t+2);
}

function renderGlobal() {
  const cv = $('#global-canvas'); const { ctx, w, h } = setupCanvas(cv);
  const M = S.featNames.length;
  const shap = new Array(M).fill(0), ig = new Array(M).fill(0);
  for (const p of S.explain.phi) for (let j=0;j<M;j++) shap[j]+=Math.abs(p[j]);
  for (const p of S.explain.ig) for (let j=0;j<M;j++) ig[j]+=Math.abs(p[j]);
  for (let j=0;j<M;j++){shap[j]/=S.explain.phi.length; ig[j]/=S.explain.ig.length;}
  const perm = S.explain.perm.slice();
  const norm=(a)=>{const mx=Math.max(...a)||1; return a.map((v)=>v/mx);};
  const nsShap=norm(shap),nsIg=norm(ig),nsPerm=norm(perm);
  const order=[...Array(M).keys()].sort((a,b)=>shap[b]-shap[a]);
  const pad={l:104,r:16,t:10,b:24};
  const rowH=(h-pad.t-pad.b)/M;
  const bw=(w-pad.l-pad.r);
  order.forEach((j,r)=>{
    const y=pad.t+r*rowH;
    const groups=[[nsShap[j],COL.amber,'SHAP'],[nsIg[j],COL.neg,'IG'],[nsPerm[j],COL.ok,'perm']];
    const gh=(rowH-6)/3;
    groups.forEach((g,gi)=>{
      ctx.fillStyle=g[1];
      ctx.fillRect(pad.l, y+2+gi*gh, Math.max(1,g[0]*bw), gh-1.5);
    });
    ctx.fillStyle=COL.ink;ctx.font='11px "IBM Plex Mono"';ctx.textAlign='right';
    ctx.fillText(truncate(S.featNames[j],12),pad.l-8,y+rowH/2+4);
  });
  // legend
  ctx.textAlign='left';ctx.font='10px "IBM Plex Mono"';
  ctx.fillStyle=COL.amber;ctx.fillText('■ mean|SHAP|',pad.l,h-8);
  ctx.fillStyle=COL.neg;ctx.fillText('■ mean|IG|',pad.l+96,h-8);
  ctx.fillStyle=COL.ok;ctx.fillText('■ perm',pad.l+176,h-8);
}

/* =====================================================================
   4. DIAGNOSTICS — deletion/insertion + faithfulness scorecard
   ===================================================================== */
async function diagnose() {
  if (!S.explain.phi.length) return;
  const prog = $('#diag-progress'); prog.hidden = false;
  const bar = prog.querySelector('.bar'), lbl = prog.querySelector('.label');
  $('#btn-diagnose').disabled = true;
  const M = S.featNames.length, N = S.explain.rows.length;

  // Deletion & insertion, averaged over instances, ranked by |phi|
  const delShap = new Array(M+1).fill(0), insShap = new Array(M+1).fill(0);
  const delRand = new Array(M+1).fill(0), insRand = new Array(M+1).fill(0);
  const rng = seeded(2024);

  for (let e = 0; e < N; e++) {
    const x = S.explain.rows[e], phi = S.explain.phi[e];
    const orderImp = [...Array(M).keys()].sort((a,b)=>Math.abs(phi[b])-Math.abs(phi[a]));
    const orderRnd = [...Array(M).keys()].sort(()=>rng()-0.5);
    accumCurve(x, orderImp, delShap, insShap);
    accumCurve(x, orderRnd, delRand, insRand);
    if (e%5===0){bar.style.width=Math.round((e/N)*70)+'%';lbl.textContent=`curves ${e+1}/${N}`;await tf.nextFrame();}
  }
  for (let s=0;s<=M;s++){delShap[s]/=N;insShap[s]/=N;delRand[s]/=N;insRand[s]/=N;}

  // Faithfulness correlation (Bhatt): over random subsets S per instance
  lbl.textContent='faithfulness correlation…';await tf.nextFrame();
  const faithCorr = faithfulnessCorrelation();

  // IG completeness gap
  const igGap = igCompletenessGap();

  // SHAP-IG rank agreement (Spearman)
  const rankAgree = shapIgRankAgreement();

  S.diag = { delShap, insShap, delRand, insRand, faithCorr, igGap, rankAgree, M };
  bar.style.width='100%';lbl.textContent='done';
  ['curves-card','faith-card'].forEach((id)=>$('#'+id).hidden=false);
  drawCurves(); renderAUC(); renderFaithScorecard();
  markStage('diag');
  toast('Diagnostics complete');
}

function accumCurve(x, order, delArr, insArr) {
  const M = x.length;
  // deletion: start from full x, progressively set top features to baseline
  const delRows = [], insRows = [];
  const cur = [...x];
  delRows.push([...cur]);
  for (let s=0;s<M;s++){ cur[order[s]] = S.baseline[order[s]]; delRows.push([...cur]); }
  // insertion: start from baseline, progressively insert top features
  const curI = [...S.baseline];
  insRows.push([...curI]);
  for (let s=0;s<M;s++){ curI[order[s]] = x[order[s]]; insRows.push([...curI]); }
  const pd = predictBatch(delRows), pi = predictBatch(insRows);
  for (let s=0;s<=M;s++){ delArr[s]+=pd[s]; insArr[s]+=pi[s]; }
}

function trapz(arr){let a=0;for(let i=1;i<arr.length;i++)a+=(arr[i]+arr[i-1])/2;return a/(arr.length-1);}

function faithfulnessCorrelation(nSubsets=24){
  const M=S.featNames.length, rng=seeded(321);
  const corrs=[];
  for(let e=0;e<S.explain.rows.length;e++){
    const x=S.explain.rows[e], phi=S.explain.phi[e], fx=S.explain.fx[e];
    const sums=[], drops=[];
    const rows=[];
    const subsets=[];
    for(let t=0;t<nSubsets;t++){
      const size=1+Math.floor(rng()*M);
      const idx=[...Array(M).keys()].sort(()=>rng()-0.5).slice(0,size);
      subsets.push(idx);
      const xp=[...x]; let ps=0;
      for(const j of idx){xp[j]=S.baseline[j]; ps+=phi[j];}
      rows.push(xp); sums.push(ps);
    }
    const preds=predictBatch(rows);
    for(let t=0;t<nSubsets;t++) drops.push(fx-preds[t]);
    corrs.push(pearson(sums,drops));
  }
  return mean(corrs.filter((v)=>isFinite(v)));
}

function igCompletenessGap(){
  let g=0;
  for(let e=0;e<S.explain.rows.length;e++){
    const sumIg=S.explain.ig[e].reduce((a,b)=>a+b,0);
    const fx=S.explain.fx[e];
    const fBase=predictBatch([S.baseline])[0];
    g+=Math.abs(sumIg-(fx-fBase));
  }
  return g/S.explain.rows.length;
}

function shapIgRankAgreement(){
  const M=S.featNames.length; const rs=[];
  for(let e=0;e<S.explain.rows.length;e++){
    const a=S.explain.phi[e].map(Math.abs), b=S.explain.ig[e].map(Math.abs);
    rs.push(spearman(a,b));
  }
  return mean(rs.filter((v)=>isFinite(v)));
}

function drawCurves(){
  const cv=$('#curves-canvas');const{ctx,w,h}=setupCanvas(cv);
  const d=S.diag;const M=d.M;
  const pad={l:46,r:14,t:12,b:28};
  const all=[...d.delShap,...d.insShap,...d.delRand,...d.insRand];
  const mn=Math.min(...all),mx=Math.max(...all);
  const X=(s)=>pad.l+(s/M)*(w-pad.l-pad.r);
  const Y=(v)=>pad.t+(1-(v-mn)/((mx-mn)||1))*(h-pad.t-pad.b);
  ctx.strokeStyle=COL.grid;for(let g=0;g<=4;g++){const y=pad.t+g/4*(h-pad.t-pad.b);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();}
  const plot=(arr,color,dash)=>{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash||[]);ctx.beginPath();arr.forEach((v,s)=>{const x=X(s),y=Y(v);s?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
  plot(d.delRand,COL.muted,[3,3]);plot(d.insRand,COL.muted,[3,3]);
  plot(d.delShap,COL.pos);plot(d.insShap,COL.ok);
  ctx.fillStyle=COL.muted;ctx.font='10px "IBM Plex Mono"';ctx.textAlign='center';
  ctx.fillText('features perturbed →',w/2,h-6);
  ctx.textAlign='left';
  ctx.fillStyle=COL.pos;ctx.fillText('deletion',pad.l+4,pad.t+10);
  ctx.fillStyle=COL.ok;ctx.fillText('insertion',pad.l+70,pad.t+10);
  ctx.fillStyle=COL.muted;ctx.fillText('random null',pad.l+150,pad.t+10);
}

function renderAUC(){
  const d=S.diag;
  const delAUC=trapz(d.delShap),insAUC=trapz(d.insShap);
  const delRnd=trapz(d.delRand),insRnd=trapz(d.insRand);
  const el=$('#auc-summary');
  const delGood=delAUC<delRnd, insGood=insAUC>insRnd;
  el.innerHTML=`
    <div class="stat ${delGood?'good':'warn'}"><b>${delAUC.toFixed(3)}</b><span>deletion AUC (↓ better · null ${delRnd.toFixed(3)})</span></div>
    <div class="stat ${insGood?'good':'warn'}"><b>${insAUC.toFixed(3)}</b><span>insertion AUC (↑ better · null ${insRnd.toFixed(3)})</span></div>`;
}

function renderFaithScorecard(){
  const d=S.diag;
  const fc=d.faithCorr, gap=d.igGap, ra=d.rankAgree;
  const fcCls=fc>=0.5?'good':fc>=0.2?'':'warn';
  const gapCls=gap<0.02?'good':gap<0.1?'':'warn';
  const raCls=ra>=0.6?'good':ra>=0.3?'':'warn';
  $('#faith-summary').innerHTML=`
    <div class="stat ${fcCls}"><b>${fc.toFixed(3)}</b><span>faithfulness r (↑ better)</span></div>
    <div class="stat ${gapCls}"><b>${gap.toExponential(1)}</b><span>IG completeness gap (→0)</span></div>
    <div class="stat ${raCls}"><b>${ra.toFixed(3)}</b><span>SHAP–IG rank ρ</span></div>`;
}

/* =====================================================================
   Small stats + helpers
   ===================================================================== */
function pearson(a,b){const n=a.length;let ma=0,mb=0;for(let i=0;i<n;i++){ma+=a[i];mb+=b[i];}ma/=n;mb/=n;
  let sab=0,saa=0,sbb=0;for(let i=0;i<n;i++){const da=a[i]-ma,db=b[i]-mb;sab+=da*db;saa+=da*da;sbb+=db*db;}
  return sab/(Math.sqrt(saa*sbb)||1e-12);}
function rankArr(a){const idx=a.map((v,i)=>[v,i]).sort((x,y)=>x[0]-y[0]);const r=new Array(a.length);
  for(let i=0;i<idx.length;i++)r[idx[i][1]]=i;return r;}
function spearman(a,b){return pearson(rankArr(a),rankArr(b));}
function hash(n){let x=Math.sin(n*127.1)*43758.5453;return x-Math.floor(x);}
function truncate(s,n){return s.length>n?s.slice(0,n-1)+'…':s;}
function lerpColor(c1,c2,t){
  const p=(c)=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];
  const a=p(c1),b=p(c2);t=clamp(t,0,1);
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}

/* export */
function exportPhi(){
  const M=S.featNames.length;
  let csv='instance,'+S.featNames.map((f)=>'shap_'+f).join(',')+','+S.featNames.map((f)=>'ig_'+f).join(',')+',fx,base\n';
  S.explain.phi.forEach((phi,i)=>{
    csv+=i+','+phi.map((v)=>v.toFixed(6)).join(',')+','+S.explain.ig[i].map((v)=>v.toFixed(6)).join(',')+','+S.explain.fx[i].toFixed(6)+','+S.explain.base.toFixed(6)+'\n';
  });
  downloadBlob(csv,'xai_attributions.csv','text/csv');
  toast('Attributions exported');
}
function exportPNG(){
  const cv=$('#global-canvas');
  cv.toBlob((b)=>{const url=URL.createObjectURL(b);const a=document.createElement('a');a.href=url;a.download='xai_global_importance.png';a.click();URL.revokeObjectURL(url);});
  toast('PNG saved');
}
function downloadBlob(text,name,type){const b=new Blob([text],{type});const url=URL.createObjectURL(b);const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}

/* =====================================================================
   Wire up events + PWA install + service worker
   ===================================================================== */
window.addEventListener('DOMContentLoaded', () => {
  $('#btn-demo').addEventListener('click', makeBenchmark);
  $('#csv-input').addEventListener('change', (e) => e.target.files[0] && handleCSV(e.target.files[0]));
  $('#btn-apply-target').addEventListener('click', applyTarget);
  $('#btn-train').addEventListener('click', () => train().catch((e)=>{console.error(e);toast('Training error');$('#btn-train').disabled=false;}));
  $('#btn-explain').addEventListener('click', () => explain().catch((e)=>{console.error(e);toast('Attribution error');$('#btn-explain').disabled=false;}));
  $('#btn-diagnose').addEventListener('click', () => diagnose().catch((e)=>{console.error(e);toast('Diagnostic error');$('#btn-diagnose').disabled=false;}));
  $('#btn-export-phi').addEventListener('click', exportPhi);
  $('#btn-export-png').addEventListener('click', exportPNG);

  // redraw on resize / orientation
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(redrawAll, 200); });

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(()=>{});
  }
});

function redrawAll(){
  if (S.history) {} // noop guard
  if (!$('#beeswarm-card').hidden) renderBeeswarm();
  if (!$('#local-card').hidden) renderLocal(+$('#instance-select').value||0);
  if (!$('#dependence-card').hidden) renderDependence();
  if (!$('#global-card').hidden) renderGlobal();
  if (S.diag){drawCurves();}
}

let deferredPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault(); deferredPrompt = e;
  const b = $('#install-btn'); b.hidden = false;
  b.onclick = async () => { b.hidden = true; deferredPrompt.prompt(); await deferredPrompt.userChoice; deferredPrompt = null; };
});
