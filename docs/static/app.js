/* The Scissors Effect - project page interactions.
   Curves in fig1.json are exported by scripts/plot_fig1_scissors.py, the same
   script that draws Figure 1, so the page and the figure cannot drift apart. */

const NS = "http://www.w3.org/2000/svg";
const el = (t, a) => { const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]); return n; };
const sgn = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const sgn1 = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1);

/* ---------- 01: the scissors ---------- */
let FIG = null, panelIdx = 0;

function drawScissors() {
  if (!FIG) return;
  const svg = document.getElementById("scissors");
  svg.innerHTML = "";
  const W = 900, H = 340, L = 62, R = 34, T = 24, B = 46;
  const panel = FIG.panels[panelIdx], ps = FIG.ps;
  const rows = panel.rows;

  let lo = 0, hi = 0;
  rows.forEach(r => r.mean.forEach((_, i) => {
    lo = Math.min(lo, r.lo[i]); hi = Math.max(hi, r.hi[i]);
  }));
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const X = p => L + (W - L - R) * p;
  const Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));

  // grid + zero line
  for (let g = 0; g <= 4; g++) {
    const v = lo + (hi - lo) * g / 4;
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v),
      stroke: "#e2dbd1", "stroke-width": 1 }));
    const tx = el("text", { x: L - 9, y: Y(v) + 4, "text-anchor": "end",
      fill: "#8c8175", "font-size": 12, "font-family": "JetBrains Mono, monospace" });
    tx.textContent = v.toFixed(0); svg.appendChild(tx);
  }
  svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0),
    stroke: "#8c8175", "stroke-width": 1.2 }));

  const cur = +document.getElementById("pSlide").value;
  const colors = ["#1f6f8b", "#b4531f", "#c98a3f"];

  rows.forEach((r, ri) => {
    const c = colors[ri % 3];
    // CI band
    let d = "";
    ps.forEach((p, i) => { d += (i ? "L" : "M") + X(p) + "," + Y(r.hi[i]); });
    for (let i = ps.length - 1; i >= 0; i--) d += "L" + X(ps[i]) + "," + Y(r.lo[i]);
    svg.appendChild(el("path", { d: d + "Z", fill: c, opacity: .13 }));
    // line up to the current p
    let dl = "";
    ps.forEach((p, i) => { if (i <= cur) dl += (i ? "L" : "M") + X(p) + "," + Y(r.mean[i]); });
    svg.appendChild(el("path", { d: dl, fill: "none", stroke: c, "stroke-width": 2.4,
      "stroke-linecap": "round" }));
    // ghost of the rest
    let dg = "";
    ps.forEach((p, i) => { if (i >= cur) dg += (i === cur ? "M" : "L") + X(p) + "," + Y(r.mean[i]); });
    svg.appendChild(el("path", { d: dg, fill: "none", stroke: c, "stroke-width": 1.2,
      opacity: .28, "stroke-dasharray": "3 3" }));
    // moving dot
    svg.appendChild(el("circle", { cx: X(ps[cur]), cy: Y(r.mean[cur]), r: 6,
      fill: "#fffdfa", stroke: c, "stroke-width": 2.4 }));
    // label
    const t = el("text", { x: X(ps[cur]) + 12, y: Y(r.mean[cur]) + 4, fill: c,
      "font-size": 13, "font-weight": 600, "font-family": "JetBrains Mono, monospace" });
    t.textContent = sgn1(r.mean[cur]); svg.appendChild(t);
  });

  // axis
  const ax = el("text", { x: (L + W - R) / 2, y: H - 10, "text-anchor": "middle",
    fill: "#5c5248", "font-size": 13 });
  ax.textContent = "diversity probability  p"; svg.appendChild(ax);
  const ay = el("text", { x: 16, y: H / 2, fill: "#5c5248", "font-size": 13,
    transform: "rotate(-90 16," + (H / 2) + ")", "text-anchor": "middle" });
  ay.textContent = "change in transfer ASR vs p=0  (pp)"; svg.appendChild(ay);
  [0, .5, 1].forEach(p => {
    const t = el("text", { x: X(p), y: H - 28, "text-anchor": "middle", fill: "#8c8175",
      "font-size": 12, "font-family": "JetBrains Mono, monospace" });
    t.textContent = p.toFixed(1); svg.appendChild(t);
  });

  // readout chips
  const box = document.getElementById("readout");
  box.innerHTML = rows.map(r => {
    const v = r.mean[cur], robust = /robust/i.test(r.label);
    return '<div class="chip ' + (robust ? "rob" : "std") + '">' +
      '<div class="k">' + r.label.replace(/\$\\epsilon\$/, "eps") + "</div>" +
      '<div class="v">' + sgn1(v) + "</div>" +
      '<div class="s">pp vs p=0 &middot; base ' + r.base.toFixed(1) + "%</div></div>";
  }).join("");
  document.getElementById("fig1src").textContent =
    "Fig. 1 · " + panel.title.trim() +
    " · N=1,000, 5 seeds, r=0.9, all-images basis, sources disjoint from every target." +
    " Bands are 95% image-paired bootstrap intervals.";
}

FIG = FIG1;              // inlined in data.js so the page works from file:// too
drawScissors();

document.getElementById("pSlide").addEventListener("input", e => {
  document.getElementById("pVal").textContent = (e.target.value / 10).toFixed(1);
  drawScissors();
});
document.querySelectorAll("[data-panel]").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("[data-panel]").forEach(x => x.classList.remove("sel"));
  b.classList.add("sel"); panelIdx = +b.dataset.panel; drawScissors();
}));

/* ---------- 02: robustness sweep ---------- */
function drawSweep() {
  const i = +document.getElementById("eSlide").value, r = EPS_SWEEP[i];
  document.getElementById("eVal").textContent = r.label;
  document.getElementById("swMI").textContent = r.mi.toFixed(1);
  document.getElementById("swDI").textContent = r.di.toFixed(1);
  document.getElementById("swD").textContent = sgn1(r.d);
  const chip = document.getElementById("swChip");
  chip.className = "chip " + (r.d > 0 ? "std" : "rob");
  document.getElementById("swNote").textContent =
    "pp · " + (r.d > 0 ? "DI helps" : "DI hurts") + (r.note ? " · " + r.note : "");

  const svg = document.getElementById("sweepPlot"); svg.innerHTML = "";
  const W = 900, H = 200, L = 62, R = 34, T = 18, B = 40;
  const lo = -16, hi = 16;
  const X = k => L + (W - L - R) * k / (EPS_SWEEP.length - 1);
  const Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0),
    stroke: "#8c8175", "stroke-width": 1.2 }));
  let d = "";
  EPS_SWEEP.forEach((row, k) => { d += (k ? "L" : "M") + X(k) + "," + Y(row.d); });
  svg.appendChild(el("path", { d, fill: "none", stroke: "#5c5248", "stroke-width": 2 }));
  EPS_SWEEP.forEach((row, k) => {
    const c = row.d > 0 ? "#1f6f8b" : "#b4531f";
    svg.appendChild(el("circle", { cx: X(k), cy: Y(row.d), r: k === i ? 8 : 4.5,
      fill: k === i ? c : "#fffdfa", stroke: c, "stroke-width": 2.2 }));
    const t = el("text", { x: X(k), y: H - 14, "text-anchor": "middle",
      fill: k === i ? "#241f1a" : "#8c8175", "font-size": 12,
      "font-weight": k === i ? 600 : 400, "font-family": "JetBrains Mono, monospace" });
    t.textContent = row.label.replace("/255", ""); svg.appendChild(t);
  });
  const t2 = el("text", { x: (L + W - R) / 2, y: H - 1, "text-anchor": "middle",
    fill: "#5c5248", "font-size": 12 });
  t2.textContent = "adversarial-training budget  (x/255)"; svg.appendChild(t2);
}
document.getElementById("eSlide").addEventListener("input", drawSweep);
drawSweep();

/* ---------- 03: factorial ---------- */
function drawFactorial(which) {
  const rows = FACTORIAL[which];
  const span = 16;
  document.getElementById("facBars").innerHTML = rows.map(r => {
    const c = r.est >= 0 ? "#1f6f8b" : "#b4531f";
    const w = Math.abs(r.est) / span * 50, left = r.est >= 0 ? 50 : 50 - w;
    const cl = (Math.min(r.lo, r.hi) / span * 50) + 50, ch = (Math.max(r.lo, r.hi) / span * 50) + 50;
    return '<div class="bar"><div>' + r.name +
      (r.key ? ' <span class="dim">&#9664;</span>' : "") +
      (r.zero ? ' <span class="dim">(contains 0)</span>' : "") + "</div>" +
      '<div class="track"><div class="zero" style="left:50%"></div>' +
      '<div class="fill" style="left:' + left + "%;width:" + w + "%;background:" + c +
      (r.zero ? ";opacity:.45" : "") + '"></div>' +
      '<div class="ci" style="left:' + cl + "%;width:" + (ch - cl) + '%"></div></div>' +
      '<div class="lab" style="color:' + c + '">' + sgn(r.est) + "</div></div>";
  }).join("");
}
document.querySelectorAll("[data-fac]").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("[data-fac]").forEach(x => x.classList.remove("sel"));
  b.classList.add("sel"); drawFactorial(b.dataset.fac);
}));
drawFactorial("robust");

/* ---------- 04: variance ratios + LGC table ---------- */
const VAR = [
  { n: "Swin-B", t: "standard", v: 0.036 }, { n: "ResNet-50", t: "standard", v: 0.057 },
  { n: "DenseNet-121", t: "standard", v: 0.131 }, { n: "ConvNeXt-B", t: "standard", v: 0.229 },
  { n: "InceptionV3", t: "standard", v: 0.434 }, { n: "ViT-B/16", t: "standard", v: 0.516 },
  { n: "Salman eps=8", t: "robust", v: 0.027 }, { n: "Engstrom", t: "robust", v: 1.384 },
  { n: "Salman eps=2", t: "robust", v: 1.788 }, { n: "Salman eps=0.5", t: "robust", v: 3.165 },
  { n: "ARES ConvNeXt-AT", t: "robust", v: 11.39 }, { n: "Mo2022", t: "robust", v: 134.5 },
];
document.getElementById("varBars").innerHTML = VAR.map(r => {
  const c = r.t === "robust" ? "#b4531f" : "#1f6f8b";
  const w = Math.min(100, (Math.log10(r.v) + 2) / 4.2 * 100);
  const one = (Math.log10(1) + 2) / 4.2 * 100;
  return '<div class="bar"><div>' + r.n + "</div>" +
    '<div class="track"><div class="zero" style="left:' + one + '%"></div>' +
    '<div class="fill" style="left:0;width:' + w + "%;background:" + c + ';opacity:.75"></div></div>' +
    '<div class="lab" style="color:' + c + '">' + r.v.toFixed(r.v < 10 ? 3 : 1) + "</div></div>";
}).join("");

document.getElementById("lgcTable").innerHTML = MODELS.map(m => {
  const c = m.type === "robust" ? "neg" : m.type === "vlm" ? "dim" : "pos";
  return "<tr><td>" + m.name + '</td><td class="num dim">' + m.hf.toFixed(2) +
    '</td><td class="num ' + c + '">' + m.lgc.toFixed(2) + "</td></tr>";
}).join("");

/* ---------- 05: CG-DI gauge ---------- */
function drawGauge() {
  const v = +document.getElementById("lSlide").value / 100;
  document.getElementById("lVal").textContent = v.toFixed(2);
  document.getElementById("lKnob").style.left = (v * 100) + "%";
  const on = v < CGDI.tau;
  const d = document.getElementById("verdict");
  d.className = "verdict " + (on ? "on" : "off");
  d.innerHTML = on
    ? "✓&nbsp; ENABLE DI &nbsp;<span style='font-weight:400;color:var(--ink-3)'>p = " + CGDI.pOn + "</span>"
    : "✕&nbsp; DISABLE DI &nbsp;<span style='font-weight:400;color:var(--ink-3)'>p = " + CGDI.pOff + "</span>";
}
document.getElementById("lSlide").addEventListener("input", drawGauge);
document.getElementById("presets").innerHTML = CGDI.presets.map(p =>
  '<span class="pill" data-lgc="' + p.lgc + '">' + p.name + " &middot; " + p.lgc.toFixed(2) + "</span>"
).join("");
document.querySelectorAll("[data-lgc]").forEach(b => b.addEventListener("click", () => {
  document.getElementById("lSlide").value = Math.round(+b.dataset.lgc * 100); drawGauge();
}));
drawGauge();

/* ---------- 05b / 06: tables ---------- */
document.getElementById("heldTable").innerHTML = HELDOUT.map(h => {
  const c = h.d >= 0 ? "pos" : "neg";
  const ok = (h.d > 0) === (h.pred === "p=0.8");
  return "<tr><td>" + h.name + '</td><td class="num">' + h.lgc.toFixed(3) +
    '</td><td class="num dim">' + h.pred + '</td><td class="num ' + c + '">' + sgn(h.d) +
    '</td><td class="num dim">[' + sgn(h.lo) + ", " + sgn(h.hi) + "]" +
    (ok ? "" : " ") + "</td></tr>";
}).join("");

document.getElementById("atkTable").innerHTML = ATTACKS.map(a =>
  "<tr><td>" + a.name + '</td><td class="dim" style="text-align:left">' + a.venue +
  '</td><td class="num neg">' + sgn(a.rob) +
  '</td><td class="num ' + (a.std >= 0 ? "pos" : "neg") + '">' + sgn(a.std) + "</td></tr>"
).join("");
