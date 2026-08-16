import { maxAbs, stats } from "../utils/math.js";

export function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

export function formatter(precision = 4) {
  return (value) => {
    if (value === Infinity) return "+∞";
    if (value === -Infinity) return "MASK";
    if (!Number.isFinite(value)) return "NaN";
    const rounded = Math.abs(value) < 0.5 * 10 ** -precision ? 0 : value;
    return rounded.toFixed(precision);
  };
}

function colorStyle(value, scale) {
  if (!Number.isFinite(value)) return "";
  const intensity = Math.min(1, Math.abs(value) / Math.max(scale, 1e-12));
  return `--cell-color:${value >= 0 ? "var(--positive)" : "var(--negative)"};--intensity:${intensity.toFixed(3)}`;
}

export function renderVector(values, { precision = 4, label = "vector", columns = values.length } = {}) {
  const fmt = formatter(precision);
  const scale = maxAbs(values);
  return `<div class="vector-row"><span class="vector-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span><div class="vector-cells" style="--columns:${Math.min(columns, 16)}">${values.map((value) => `<span class="value-cell" style="${colorStyle(value, scale)}">${fmt(value)}</span>`).join("")}</div></div>`;
}

export function renderTensorRows(tensor, rowLabels, { precision = 4, title = "" } = {}) {
  const [rows, cols] = tensor.shape;
  let html = title ? `<h3>${escapeHtml(title)} <small>[${tensor.shape.join(" × ")}]</small></h3>` : "";
  for (let row = 0; row < rows; row += 1) html += renderVector(tensor.data.slice(row * cols, (row + 1) * cols), { precision, label: rowLabels[row] ?? `row ${row}`, columns: cols });
  return html;
}

export function renderMatrix(tensor, rowLabels, colLabels, { precision = 4, interactive = false, selected = null } = {}) {
  const fmt = formatter(precision);
  const [rows, cols] = tensor.shape;
  const finite = tensor.data.filter(Number.isFinite);
  const scale = maxAbs(finite);
  let html = `<table class="matrix-table"><thead><tr><th>Q \ K</th>${colLabels.map((label) => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>`;
  for (let row = 0; row < rows; row += 1) {
    html += `<tr><td>${escapeHtml(rowLabels[row] ?? row)}</td>`;
    for (let col = 0; col < cols; col += 1) {
      const value = tensor.data[row * cols + col];
      const mask = value === -Infinity;
      const attrs = interactive ? ` data-attention-row="${row}" data-attention-col="${col}"` : "";
      const selectedClass = selected && selected.row === row && selected.col === col ? " selected" : "";
      html += `<td class="${mask ? "mask" : `numeric${selectedClass}`}"${attrs} style="${colorStyle(value, scale)}">${fmt(value)}</td>`;
    }
    html += "</tr>";
  }
  return `${html}</tbody></table>`;
}

export function renderPrediction(trace, vocabulary, precision = 4, topN = 8) {
  const [rows, cols] = trace.probabilities.shape;
  const last = trace.probabilities.data.slice((rows - 1) * cols, rows * cols);
  const logits = trace.logits.data.slice((rows - 1) * cols, rows * cols);
  const ranked = last.map((probability, id) => ({ id, token: vocabulary[id], probability, logit: logits[id] })).sort((a, b) => b.probability - a.probability);
  const fmt = formatter(precision);
  const list = ranked.slice(0, topN).map((item) => `<div class="prediction-row"><span>${escapeHtml(item.token)}</span><span class="prob-bar"><i style="width:${(item.probability * 100).toFixed(2)}%"></i></span><span>${(item.probability * 100).toFixed(2)}%</span></div>`).join("");
  const sum = last.reduce((total, value) => total + value, 0);
  return `<div class="prediction-list">${list}</div><div class="sum-check">argmax = ${escapeHtml(ranked[0].token)} · logit ${fmt(ranked[0].logit)}<br>ΣP = ${sum.toFixed(6)}</div>`;
}

export function attentionCellDetails(trace, headIndex, row, col, precision = 4) {
  const fmt = formatter(precision);
  const head = trace.heads[headIndex];
  const dHead = head.q.shape[1];
  const size = head.rawScores.shape[1];
  const q = head.q.data.slice(row * dHead, (row + 1) * dHead);
  const k = head.k.data.slice(col * dHead, (col + 1) * dHead);
  const products = q.map((value, index) => value * k[index]);
  const dot = products.reduce((sum, value) => sum + value, 0);
  const scaled = dot / Math.sqrt(dHead);
  const masked = col > row;
  const allowed = head.rawScores.data.slice(row * size, row * size + row + 1);
  const max = Math.max(...allowed);
  const numerator = masked ? 0 : Math.exp(scaled - max);
  const denominator = allowed.reduce((sum, value) => sum + Math.exp(value - max), 0);
  const weight = head.attentionWeights.data[row * size + col];
  return `<div class="trace-list">
    <div class="trace-item"><strong>対象</strong><code>Head ${headIndex + 1} / Query ${escapeHtml(trace.tokens[row].text)} → Key ${escapeHtml(trace.tokens[col].text)}</code></div>
    <div class="trace-item"><strong>Q vector</strong><code>[${q.map(fmt).join(", ")}]</code></div>
    <div class="trace-item"><strong>K vector</strong><code>[${k.map(fmt).join(", ")}]</code></div>
    <div class="trace-item"><strong>要素積</strong><code>[${products.map(fmt).join(", ")}]</code></div>
  </div>
  <pre class="formula">Q · K
= ${products.map(fmt).join(" + ")}
= ${fmt(dot)}

scaled score
= ${fmt(dot)} / √${dHead}
= ${fmt(scaled)}

mask status = ${masked ? "MASKED (j > i)" : "ALLOWED (j ≤ i)"}
softmax numerator = ${fmt(numerator)}
softmax denominator = ${fmt(denominator)}
attention weight = ${fmt(weight)}</pre>`;
}

export function renderTokenTrace(trace, tokenIndex, precision = 4) {
  const token = trace.tokens[tokenIndex];
  if (!token) return "";
  const pick = (tensor) => {
    const cols = tensor.shape[1];
    return tensor.data.slice(tokenIndex * cols, (tokenIndex + 1) * cols);
  };
  const stages = [
    ["Token embedding", pick(trace.tokenEmbeddings)],
    ["+ Position → Initial", pick(trace.initialRepresentation)],
    ["After Attention projection", pick(trace.attentionOutput)],
    ["After Residual 1", pick(trace.residual1)],
    ["After MLP", pick(trace.mlpOutput)],
    ["After Residual 2", pick(trace.residual2)],
    ["Final representation", pick(trace.finalNorm)],
  ];
  const fmt = formatter(precision);
  return `<p><strong>${escapeHtml(token.text)}</strong> · ID ${token.id} · POS ${token.position}</p><div class="trace-list">${stages.map(([label, values]) => `<div class="trace-item"><strong>${label}</strong><code>[${values.map(fmt).join(", ")}]</code></div>`).join("")}</div>`;
}

export function renderParameterTable(parameters, lastUpdate, precision = 4, selectedName = "") {
  const fmt = formatter(precision);
  return parameters.map(({ name, tensor }) => {
    const valueStats = stats(tensor.data);
    const gradient = lastUpdate?.[name]?.gradient ?? tensor.grad;
    const gradNorm = Math.sqrt(gradient.reduce((sum, value) => sum + value * value, 0));
    return `<tr data-parameter="${escapeHtml(name)}" class="${name === selectedName ? "selected" : ""}"><td>${escapeHtml(name)}</td><td>[${tensor.shape.join(" × ")}]</td><td>${tensor.data.length}</td><td>${fmt(valueStats.mean)}</td><td>${fmt(valueStats.std)}</td><td>${fmt(valueStats.min)}</td><td>${fmt(valueStats.max)}</td><td>${fmt(gradNorm)}</td></tr>`;
  }).join("");
}

export function renderParameterInspector(name, tensor, update, learningRate, precision = 4) {
  const fmt = formatter(precision);
  const values = update?.oldValue ?? tensor.data;
  const gradients = update?.gradient ?? tensor.grad;
  const deltas = update?.update ?? gradients.map((value) => -learningRate * value);
  const current = update?.newValue ?? tensor.data;
  const scale = Math.max(maxAbs(values), maxAbs(gradients), maxAbs(deltas), 1e-12);
  const columns = tensor.shape.length === 2 ? tensor.shape[1] : Math.min(8, tensor.shape[0]);
  const cells = current.map((value, index) => `<button class="parameter-cell" data-param-index="${index}" style="--cell-color:${value >= 0 ? "var(--positive)" : "var(--negative)"};--intensity:${Math.min(1, Math.abs(value) / scale).toFixed(3)}" title="index ${index}\nold ${fmt(values[index])}\ngrad ${fmt(gradients[index])}\ndelta ${fmt(deltas[index])}\nnew ${fmt(value)}">${fmt(value)}</button>`).join("");
  return `<p><strong>${escapeHtml(name)}</strong><br><span class="microcopy">Shape [${tensor.shape.join(" × ")}] · ${tensor.data.length} values</span></p>
    <pre class="formula">new = old - learningRate × clippedGradient
learningRate = ${fmt(learningRate)}

セルにマウスを置くと old / gradient / delta / new を確認できます。</pre>
    <div class="parameter-heatmap" style="--columns:${columns}">${cells}</div>`;
}

export function renderLossChart(history) {
  if (!history.length) return `<text x="360" y="112" text-anchor="middle" fill="#9aa8ac" font-size="13">Training data not available</text>`;
  const width = 720, height = 220, pad = 28;
  const values = history.map((item) => item.loss);
  const min = Math.min(...values), max = Math.max(...values);
  const range = Math.max(max - min, 1e-9);
  const points = history.map((item, index) => {
    const x = pad + (width - pad * 2) * (history.length === 1 ? 0 : index / (history.length - 1));
    const y = pad + (height - pad * 2) * (1 - (item.loss - min) / range);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#344048"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#344048"/><polyline points="${points}" fill="none" stroke="#7dc7bd" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="${pad}" y="18" fill="#9aa8ac" font-size="11">max ${max.toFixed(4)}</text><text x="${pad}" y="${height - 7}" fill="#9aa8ac" font-size="11">min ${min.toFixed(4)}</text><text x="${width - pad}" y="${height - 7}" text-anchor="end" fill="#9aa8ac" font-size="11">step ${history.at(-1).step}</text>`;
}
