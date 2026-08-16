import { assertFinite, assertShape, sizeOf } from "../utils/math.js";

function zeros(length) {
  return new Array(length).fill(0);
}

function sameShape(a, b) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function broadcastKind(a, b) {
  if (sameShape(a.shape, b.shape)) return "same";
  if (b.data.length === 1) return "b-scalar";
  if (a.data.length === 1) return "a-scalar";
  if (b.shape.length === 1 && a.shape.at(-1) === b.shape[0]) return "b-last";
  if (a.shape.length === 1 && b.shape.at(-1) === a.shape[0]) return "a-last";
  throw new Error(`Unsupported broadcast [${a.shape}] with [${b.shape}]`);
}

function valueAt(tensor, index, kind, side, outputShape) {
  if ((kind === "b-scalar" && side === "b") || (kind === "a-scalar" && side === "a")) return tensor.data[0];
  if ((kind === "b-last" && side === "b") || (kind === "a-last" && side === "a")) return tensor.data[index % outputShape.at(-1)];
  return tensor.data[index];
}

function accumulateBroadcast(tensor, index, value, kind, side, outputShape) {
  if (!tensor.requiresGrad) return;
  if ((kind === "b-scalar" && side === "b") || (kind === "a-scalar" && side === "a")) tensor.grad[0] += value;
  else if ((kind === "b-last" && side === "b") || (kind === "a-last" && side === "a")) tensor.grad[index % outputShape.at(-1)] += value;
  else tensor.grad[index] += value;
}

export class Tensor {
  constructor(data, shape, { requiresGrad = false, name = "", parents = [], backward = null } = {}) {
    this.data = Array.from(data, Number);
    this.shape = [...shape];
    assertShape(this.data, this.shape, name || "tensor");
    assertFinite(this.data, name || "tensor");
    this.grad = zeros(this.data.length);
    this.requiresGrad = requiresGrad;
    this.name = name;
    this.parents = parents;
    this._backward = backward ?? (() => {});
  }

  cloneData() {
    return { data: [...this.data], shape: [...this.shape] };
  }

  zeroGrad() {
    this.grad.fill(0);
  }

  backward(seed = 1) {
    if (this.data.length !== 1) throw new Error("backward() requires a scalar output");
    const order = [];
    const visited = new Set();
    const visit = (node) => {
      if (visited.has(node)) return;
      visited.add(node);
      node.parents.forEach(visit);
      order.push(node);
    };
    visit(this);
    order.forEach((node) => node.grad.fill(0));
    this.grad[0] = seed;
    for (let i = order.length - 1; i >= 0; i -= 1) order[i]._backward();
  }
}

export function scalar(value, options = {}) {
  return new Tensor([value], [1], options);
}

export function parameter(data, shape, name) {
  return new Tensor(data, shape, { requiresGrad: true, name });
}

function elementwise(a, b, forwardFn, backwardFn, name) {
  const kind = broadcastKind(a, b);
  const outputShape = kind.startsWith("a-") ? b.shape : a.shape;
  const length = sizeOf(outputShape);
  const data = new Array(length);
  for (let i = 0; i < length; i += 1) data[i] = forwardFn(valueAt(a, i, kind, "a", outputShape), valueAt(b, i, kind, "b", outputShape));
  let out;
  out = new Tensor(data, outputShape, {
    requiresGrad: a.requiresGrad || b.requiresGrad,
    parents: [a, b],
    name,
    backward: () => {
      for (let i = 0; i < length; i += 1) {
        const av = valueAt(a, i, kind, "a", outputShape);
        const bv = valueAt(b, i, kind, "b", outputShape);
        const [da, db] = backwardFn(out.grad[i], av, bv);
        accumulateBroadcast(a, i, da, kind, "a", outputShape);
        accumulateBroadcast(b, i, db, kind, "b", outputShape);
      }
    },
  });
  return out;
}

export function add(a, b) {
  return elementwise(a, b, (x, y) => x + y, (g) => [g, g], "add");
}

export function subtract(a, b) {
  return elementwise(a, b, (x, y) => x - y, (g) => [g, -g], "subtract");
}

export function multiply(a, b) {
  return elementwise(a, b, (x, y) => x * y, (g, x, y) => [g * y, g * x], "multiply");
}

export function divide(a, b) {
  return elementwise(a, b, (x, y) => x / y, (g, x, y) => [g / y, -g * x / (y * y)], "divide");
}

export function matmul(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) {
    throw new Error(`matmul shape mismatch [${a.shape}] x [${b.shape}]`);
  }
  const [rows, inner] = a.shape;
  const cols = b.shape[1];
  const data = zeros(rows * cols);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      let sum = 0;
      for (let k = 0; k < inner; k += 1) sum += a.data[i * inner + k] * b.data[k * cols + j];
      data[i * cols + j] = sum;
    }
  }
  let out;
  out = new Tensor(data, [rows, cols], {
    requiresGrad: a.requiresGrad || b.requiresGrad,
    parents: [a, b],
    name: "matmul",
    backward: () => {
      for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
          const g = out.grad[i * cols + j];
          for (let k = 0; k < inner; k += 1) {
            if (a.requiresGrad) a.grad[i * inner + k] += g * b.data[k * cols + j];
            if (b.requiresGrad) b.grad[k * cols + j] += a.data[i * inner + k] * g;
          }
        }
      }
    },
  });
  return out;
}

export function transpose2D(a) {
  if (a.shape.length !== 2) throw new Error("transpose2D requires a matrix");
  const [rows, cols] = a.shape;
  const data = zeros(a.data.length);
  for (let i = 0; i < rows; i += 1) for (let j = 0; j < cols; j += 1) data[j * rows + i] = a.data[i * cols + j];
  let out;
  out = new Tensor(data, [cols, rows], {
    requiresGrad: a.requiresGrad,
    parents: [a],
    name: "transpose",
    backward: () => {
      if (!a.requiresGrad) return;
      for (let i = 0; i < rows; i += 1) for (let j = 0; j < cols; j += 1) a.grad[i * cols + j] += out.grad[j * rows + i];
    },
  });
  return out;
}

export function concatColumns(tensors) {
  if (!tensors.length || tensors.some((tensor) => tensor.shape.length !== 2 || tensor.shape[0] !== tensors[0].shape[0])) {
    throw new Error("concatColumns requires matrices with equal row count");
  }
  const rows = tensors[0].shape[0];
  const widths = tensors.map((tensor) => tensor.shape[1]);
  const cols = widths.reduce((sum, value) => sum + value, 0);
  const data = zeros(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    let offset = 0;
    tensors.forEach((tensor, index) => {
      for (let col = 0; col < widths[index]; col += 1) data[row * cols + offset + col] = tensor.data[row * widths[index] + col];
      offset += widths[index];
    });
  }
  let out;
  out = new Tensor(data, [rows, cols], {
    requiresGrad: tensors.some((tensor) => tensor.requiresGrad),
    parents: tensors,
    name: "concat",
    backward: () => {
      for (let row = 0; row < rows; row += 1) {
        let offset = 0;
        tensors.forEach((tensor, index) => {
          if (tensor.requiresGrad) for (let col = 0; col < widths[index]; col += 1) tensor.grad[row * widths[index] + col] += out.grad[row * cols + offset + col];
          offset += widths[index];
        });
      }
    },
  });
  return out;
}

export function embedding(table, ids) {
  if (table.shape.length !== 2) throw new Error("embedding table must be a matrix");
  const width = table.shape[1];
  const data = [];
  ids.forEach((id) => {
    if (!Number.isInteger(id) || id < 0 || id >= table.shape[0]) throw new Error(`invalid embedding id ${id}`);
    data.push(...table.data.slice(id * width, (id + 1) * width));
  });
  let out;
  out = new Tensor(data, [ids.length, width], {
    requiresGrad: table.requiresGrad,
    parents: [table],
    name: "embedding",
    backward: () => {
      if (!table.requiresGrad) return;
      ids.forEach((id, row) => {
        for (let col = 0; col < width; col += 1) table.grad[id * width + col] += out.grad[row * width + col];
      });
    },
  });
  return out;
}

export function layerNorm(x, gamma, beta, epsilon = 1e-5) {
  if (x.shape.length !== 2 || gamma.shape[0] !== x.shape[1] || beta.shape[0] !== x.shape[1]) throw new Error("layerNorm shape mismatch");
  const [rows, cols] = x.shape;
  const data = zeros(x.data.length);
  const normalized = zeros(x.data.length);
  const invStd = zeros(rows);
  for (let row = 0; row < rows; row += 1) {
    let mean = 0;
    for (let col = 0; col < cols; col += 1) mean += x.data[row * cols + col];
    mean /= cols;
    let variance = 0;
    for (let col = 0; col < cols; col += 1) variance += (x.data[row * cols + col] - mean) ** 2;
    variance /= cols;
    invStd[row] = 1 / Math.sqrt(variance + epsilon);
    for (let col = 0; col < cols; col += 1) {
      const index = row * cols + col;
      normalized[index] = (x.data[index] - mean) * invStd[row];
      data[index] = normalized[index] * gamma.data[col] + beta.data[col];
    }
  }
  let out;
  out = new Tensor(data, x.shape, {
    requiresGrad: x.requiresGrad || gamma.requiresGrad || beta.requiresGrad,
    parents: [x, gamma, beta],
    name: "layerNorm",
    backward: () => {
      for (let row = 0; row < rows; row += 1) {
        let sumDxHat = 0;
        let sumDxHatNorm = 0;
        for (let col = 0; col < cols; col += 1) {
          const index = row * cols + col;
          const dxHat = out.grad[index] * gamma.data[col];
          sumDxHat += dxHat;
          sumDxHatNorm += dxHat * normalized[index];
          if (gamma.requiresGrad) gamma.grad[col] += out.grad[index] * normalized[index];
          if (beta.requiresGrad) beta.grad[col] += out.grad[index];
        }
        if (x.requiresGrad) {
          for (let col = 0; col < cols; col += 1) {
            const index = row * cols + col;
            const dxHat = out.grad[index] * gamma.data[col];
            x.grad[index] += (invStd[row] / cols) * (cols * dxHat - sumDxHat - normalized[index] * sumDxHatNorm);
          }
        }
      }
    },
  });
  return out;
}

export function gelu(x) {
  const c = Math.sqrt(2 / Math.PI);
  const data = x.data.map((value) => 0.5 * value * (1 + Math.tanh(c * (value + 0.044715 * value ** 3))));
  let out;
  out = new Tensor(data, x.shape, {
    requiresGrad: x.requiresGrad,
    parents: [x],
    name: "gelu",
    backward: () => {
      if (!x.requiresGrad) return;
      x.data.forEach((value, index) => {
        const inner = c * (value + 0.044715 * value ** 3);
        const tanhInner = Math.tanh(inner);
        const derivative = 0.5 * (1 + tanhInner) + 0.5 * value * (1 - tanhInner ** 2) * c * (1 + 3 * 0.044715 * value ** 2);
        x.grad[index] += out.grad[index] * derivative;
      });
    },
  });
  return out;
}

export function maskedSoftmaxRows(scores, causal = true) {
  if (scores.shape.length !== 2) throw new Error("maskedSoftmaxRows requires a matrix");
  const [rows, cols] = scores.shape;
  const data = zeros(scores.data.length);
  const maskedScores = [...scores.data];
  for (let row = 0; row < rows; row += 1) {
    const allowed = [];
    for (let col = 0; col < cols; col += 1) {
      const index = row * cols + col;
      if (causal && col > row) maskedScores[index] = -Infinity;
      else allowed.push(index);
    }
    const max = Math.max(...allowed.map((index) => scores.data[index]));
    let denominator = 0;
    allowed.forEach((index) => {
      data[index] = Math.exp(scores.data[index] - max);
      denominator += data[index];
    });
    allowed.forEach((index) => { data[index] /= denominator; });
  }
  let out;
  out = new Tensor(data, scores.shape, {
    requiresGrad: scores.requiresGrad,
    parents: [scores],
    name: "maskedSoftmax",
    backward: () => {
      if (!scores.requiresGrad) return;
      for (let row = 0; row < rows; row += 1) {
        let dot = 0;
        for (let col = 0; col < cols; col += 1) dot += out.grad[row * cols + col] * out.data[row * cols + col];
        for (let col = 0; col < cols; col += 1) {
          const index = row * cols + col;
          if (!causal || col <= row) scores.grad[index] += out.data[index] * (out.grad[index] - dot);
        }
      }
    },
  });
  out.maskedScores = maskedScores;
  return out;
}

export function crossEntropy(logits, targets) {
  if (logits.shape.length !== 2 || targets.length !== logits.shape[0]) throw new Error("crossEntropy shape mismatch");
  const [rows, cols] = logits.shape;
  const probabilities = zeros(logits.data.length);
  let loss = 0;
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    const rowValues = logits.data.slice(offset, offset + cols);
    const max = Math.max(...rowValues);
    const exps = rowValues.map((value) => Math.exp(value - max));
    const denominator = exps.reduce((sum, value) => sum + value, 0);
    exps.forEach((value, col) => { probabilities[offset + col] = value / denominator; });
    if (!Number.isInteger(targets[row]) || targets[row] < 0 || targets[row] >= cols) throw new Error(`invalid target ${targets[row]}`);
    loss -= Math.log(Math.max(probabilities[offset + targets[row]], 1e-300));
  }
  loss /= rows;
  let out;
  out = new Tensor([loss], [1], {
    requiresGrad: logits.requiresGrad,
    parents: [logits],
    name: "crossEntropy",
    backward: () => {
      if (!logits.requiresGrad) return;
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const index = row * cols + col;
          logits.grad[index] += out.grad[0] * (probabilities[index] - (col === targets[row] ? 1 : 0)) / rows;
        }
      }
    },
  });
  out.probabilities = probabilities;
  out.lossByPosition = targets.map((target, row) => -Math.log(Math.max(probabilities[row * cols + target], 1e-300)));
  return out;
}
