export function sizeOf(shape) {
  return shape.reduce((product, value) => product * value, 1);
}

export function assertShape(data, shape, label = "tensor") {
  if (!Array.isArray(shape) || shape.length === 0 || shape.some((n) => !Number.isInteger(n) || n <= 0)) {
    throw new Error(`${label}: invalid shape ${JSON.stringify(shape)}`);
  }
  if (data.length !== sizeOf(shape)) {
    throw new Error(`${label}: data length ${data.length} does not match shape [${shape}]`);
  }
}

export function assertFinite(values, label = "values") {
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) throw new Error(`${label}: non-finite value at index ${i}`);
  }
}

export function stableSoftmax(values, temperature = 1) {
  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error("temperature must be finite and positive");
  const scaled = values.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

export function stats(values) {
  if (!values.length) return { count: 0, mean: 0, std: 0, min: 0, max: 0, norm: 0 };
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return {
    count: values.length,
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values),
    norm: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)),
  };
}

export function maxAbs(values) {
  return values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
}

export function relativeError(a, b) {
  return Math.abs(a - b) / Math.max(1e-12, Math.abs(a) + Math.abs(b));
}
