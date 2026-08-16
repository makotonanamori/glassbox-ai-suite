import { assertFinite } from "../utils/math.js";

export function globalGradientNorm(parameters) {
  return Math.sqrt(parameters.reduce((sum, { tensor }) => sum + tensor.grad.reduce((inner, value) => inner + value * value, 0), 0));
}

export function globalParameterNorm(parameters) {
  return Math.sqrt(parameters.reduce((sum, { tensor }) => sum + tensor.data.reduce((inner, value) => inner + value * value, 0), 0));
}

export function sgdStep(parameters, learningRate = 0.03, clipNorm = 1) {
  if (!(learningRate > 0) || !Number.isFinite(learningRate)) throw new Error("learningRate must be finite and positive");
  if (!(clipNorm > 0) || !Number.isFinite(clipNorm)) throw new Error("clipNorm must be finite and positive");
  const rawGradientNorm = globalGradientNorm(parameters);
  const scale = rawGradientNorm > clipNorm ? clipNorm / rawGradientNorm : 1;
  const updates = {};
  parameters.forEach(({ name, tensor }) => {
    assertFinite(tensor.grad, `${name}.grad`);
    const oldValue = [...tensor.data];
    const gradient = [...tensor.grad];
    const update = gradient.map((value) => -learningRate * value * scale);
    tensor.data = tensor.data.map((value, index) => value + update[index]);
    assertFinite(tensor.data, name);
    updates[name] = { shape: [...tensor.shape], oldValue, gradient, update, newValue: [...tensor.data] };
  });
  return { rawGradientNorm, clippedGradientNorm: rawGradientNorm * scale, clipScale: scale, updates };
}
