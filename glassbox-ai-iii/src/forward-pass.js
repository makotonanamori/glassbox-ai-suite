import { NETWORK_SHAPE } from './neural-network.js';

export function stableSoftmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) {
    throw new Error('softmaxには1個以上のlogitが必要です。');
  }
  if (!logits.every(Number.isFinite)) {
    throw new Error('softmaxのlogitは有限数である必要があります。');
  }

  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / denominator);
}

export function crossEntropyLoss(probabilities, targetIndex) {
  if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= probabilities.length) {
    throw new Error('正解クラスの番号が範囲外です。');
  }
  const probability = Math.max(probabilities[targetIndex], Number.EPSILON);
  return -Math.log(probability);
}

export function argmax(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error('argmaxには1個以上の値が必要です。');
  }
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}

export function forwardPass(network, inputs) {
  if (!Array.isArray(inputs) || inputs.length !== NETWORK_SHAPE.input || !inputs.every(Number.isFinite)) {
    throw new Error(`入力は有限数${NETWORK_SHAPE.input}個である必要があります。`);
  }

  const hiddenProducts = network.weightsIH.map((weights, hidden) =>
    weights.map((weight, input) => inputs[input] * weight),
  );
  const hiddenPreActivations = hiddenProducts.map(
    (products, hidden) => products.reduce((sum, value) => sum + value, 0) + network.biasH[hidden],
  );
  const hiddenActivations = hiddenPreActivations.map((value) => Math.tanh(value));

  const outputProducts = network.weightsHO.map((weights, output) =>
    weights.map((weight, hidden) => hiddenActivations[hidden] * weight),
  );
  const logits = outputProducts.map(
    (products, output) => products.reduce((sum, value) => sum + value, 0) + network.biasO[output],
  );
  const probabilities = stableSoftmax(logits);

  return {
    inputs: [...inputs],
    hiddenProducts,
    hiddenPreActivations,
    hiddenActivations,
    outputProducts,
    logits,
    probabilities,
    selectedIndex: argmax(probabilities),
  };
}
