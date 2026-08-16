import {
  PARAMETER_SPECS,
  cloneNetwork,
  getParameterValue,
  setParameterValue,
} from './neural-network.js';
import { oneHot, applyGradients, getGradientValue } from './backpropagation.js';
import { forwardPass, stableSoftmax, argmax } from './forward-pass.js';

function validateTemperature(temperature) {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error('方策温度は0より大きい有限数である必要があります。');
  }
}

export function temperaturePolicy(logits, temperature = 1) {
  validateTemperature(temperature);
  const scaledLogits = logits.map((logit) => logit / temperature);
  const probabilities = stableSoftmax(scaledLogits);
  const entropy = -probabilities.reduce(
    (sum, probability) => sum + probability * Math.log(Math.max(probability, Number.EPSILON)),
    0,
  );
  return {
    temperature,
    scaledLogits,
    probabilities,
    entropy,
    greedyActionIndex: argmax(probabilities),
  };
}

export function samplePolicy(probabilities, randomValue) {
  if (!Array.isArray(probabilities) || probabilities.length === 0) {
    throw new Error('行動サンプリングには1個以上の確率が必要です。');
  }
  if (!probabilities.every((value) => Number.isFinite(value) && value >= 0)) {
    throw new Error('行動確率は0以上の有限数である必要があります。');
  }
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 1e-10) throw new Error('行動確率の合計は1である必要があります。');
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new Error('サンプリング乱数は0以上1未満である必要があります。');
  }

  let lower = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const upper = index === probabilities.length - 1 ? 1 : lower + probabilities[index];
    if (randomValue < upper) return { actionIndex: index, lower, upper, randomValue };
    lower = upper;
  }
  return { actionIndex: probabilities.length - 1, lower, upper: 1, randomValue };
}

export function discountedReturns(rewards, gamma) {
  if (!Array.isArray(rewards) || !rewards.every(Number.isFinite)) {
    throw new Error('報酬列は有限数の配列である必要があります。');
  }
  if (!Number.isFinite(gamma) || gamma < 0 || gamma > 1) {
    throw new Error('割引率γは0以上1以下である必要があります。');
  }
  const returns = Array(rewards.length).fill(0);
  let future = 0;
  for (let index = rewards.length - 1; index >= 0; index -= 1) {
    future = rewards[index] + gamma * future;
    returns[index] = future;
  }
  return returns;
}

export function createZeroGradients() {
  return {
    weightsIH: Array.from({ length: 4 }, () => Array(5).fill(0)),
    biasH: Array(4).fill(0),
    weightsHO: Array.from({ length: 3 }, () => Array(4).fill(0)),
    biasO: Array(3).fill(0),
  };
}

export function addGradients(target, source, scale = 1) {
  for (const spec of PARAMETER_SPECS) {
    const value = getGradientValue(target, spec) + getGradientValue(source, spec) * scale;
    if (spec.layer === 'IH') target.weightsIH[spec.row][spec.column] = value;
    else if (spec.layer === 'BH') target.biasH[spec.row] = value;
    else if (spec.layer === 'HO') target.weightsHO[spec.row][spec.column] = value;
    else target.biasO[spec.row] = value;
  }
  return target;
}

export function computePolicyGradient(
  network,
  inputs,
  actionIndex,
  returnToGo,
  temperature = 1,
  forward = forwardPass(network, inputs),
) {
  if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= forward.logits.length) {
    throw new Error('方策勾配の行動番号が範囲外です。');
  }
  if (!Number.isFinite(returnToGo)) throw new Error('リターンは有限数である必要があります。');
  const policy = temperaturePolicy(forward.logits, temperature);
  const target = oneHot(actionIndex, policy.probabilities.length);
  const outputDeltas = policy.probabilities.map(
    (probability, output) => (returnToGo / temperature) * (probability - target[output]),
  );
  const weightsHO = outputDeltas.map((delta) =>
    forward.hiddenActivations.map((activation) => delta * activation),
  );
  const biasO = [...outputDeltas];
  const propagatedHiddenErrors = forward.hiddenActivations.map((_, hidden) =>
    outputDeltas.reduce(
      (sum, outputDelta, output) => sum + network.weightsHO[output][hidden] * outputDelta,
      0,
    ),
  );
  const tanhDerivatives = forward.hiddenActivations.map((activation) => 1 - activation * activation);
  const hiddenDeltas = propagatedHiddenErrors.map(
    (error, hidden) => error * tanhDerivatives[hidden],
  );
  const weightsIH = hiddenDeltas.map((delta) => inputs.map((input) => delta * input));
  const biasH = [...hiddenDeltas];
  const chosenProbability = Math.max(policy.probabilities[actionIndex], Number.EPSILON);
  const policyLoss = -returnToGo * Math.log(chosenProbability);
  return {
    actionIndex,
    returnToGo,
    temperature,
    policyLoss,
    forward,
    policy,
    target,
    outputDeltas,
    weightsHO,
    biasO,
    propagatedHiddenErrors,
    tanhDerivatives,
    hiddenDeltas,
    weightsIH,
    biasH,
  };
}

export function aggregatePolicyGradients(network, experiences, returns, temperature = 1) {
  if (!experiences.length || experiences.length !== returns.length) {
    throw new Error('経験とリターンは同じ1件以上の長さである必要があります。');
  }
  const individual = experiences.map((experience, index) =>
    computePolicyGradient(
      network,
      experience.inputs,
      experience.actionIndex,
      returns[index],
      temperature,
      experience.forward,
    ),
  );
  const average = createZeroGradients();
  const scale = 1 / experiences.length;
  individual.forEach((gradient) => addGradients(average, gradient, scale));
  return { individual, average };
}

export function applyPolicyGradient(network, averageGradient, learningRate) {
  return applyGradients(network, averageGradient, learningRate);
}

function policyLossFor(network, inputs, actionIndex, returnToGo, temperature) {
  const forward = forwardPass(network, inputs);
  const policy = temperaturePolicy(forward.logits, temperature);
  return -returnToGo * Math.log(Math.max(policy.probabilities[actionIndex], Number.EPSILON));
}

export function policyGradientCheck(network, inputs, actionIndex, returnToGo, options = {}) {
  const epsilon = options.epsilon ?? 1e-5;
  const tolerance = options.tolerance ?? 1e-5;
  const temperature = options.temperature ?? 1;
  const analytic = computePolicyGradient(network, inputs, actionIndex, returnToGo, temperature);
  const results = PARAMETER_SPECS.map((spec) => {
    const plus = cloneNetwork(network);
    const minus = cloneNetwork(network);
    const original = getParameterValue(network, spec);
    setParameterValue(plus, spec, original + epsilon);
    setParameterValue(minus, spec, original - epsilon);
    const numerical = (
      policyLossFor(plus, inputs, actionIndex, returnToGo, temperature) -
      policyLossFor(minus, inputs, actionIndex, returnToGo, temperature)
    ) / (2 * epsilon);
    const analyticValue = getGradientValue(analytic, spec);
    const absoluteError = Math.abs(numerical - analyticValue);
    const normalizedError = absoluteError / Math.max(1, Math.abs(numerical), Math.abs(analyticValue));
    return {
      parameter: spec.name,
      analytic: analyticValue,
      numerical,
      absoluteError,
      normalizedError,
      passed: normalizedError < tolerance,
    };
  });
  return {
    epsilon,
    tolerance,
    temperature,
    checkedParameters: results.length,
    maximumNormalizedError: Math.max(...results.map((result) => result.normalizedError)),
    maximumAbsoluteError: Math.max(...results.map((result) => result.absoluteError)),
    passed: results.every((result) => result.passed),
    results,
  };
}
