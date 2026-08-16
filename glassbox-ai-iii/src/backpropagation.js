import { cloneNetwork, PARAMETER_SPECS, getParameterValue, setParameterValue } from './neural-network.js';
import { crossEntropyLoss, forwardPass } from './forward-pass.js';

export function oneHot(targetIndex, size = 3) {
  if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= size) {
    throw new Error('one-hotへ変換するクラス番号が範囲外です。');
  }
  return Array.from({ length: size }, (_, index) => (index === targetIndex ? 1 : 0));
}

export function computeGradients(network, inputs, targetIndex, forward = forwardPass(network, inputs)) {
  const target = oneHot(targetIndex, forward.probabilities.length);
  const outputErrors = forward.probabilities.map((probability, output) => probability - target[output]);

  const weightsHO = outputErrors.map((error) =>
    forward.hiddenActivations.map((activation) => error * activation),
  );
  const biasO = [...outputErrors];

  const propagatedHiddenErrors = forward.hiddenActivations.map((_, hidden) =>
    outputErrors.reduce(
      (sum, outputError, output) => sum + network.weightsHO[output][hidden] * outputError,
      0,
    ),
  );
  const tanhDerivatives = forward.hiddenActivations.map((activation) => 1 - activation * activation);
  const hiddenDeltas = propagatedHiddenErrors.map(
    (error, hidden) => error * tanhDerivatives[hidden],
  );

  const weightsIH = hiddenDeltas.map((delta) => inputs.map((input) => delta * input));
  const biasH = [...hiddenDeltas];

  return {
    target,
    loss: crossEntropyLoss(forward.probabilities, targetIndex),
    outputErrors,
    weightsHO,
    biasO,
    propagatedHiddenErrors,
    tanhDerivatives,
    hiddenDeltas,
    weightsIH,
    biasH,
  };
}

export function getGradientValue(gradients, spec) {
  if (spec.layer === 'IH') return gradients.weightsIH[spec.row][spec.column];
  if (spec.layer === 'BH') return gradients.biasH[spec.row];
  if (spec.layer === 'HO') return gradients.weightsHO[spec.row][spec.column];
  if (spec.layer === 'BO') return gradients.biasO[spec.row];
  throw new Error(`未知の勾配層です: ${spec.layer}`);
}

export function applyGradients(network, gradients, learningRate) {
  if (!Number.isFinite(learningRate) || learningRate <= 0) {
    throw new Error('学習率は0より大きい有限数である必要があります。');
  }
  const updated = cloneNetwork(network);
  const changes = {};

  for (const spec of PARAMETER_SPECS) {
    const before = getParameterValue(updated, spec);
    const gradient = getGradientValue(gradients, spec);
    const update = -learningRate * gradient;
    const after = before + update;
    setParameterValue(updated, spec, after);
    changes[spec.name] = { before, gradient, learningRate, update, after };
  }
  updated.learningCount += 1;

  return { network: updated, changes };
}

export function compareTraining(beforeForward, afterForward, targetIndex, changes = {}) {
  const beforeLoss = crossEntropyLoss(beforeForward.probabilities, targetIndex);
  const afterLoss = crossEntropyLoss(afterForward.probabilities, targetIndex);
  const changeValues = Object.values(changes).map((change) => Math.abs(change.update));
  const weightChangeValues = Object.entries(changes)
    .filter(([name]) => name.startsWith('w_'))
    .map(([, change]) => Math.abs(change.update));

  return {
    targetIndex,
    beforeProbabilities: [...beforeForward.probabilities],
    afterProbabilities: [...afterForward.probabilities],
    beforeLoss,
    afterLoss,
    targetProbabilityBefore: beforeForward.probabilities[targetIndex],
    targetProbabilityAfter: afterForward.probabilities[targetIndex],
    targetProbabilityChange: afterForward.probabilities[targetIndex] - beforeForward.probabilities[targetIndex],
    changedParameterCount: changeValues.filter((value) => value > 0).length,
    changedWeightCount: weightChangeValues.filter((value) => value > 0).length,
    maximumAbsoluteChange: changeValues.length ? Math.max(...changeValues) : 0,
    lossDecreased: afterLoss < beforeLoss,
  };
}

export function trainOne(network, inputs, targetIndex, learningRate) {
  const beforeForward = forwardPass(network, inputs);
  const gradients = computeGradients(network, inputs, targetIndex, beforeForward);
  const applied = applyGradients(network, gradients, learningRate);
  const afterForward = forwardPass(applied.network, inputs);
  const comparison = compareTraining(beforeForward, afterForward, targetIndex, applied.changes);
  return { beforeForward, gradients, ...applied, afterForward, comparison };
}
