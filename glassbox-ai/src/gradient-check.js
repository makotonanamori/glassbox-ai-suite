import {
  PARAMETER_SPECS,
  cloneNetwork,
  getParameterValue,
  setParameterValue,
} from './neural-network.js';
import { crossEntropyLoss, forwardPass } from './forward-pass.js';
import { computeGradients, getGradientValue } from './backpropagation.js';

export function gradientCheck(network, inputs, targetIndex, options = {}) {
  const epsilon = options.epsilon ?? 1e-5;
  const tolerance = options.tolerance ?? 1e-5;
  const analytic = computeGradients(network, inputs, targetIndex);
  const results = [];

  for (const spec of PARAMETER_SPECS) {
    const plusNetwork = cloneNetwork(network);
    const minusNetwork = cloneNetwork(network);
    const original = getParameterValue(network, spec);
    setParameterValue(plusNetwork, spec, original + epsilon);
    setParameterValue(minusNetwork, spec, original - epsilon);

    const plusLoss = crossEntropyLoss(forwardPass(plusNetwork, inputs).probabilities, targetIndex);
    const minusLoss = crossEntropyLoss(forwardPass(minusNetwork, inputs).probabilities, targetIndex);
    const numerical = (plusLoss - minusLoss) / (2 * epsilon);
    const analyticValue = getGradientValue(analytic, spec);
    const absoluteError = Math.abs(numerical - analyticValue);
    const normalizedError = absoluteError / Math.max(1, Math.abs(numerical), Math.abs(analyticValue));

    results.push({
      parameter: spec.name,
      analytic: analyticValue,
      numerical,
      absoluteError,
      normalizedError,
      passed: normalizedError < tolerance,
    });
  }

  const maximumNormalizedError = Math.max(...results.map((result) => result.normalizedError));
  const maximumAbsoluteError = Math.max(...results.map((result) => result.absoluteError));
  return {
    epsilon,
    tolerance,
    checkedParameters: results.length,
    maximumNormalizedError,
    maximumAbsoluteError,
    passed: results.every((result) => result.passed),
    results,
  };
}
