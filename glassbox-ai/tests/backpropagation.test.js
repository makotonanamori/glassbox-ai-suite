import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, getParameterValue, PARAMETER_SPECS } from '../src/neural-network.js';
import { forwardPass } from '../src/forward-pass.js';
import { applyGradients, computeGradients } from '../src/backpropagation.js';

const inputs = [0.7, -0.5, 0.3, 0, 0.8];

test('出力層と入力層の解析的勾配が連鎖律の式と一致する', () => {
  const network = createNetwork('backprop-test');
  const forward = forwardPass(network, inputs);
  const gradients = computeGradients(network, inputs, 1, forward);
  const outputError = forward.probabilities[1] - 1;
  assert.ok(Math.abs(gradients.outputErrors[1] - outputError) < 1e-15);
  assert.ok(Math.abs(gradients.weightsHO[1][2] - outputError * forward.hiddenActivations[2]) < 1e-15);

  const propagated = gradients.outputErrors.reduce(
    (sum, error, output) => sum + network.weightsHO[output][0] * error,
    0,
  );
  const delta = propagated * (1 - forward.hiddenActivations[0] ** 2);
  assert.ok(Math.abs(gradients.weightsIH[0][4] - delta * inputs[4]) < 1e-15);
  assert.ok(Math.abs(gradients.biasH[0] - delta) < 1e-15);
});

test('重み更新式 current - learningRate * gradient が全39パラメータへ適用される', () => {
  const network = createNetwork('update-test');
  const gradients = computeGradients(network, inputs, 2);
  const learningRate = 0.1;
  const { network: updated, changes } = applyGradients(network, gradients, learningRate);
  assert.equal(Object.keys(changes).length, 39);
  for (const spec of PARAMETER_SPECS) {
    const expected = getParameterValue(network, spec) - learningRate * changes[spec.name].gradient;
    assert.ok(Math.abs(getParameterValue(updated, spec) - expected) < 1e-15);
  }
  assert.equal(updated.learningCount, network.learningCount + 1);
});
