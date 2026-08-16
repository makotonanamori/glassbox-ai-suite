import test from 'node:test';
import assert from 'node:assert/strict';
import { crossEntropyLoss, forwardPass } from '../src/forward-pass.js';

const knownNetwork = {
  structure: { input: 5, hidden: 4, output: 3 },
  seed: 'manual',
  learningCount: 0,
  weightsIH: [
    [0.2, -0.4, 0.1, 0.3, -0.2],
    [0, 0, 0, 0, 0],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [-0.2, 0.3, 0, -0.1, 0.4],
  ],
  biasH: [0.1, -0.2, 0, 0.05],
  weightsHO: [
    [0.3, -0.1, 0.2, 0.4],
    [-0.2, 0.5, -0.3, 0.1],
    [0.1, 0.2, 0.4, -0.5],
  ],
  biasO: [0.05, -0.05, 0.1],
};

const inputs = [1, -0.5, 0, 0.25, -1];

test('順伝播の中間和・tanh・logitが手計算と一致する', () => {
  const result = forwardPass(knownNetwork, inputs);
  const expectedHiddenPre = [0.775, -0.2, -0.025, -0.725];
  expectedHiddenPre.forEach((value, index) => {
    assert.ok(Math.abs(result.hiddenPreActivations[index] - value) < 1e-15);
  });
  const expectedHidden = expectedHiddenPre.map(Math.tanh);
  expectedHidden.forEach((value, index) => {
    assert.ok(Math.abs(result.hiddenActivations[index] - value) < 1e-15);
  });
  const expectedLogit0 =
    expectedHidden[0] * 0.3 + expectedHidden[1] * -0.1 + expectedHidden[2] * 0.2 + expectedHidden[3] * 0.4 + 0.05;
  assert.ok(Math.abs(result.logits[0] - expectedLogit0) < 1e-15);
});

test('クロスエントロピー損失は正解確率の負の対数と一致する', () => {
  const result = forwardPass(knownNetwork, inputs);
  assert.equal(crossEntropyLoss(result.probabilities, 1), -Math.log(result.probabilities[1]));
});
