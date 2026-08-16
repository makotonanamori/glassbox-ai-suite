import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { forwardPass } from '../src/forward-pass.js';
import {
  applyPolicyGradient,
  computePolicyGradient,
  discountedReturns,
  policyGradientCheck,
  samplePolicy,
  temperaturePolicy,
} from '../src/policy-gradient.js';

test('温度付きsoftmaxは合計1を保ち、高温ほど一様分布へ近づく', () => {
  const logits = [2, 0.5, -1];
  const cold = temperaturePolicy(logits, 0.5);
  const hot = temperaturePolicy(logits, 4);
  assert.ok(Math.abs(cold.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(Math.abs(hot.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  const coldSpread = Math.max(...cold.probabilities) - Math.min(...cold.probabilities);
  const hotSpread = Math.max(...hot.probabilities) - Math.min(...hot.probabilities);
  assert.ok(hotSpread < coldSpread);
  assert.ok(hot.entropy > cold.entropy);
});

test('累積確率区間から行動と使用区間を正しく選ぶ', () => {
  const probabilities = [0.2, 0.3, 0.5];
  assert.deepEqual(samplePolicy(probabilities, 0.1), {
    actionIndex: 0,
    lower: 0,
    upper: 0.2,
    randomValue: 0.1,
  });
  assert.equal(samplePolicy(probabilities, 0.2).actionIndex, 1);
  assert.equal(samplePolicy(probabilities, 0.499999).actionIndex, 1);
  assert.equal(samplePolicy(probabilities, 0.5).actionIndex, 2);
});

test('割引リターンを終端から逆順に計算する', () => {
  const returns = discountedReturns([1, 2, 3], 0.5);
  assert.deepEqual(returns, [2.75, 3.5, 3]);
});

test('REINFORCEの全39解析勾配が中央差分数値勾配と一致する', () => {
  const result = policyGradientCheck(
    createNetwork('policy-gradient-check'),
    [0.7, -0.5, 0.3, 0, 0.8],
    2,
    0.73,
    { epsilon: 1e-5, tolerance: 1e-5, temperature: 1.4 },
  );
  assert.equal(result.checkedParameters, 39);
  assert.equal(result.passed, true);
  assert.ok(result.maximumNormalizedError < 1e-5);
});

test('正のリターンは選択行動確率を上げ、負のリターンは下げる', () => {
  const network = createNetwork('policy-return-direction');
  const inputs = [-0.9, 0.4, -0.6, 0.2, 0.1];
  const actionIndex = 1;
  const temperature = 1.1;
  const before = temperaturePolicy(forwardPass(network, inputs).logits, temperature).probabilities[actionIndex];

  const positiveGradient = computePolicyGradient(network, inputs, actionIndex, 1, temperature);
  const positiveNetwork = applyPolicyGradient(network, positiveGradient, 0.05).network;
  const positive = temperaturePolicy(
    forwardPass(positiveNetwork, inputs).logits,
    temperature,
  ).probabilities[actionIndex];

  const negativeGradient = computePolicyGradient(network, inputs, actionIndex, -1, temperature);
  const negativeNetwork = applyPolicyGradient(network, negativeGradient, 0.05).network;
  const negative = temperaturePolicy(
    forwardPass(negativeNetwork, inputs).logits,
    temperature,
  ).probabilities[actionIndex];

  assert.ok(positive > before);
  assert.ok(negative < before);
});
