import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { gradientCheck } from '../src/gradient-check.js';

test('全39パラメータの解析的勾配と中央差分の数値勾配が一致する', () => {
  const result = gradientCheck(
    createNetwork('gradient-check'),
    [-0.9, 0.4, -0.6, 0.2, 0.1],
    2,
    { epsilon: 1e-5, tolerance: 1e-5 },
  );
  assert.equal(result.checkedParameters, 39);
  assert.equal(result.passed, true);
  assert.ok(result.maximumNormalizedError < 1e-5);
});
