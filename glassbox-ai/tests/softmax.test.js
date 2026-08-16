import test from 'node:test';
import assert from 'node:assert/strict';
import { stableSoftmax } from '../src/forward-pass.js';

test('softmaxの出力合計は1になる', () => {
  const probabilities = stableSoftmax([1.2, -0.8, 0.35]);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-15);
  assert.ok(probabilities.every((value) => value > 0 && value < 1));
});

test('softmaxは極端に大きいlogitでもInfinityやNaNを生じない', () => {
  const probabilities = stableSoftmax([10000, 9999, -10000]);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-15);
  assert.ok(probabilities[0] > probabilities[1]);
});
