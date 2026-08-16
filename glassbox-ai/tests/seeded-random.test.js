import test from 'node:test';
import assert from 'node:assert/strict';
import { createSeededRandom } from '../src/seeded-random.js';
import { PARAMETER_SPECS, createNetwork, getParameterValue } from '../src/neural-network.js';

test('同じシードから同じ疑似乱数列と39パラメータが生成される', () => {
  const firstRandom = createSeededRandom('reproducible-seed');
  const secondRandom = createSeededRandom('reproducible-seed');
  const firstSequence = Array.from({ length: 10 }, () => firstRandom.next());
  const secondSequence = Array.from({ length: 10 }, () => secondRandom.next());
  assert.deepEqual(firstSequence, secondSequence);

  const firstNetwork = createNetwork('reproducible-seed');
  const secondNetwork = createNetwork('reproducible-seed');
  assert.equal(PARAMETER_SPECS.length, 39);
  assert.deepEqual(
    PARAMETER_SPECS.map((spec) => getParameterValue(firstNetwork, spec)),
    PARAMETER_SPECS.map((spec) => getParameterValue(secondNetwork, spec)),
  );
});

test('異なるシードは異なる初期値を生成する', () => {
  assert.notDeepEqual(createNetwork('seed-a').weightsIH, createNetwork('seed-b').weightsIH);
});
