import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { trainOne } from '../src/backpropagation.js';
import {
  GRID_ACTIONS,
  applyGridAction,
  cloneGridWorld,
  createGridWorld,
  getTeacherAction,
  senseGridWorld,
} from '../src/grid-world.js';

test('同じ世界シードから同じ初期位置・向き・餌が再現される', () => {
  const first = createGridWorld('world-reproducible');
  const second = createGridWorld('world-reproducible');
  assert.deepEqual(first, second);
});

test('5センサーは有限数で-1から+1の範囲に収まる', () => {
  const sensed = senseGridWorld(createGridWorld('sensor-range'));
  assert.equal(sensed.values.length, 5);
  assert.ok(sensed.values.every((value) => Number.isFinite(value) && value >= -1 && value <= 1));
});

test('左折と右折は位置を変えず向きを90度変更する', () => {
  const world = createGridWorld('turn-test');
  const left = applyGridAction(world, 1);
  const right = applyGridAction(world, 2);
  assert.equal(left.agent.row, world.agent.row);
  assert.equal(left.agent.column, world.agent.column);
  assert.equal(left.agent.direction, (world.agent.direction + 3) % 4);
  assert.equal(right.agent.direction, (world.agent.direction + 1) % 4);
  assert.equal(left.lastTransition.reward, -0.02);
});

test('壁への前進は位置を変えず衝突を記録する', () => {
  const world = createGridWorld('collision-test');
  world.agent = { row: 1, column: 2, direction: 1 };
  const next = applyGridAction(world, 0);
  assert.deepEqual(next.agent, world.agent);
  assert.equal(next.counters.collisions, 1);
  assert.equal(next.lastTransition.reward, -0.25);
  assert.equal(next.lastTransition.moved, false);
});

test('危険セルへの前進は移動と危険接触を記録する', () => {
  const world = createGridWorld('danger-test');
  world.agent = { row: 2, column: 4, direction: 1 };
  const next = applyGridAction(world, 0);
  assert.equal(next.agent.row, 2);
  assert.equal(next.agent.column, 5);
  assert.equal(next.counters.dangerHits, 1);
  assert.equal(next.lastTransition.reward, -1);
});

test('餌への前進は取得を記録し、次の餌を再現可能に配置する', () => {
  const first = createGridWorld('food-test');
  first.agent = { row: 1, column: 4, direction: 1 };
  first.food = { row: 1, column: 5 };
  const firstResult = applyGridAction(first, 0);

  const second = cloneGridWorld(first);
  const secondResult = applyGridAction(second, 0);
  assert.equal(firstResult.counters.foods, 1);
  assert.equal(firstResult.lastTransition.reward, 1);
  assert.deepEqual(firstResult.food, secondResult.food);
  assert.notDeepEqual(firstResult.food, firstResult.agent);
});

test('教師は壁と危険を避け、決定的な行動列で餌へ到達する', () => {
  for (const seed of ['teacher-a', 'teacher-b', 'teacher-c', 'teacher-d']) {
    let world = createGridWorld(seed);
    const firstPlan = getTeacherAction(world);
    assert.equal(firstPlan.safePathFound, true);
    assert.ok(firstPlan.pathLength > 0);
    assert.deepEqual(firstPlan, getTeacherAction(world));

    for (let step = 0; step < 80 && world.counters.foods === 0; step += 1) {
      const teacher = getTeacherAction(world);
      assert.notEqual(teacher.actionIndex, null);
      world = applyGridAction(world, teacher.actionIndex);
      assert.equal(world.counters.collisions, 0);
      assert.equal(world.counters.dangerHits, 0);
    }
    assert.equal(world.counters.foods, 1, seed);
  }
});

test('教師ラベルによる一回のSGDで同じ観測の教師行動確率が上がる', () => {
  const world = createGridWorld('teacher-learning');
  const sensors = senseGridWorld(world).values;
  const teacher = getTeacherAction(world);
  assert.ok(GRID_ACTIONS[teacher.actionIndex]);
  const result = trainOne(createNetwork('teacher-network'), sensors, teacher.actionIndex, 0.1);
  assert.ok(
    result.afterForward.probabilities[teacher.actionIndex] >
      result.beforeForward.probabilities[teacher.actionIndex],
  );
  assert.ok(result.comparison.afterLoss < result.comparison.beforeLoss);
});
