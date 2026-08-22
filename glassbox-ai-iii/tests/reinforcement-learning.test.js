import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { createGridWorld } from '../src/grid-world.js';
import {
  REWARD_PROFILES,
  applyActionWithRewardProfile,
  diagnoseEpisode,
} from '../src/reward-profiles.js';
import {
  ReinforcementStepEngine,
  buildReinforcementTimeline,
} from '../src/reinforcement-learning.js';
import {
  advanceToReinforcementVisualBoundary,
  completePolicyReference,
  continuousEpisodeDelay,
  createPolicyReference,
} from '../src/continuous-episode-runner.js';

test('同じ観測で学習前後の方策を再計算して比較する', () => {
  const network = createNetwork('policy-reference-network');
  const world = createGridWorld('policy-reference-world');
  const built = buildReinforcementTimeline(network, world, {
    randomSeed: 'policy-reference-random',
    maxSteps: 6,
    learningRate: 0.05,
  });
  const firstExperience = built.experiences[0];
  const reference = createPolicyReference(
    network,
    firstExperience.beforeWorld,
    firstExperience.inputs,
    built.config.temperature,
  );
  const completed = completePolicyReference(reference, built.finalNetwork);

  assert.notEqual(completed.world, firstExperience.beforeWorld);
  assert.deepEqual(completed.inputs, firstExperience.inputs);
  assert.ok(Math.abs(completed.before.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(Math.abs(completed.after.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.deepEqual(
    completed.deltas,
    completed.after.map((value, index) => value - completed.before[index]),
  );
  assert.ok(completed.deltas.some((value) => Math.abs(value) > 1e-12));
});

test('4報酬プリセットが実際の遷移種別へ異なる値を適用する', () => {
  assert.equal(Object.keys(REWARD_PROFILES).length, 4);
  const turnWorld = createGridWorld('reward-turn');
  assert.equal(applyActionWithRewardProfile(turnWorld, 1, 'balanced').lastTransition.reward, -0.02);
  assert.equal(applyActionWithRewardProfile(turnWorld, 1, 'sparse').lastTransition.reward, 0);
  assert.equal(applyActionWithRewardProfile(turnWorld, 1, 'turn-bonus').lastTransition.reward, 0.08);

  const collisionWorld = createGridWorld('reward-collision');
  collisionWorld.agent = { row: 1, column: 2, direction: 1 };
  const collision = applyActionWithRewardProfile(collisionWorld, 0, 'collision-bonus');
  assert.equal(collision.lastTransition.rewardCategory, 'collision');
  assert.equal(collision.lastTransition.reward, 0.3);
  assert.equal(collision.lastTransition.baseReward, -0.25);
});

test('実測比率から旋回報酬ハッキング候補を診断する', () => {
  const experiences = Array.from({ length: 10 }, (_, index) => ({
    stateKey: `state-${index % 2}`,
    actionIndex: index % 2 ? 1 : 2,
    rewardCategory: 'turn',
  }));
  const diagnostic = diagnoseEpisode(
    { foods: 0, cumulativeReward: 0.8 },
    experiences,
    'turn-bonus',
  );
  assert.equal(diagnostic.key, 'reward-hacking-turn');
  assert.equal(diagnostic.level, 'danger');
});

test('同じ世界・RLシードから同じ行動列、報酬、最終ネットワークを再現する', () => {
  const network = createNetwork('rl-repro-network');
  const world = createGridWorld('rl-repro-world');
  const config = {
    randomSeed: 'rl-repro-random',
    episodeNumber: 3,
    maxSteps: 8,
    temperature: 1.3,
    gamma: 0.9,
    learningRate: 0.05,
    rewardProfile: 'balanced',
  };
  const first = buildReinforcementTimeline(network, world, config);
  const second = buildReinforcementTimeline(network, world, config);
  assert.deepEqual(
    first.experiences.map((experience) => [experience.actionIndex, experience.reward]),
    second.experiences.map((experience) => [experience.actionIndex, experience.reward]),
  );
  assert.deepEqual(first.finalNetwork, second.finalNetwork);
  assert.deepEqual(first.finalWorld, second.finalWorld);
});

test('強化学習タイムラインに観測から全39更新と比較までが含まれる', () => {
  const built = buildReinforcementTimeline(
    createNetwork('rl-stages-network'),
    createGridWorld('rl-stages-world'),
    { randomSeed: 'rl-stages', maxSteps: 4 },
  );
  const stages = new Set(built.steps.map((step) => step.stage));
  for (const required of [
    'rl-ready', 'rl-observation', 'rl-policy', 'rl-sample', 'rl-transition',
    'rl-return', 'rl-gradient', 'rl-aggregate', 'rl-update', 'rl-comparison',
  ]) {
    assert.equal(stages.has(required), true, required);
  }
  assert.equal(built.steps.filter((step) => step.stage === 'rl-update').length, 39);
  assert.equal(built.finalNetwork.learningCount, 1);
});

test('RLステップ実行と一括実行の最終スナップショットが一致する', () => {
  const network = createNetwork('rl-step-equivalence');
  const world = createGridWorld('rl-step-world');
  const options = { randomSeed: 'rl-step-random', maxSteps: 6, rewardProfile: 'turn-bonus' };
  const stepped = new ReinforcementStepEngine(network, world, options);
  const bulk = new ReinforcementStepEngine(network, world, options);
  while (stepped.canGoNext) stepped.next();
  bulk.runToEnd();
  assert.deepEqual(stepped.current.network, bulk.current.network);
  assert.deepEqual(stepped.current.world, bulk.current.world);
  assert.deepEqual(stepped.summary, bulk.summary);
});

test('連続表示の実移動境界を通っても一括実行と同じ最終状態になる', () => {
  const network = createNetwork('rl-visible-boundary-equivalence');
  const world = createGridWorld('rl-visible-boundary-world');
  const options = { randomSeed: 'rl-visible-boundary-random', maxSteps: 7, rewardProfile: 'balanced' };
  const animated = new ReinforcementStepEngine(network, world, options);
  const bulk = new ReinforcementStepEngine(network, world, options);
  const stages = [];

  while (animated.canGoNext) {
    const boundary = advanceToReinforcementVisualBoundary(animated, (step) => stages.push(step.stage));
    assert.equal(['rl-transition', 'rl-comparison'].includes(boundary.step.stage), true);
  }
  bulk.runToEnd();

  assert.equal(stages.filter((stage) => stage === 'rl-transition').length, animated.summary.steps);
  assert.equal(stages.includes('rl-update'), true);
  assert.deepEqual(animated.current.network, bulk.current.network);
  assert.deepEqual(animated.current.world, bulk.current.world);
  assert.deepEqual(animated.summary, bulk.summary);
});

test('連続表示速度は実計算と独立した正の待機時間へ変換される', () => {
  assert.equal(continuousEpisodeDelay(4) < continuousEpisodeDelay(1), true);
  assert.equal(continuousEpisodeDelay(2, 'episode') > continuousEpisodeDelay(2), true);
  assert.throws(() => continuousEpisodeDelay(0), /0より大きい/);
});

test('前のRLステップで世界・ネットワーク・部分更新を完全復元する', () => {
  const engine = new ReinforcementStepEngine(
    createNetwork('rl-history-network'),
    createGridWorld('rl-history-world'),
    { randomSeed: 'rl-history', maxSteps: 5 },
  );
  const updateIndex = engine.steps.findIndex((step) => step.stage === 'rl-update') + 5;
  while (engine.index < updateIndex) engine.next();
  const expectedPrevious = engine.steps[engine.index - 1];
  engine.previous();
  assert.deepEqual(engine.current.network, expectedPrevious.network);
  assert.deepEqual(engine.current.world, expectedPrevious.world);
  assert.deepEqual(engine.current.parameterInfo, expectedPrevious.parameterInfo);
});

test('各経験が遷移前後の世界と逐次累積報酬を完全保存する', () => {
  const built = buildReinforcementTimeline(
    createNetwork('rl-environment-history-network'),
    createGridWorld('rl-environment-history-world'),
    { randomSeed: 'rl-environment-history', maxSteps: 8, rewardProfile: 'balanced' },
  );
  let cumulativeReward = 0;
  built.experiences.forEach((experience, index) => {
    cumulativeReward += experience.reward;
    assert.equal(experience.time, index);
    assert.equal(experience.cumulativeReward, cumulativeReward);
    assert.equal(experience.beforeWorld.counters.steps + 1, experience.afterWorld.counters.steps);
    if (index > 0) assert.deepEqual(experience.beforeWorld, built.experiences[index - 1].afterWorld);
  });
  const preservedRow = built.experiences[0].beforeWorld.agent.row;
  built.finalWorld.agent.row = (built.finalWorld.agent.row + 1) % built.finalWorld.size;
  assert.equal(built.experiences[0].beforeWorld.agent.row, preservedRow);
});
