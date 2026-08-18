import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { StepEngine } from '../src/step-engine.js';
import {
  SUPERVISED_PLAYBACK_DEFAULTS,
  advanceSupervisedPlayback,
  supervisedPlaybackDelay,
} from '../src/supervised-playback.js';

test('教師あり自動再生は139 snapshotを通って一括実行と同じ結果になる', () => {
  const inputs = [0.8, 0, 0, 0, 0];
  const learningRate = 0.1;
  const targetIndex = 0;
  const animated = new StepEngine(createNetwork('supervised-playback'), inputs);
  const bulk = new StepEngine(createNetwork('supervised-playback'), inputs);
  let displayedSteps = 0;
  let appendedLearningCount = 0;

  while (true) {
    const result = advanceSupervisedPlayback(animated, {
      targetIndex,
      learningRate,
      onStep: () => { displayedSteps += 1; },
    });
    if (result.appendedLearning) appendedLearningCount += 1;
    if (result.complete) break;
  }

  while (bulk.canGoNext) bulk.next();
  bulk.appendLearning(targetIndex, learningRate);
  bulk.last();

  assert.equal(displayedSteps, SUPERVISED_PLAYBACK_DEFAULTS.totalSteps);
  assert.equal(animated.index, SUPERVISED_PLAYBACK_DEFAULTS.totalSteps);
  assert.equal(appendedLearningCount, 1);
  assert.deepEqual(animated.current.network, bulk.current.network);
  assert.deepEqual(animated.current.training.comparison, bulk.current.training.comparison);
});

test('教師あり再生速度は数学結果と独立した正の待機時間へ変換される', () => {
  assert.equal(supervisedPlaybackDelay(4) < supervisedPlaybackDelay(1), true);
  assert.equal(supervisedPlaybackDelay(4), 120);
  assert.throws(() => supervisedPlaybackDelay(0), /0より大きい/);
});
