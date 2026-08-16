import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork } from '../src/neural-network.js';
import { createGridWorld } from '../src/grid-world.js';
import { buildReinforcementTimeline } from '../src/reinforcement-learning.js';
import {
  createReinforcementHistoryDocument,
  parseReinforcementHistoryJson,
  serializeReinforcementHistory,
} from '../src/reinforcement-history.js';

function episodeRecord(seed = 'history-export') {
  const built = buildReinforcementTimeline(
    createNetwork(`${seed}:network`),
    createGridWorld(`${seed}:world`),
    { randomSeed: `${seed}:policy`, maxSteps: 5 },
  );
  return {
    config: built.config,
    profile: built.profile,
    initialWorld: built.initialWorld,
    finalWorld: built.finalWorld,
    experiences: built.experiences,
    returns: built.returns,
    summary: built.summary,
    initialNetwork: built.steps[0].network,
    finalNetwork: built.finalNetwork,
    averageGradient: built.averageGradient,
    parameterInfo: built.steps.at(-1).parameterInfo,
  };
}

test('RL履歴JSONを同じ環境・報酬・方策更新情報へ復元できる', () => {
  const record = episodeRecord();
  const exportedAt = '2026-08-15T00:00:00.000Z';
  const canonical = createReinforcementHistoryDocument([record], exportedAt);
  const json = serializeReinforcementHistory([record], exportedAt);
  const restored = parseReinforcementHistoryJson(json);
  assert.equal(restored.format, 'glassbox-ai-reinforcement-history');
  assert.equal(restored.episodeCount, 1);
  assert.deepEqual(restored.episodes[0], canonical.episodes[0]);
});

test('RL履歴Documentは元のエピソードから独立したcloneである', () => {
  const record = episodeRecord('history-clone');
  const document = createReinforcementHistoryDocument([record], 'fixed');
  document.episodes[0].experiences[0].beforeWorld.agent.row = 99;
  assert.notEqual(record.experiences[0].beforeWorld.agent.row, 99);
});

test('形式不一致、件数不一致、非有限報酬のRL履歴を拒否する', () => {
  assert.throws(() => parseReinforcementHistoryJson('{}'), /対応していない/);
  const record = episodeRecord('history-invalid');
  const document = createReinforcementHistoryDocument([record], 'fixed');
  document.episodeCount = 2;
  assert.throws(() => parseReinforcementHistoryJson(JSON.stringify(document)), /件数/);
  document.episodeCount = 1;
  document.episodes[0].experiences[0].reward = null;
  assert.throws(() => parseReinforcementHistoryJson(JSON.stringify(document)), /報酬/);
});
