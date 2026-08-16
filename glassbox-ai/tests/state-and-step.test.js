import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, PARAMETER_SPECS, getParameterValue } from '../src/neural-network.js';
import { forwardPass } from '../src/forward-pass.js';
import { trainOne } from '../src/backpropagation.js';
import { parseStateJson, serializeState } from '../src/serializer.js';
import { StepEngine } from '../src/step-engine.js';
import { OperationLog } from '../src/state-manager.js';

const inputs = [0.8, 0, 0, 0, 0];

test('保存した状態を読み込むと同じ出力が再現される', () => {
  const network = createNetwork('serialization-test');
  const before = forwardPass(network, inputs);
  const json = serializeState({ network, learningRate: 0.1, inputs, targetIndex: 0 });
  const restored = parseStateJson(json);
  const after = forwardPass(restored.network, restored.inputs);
  assert.deepEqual(after.probabilities, before.probabilities);
  assert.equal(restored.learningRate, 0.1);
  assert.equal(restored.targetIndex, 0);
});

test('構造の異なる保存データは安全に拒否される', () => {
  const network = createNetwork('invalid-state');
  const payload = JSON.parse(serializeState({ network, learningRate: 0.1, inputs, targetIndex: 0 }));
  payload.structure.hidden = 5;
  assert.throws(() => parseStateJson(JSON.stringify(payload)), /ネットワーク構造/);
});

test('操作ログをテキストとJSONへ同じ内容で書き出せる', () => {
  const log = new OperationLog();
  log.add('[推論] 入力値を確定');
  log.add('[判断] 行動Aを選択');
  assert.match(log.toText(), /\[推論\] 入力値を確定/);
  const json = JSON.parse(log.toJson());
  assert.equal(json.format, 'glassbox-ai-log');
  assert.deepEqual(json.entries.map((entry) => entry.message), [
    '[推論] 入力値を確定',
    '[判断] 行動Aを選択',
  ]);
});

test('推論のステップ実行と一括順伝播の最終結果が一致する', () => {
  const network = createNetwork('step-inference');
  const expected = forwardPass(network, inputs);
  const engine = new StepEngine(network, inputs);
  engine.last();
  assert.deepEqual(engine.current.forward.probabilities, expected.probabilities);
  assert.equal(engine.current.forward.selectedIndex, expected.selectedIndex);
});

test('学習のステップ実行と一括学習の最終ネットワークが一致する', () => {
  const network = createNetwork('step-training');
  const learningRate = 0.1;
  const expected = trainOne(network, inputs, 0, learningRate);
  const engine = new StepEngine(network, inputs);
  engine.last();
  engine.appendLearning(0, learningRate);
  assert.equal(engine.length - 1, 139, '初期スナップショットを除く既存の教師あり数学タイムラインは139ステップを維持する');
  engine.last();

  for (const spec of PARAMETER_SPECS) {
    assert.ok(
      Math.abs(getParameterValue(engine.current.network, spec) - getParameterValue(expected.network, spec)) < 1e-15,
      spec.name,
    );
  }
  assert.deepEqual(engine.current.forward.probabilities, expected.afterForward.probabilities);
  assert.deepEqual(engine.current.training.comparison, expected.comparison);
});
