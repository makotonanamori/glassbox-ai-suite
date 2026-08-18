import test from "node:test";
import assert from "node:assert/strict";
import { LanguageDataset } from "../training/dataset.js";
import { TinyTransformer } from "../model/transformer.js";
import { Trainer } from "../training/trainer.js";

function setup(seed = 42) {
  const dataset = new LanguageDataset();
  const model = new TinyTransformer(dataset.tokenizer.vocabulary, { seed });
  const trainer = new Trainer(model, dataset, { seed, learningRate: 0.03, clipNorm: 1 });
  return { dataset, model, trainer };
}

test("training lowers evaluation loss and remains finite for 500 steps", () => {
  const { trainer } = setup();
  const initial = trainer.evaluateLoss();
  trainer.trainSteps(500);
  const trained = trainer.evaluateLoss();
  assert.ok(Number.isFinite(trained));
  assert.ok(trained < initial * 0.75, `initial=${initial}, trained=${trained}`);
  assert.equal(trainer.step, 500);
  assert.ok(trainer.snapshots.has("STEP 500"));
});

test("same seed and steps reproduce the same model", () => {
  const a = setup(77);
  const b = setup(77);
  a.trainer.trainSteps(25);
  b.trainer.trainSteps(25);
  assert.deepEqual(a.model.exportState().parameters, b.model.exportState().parameters);
  assert.deepEqual(a.trainer.lossHistory, b.trainer.lossHistory);
});

test("staged training follows the same real computation as trainOneStep", () => {
  const direct = setup(91);
  const staged = setup(91);
  const directResult = direct.trainer.trainOneStep();

  const beforeParameters = staged.model.exportState().parameters;
  const session = staged.trainer.beginStep();
  assert.deepEqual(staged.model.exportState().parameters, beforeParameters);
  assert.equal(staged.trainer.step, 0);

  staged.trainer.backwardStep(session);
  assert.ok(session.rawGradientNorm > 0);
  assert.deepEqual(staged.model.exportState().parameters, beforeParameters);

  staged.trainer.updateStep(session);
  assert.notDeepEqual(staged.model.exportState().parameters, beforeParameters);
  const stagedResult = staged.trainer.finishStep(session);

  assert.deepEqual(staged.model.exportState().parameters, direct.model.exportState().parameters);
  assert.deepEqual(staged.trainer.lossHistory, direct.trainer.lossHistory);
  assert.equal(stagedResult.lossBefore, directResult.lossBefore);
  assert.equal(stagedResult.lossAfter, directResult.lossAfter);
  assert.equal(stagedResult.rawGradientNorm, directResult.rawGradientNorm);
  assert.equal(stagedResult.clipScale, directResult.clipScale);
  assert.deepEqual(stagedResult.updates, directResult.updates);
});
