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
