import test from "node:test";
import assert from "node:assert/strict";
import { LanguageDataset } from "../training/dataset.js";
import { TinyTransformer } from "../model/transformer.js";
import { Trainer } from "../training/trainer.js";
import {
  captureLearningObservation, learningCheckpoints, learningObserverDelay,
} from "../js/learning-observer.js";

test("learning observer divides 500 real updates into visible checkpoints", () => {
  assert.deepEqual(learningCheckpoints(500, 50), [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]);
  assert.deepEqual(learningCheckpoints(125, 50), [50, 100, 125]);
  assert.equal(learningObserverDelay(0.5), 840);
  assert.equal(learningObserverDelay(2), 210);
  assert.equal(learningObserverDelay(4), 105);
});

test("real training changes the fixed prompt from random continuation to corpus pattern", () => {
  const dataset = new LanguageDataset();
  const model = new TinyTransformer(dataset.tokenizer.vocabulary, { seed: 42 });
  const trainer = new Trainer(model, dataset, { seed: 42, learningRate: 0.03, clipNorm: 1 });
  const lossBefore = trainer.evaluateLoss();
  const before = captureLearningObservation(model, dataset.tokenizer);

  trainer.trainSteps(500);

  const after = captureLearningObservation(model, dataset.tokenizer);
  const lossAfter = trainer.evaluateLoss();
  assert.equal(before.generatedText, "the cat eats the dog cats milk the");
  assert.equal(after.generatedText, "the cat eats fish.");
  assert.equal(before.selectedToken, "the");
  assert.equal(after.selectedToken, "fish");
  assert.ok(after.targetProbability > before.targetProbability * 2);
  assert.ok(lossAfter < lossBefore * 0.4, `before=${lossBefore}, after=${lossAfter}`);
});
