import test from "node:test";
import assert from "node:assert/strict";
import { LanguageDataset } from "../training/dataset.js";
import { TinyTransformer } from "../model/transformer.js";
import { ForwardStepEngine, FORWARD_STAGES } from "../model/step-engine.js";
import { importApplicationState, exportApplicationState } from "../utils/serialization.js";
import { Trainer } from "../training/trainer.js";

function setup(seed = 42) {
  const dataset = new LanguageDataset();
  const model = new TinyTransformer(dataset.tokenizer.vocabulary, { seed });
  return { dataset, model };
}

test("forward trace has expected shapes and probability sums", () => {
  const { dataset, model } = setup();
  const sample = dataset.samples[0];
  const { trace } = model.forward(sample.inputIds, { targets: sample.targetIds });
  const length = sample.inputIds.length;
  const vocab = dataset.tokenizer.vocabulary.length;
  assert.deepEqual(trace.tokenEmbeddings.shape, [length, 8]);
  assert.deepEqual(trace.heads[0].q.shape, [length, 4]);
  assert.deepEqual(trace.heads[0].rawScores.shape, [length, length]);
  assert.deepEqual(trace.mlpActivation.shape, [length, 16]);
  assert.deepEqual(trace.logits.shape, [length, vocab]);
  for (let row = 0; row < length; row += 1) {
    const sum = trace.probabilities.data.slice(row * vocab, (row + 1) * vocab).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-12);
    trace.heads.forEach((head) => {
      const attentionSum = head.attentionWeights.data.slice(row * length, (row + 1) * length).reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(attentionSum - 1) < 1e-12);
      for (let col = row + 1; col < length; col += 1) assert.equal(head.attentionWeights.data[row * length + col], 0);
    });
  }
});

test("same seed produces identical parameters and logits", () => {
  const a = setup(123);
  const b = setup(123);
  assert.deepEqual(a.model.exportState().parameters, b.model.exportState().parameters);
  assert.deepEqual(a.model.forward(a.dataset.samples[2].inputIds).trace.logits, b.model.forward(b.dataset.samples[2].inputIds).trace.logits);
});

test("serialization restores identical logits", () => {
  const { dataset, model } = setup();
  const trainer = new Trainer(model, dataset);
  trainer.trainSteps(3);
  const sample = dataset.samples[1];
  const before = model.forward(sample.inputIds).trace.logits.data;
  const restored = importApplicationState(exportApplicationState(model, trainer, dataset));
  const after = restored.model.forward(sample.inputIds).trace.logits.data;
  assert.deepEqual(after, before);
  assert.equal(restored.trainer.step, 3);
});

test("trace snapshots and step engine remain internally consistent", () => {
  const { dataset, model } = setup();
  const trace = model.forward(dataset.samples[0].inputIds).trace;
  const preserved = trace.logits.data[0];
  model.parameters()[0].tensor.data[0] += 1;
  assert.equal(trace.logits.data[0], preserved);
  const engine = new ForwardStepEngine(trace);
  assert.equal(engine.current().key, "tokenizer");
  engine.run();
  assert.equal(engine.current().key, "probabilities");
  for (let i = 1; i < FORWARD_STAGES.length; i += 1) engine.previous();
  assert.equal(engine.current().key, "tokenizer");
});
