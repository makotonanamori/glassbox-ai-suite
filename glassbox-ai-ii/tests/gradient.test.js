import test from "node:test";
import assert from "node:assert/strict";
import { LanguageDataset } from "../training/dataset.js";
import { TinyTransformer } from "../model/transformer.js";
import { relativeError } from "../utils/math.js";

function check(name, index, epsilon = 1e-5) {
  const dataset = new LanguageDataset();
  const model = new TinyTransformer(dataset.tokenizer.vocabulary, { seed: 42 });
  const sample = dataset.samples[0];
  model.zeroGrad();
  const forward = model.forward(sample.inputIds, { targets: sample.targetIds });
  model.backward(forward.lossTensor);
  const tensor = model.parameterMap.get(name);
  const analytical = tensor.grad[index];
  const original = tensor.data[index];
  tensor.data[index] = original + epsilon;
  const plus = model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss;
  tensor.data[index] = original - epsilon;
  const minus = model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss;
  tensor.data[index] = original;
  const numerical = (plus - minus) / (2 * epsilon);
  return { analytical, numerical, error: relativeError(analytical, numerical) };
}

for (const [name, index] of [
  ["vocabulary.w", 7],
  ["block.attention.head1.wq", 5],
  ["block.attention.head2.wv", 11],
  ["block.mlp.w1", 23],
  ["embeddings.token", 3],
]) {
  test(`gradient check ${name}[${index}]`, () => {
    const result = check(name, index);
    assert.ok(result.error < 1e-3, JSON.stringify(result));
  });
}
