import test from "node:test";
import assert from "node:assert/strict";
import { Tensor, add, crossEntropy, gelu, layerNorm, maskedSoftmaxRows, matmul, parameter } from "../model/tensor.js";

test("matmul forward and backward match hand calculation", () => {
  const a = parameter([1, 2, 3, 4], [2, 2], "a");
  const b = parameter([2, 0, 1, 2], [2, 2], "b");
  const product = matmul(a, b);
  assert.deepEqual(product.data, [4, 4, 10, 8]);
  const logits = add(product, new Tensor([0, 0], [2]));
  const loss = crossEntropy(logits, [0, 1]);
  loss.backward();
  assert.ok(a.grad.every(Number.isFinite));
  assert.ok(b.grad.some((value) => Math.abs(value) > 0));
});

test("softmax is stable, sums to one, and causal mask is exact", () => {
  const scores = parameter([1000, -1000, 4, 5], [2, 2], "scores");
  const weights = maskedSoftmaxRows(scores, true);
  assert.equal(weights.data[1], 0);
  assert.ok(Math.abs(weights.data[0] - 1) < 1e-12);
  assert.ok(Math.abs(weights.data[2] + weights.data[3] - 1) < 1e-12);
  assert.ok(weights.data.every(Number.isFinite));
});

test("LayerNorm and GELU propagate finite gradients", () => {
  const x = parameter([1, 2, 3, 4, -1, 0, 1, 2], [2, 4], "x");
  const gamma = parameter([1, 1, 1, 1], [4], "gamma");
  const beta = parameter([0, 0, 0, 0], [4], "beta");
  const norm = layerNorm(x, gamma, beta);
  for (let row = 0; row < 2; row += 1) {
    const mean = norm.data.slice(row * 4, row * 4 + 4).reduce((sum, value) => sum + value, 0) / 4;
    assert.ok(Math.abs(mean) < 1e-12);
  }
  const activated = gelu(norm);
  const loss = crossEntropy(activated, [0, 1]);
  loss.backward();
  assert.ok(x.grad.every(Number.isFinite));
  assert.ok(gamma.grad.every(Number.isFinite));
});
