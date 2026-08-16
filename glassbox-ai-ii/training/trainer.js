import { TinyTransformer } from "../model/transformer.js";
import { globalParameterNorm, sgdStep } from "../model/optimizer.js";
import { assertFinite } from "../utils/math.js";

function deepClone(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

export class Trainer {
  constructor(model, dataset, { learningRate = 0.03, clipNorm = 1, seed = 42 } = {}) {
    this.model = model;
    this.dataset = dataset;
    this.learningRate = learningRate;
    this.clipNorm = clipNorm;
    this.seed = seed;
    this.step = 0;
    this.lossHistory = [];
    this.lastUpdate = null;
    this.snapshots = new Map();
    this.captureSnapshot("STEP 0");
  }

  trainOneStep() {
    const sample = this.dataset.sampleAt(this.step, this.seed);
    this.model.zeroGrad();
    const before = this.model.forward(sample.inputIds, { targets: sample.targetIds });
    this.model.backward(before.lossTensor);
    const parameterNormBefore = globalParameterNorm(this.model.parameters());
    const update = sgdStep(this.model.parameters(), this.learningRate, this.clipNorm);
    const after = this.model.forward(sample.inputIds, { targets: sample.targetIds });
    this.step += 1;
    this.lossHistory.push({ step: this.step, loss: before.trace.loss });
    this.lastUpdate = deepClone(update.updates);
    const result = {
      step: this.step,
      sample: deepClone(sample),
      beforeTrace: before.trace,
      afterTrace: after.trace,
      lossBefore: before.trace.loss,
      lossAfter: after.trace.loss,
      parameterNormBefore,
      parameterNormAfter: globalParameterNorm(this.model.parameters()),
      ...update,
    };
    assertFinite([result.lossBefore, result.lossAfter, result.rawGradientNorm, result.parameterNormAfter], "training diagnostics");
    if ([10, 100, 500].includes(this.step)) this.captureSnapshot(`STEP ${this.step}`);
    return result;
  }

  trainSteps(count) {
    let result = null;
    for (let i = 0; i < count; i += 1) result = this.trainOneStep();
    return result;
  }

  evaluateLoss() {
    const losses = this.dataset.samples.map((sample) => this.model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss);
    return losses.reduce((sum, value) => sum + value, 0) / losses.length;
  }

  captureSnapshot(label = `STEP ${this.step}`) {
    const state = {
      label,
      step: this.step,
      model: this.model.exportState(),
      lossHistory: deepClone(this.lossHistory),
      settings: { learningRate: this.learningRate, clipNorm: this.clipNorm, seed: this.seed },
    };
    this.snapshots.set(label, state);
    return state;
  }

  traceFromSnapshot(label, tokenIds, targets = null) {
    const state = this.snapshots.get(label);
    if (!state) throw new Error(`Snapshot not found: ${label}`);
    const model = new TinyTransformer(state.model.vocabulary, { seed: state.model.seed, config: state.model.config });
    model.importParameters(state.model.parameters);
    return model.forward(tokenIds, { targets }).trace;
  }
}
