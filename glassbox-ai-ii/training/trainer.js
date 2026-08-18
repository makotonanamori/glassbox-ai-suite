import { TinyTransformer } from "../model/transformer.js";
import { globalGradientNorm, globalParameterNorm, sgdStep } from "../model/optimizer.js";
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
    const session = this.beginStep();
    this.backwardStep(session);
    this.updateStep(session);
    return this.finishStep(session);
  }

  beginStep() {
    const sample = this.dataset.sampleAt(this.step, this.seed);
    this.model.zeroGrad();
    const before = this.model.forward(sample.inputIds, { targets: sample.targetIds });
    const parameterNormBefore = globalParameterNorm(this.model.parameters());
    return {
      stepBefore: this.step,
      sample: deepClone(sample),
      lossTensor: before.lossTensor,
      beforeTrace: before.trace,
      lossBefore: before.trace.loss,
      parameterNormBefore,
      backwardComplete: false,
      updateComplete: false,
      finished: false,
    };
  }

  backwardStep(session) {
    this.#assertSession(session, "backward");
    if (session.backwardComplete) return session;
    this.model.backward(session.lossTensor);
    session.rawGradientNorm = globalGradientNorm(this.model.parameters());
    session.clipScale = session.rawGradientNorm > this.clipNorm ? this.clipNorm / session.rawGradientNorm : 1;
    session.clippedGradientNorm = session.rawGradientNorm * session.clipScale;
    session.backwardComplete = true;
    assertFinite([session.lossBefore, session.rawGradientNorm, session.clippedGradientNorm], "backward diagnostics");
    return session;
  }

  updateStep(session) {
    this.#assertSession(session, "update");
    if (!session.backwardComplete) throw new Error("Backwardを完了してからSGD Updateを実行してください。");
    if (session.updateComplete) return session;
    const update = sgdStep(this.model.parameters(), this.learningRate, this.clipNorm);
    Object.assign(session, update);
    session.updateComplete = true;
    this.lastUpdate = deepClone(update.updates);
    return session;
  }

  finishStep(session) {
    this.#assertSession(session, "finish");
    if (!session.updateComplete) throw new Error("SGD Updateを完了してからTraining Stepを確定してください。");
    if (session.finished) return session.result;
    const after = this.model.forward(session.sample.inputIds, { targets: session.sample.targetIds });
    this.step += 1;
    this.lossHistory.push({ step: this.step, loss: session.lossBefore });
    const result = {
      step: this.step,
      sample: deepClone(session.sample),
      beforeTrace: session.beforeTrace,
      afterTrace: after.trace,
      lossBefore: session.lossBefore,
      lossAfter: after.trace.loss,
      parameterNormBefore: session.parameterNormBefore,
      parameterNormAfter: globalParameterNorm(this.model.parameters()),
      rawGradientNorm: session.rawGradientNorm,
      clippedGradientNorm: session.clippedGradientNorm,
      clipScale: session.clipScale,
      updates: session.updates,
    };
    assertFinite([result.lossBefore, result.lossAfter, result.rawGradientNorm, result.parameterNormAfter], "training diagnostics");
    if ([10, 100, 500].includes(this.step)) this.captureSnapshot(`STEP ${this.step}`);
    session.finished = true;
    session.result = result;
    return result;
  }

  #assertSession(session, action) {
    if (!session || session.stepBefore !== this.step) {
      throw new Error(`${action}対象のTraining Sessionが現在のStepと一致しません。`);
    }
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
