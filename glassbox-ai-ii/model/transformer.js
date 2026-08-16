import {
  Tensor, add, concatColumns, crossEntropy, divide, embedding, gelu, layerNorm,
  maskedSoftmaxRows, matmul, parameter, scalar, transpose2D,
} from "./tensor.js";
import { SeededRandom } from "../utils/rng.js";
import { assertFinite, maxAbs } from "../utils/math.js";

export const DEFAULT_CONFIG = Object.freeze({
  contextLength: 8,
  dModel: 8,
  heads: 2,
  dHead: 4,
  mlpHidden: 16,
  epsilon: 1e-5,
});

function snapshot(tensor) {
  return { data: [...tensor.data], shape: [...tensor.shape] };
}

function xavier(rng, fanIn, fanOut, length) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return Array.from({ length }, () => rng.uniform(-limit, limit));
}

function zeros(length) {
  return new Array(length).fill(0);
}

function ones(length) {
  return new Array(length).fill(1);
}

function validateConfig(config) {
  const keys = ["contextLength", "dModel", "heads", "dHead", "mlpHidden"];
  keys.forEach((key) => {
    if (!Number.isInteger(config[key]) || config[key] <= 0) throw new Error(`Invalid config.${key}`);
  });
  if (config.heads * config.dHead !== config.dModel) throw new Error("heads × dHead must equal dModel");
  if (!(config.epsilon > 0)) throw new Error("epsilon must be positive");
}

export class TinyTransformer {
  constructor(vocabulary, { seed = 42, config = {} } = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config, vocabSize: vocabulary.length };
    validateConfig(this.config);
    this.vocabulary = [...vocabulary];
    this.seed = seed;
    this.parameterMap = new Map();
    const rng = new SeededRandom(seed);
    const c = this.config;
    const make = (name, shape, data) => {
      const tensor = parameter(data, shape, name);
      this.parameterMap.set(name, tensor);
      return tensor;
    };

    this.tokenEmbedding = make("embeddings.token", [c.vocabSize, c.dModel], xavier(rng, c.vocabSize, c.dModel, c.vocabSize * c.dModel));
    this.positionEmbedding = make("embeddings.position", [c.contextLength, c.dModel], xavier(rng, c.contextLength, c.dModel, c.contextLength * c.dModel));
    this.ln1Gamma = make("block.ln1.gamma", [c.dModel], ones(c.dModel));
    this.ln1Beta = make("block.ln1.beta", [c.dModel], zeros(c.dModel));

    this.headParameters = Array.from({ length: c.heads }, (_, head) => ({
      wq: make(`block.attention.head${head + 1}.wq`, [c.dModel, c.dHead], xavier(rng, c.dModel, c.dHead, c.dModel * c.dHead)),
      bq: make(`block.attention.head${head + 1}.bq`, [c.dHead], zeros(c.dHead)),
      wk: make(`block.attention.head${head + 1}.wk`, [c.dModel, c.dHead], xavier(rng, c.dModel, c.dHead, c.dModel * c.dHead)),
      bk: make(`block.attention.head${head + 1}.bk`, [c.dHead], zeros(c.dHead)),
      wv: make(`block.attention.head${head + 1}.wv`, [c.dModel, c.dHead], xavier(rng, c.dModel, c.dHead, c.dModel * c.dHead)),
      bv: make(`block.attention.head${head + 1}.bv`, [c.dHead], zeros(c.dHead)),
    }));
    this.wo = make("block.attention.wo", [c.dModel, c.dModel], xavier(rng, c.dModel, c.dModel, c.dModel * c.dModel));
    this.bo = make("block.attention.bo", [c.dModel], zeros(c.dModel));
    this.ln2Gamma = make("block.ln2.gamma", [c.dModel], ones(c.dModel));
    this.ln2Beta = make("block.ln2.beta", [c.dModel], zeros(c.dModel));
    this.mlpW1 = make("block.mlp.w1", [c.dModel, c.mlpHidden], xavier(rng, c.dModel, c.mlpHidden, c.dModel * c.mlpHidden));
    this.mlpB1 = make("block.mlp.b1", [c.mlpHidden], zeros(c.mlpHidden));
    this.mlpW2 = make("block.mlp.w2", [c.mlpHidden, c.dModel], xavier(rng, c.mlpHidden, c.dModel, c.mlpHidden * c.dModel));
    this.mlpB2 = make("block.mlp.b2", [c.dModel], zeros(c.dModel));
    this.finalGamma = make("final.ln.gamma", [c.dModel], ones(c.dModel));
    this.finalBeta = make("final.ln.beta", [c.dModel], zeros(c.dModel));
    this.vocabW = make("vocabulary.w", [c.dModel, c.vocabSize], xavier(rng, c.dModel, c.vocabSize, c.dModel * c.vocabSize));
    this.vocabB = make("vocabulary.b", [c.vocabSize], zeros(c.vocabSize));
  }

  parameters() {
    return [...this.parameterMap.entries()].map(([name, tensor]) => ({ name, tensor }));
  }

  parameterCount() {
    return this.parameters().reduce((sum, { tensor }) => sum + tensor.data.length, 0);
  }

  zeroGrad() {
    this.parameters().forEach(({ tensor }) => tensor.zeroGrad());
  }

  forward(tokenIds, { targets = null } = {}) {
    const c = this.config;
    if (!Array.isArray(tokenIds) || tokenIds.length < 1 || tokenIds.length > c.contextLength) throw new Error(`tokenIds length must be 1..${c.contextLength}`);
    tokenIds.forEach((id) => {
      if (!Number.isInteger(id) || id < 0 || id >= c.vocabSize) throw new Error(`Invalid token id ${id}`);
    });
    if (targets && targets.length !== tokenIds.length) throw new Error("targets length must equal tokenIds length");
    const positions = tokenIds.map((_, index) => index);
    const tokenEmbeddings = embedding(this.tokenEmbedding, tokenIds);
    const positionEmbeddings = embedding(this.positionEmbedding, positions);
    const initialRepresentation = add(tokenEmbeddings, positionEmbeddings);
    const layerNorm1 = layerNorm(initialRepresentation, this.ln1Gamma, this.ln1Beta, c.epsilon);
    const heads = this.headParameters.map((params) => {
      const q = add(matmul(layerNorm1, params.wq), params.bq);
      const k = add(matmul(layerNorm1, params.wk), params.bk);
      const v = add(matmul(layerNorm1, params.wv), params.bv);
      const rawScores = divide(matmul(q, transpose2D(k)), scalar(Math.sqrt(c.dHead)));
      const attentionWeights = maskedSoftmaxRows(rawScores, true);
      const output = matmul(attentionWeights, v);
      return { q, k, v, rawScores, attentionWeights, output };
    });
    const concatenatedHeads = concatColumns(heads.map((head) => head.output));
    const attentionOutput = add(matmul(concatenatedHeads, this.wo), this.bo);
    const residual1 = add(initialRepresentation, attentionOutput);
    const layerNorm2 = layerNorm(residual1, this.ln2Gamma, this.ln2Beta, c.epsilon);
    const mlpPreActivation = add(matmul(layerNorm2, this.mlpW1), this.mlpB1);
    const mlpActivation = gelu(mlpPreActivation);
    const mlpOutput = add(matmul(mlpActivation, this.mlpW2), this.mlpB2);
    const residual2 = add(residual1, mlpOutput);
    const finalNorm = layerNorm(residual2, this.finalGamma, this.finalBeta, c.epsilon);
    const logits = add(matmul(finalNorm, this.vocabW), this.vocabB);
    const probabilities = maskedSoftmaxRows(logits, false);
    const lossTensor = targets ? crossEntropy(logits, targets) : null;

    const trace = {
      tokens: tokenIds.map((id, position) => ({ id, text: this.vocabulary[id], position })),
      tokenIds: [...tokenIds],
      targets: targets ? [...targets] : null,
      tokenEmbeddings: snapshot(tokenEmbeddings),
      positionEmbeddings: snapshot(positionEmbeddings),
      initialRepresentation: snapshot(initialRepresentation),
      layerNorm1: snapshot(layerNorm1),
      heads: heads.map((head) => ({
        q: snapshot(head.q), k: snapshot(head.k), v: snapshot(head.v),
        rawScores: snapshot(head.rawScores),
        maskedScores: { data: [...head.attentionWeights.maskedScores], shape: [...head.rawScores.shape] },
        attentionWeights: snapshot(head.attentionWeights), output: snapshot(head.output),
      })),
      concatenatedHeads: snapshot(concatenatedHeads),
      attentionOutput: snapshot(attentionOutput),
      residual1: snapshot(residual1),
      layerNorm2: snapshot(layerNorm2),
      mlpPreActivation: snapshot(mlpPreActivation),
      mlpActivation: snapshot(mlpActivation),
      mlpOutput: snapshot(mlpOutput),
      residual2: snapshot(residual2),
      finalNorm: snapshot(finalNorm),
      logits: snapshot(logits),
      probabilities: snapshot(probabilities),
      loss: lossTensor?.data[0] ?? null,
      lossByPosition: lossTensor ? [...lossTensor.lossByPosition] : null,
    };
    const activationArrays = [
      trace.initialRepresentation.data, trace.layerNorm1.data, trace.attentionOutput.data,
      trace.residual1.data, trace.mlpActivation.data, trace.mlpOutput.data,
      trace.residual2.data, trace.finalNorm.data, trace.logits.data,
    ];
    trace.maxAbsoluteActivation = Math.max(...activationArrays.map(maxAbs));
    trace.nanCount = activationArrays.flat().filter((value) => !Number.isFinite(value)).length;
    if (trace.nanCount) throw new Error(`Forward stopped: ${trace.nanCount} non-finite activation(s)`);
    return { trace, logits, probabilities, lossTensor };
  }

  backward(lossTensor) {
    if (!(lossTensor instanceof Tensor)) throw new Error("backward requires a loss Tensor");
    lossTensor.backward();
    this.parameters().forEach(({ name, tensor }) => assertFinite(tensor.grad, `${name}.grad`));
  }

  exportState() {
    return {
      format: "glassbox-ai-ii/model-v1",
      config: { ...this.config },
      vocabulary: [...this.vocabulary],
      seed: this.seed,
      parameters: Object.fromEntries(this.parameters().map(({ name, tensor }) => [name, { shape: [...tensor.shape], data: [...tensor.data] }])),
    };
  }

  importParameters(parameterState) {
    const expected = new Set(this.parameterMap.keys());
    if (!parameterState || typeof parameterState !== "object" || Array.isArray(parameterState)) throw new Error("parameters must be an object");
    if (Object.keys(parameterState).length !== expected.size) throw new Error("parameter count mismatch");
    for (const [name, tensor] of this.parameterMap) {
      const saved = parameterState[name];
      if (!saved || !Array.isArray(saved.shape) || !Array.isArray(saved.data)) throw new Error(`missing parameter ${name}`);
      if (saved.shape.length !== tensor.shape.length || saved.shape.some((value, index) => value !== tensor.shape[index])) throw new Error(`shape mismatch for ${name}`);
      if (saved.data.length !== tensor.data.length) throw new Error(`data length mismatch for ${name}`);
      assertFinite(saved.data, name);
    }
    for (const [name, tensor] of this.parameterMap) tensor.data.splice(0, tensor.data.length, ...parameterState[name].data);
  }
}
