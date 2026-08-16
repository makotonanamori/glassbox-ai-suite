import { TinyTransformer } from "../model/transformer.js";
import { SimpleTokenizer, SPECIAL_TOKENS } from "../model/tokenizer.js";
import { LanguageDataset } from "../training/dataset.js";
import { Trainer } from "../training/trainer.js";

export function exportApplicationState(model, trainer, dataset) {
  return JSON.stringify({
    format: "glassbox-ai-ii/application-v1",
    createdAt: new Date().toISOString(),
    model: model.exportState(),
    corpus: [...dataset.corpus],
    training: {
      step: trainer.step,
      learningRate: trainer.learningRate,
      clipNorm: trainer.clipNorm,
      seed: trainer.seed,
      lossHistory: trainer.lossHistory.map((entry) => ({ ...entry })),
    },
  }, null, 2);
}

function validateApplicationState(state) {
  if (!state || state.format !== "glassbox-ai-ii/application-v1") throw new Error("Unsupported or missing application format");
  if (!state.model || state.model.format !== "glassbox-ai-ii/model-v1") throw new Error("Unsupported or missing model format");
  if (!Array.isArray(state.model.vocabulary) || SPECIAL_TOKENS.some((token, index) => state.model.vocabulary[index] !== token)) throw new Error("Invalid vocabulary");
  if (!Array.isArray(state.corpus) || !state.corpus.length || state.corpus.some((line) => typeof line !== "string")) throw new Error("Invalid corpus");
  if (!state.training || !Number.isInteger(state.training.step) || state.training.step < 0) throw new Error("Invalid training step");
  if (!(state.training.learningRate > 0) || !(state.training.clipNorm > 0)) throw new Error("Invalid training settings");
  if (!Array.isArray(state.training.lossHistory) || state.training.lossHistory.some((item) => !Number.isInteger(item.step) || !Number.isFinite(item.loss))) throw new Error("Invalid loss history");
}

export function importApplicationState(json) {
  let state;
  try {
    state = typeof json === "string" ? JSON.parse(json) : json;
  } catch (error) {
    throw new Error(`Invalid JSON: ${error.message}`);
  }
  validateApplicationState(state);
  const tokenizer = new SimpleTokenizer(state.corpus, state.model.vocabulary);
  const dataset = new LanguageDataset(state.corpus, state.model.config.contextLength, tokenizer);
  const model = new TinyTransformer(state.model.vocabulary, { seed: state.model.seed, config: state.model.config });
  model.importParameters(state.model.parameters);
  const trainer = new Trainer(model, dataset, {
    learningRate: state.training.learningRate,
    clipNorm: state.training.clipNorm,
    seed: state.training.seed,
  });
  trainer.step = state.training.step;
  trainer.lossHistory = state.training.lossHistory.map((entry) => ({ ...entry }));
  trainer.snapshots.clear();
  trainer.captureSnapshot(`IMPORTED STEP ${trainer.step}`);
  return { model, trainer, dataset, tokenizer };
}
