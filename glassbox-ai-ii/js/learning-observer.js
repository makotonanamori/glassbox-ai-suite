import {
  advanceGenerationContext, generationDistribution, selectGenerationToken, topGenerationCandidates,
} from "./auto-playback.js";

export const LEARNING_OBSERVER_DEFAULTS = Object.freeze({
  prompt: "the cat eats",
  targetToken: "fish",
  totalSteps: 500,
  checkpointEvery: 50,
  generationTokens: 5,
  speed: 2,
  checkpointDelayMs: 420,
});

export function learningCheckpoints(totalSteps = LEARNING_OBSERVER_DEFAULTS.totalSteps, checkpointEvery = LEARNING_OBSERVER_DEFAULTS.checkpointEvery) {
  if (!Number.isInteger(totalSteps) || totalSteps <= 0) throw new RangeError("Training Step数は1以上の整数にしてください。");
  if (!Number.isInteger(checkpointEvery) || checkpointEvery <= 0) throw new RangeError("Checkpoint間隔は1以上の整数にしてください。");
  const checkpoints = [];
  for (let step = Math.min(checkpointEvery, totalSteps); step < totalSteps; step += checkpointEvery) checkpoints.push(step);
  checkpoints.push(totalSteps);
  return checkpoints;
}

export function learningObserverDelay(speed) {
  const numericSpeed = Number(speed);
  if (!Number.isFinite(numericSpeed) || numericSpeed <= 0) throw new RangeError("再生速度は0より大きくしてください。");
  return Math.max(16, Math.round(LEARNING_OBSERVER_DEFAULTS.checkpointDelayMs / numericSpeed));
}

function nextDistribution(model, tokenizer, contextIds) {
  const trace = model.forward(contextIds).trace;
  const [rows, cols] = trace.logits.shape;
  const logits = trace.logits.data.slice((rows - 1) * cols, rows * cols);
  return generationDistribution(logits, { bosId: tokenizer.tokenToId.get("<BOS>") });
}

export function captureLearningObservation(model, tokenizer, {
  prompt = LEARNING_OBSERVER_DEFAULTS.prompt,
  targetToken = LEARNING_OBSERVER_DEFAULTS.targetToken,
  generationTokens = LEARNING_OBSERVER_DEFAULTS.generationTokens,
} = {}) {
  if (!model || !tokenizer) throw new TypeError("ModelとTokenizerが必要です。");
  const targetId = tokenizer.tokenToId.get(targetToken);
  if (!Number.isInteger(targetId)) throw new Error(`Vocabularyに対象Tokenがありません: ${targetToken}`);
  if (!Number.isInteger(generationTokens) || generationTokens <= 0) throw new RangeError("生成Token数は1以上の整数にしてください。");

  let contextIds = tokenizer.encode(prompt, { addBos: true }).slice(-model.config.contextLength);
  if (!contextIds.length) contextIds = [tokenizer.tokenToId.get("<BOS>")];
  const promptTokens = tokenizer.tokenize(prompt);
  const probabilities = nextDistribution(model, tokenizer, contextIds);
  const firstSelectedId = selectGenerationToken(probabilities, { mode: "greedy" });
  const visibleTokens = [...promptTokens];
  const generatedTokens = [];
  const eosId = tokenizer.tokenToId.get("<EOS>");

  for (let index = 0; index < generationTokens; index += 1) {
    const distribution = index === 0 ? probabilities : nextDistribution(model, tokenizer, contextIds);
    const nextId = selectGenerationToken(distribution, { mode: "greedy" });
    if (nextId === eosId) break;
    const token = tokenizer.vocabulary[nextId];
    generatedTokens.push(token);
    visibleTokens.push(token);
    contextIds = advanceGenerationContext(contextIds, nextId, model.config.contextLength).contextIds;
  }

  return {
    prompt,
    promptTokens,
    targetToken,
    targetId,
    targetProbability: probabilities[targetId],
    probabilities,
    candidates: topGenerationCandidates(probabilities, 5),
    selectedId: firstSelectedId,
    selectedToken: tokenizer.vocabulary[firstSelectedId],
    generatedTokens,
    generatedText: tokenizer.detokenize(visibleTokens),
  };
}
