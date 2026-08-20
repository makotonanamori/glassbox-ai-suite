import { stableSoftmax } from "../utils/math.js";

export const LANGUAGE_PLAYBACK_DEFAULTS = Object.freeze({
  tokens: 5,
  speed: 2,
  stageDelayMs: 480,
});

export const GENERATION_PHASES = Object.freeze([
  { key: "candidates", label: "候補の確率が出る" },
  { key: "selection", label: "1 Token選ぶ" },
  { key: "append", label: "文の末尾へ足す" },
  { key: "repeat", label: "次の予測へ戻る" },
]);

export function languagePlaybackStageCount(tokenCount = LANGUAGE_PLAYBACK_DEFAULTS.tokens) {
  const count = Number(tokenCount);
  if (!Number.isInteger(count) || count <= 0) throw new RangeError("生成Token数は1以上の整数にしてください。");
  return count * GENERATION_PHASES.length;
}

export function languagePlaybackDelay(speed) {
  const numericSpeed = Number(speed);
  if (!Number.isFinite(numericSpeed) || numericSpeed <= 0) {
    throw new RangeError("言語モデルの再生速度は0より大きい有限数にしてください。");
  }
  return Math.max(16, Math.round(LANGUAGE_PLAYBACK_DEFAULTS.stageDelayMs / numericSpeed));
}

export function generationDistribution(logits, { bosId, temperature = 1 } = {}) {
  if (!Array.isArray(logits) || logits.length === 0 || logits.some((value) => !Number.isFinite(value))) {
    throw new TypeError("有限なlogits配列が必要です。");
  }
  if (!Number.isInteger(bosId) || bosId < 0 || bosId >= logits.length) {
    throw new RangeError("<BOS> Token IDがlogitsの範囲外です。");
  }
  const selectable = [...logits];
  selectable[bosId] = -Infinity;
  return stableSoftmax(selectable, temperature);
}

export function selectGenerationToken(probabilities, { mode = "greedy", draw = 0 } = {}) {
  if (!Array.isArray(probabilities) || probabilities.length === 0 || probabilities.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new TypeError("0以上の有限な確率配列が必要です。");
  }
  if (mode === "greedy") {
    let best = 0;
    probabilities.forEach((value, index) => { if (value > probabilities[best]) best = index; });
    return best;
  }
  if (mode !== "temperature") throw new Error(`未対応のSampling modeです: ${mode}`);
  if (!(draw >= 0 && draw < 1) || !Number.isFinite(draw)) throw new RangeError("乱数は0以上1未満にしてください。");
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i += 1) {
    cumulative += probabilities[i];
    if (draw < cumulative) return i;
  }
  return probabilities.length - 1;
}

export function topGenerationCandidates(probabilities, count = 5) {
  return probabilities
    .map((probability, id) => ({ id, probability }))
    .sort((a, b) => b.probability - a.probability || a.id - b.id)
    .slice(0, Math.max(1, Math.floor(count)));
}

export function advanceGenerationContext(contextIds, nextId, contextLength) {
  if (!Array.isArray(contextIds) || contextIds.some((id) => !Number.isInteger(id))) {
    throw new TypeError("ContextにはToken IDの配列が必要です。");
  }
  if (!Number.isInteger(nextId) || !Number.isInteger(contextLength) || contextLength <= 0) {
    throw new RangeError("次Token IDと正のContext Lengthが必要です。");
  }
  const expanded = [...contextIds, nextId];
  const overflow = Math.max(0, expanded.length - contextLength);
  return {
    contextIds: expanded.slice(-contextLength),
    droppedIds: expanded.slice(0, overflow),
  };
}
