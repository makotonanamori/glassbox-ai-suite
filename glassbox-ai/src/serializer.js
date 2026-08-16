import { NETWORK_SHAPE, cloneNetwork } from './neural-network.js';

export const STATE_FORMAT = 'glassbox-ai-state';
export const STATE_VERSION = 1;

function assertFiniteArray(value, length, label) {
  if (!Array.isArray(value) || value.length !== length || !value.every(Number.isFinite)) {
    throw new Error(`${label}は有限数${length}個の配列である必要があります。`);
  }
}

function assertMatrix(value, rows, columns, label) {
  if (!Array.isArray(value) || value.length !== rows) {
    throw new Error(`${label}の行数が固定構造と一致しません。`);
  }
  value.forEach((row, index) => assertFiniteArray(row, columns, `${label}[${index}]`));
}

export function createStatePayload({ network, learningRate, inputs, targetIndex }) {
  return {
    format: STATE_FORMAT,
    version: STATE_VERSION,
    savedAt: new Date().toISOString(),
    structure: { ...network.structure },
    network: cloneNetwork(network),
    learningRate,
    seed: String(network.seed),
    inputs: [...inputs],
    targetIndex,
    learningCount: Number(network.learningCount),
  };
}

export function validateStatePayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('保存データの最上位はオブジェクトである必要があります。');
  }
  if (payload.format !== STATE_FORMAT || payload.version !== STATE_VERSION) {
    throw new Error('Glassbox AI初版の保存形式ではありません。');
  }
  const expected = NETWORK_SHAPE;
  const structure = payload.structure;
  if (!structure || structure.input !== expected.input || structure.hidden !== expected.hidden || structure.output !== expected.output) {
    throw new Error('保存データのネットワーク構造が5入力・4中間・3出力ではありません。');
  }
  const network = payload.network;
  if (!network || typeof network !== 'object') throw new Error('networkがありません。');
  if (
    !network.structure ||
    network.structure.input !== expected.input ||
    network.structure.hidden !== expected.hidden ||
    network.structure.output !== expected.output
  ) {
    throw new Error('network内の構造情報が固定構造と一致しません。');
  }
  assertMatrix(network.weightsIH, expected.hidden, expected.input, 'weightsIH');
  assertFiniteArray(network.biasH, expected.hidden, 'biasH');
  assertMatrix(network.weightsHO, expected.output, expected.hidden, 'weightsHO');
  assertFiniteArray(network.biasO, expected.output, 'biasO');
  if (String(network.seed) !== String(payload.seed)) throw new Error('シード情報が一致しません。');
  if (!Number.isInteger(payload.learningCount) || payload.learningCount < 0) {
    throw new Error('学習回数は0以上の整数である必要があります。');
  }
  if (network.learningCount !== payload.learningCount) {
    throw new Error('network内外の学習回数が一致しません。');
  }
  if (!Number.isFinite(payload.learningRate) || payload.learningRate <= 0) {
    throw new Error('学習率は0より大きい有限数である必要があります。');
  }
  assertFiniteArray(payload.inputs, expected.input, 'inputs');
  if (!payload.inputs.every((value) => value >= -1 && value <= 1)) {
    throw new Error('入力値は-1.0から+1.0の範囲である必要があります。');
  }
  if (!Number.isInteger(payload.targetIndex) || payload.targetIndex < 0 || payload.targetIndex >= expected.output) {
    throw new Error('正解クラスが範囲外です。');
  }
  return true;
}

export function parseStateJson(jsonText) {
  let payload;
  try {
    payload = JSON.parse(jsonText);
  } catch (error) {
    throw new Error(`JSONを解析できません: ${error.message}`);
  }
  validateStatePayload(payload);
  const network = cloneNetwork(payload.network);
  network.structure = { ...NETWORK_SHAPE };
  network.seed = String(payload.seed);
  network.learningCount = payload.learningCount;
  return {
    network,
    learningRate: payload.learningRate,
    inputs: [...payload.inputs],
    targetIndex: payload.targetIndex,
  };
}

export function serializeState(state) {
  return JSON.stringify(createStatePayload(state), null, 2);
}
