import { createSeededRandom } from './seeded-random.js';

export const NETWORK_SHAPE = Object.freeze({ input: 5, hidden: 4, output: 3 });
export const INPUT_NAMES = Object.freeze(['入力1', '入力2', '入力3', '入力4', '入力5']);
export const OUTPUT_NAMES = Object.freeze(['行動A', '行動B', '行動C']);

export function cloneNetwork(network) {
  return {
    structure: { ...network.structure },
    seed: String(network.seed),
    weightsIH: network.weightsIH.map((row) => [...row]),
    biasH: [...network.biasH],
    weightsHO: network.weightsHO.map((row) => [...row]),
    biasO: [...network.biasO],
    learningCount: Number(network.learningCount ?? 0),
  };
}

export function createNetwork(seed = 'glassbox-1') {
  const random = createSeededRandom(seed);
  const weightsIH = Array.from({ length: NETWORK_SHAPE.hidden }, () =>
    Array.from({ length: NETWORK_SHAPE.input }, () => random.uniform(-0.45, 0.45)),
  );
  const biasH = Array.from({ length: NETWORK_SHAPE.hidden }, () => random.uniform(-0.1, 0.1));
  const weightsHO = Array.from({ length: NETWORK_SHAPE.output }, () =>
    Array.from({ length: NETWORK_SHAPE.hidden }, () => random.uniform(-0.45, 0.45)),
  );
  const biasO = Array.from({ length: NETWORK_SHAPE.output }, () => random.uniform(-0.1, 0.1));

  return {
    structure: { ...NETWORK_SHAPE },
    seed: String(seed),
    weightsIH,
    biasH,
    weightsHO,
    biasO,
    learningCount: 0,
  };
}

export function getParameterSpecs() {
  const specs = [];

  for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
    for (let input = 0; input < NETWORK_SHAPE.input; input += 1) {
      specs.push({
        name: `w_I${input + 1}_H${hidden + 1}`,
        kind: 'weight',
        layer: 'IH',
        row: hidden,
        column: input,
        from: `I${input + 1}`,
        to: `H${hidden + 1}`,
      });
    }
  }

  for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
    specs.push({
      name: `b_H${hidden + 1}`,
      kind: 'bias',
      layer: 'BH',
      row: hidden,
      column: null,
      from: 'バイアス',
      to: `H${hidden + 1}`,
    });
  }

  for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      specs.push({
        name: `w_H${hidden + 1}_O${output + 1}`,
        kind: 'weight',
        layer: 'HO',
        row: output,
        column: hidden,
        from: `H${hidden + 1}`,
        to: `O${output + 1}`,
      });
    }
  }

  for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
    specs.push({
      name: `b_O${output + 1}`,
      kind: 'bias',
      layer: 'BO',
      row: output,
      column: null,
      from: 'バイアス',
      to: `O${output + 1}`,
    });
  }

  return specs;
}

export const PARAMETER_SPECS = Object.freeze(getParameterSpecs().map((spec) => Object.freeze(spec)));

export function getParameterValue(network, spec) {
  if (spec.layer === 'IH') return network.weightsIH[spec.row][spec.column];
  if (spec.layer === 'BH') return network.biasH[spec.row];
  if (spec.layer === 'HO') return network.weightsHO[spec.row][spec.column];
  if (spec.layer === 'BO') return network.biasO[spec.row];
  throw new Error(`未知のパラメータ層です: ${spec.layer}`);
}

export function setParameterValue(network, spec, value) {
  if (spec.layer === 'IH') network.weightsIH[spec.row][spec.column] = value;
  else if (spec.layer === 'BH') network.biasH[spec.row] = value;
  else if (spec.layer === 'HO') network.weightsHO[spec.row][spec.column] = value;
  else if (spec.layer === 'BO') network.biasO[spec.row] = value;
  else throw new Error(`未知のパラメータ層です: ${spec.layer}`);
}

export function findParameterSpec(name) {
  return PARAMETER_SPECS.find((spec) => spec.name === name) ?? null;
}
