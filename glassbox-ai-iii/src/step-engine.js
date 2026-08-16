import {
  NETWORK_SHAPE,
  OUTPUT_NAMES,
  PARAMETER_SPECS,
  cloneNetwork,
  getParameterValue,
  setParameterValue,
} from './neural-network.js';
import { argmax, crossEntropyLoss, forwardPass, stableSoftmax } from './forward-pass.js';
import { compareTraining, computeGradients, getGradientValue } from './backpropagation.js';

const cloneData = (value) => JSON.parse(JSON.stringify(value));
const numberPart = (value, options = {}) => ({ type: 'number', value, ...options });
const textPart = (value) => ({ type: 'text', value });
const formulaTerm = (status, ...parts) => ({ status, parts });

function emptyMatrix(rows, columns) {
  return Array.from({ length: rows }, () => Array(columns).fill(null));
}

function createEmptyForward(inputs) {
  return {
    inputs: [...inputs],
    inputsCommitted: false,
    hiddenProducts: emptyMatrix(NETWORK_SHAPE.hidden, NETWORK_SHAPE.input),
    hiddenPreActivations: Array(NETWORK_SHAPE.hidden).fill(null),
    hiddenActivations: Array(NETWORK_SHAPE.hidden).fill(null),
    outputProducts: emptyMatrix(NETWORK_SHAPE.output, NETWORK_SHAPE.hidden),
    logits: Array(NETWORK_SHAPE.output).fill(null),
    probabilities: Array(NETWORK_SHAPE.output).fill(null),
    selectedIndex: null,
  };
}

function createEmptyTraining() {
  return {
    targetIndex: null,
    oneHot: null,
    loss: null,
    outputErrors: Array(NETWORK_SHAPE.output).fill(null),
    gradientsHO: emptyMatrix(NETWORK_SHAPE.output, NETWORK_SHAPE.hidden),
    gradientsBO: Array(NETWORK_SHAPE.output).fill(null),
    propagatedHiddenErrors: Array(NETWORK_SHAPE.hidden).fill(null),
    tanhDerivatives: Array(NETWORK_SHAPE.hidden).fill(null),
    hiddenDeltas: Array(NETWORK_SHAPE.hidden).fill(null),
    gradientsIH: emptyMatrix(NETWORK_SHAPE.hidden, NETWORK_SHAPE.input),
    gradientsBH: Array(NETWORK_SHAPE.hidden).fill(null),
    parameterInfo: {},
    beforeForward: null,
    afterForward: null,
    comparison: null,
  };
}

function connectionTerms(layer, targetIndex, currentIndex, inputValues, weights, products, sourcePrefix, weightNames) {
  return inputValues.map((inputValue, sourceIndex) => {
    let status = 'pending';
    if (sourceIndex < currentIndex) status = 'done';
    if (sourceIndex === currentIndex) status = 'current';
    const product = products[sourceIndex];
    const parts = [
      textPart(`${sourcePrefix}${sourceIndex + 1} × ${weightNames(sourceIndex, targetIndex)} = `),
      numberPart(inputValue),
      textPart(' × '),
      numberPart(weights[sourceIndex]),
    ];
    if (product !== null) parts.push(textPart(' = '), numberPart(product));
    return formulaTerm(status, ...parts);
  });
}

function parameterFor(layer, row, column = null) {
  return PARAMETER_SPECS.find(
    (spec) => spec.layer === layer && spec.row === row && spec.column === column,
  );
}

export class StepEngine {
  constructor(network, inputs) {
    this.steps = [];
    this.index = 0;
    this.hasLearningTimeline = false;
    this.preLearningNetwork = null;
    this.learningRate = null;
    this.targetIndex = null;
    this.buildInference(network, inputs);
  }

  buildInference(network, inputs) {
    this.steps = [];
    this.index = 0;
    this.hasLearningTimeline = false;
    this.preLearningNetwork = null;
    const workingNetwork = cloneNetwork(network);
    const complete = forwardPass(workingNetwork, inputs);
    const forward = createEmptyForward(inputs);
    const training = createEmptyTraining();

    const push = (details) => {
      this.steps.push({
        phase: '推論',
        stage: 'ready',
        title: '計算開始前',
        formulaTerms: [],
        explanationKey: 'ready',
        active: {},
        logMessage: null,
        ...details,
        network: cloneNetwork(workingNetwork),
        forward: cloneData(forward),
        training: cloneData(training),
      });
    };

    push({
      phase: '準備',
      title: '計算開始前',
      description: '入力と現在の39パラメータを確認し、「次のステップ」で順伝播を開始します。',
    });

    forward.inputsCommitted = true;
    push({
      stage: 'input',
      title: '入力値を確定',
      description: '5個の実数を入力ベクトルとして固定しました。この値は丸めずに計算へ渡します。',
      explanationKey: 'input',
      formulaTerms: inputs.map((value, input) =>
        formulaTerm('done', textPart(`x${input + 1} = `), numberPart(value)),
      ),
      active: { node: { layer: 'input', index: null } },
      logMessage: '[推論] 入力値を確定',
    });

    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      for (let input = 0; input < NETWORK_SHAPE.input; input += 1) {
        forward.hiddenProducts[hidden][input] = complete.hiddenProducts[hidden][input];
        const spec = parameterFor('IH', hidden, input);
        push({
          stage: 'hidden-product',
          title: `中間ノードH${hidden + 1}への積 ${input + 1}/${NETWORK_SHAPE.input}`,
          description: `入力x${input + 1}に、H${hidden + 1}へ向かう重みを掛けています。`,
          explanationKey: 'weighted-product',
          formulaTerms: connectionTerms(
            'IH',
            hidden,
            input,
            inputs,
            workingNetwork.weightsIH[hidden],
            forward.hiddenProducts[hidden],
            'x',
            (source, target) => `w_I${source + 1}_H${target + 1}`,
          ),
          active: {
            connection: { layer: 'IH', from: input, to: hidden },
            node: { layer: 'hidden', index: hidden },
            parameter: spec.name,
          },
          logMessage: `[推論] x${input + 1} × ${spec.name} = ${complete.hiddenProducts[hidden][input]}`,
        });
      }

      forward.hiddenPreActivations[hidden] = complete.hiddenPreActivations[hidden];
      const sumParts = [textPart(`z_H${hidden + 1} = `)];
      complete.hiddenProducts[hidden].forEach((value, input) => {
        if (input > 0) sumParts.push(textPart(' + '));
        sumParts.push(numberPart(value));
      });
      sumParts.push(
        textPart(' + b = '),
        numberPart(workingNetwork.biasH[hidden]),
        textPart(' を加えて '),
        numberPart(complete.hiddenPreActivations[hidden]),
      );
      push({
        stage: 'hidden-sum',
        title: `H${hidden + 1}の重み付き和とバイアス`,
        description: '5個の積を合計し、中間ノード固有のバイアスを加えました。',
        explanationKey: 'weighted-sum',
        formulaTerms: [formulaTerm('current', ...sumParts)],
        active: {
          node: { layer: 'hidden', index: hidden },
          parameter: parameterFor('BH', hidden).name,
        },
        logMessage: `[推論] H${hidden + 1}の重み付き和 = ${complete.hiddenPreActivations[hidden]}`,
      });

      forward.hiddenActivations[hidden] = complete.hiddenActivations[hidden];
      push({
        stage: 'tanh',
        title: `H${hidden + 1}へtanhを適用`,
        description: '重み付き和を-1から+1の範囲へ滑らかに変換しました。',
        explanationKey: 'activation',
        formulaTerms: [
          formulaTerm(
            'current',
            textPart(`a_H${hidden + 1} = tanh(z_H${hidden + 1}) = tanh(`),
            numberPart(complete.hiddenPreActivations[hidden]),
            textPart(') = '),
            numberPart(complete.hiddenActivations[hidden]),
          ),
        ],
        active: { node: { layer: 'hidden', index: hidden } },
        logMessage: `[推論] H${hidden + 1}のtanh出力 = ${complete.hiddenActivations[hidden]}`,
      });
    }

    for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
      for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
        forward.outputProducts[output][hidden] = complete.outputProducts[output][hidden];
        const spec = parameterFor('HO', output, hidden);
        push({
          stage: 'output-product',
          title: `${OUTPUT_NAMES[output]}への積 ${hidden + 1}/${NETWORK_SHAPE.hidden}`,
          description: `H${hidden + 1}の活性値に、${OUTPUT_NAMES[output]}へ向かう重みを掛けています。`,
          explanationKey: 'weighted-product',
          formulaTerms: connectionTerms(
            'HO',
            output,
            hidden,
            complete.hiddenActivations,
            workingNetwork.weightsHO[output],
            forward.outputProducts[output],
            'a_H',
            (source, target) => `w_H${source + 1}_O${target + 1}`,
          ),
          active: {
            connection: { layer: 'HO', from: hidden, to: output },
            node: { layer: 'output', index: output },
            parameter: spec.name,
          },
          logMessage: `[推論] a_H${hidden + 1} × ${spec.name} = ${complete.outputProducts[output][hidden]}`,
        });
      }

      forward.logits[output] = complete.logits[output];
      const logitParts = [textPart(`logit_O${output + 1} = `)];
      complete.outputProducts[output].forEach((value, hidden) => {
        if (hidden > 0) logitParts.push(textPart(' + '));
        logitParts.push(numberPart(value));
      });
      logitParts.push(
        textPart(' + b = '),
        numberPart(workingNetwork.biasO[output]),
        textPart(' を加えて '),
        numberPart(complete.logits[output]),
      );
      push({
        stage: 'logit',
        title: `${OUTPUT_NAMES[output]}のlogitとバイアス`,
        description: '出力層へ入る4個の積とバイアスから、確率化前の得点を求めました。',
        explanationKey: 'logit',
        formulaTerms: [formulaTerm('current', ...logitParts)],
        active: {
          node: { layer: 'output', index: output },
          parameter: parameterFor('BO', output).name,
        },
        logMessage: `[推論] ${OUTPUT_NAMES[output]}のlogit = ${complete.logits[output]}`,
      });
    }

    forward.probabilities = [...complete.probabilities];
    const maximum = Math.max(...complete.logits);
    const exponentials = complete.logits.map((value) => Math.exp(value - maximum));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    push({
      stage: 'softmax',
      title: 'softmaxで3クラスの確率を計算',
      description: '最大logitを引いてから指数化し、合計で割ることで、合計1の確率へ変換しました。',
      explanationKey: 'softmax',
      formulaTerms: [
        formulaTerm('done', textPart('max(logits) = '), numberPart(maximum)),
        ...complete.logits.map((logit, output) =>
          formulaTerm(
            'done',
            textPart(`exp(logit_O${output + 1} - max) = exp(`),
            numberPart(logit),
            textPart(' - '),
            numberPart(maximum),
            textPart(') = '),
            numberPart(exponentials[output]),
          ),
        ),
        formulaTerm('done', textPart('指数の合計 = '), numberPart(denominator)),
        ...complete.probabilities.map((probability, output) =>
          formulaTerm(
            'current',
            textPart(`P(${OUTPUT_NAMES[output]}) = `),
            numberPart(exponentials[output]),
            textPart(' ÷ '),
            numberPart(denominator),
            textPart(' = '),
            numberPart(probability, { percent: true }),
          ),
        ),
      ],
      active: { node: { layer: 'output', index: null } },
      logMessage: `[推論] softmaxを計算：${complete.probabilities.join(', ')}`,
    });

    forward.selectedIndex = argmax(complete.probabilities);
    push({
      stage: 'argmax',
      title: `選択された判断：${OUTPUT_NAMES[forward.selectedIndex]}`,
      description: '確率的に抽選せず、最大確率のクラスを選ぶargmaxで判断しました。',
      explanationKey: 'argmax',
      formulaTerms: [
        ...complete.probabilities.map((probability, output) =>
          formulaTerm(
            output === forward.selectedIndex ? 'current' : 'done',
            textPart(`${OUTPUT_NAMES[output]} = `),
            numberPart(probability, { percent: true }),
          ),
        ),
        formulaTerm('current', textPart(`argmax(probabilities) = ${OUTPUT_NAMES[forward.selectedIndex]}`)),
      ],
      active: { node: { layer: 'output', index: forward.selectedIndex } },
      logMessage: `[判断] ${OUTPUT_NAMES[forward.selectedIndex]}を選択`,
    });

    this.inferenceEndIndex = this.steps.length - 1;
  }

  appendLearning(targetIndex, learningRate) {
    if (this.index !== this.inferenceEndIndex) {
      throw new Error('順伝播の最後まで進めてから学習を開始してください。');
    }
    if (this.hasLearningTimeline) throw new Error('この履歴には既に学習ステップがあります。');
    if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('学習率は0より大きくしてください。');

    this.hasLearningTimeline = true;
    this.learningRate = learningRate;
    this.targetIndex = targetIndex;
    const starting = this.steps[this.index];
    const workingNetwork = cloneNetwork(starting.network);
    this.preLearningNetwork = cloneNetwork(workingNetwork);
    const beforeForward = forwardPass(workingNetwork, starting.forward.inputs);
    const complete = computeGradients(workingNetwork, starting.forward.inputs, targetIndex, beforeForward);
    const forward = cloneData(starting.forward);
    const training = createEmptyTraining();
    training.targetIndex = targetIndex;
    training.beforeForward = cloneData(beforeForward);

    const push = (details) => {
      this.steps.push({
        phase: '学習',
        stage: 'learning',
        title: '学習',
        formulaTerms: [],
        explanationKey: 'learning',
        active: {},
        logMessage: null,
        ...details,
        network: cloneNetwork(workingNetwork),
        forward: cloneData(forward),
        training: cloneData(training),
      });
    };

    training.oneHot = [...complete.target];
    push({
      stage: 'one-hot',
      title: `正解ラベルをone-hotへ変換：${OUTPUT_NAMES[targetIndex]}`,
      description: '正解だけを1、それ以外を0にした教師ベクトルを作りました。',
      explanationKey: 'one-hot',
      formulaTerms: [formulaTerm('current', textPart(`${OUTPUT_NAMES[targetIndex]} → [${complete.target.join(', ')}]`))],
      active: { node: { layer: 'output', index: targetIndex } },
      logMessage: `[学習] 正解クラスを${OUTPUT_NAMES[targetIndex]}に設定：one-hot = [${complete.target.join(', ')}]`,
    });

    training.loss = complete.loss;
    push({
      stage: 'loss',
      title: 'クロスエントロピー損失を計算',
      description: '正解クラスへ割り当てた確率の負の自然対数を、今回の誤差の大きさとします。',
      explanationKey: 'loss',
      formulaTerms: [
        formulaTerm(
          'current',
          textPart(`L = -log(P(${OUTPUT_NAMES[targetIndex]})) = -log(`),
          numberPart(beforeForward.probabilities[targetIndex]),
          textPart(') = '),
          numberPart(complete.loss),
        ),
      ],
      active: { node: { layer: 'output', index: targetIndex } },
      logMessage: `[学習] 損失を計算：${complete.loss}`,
    });

    for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
      training.outputErrors[output] = complete.outputErrors[output];
      push({
        stage: 'output-error',
        title: `${OUTPUT_NAMES[output]}の出力誤差`,
        description: 'softmaxとクロスエントロピーを組み合わせた微分は、予測確率から教師値を引いた形になります。',
        explanationKey: 'output-error',
        formulaTerms: [
          formulaTerm(
            'current',
            textPart(`δ_O${output + 1} = p_O${output + 1} - y_O${output + 1} = `),
            numberPart(beforeForward.probabilities[output]),
            textPart(' - '),
            numberPart(complete.target[output]),
            textPart(' = '),
            numberPart(complete.outputErrors[output]),
          ),
        ],
        active: { node: { layer: 'output', index: output } },
        logMessage: `[逆伝播] O${output + 1}の誤差 = ${complete.outputErrors[output]}`,
      });
    }

    for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
      for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
        const gradient = complete.weightsHO[output][hidden];
        const spec = parameterFor('HO', output, hidden);
        training.gradientsHO[output][hidden] = gradient;
        training.parameterInfo[spec.name] = { gradient };
        push({
          stage: 'gradient-ho',
          title: `出力層重み ${spec.name} の勾配`,
          description: '出力誤差に、その接続へ入った中間活性値を掛けています。',
          explanationKey: 'gradient',
          formulaTerms: [
            formulaTerm(
              'current',
              textPart(`∂L/∂${spec.name} = δ_O${output + 1} × a_H${hidden + 1} = `),
              numberPart(complete.outputErrors[output]),
              textPart(' × '),
              numberPart(beforeForward.hiddenActivations[hidden]),
              textPart(' = '),
              numberPart(gradient),
            ),
          ],
          active: {
            connection: { layer: 'HO', from: hidden, to: output },
            node: { layer: 'output', index: output },
            parameter: spec.name,
          },
          logMessage: `[逆伝播] ${spec.name}の勾配 = ${gradient}`,
        });
      }
    }

    for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
      const gradient = complete.biasO[output];
      const spec = parameterFor('BO', output);
      training.gradientsBO[output] = gradient;
      training.parameterInfo[spec.name] = { gradient };
      push({
        stage: 'gradient-bo',
        title: `出力バイアス ${spec.name} の勾配`,
        description: 'バイアスは係数1で加算されるため、勾配はその出力誤差と同じです。',
        explanationKey: 'gradient',
        formulaTerms: [
          formulaTerm(
            'current',
            textPart(`∂L/∂${spec.name} = δ_O${output + 1} = `),
            numberPart(gradient),
          ),
        ],
        active: { node: { layer: 'output', index: output }, parameter: spec.name },
        logMessage: `[逆伝播] ${spec.name}の勾配 = ${gradient}`,
      });
    }

    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      training.propagatedHiddenErrors[hidden] = complete.propagatedHiddenErrors[hidden];
      const parts = [textPart(`e_H${hidden + 1} = `)];
      for (let output = 0; output < NETWORK_SHAPE.output; output += 1) {
        if (output > 0) parts.push(textPart(' + '));
        parts.push(
          textPart(`w_H${hidden + 1}_O${output + 1} × δ_O${output + 1} (`),
          numberPart(workingNetwork.weightsHO[output][hidden]),
          textPart(' × '),
          numberPart(complete.outputErrors[output]),
          textPart(')'),
        );
      }
      parts.push(textPart(' = '), numberPart(complete.propagatedHiddenErrors[hidden]));
      push({
        stage: 'propagate-hidden',
        title: `誤差をH${hidden + 1}へ伝播`,
        description: '更新前の出力層重みを通して、3出力の誤差がHノードへ与えた影響を合計します。',
        explanationKey: 'backpropagation',
        formulaTerms: [formulaTerm('current', ...parts)],
        active: { node: { layer: 'hidden', index: hidden } },
        logMessage: `[逆伝播] H${hidden + 1}へ伝播した誤差 = ${complete.propagatedHiddenErrors[hidden]}`,
      });
    }

    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      training.tanhDerivatives[hidden] = complete.tanhDerivatives[hidden];
      training.hiddenDeltas[hidden] = complete.hiddenDeltas[hidden];
      push({
        stage: 'tanh-derivative',
        title: `H${hidden + 1}でtanhの微分を適用`,
        description: '活性化関数の局所的な傾きを掛け、重み付き和に対する誤差へ変換します。',
        explanationKey: 'activation-derivative',
        formulaTerms: [
          formulaTerm(
            'done',
            textPart(`tanh'(z_H${hidden + 1}) = 1 - a_H${hidden + 1}² = 1 - (`),
            numberPart(beforeForward.hiddenActivations[hidden]),
            textPart(')² = '),
            numberPart(complete.tanhDerivatives[hidden]),
          ),
          formulaTerm(
            'current',
            textPart(`δ_H${hidden + 1} = e_H${hidden + 1} × tanh'(z_H${hidden + 1}) = `),
            numberPart(complete.propagatedHiddenErrors[hidden]),
            textPart(' × '),
            numberPart(complete.tanhDerivatives[hidden]),
            textPart(' = '),
            numberPart(complete.hiddenDeltas[hidden]),
          ),
        ],
        active: { node: { layer: 'hidden', index: hidden } },
        logMessage: `[逆伝播] H${hidden + 1}のdelta = ${complete.hiddenDeltas[hidden]}`,
      });
    }

    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      for (let input = 0; input < NETWORK_SHAPE.input; input += 1) {
        const gradient = complete.weightsIH[hidden][input];
        const spec = parameterFor('IH', hidden, input);
        training.gradientsIH[hidden][input] = gradient;
        training.parameterInfo[spec.name] = { gradient };
        push({
          stage: 'gradient-ih',
          title: `入力層重み ${spec.name} の勾配`,
          description: '中間ノードのdeltaに、その接続へ入った入力値を掛けています。',
          explanationKey: 'gradient',
          formulaTerms: [
            formulaTerm(
              'current',
              textPart(`∂L/∂${spec.name} = δ_H${hidden + 1} × x${input + 1} = `),
              numberPart(complete.hiddenDeltas[hidden]),
              textPart(' × '),
              numberPart(starting.forward.inputs[input]),
              textPart(' = '),
              numberPart(gradient),
            ),
          ],
          active: {
            connection: { layer: 'IH', from: input, to: hidden },
            node: { layer: 'hidden', index: hidden },
            parameter: spec.name,
          },
          logMessage: `[逆伝播] ${spec.name}の勾配 = ${gradient}`,
        });
      }
    }

    for (let hidden = 0; hidden < NETWORK_SHAPE.hidden; hidden += 1) {
      const gradient = complete.biasH[hidden];
      const spec = parameterFor('BH', hidden);
      training.gradientsBH[hidden] = gradient;
      training.parameterInfo[spec.name] = { gradient };
      push({
        stage: 'gradient-bh',
        title: `中間バイアス ${spec.name} の勾配`,
        description: '中間バイアスの勾配は、そのHノードのdeltaと同じです。',
        explanationKey: 'gradient',
        formulaTerms: [
          formulaTerm(
            'current',
            textPart(`∂L/∂${spec.name} = δ_H${hidden + 1} = `),
            numberPart(gradient),
          ),
        ],
        active: { node: { layer: 'hidden', index: hidden }, parameter: spec.name },
        logMessage: `[逆伝播] ${spec.name}の勾配 = ${gradient}`,
      });
    }

    const updateOrder = [
      ...PARAMETER_SPECS.filter((spec) => spec.kind === 'weight'),
      ...PARAMETER_SPECS.filter((spec) => spec.kind === 'bias'),
    ];
    const changes = {};
    updateOrder.forEach((spec, updateIndex) => {
      const before = getParameterValue(workingNetwork, spec);
      const gradient = getGradientValue(complete, spec);
      const update = -learningRate * gradient;
      const after = before + update;
      setParameterValue(workingNetwork, spec, after);
      if (updateIndex === updateOrder.length - 1) workingNetwork.learningCount += 1;
      const info = { before, gradient, learningRate, update, after };
      changes[spec.name] = info;
      training.parameterInfo[spec.name] = info;
      const isWeight = spec.kind === 'weight';
      push({
        stage: isWeight ? 'update-weight' : 'update-bias',
        title: `${isWeight ? '重み' : 'バイアス'} ${spec.name} を更新`,
        description: '現在値から「学習率×勾配」を引き、損失が小さくなる方向へパラメータを動かします。',
        explanationKey: 'update',
        formulaTerms: [
          formulaTerm('done', textPart('更新前 = '), numberPart(before)),
          formulaTerm('done', textPart('勾配 = '), numberPart(gradient)),
          formulaTerm('done', textPart('学習率 = '), numberPart(learningRate)),
          formulaTerm(
            'current',
            textPart(`${spec.name}_new = `),
            numberPart(before),
            textPart(' - '),
            numberPart(learningRate),
            textPart(' × '),
            numberPart(gradient),
            textPart(' = '),
            numberPart(after),
          ),
          formulaTerm('current', textPart('更新量 = '), numberPart(update)),
        ],
        active: {
          connection:
            spec.layer === 'IH'
              ? { layer: 'IH', from: spec.column, to: spec.row }
              : spec.layer === 'HO'
                ? { layer: 'HO', from: spec.column, to: spec.row }
                : null,
          node:
            spec.layer === 'BH'
              ? { layer: 'hidden', index: spec.row }
              : spec.layer === 'BO'
                ? { layer: 'output', index: spec.row }
                : null,
          parameter: spec.name,
        },
        logMessage: `[更新] ${spec.name}：${before} → ${after}（勾配 ${gradient}）`,
      });
    });

    const afterForward = forwardPass(workingNetwork, starting.forward.inputs);
    training.afterForward = cloneData(afterForward);
    Object.assign(forward, cloneData(afterForward), { inputsCommitted: true });
    push({
      stage: 'forward-after-update',
      title: '更新後のネットワークで再度順伝播',
      description: '更新済みの全39パラメータを使って同じ入力を計算し直しました。',
      explanationKey: 'recompute',
      formulaTerms: [
        ...afterForward.probabilities.map((probability, output) =>
          formulaTerm(
            output === afterForward.selectedIndex ? 'current' : 'done',
            textPart(`更新後 P(${OUTPUT_NAMES[output]}) = `),
            numberPart(probability, { percent: true }),
          ),
        ),
      ],
      active: { node: { layer: 'output', index: afterForward.selectedIndex } },
      logMessage: `[学習] 更新後に再順伝播：${afterForward.probabilities.join(', ')}`,
    });

    const comparison = compareTraining(beforeForward, afterForward, targetIndex, changes);
    training.comparison = comparison;
    push({
      phase: '比較',
      stage: 'comparison',
      title: '学習前後の出力と損失を比較',
      description: comparison.lossDecreased
        ? '今回の更新では正解確率が上がり、交差エントロピー損失が減少しました。'
        : '今回の設定では損失が減少しませんでした。値を隠さず、そのまま表示しています。',
      explanationKey: 'comparison',
      formulaTerms: [
        formulaTerm(
          'done',
          textPart('学習前の損失 = '),
          numberPart(comparison.beforeLoss),
          textPart('、学習後 = '),
          numberPart(comparison.afterLoss),
        ),
        formulaTerm(
          'current',
          textPart(`${OUTPUT_NAMES[targetIndex]}の確率変化 = `),
          numberPart(comparison.targetProbabilityBefore, { percent: true }),
          textPart(' → '),
          numberPart(comparison.targetProbabilityAfter, { percent: true }),
          textPart('（差 '),
          numberPart(comparison.targetProbabilityChange, { percentagePoint: true }),
          textPart('）'),
        ),
      ],
      active: { node: { layer: 'output', index: targetIndex } },
      logMessage: `[比較] 損失 ${comparison.beforeLoss} → ${comparison.afterLoss}、正解確率差 ${comparison.targetProbabilityChange}`,
    });
  }

  get current() {
    return this.steps[this.index];
  }

  get length() {
    return this.steps.length;
  }

  get canGoPrevious() {
    return this.index > 0;
  }

  get canGoNext() {
    return this.index < this.steps.length - 1;
  }

  get canStartLearning() {
    return this.index === this.inferenceEndIndex && !this.hasLearningTimeline;
  }

  next() {
    if (this.canGoNext) this.index += 1;
    return this.current;
  }

  previous() {
    if (this.canGoPrevious) this.index -= 1;
    return this.current;
  }

  first() {
    this.index = 0;
    return this.current;
  }

  last() {
    this.index = this.steps.length - 1;
    return this.current;
  }

  getUndoNetwork() {
    return this.preLearningNetwork ? cloneNetwork(this.preLearningNetwork) : null;
  }
}

export function runInferenceInOneStep(network, inputs) {
  return forwardPass(network, inputs);
}

export function calculateSoftmaxForStep(logits) {
  return stableSoftmax(logits);
}

export function calculateLossForStep(probabilities, targetIndex) {
  return crossEntropyLoss(probabilities, targetIndex);
}
