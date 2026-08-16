import {
  PARAMETER_SPECS,
  cloneNetwork,
  getParameterValue,
  setParameterValue,
} from './neural-network.js';
import { getGradientValue } from './backpropagation.js';
import { forwardPass } from './forward-pass.js';
import { createSeededRandom } from './seeded-random.js';
import { cloneGridWorld, senseGridWorld } from './grid-world.js';
import {
  applyActionWithRewardProfile,
  diagnoseEpisode,
  getRewardProfile,
} from './reward-profiles.js';
import {
  aggregatePolicyGradients,
  discountedReturns,
  samplePolicy,
  temperaturePolicy,
} from './policy-gradient.js';

export const RL_DEFAULTS = Object.freeze({
  gamma: 0.9,
  temperature: 1.2,
  learningRate: 0.05,
  maxSteps: 16,
  randomSeed: 'rl-1',
  rewardProfile: 'balanced',
  episodeNumber: 1,
});

function validateConfig(options) {
  const config = { ...RL_DEFAULTS, ...options };
  if (!Number.isFinite(config.gamma) || config.gamma < 0 || config.gamma > 1) {
    throw new Error('割引率γは0以上1以下にしてください。');
  }
  if (!Number.isFinite(config.temperature) || config.temperature <= 0) {
    throw new Error('方策温度τは0より大きくしてください。');
  }
  if (!Number.isFinite(config.learningRate) || config.learningRate <= 0) {
    throw new Error('強化学習率は0より大きくしてください。');
  }
  if (!Number.isInteger(config.maxSteps) || config.maxSteps < 1 || config.maxSteps > 60) {
    throw new Error('最大環境ステップは1以上60以下の整数にしてください。');
  }
  if (!Number.isInteger(config.episodeNumber) || config.episodeNumber < 1) {
    throw new Error('エピソード番号は1以上の整数である必要があります。');
  }
  getRewardProfile(config.rewardProfile);
  config.randomSeed = String(config.randomSeed || RL_DEFAULTS.randomSeed);
  return config;
}

function cloneExperience(experience) {
  return {
    ...experience,
    inputs: [...experience.inputs],
    logits: [...experience.logits],
    probabilities: [...experience.probabilities],
    beforeWorld: cloneGridWorld(experience.beforeWorld),
    afterWorld: cloneGridWorld(experience.afterWorld),
    forward: {
      ...experience.forward,
      inputs: [...experience.forward.inputs],
      hiddenProducts: experience.forward.hiddenProducts.map((row) => [...row]),
      hiddenPreActivations: [...experience.forward.hiddenPreActivations],
      hiddenActivations: [...experience.forward.hiddenActivations],
      outputProducts: experience.forward.outputProducts.map((row) => [...row]),
      logits: [...experience.forward.logits],
      probabilities: [...experience.forward.probabilities],
    },
  };
}

function cloneGradient(gradient) {
  if (!gradient) return null;
  return {
    weightsIH: gradient.weightsIH.map((row) => [...row]),
    biasH: [...gradient.biasH],
    weightsHO: gradient.weightsHO.map((row) => [...row]),
    biasO: [...gradient.biasO],
  };
}

function stateKey(world) {
  return [
    world.agent.row,
    world.agent.column,
    world.agent.direction,
    world.food.row,
    world.food.column,
  ].join(',');
}

function cloneParameterInfo(parameterInfo) {
  return Object.fromEntries(
    Object.entries(parameterInfo).map(([name, info]) => [name, { ...info }]),
  );
}

function maximumGradient(gradient) {
  return Math.max(...PARAMETER_SPECS.map((spec) => Math.abs(getGradientValue(gradient, spec))));
}

function gradientNorm(gradient) {
  return Math.sqrt(
    PARAMETER_SPECS.reduce((sum, spec) => {
      const value = getGradientValue(gradient, spec);
      return sum + value * value;
    }, 0),
  );
}

export function buildReinforcementTimeline(network, initialWorld, options = {}) {
  const config = validateConfig(options);
  const frozenNetwork = cloneNetwork(network);
  const startingWorld = cloneGridWorld(initialWorld);
  const random = createSeededRandom(`${config.randomSeed}:episode:${config.episodeNumber}`);
  const profile = getRewardProfile(config.rewardProfile);
  const steps = [];
  const experiences = [];
  let world = cloneGridWorld(startingWorld);
  let cumulativeReward = 0;
  let explorationCount = 0;
  let exploitationCount = 0;

  const pushStep = ({
    stage,
    title,
    description,
    explanation,
    logMessage,
    details = {},
    stepNetwork = frozenNetwork,
    stepWorld = world,
    stepExperiences = experiences,
    returns = [],
    aggregateGradient = null,
    parameterInfo = {},
    active = {},
  }) => {
    steps.push({
      phase: '強化学習',
      stage,
      title,
      description,
      explanation,
      logMessage,
      details,
      network: cloneNetwork(stepNetwork),
      world: cloneGridWorld(stepWorld),
      experiences: stepExperiences.map(cloneExperience),
      returns: [...returns],
      cumulativeReward,
      explorationCount,
      exploitationCount,
      aggregateGradient: cloneGradient(aggregateGradient),
      parameterInfo: cloneParameterInfo(parameterInfo),
      active: { ...active },
    });
  };

  pushStep({
    stage: 'rl-ready',
    title: `エピソード${config.episodeNumber}開始前`,
    description: `${profile.label}、最大${config.maxSteps}環境ステップで方策を固定して経験を集めます。`,
    explanation: 'まだ環境遷移はありません。次のステップで現在世界を5センサーへ変換します。',
    logMessage: `[RL] エピソード${config.episodeNumber}を準備`,
  });

  let doneReason = 'maximum-steps';
  for (let time = 0; time < config.maxSteps; time += 1) {
    const sensed = senseGridWorld(world);
    const inputs = [...sensed.values];
    const forward = forwardPass(frozenNetwork, inputs);
    const policy = temperaturePolicy(forward.logits, config.temperature);
    const randomValue = random.next();
    const sample = samplePolicy(policy.probabilities, randomValue);
    const exploration = sample.actionIndex !== policy.greedyActionIndex;

    pushStep({
      stage: 'rl-observation',
      title: `時刻t=${time}：状態を観測`,
      description: '世界の位置、向き、餌、前方障害を5個の実数へ変換します。',
      explanation: 'この5値だけがネットワークへ渡ります。安全教師の経路情報は使いません。',
      logMessage: `[RL] t=${time} 状態を5センサーへ変換`,
      details: { time, inputs, sensed },
      active: { experienceIndex: time },
    });

    pushStep({
      stage: 'rl-policy',
      title: `時刻t=${time}：温度付き確率方策`,
      description: `3logitを温度τ=${config.temperature}で割り、行動確率を計算します。`,
      explanation: '温度が高いほど確率差が小さくなり、低いほど最大logitの行動へ集中します。',
      logMessage: `[RL] t=${time} 温度付きsoftmaxを計算`,
      details: { time, inputs, forward, policy },
      active: { experienceIndex: time },
    });

    pushStep({
      stage: 'rl-sample',
      title: `時刻t=${time}：行動をサンプリング`,
      description: `シード付き乱数u=${randomValue}が入る累積確率区間を選びます。`,
      explanation: exploration
        ? '選択行動はargmaxと異なるため、この遷移を「探索」と記録します。'
        : '選択行動はargmaxと同じため、この遷移を「活用」と記録します。',
      logMessage: `[RL] t=${time} 行動${sample.actionIndex}を${exploration ? '探索' : '活用'}として選択`,
      details: { time, inputs, forward, policy, sample, exploration },
      active: { experienceIndex: time, actionIndex: sample.actionIndex },
    });

    const beforeWorld = cloneGridWorld(world);
    const nextWorld = applyActionWithRewardProfile(world, sample.actionIndex, config.rewardProfile);
    const transition = nextWorld.lastTransition;
    cumulativeReward += transition.reward;
    if (exploration) explorationCount += 1;
    else exploitationCount += 1;
    const experience = {
      time,
      stateKey: stateKey(beforeWorld),
      inputs,
      forward,
      logits: [...forward.logits],
      probabilities: [...policy.probabilities],
      entropy: policy.entropy,
      greedyActionIndex: policy.greedyActionIndex,
      actionIndex: sample.actionIndex,
      exploration,
      randomValue,
      intervalLower: sample.lower,
      intervalUpper: sample.upper,
      reward: transition.reward,
      cumulativeReward,
      baseReward: transition.baseReward,
      rewardCategory: transition.rewardCategory,
      event: transition.event,
      beforeWorld,
      afterWorld: cloneGridWorld(nextWorld),
    };
    experiences.push(experience);
    world = nextWorld;

    pushStep({
      stage: 'rl-transition',
      title: `時刻t=${time}：環境遷移と報酬`,
      description: transition.event,
      explanation: `報酬プリセット「${profile.shortLabel}」の${transition.rewardCategory}値を実際の遷移へ適用しました。`,
      logMessage: `[RL] t=${time} ${transition.event} 報酬${transition.reward}`,
      details: { time, experience, transition, cumulativeReward },
      active: { experienceIndex: time, actionIndex: sample.actionIndex },
    });

    if (world.counters.foods > startingWorld.counters.foods) {
      doneReason = 'food';
      break;
    }
  }

  const returns = discountedReturns(experiences.map((experience) => experience.reward), config.gamma);
  const revealedReturns = Array(returns.length).fill(null);
  for (let index = returns.length - 1; index >= 0; index -= 1) {
    revealedReturns[index] = returns[index];
    pushStep({
      stage: 'rl-return',
      title: `時刻t=${index}：割引リターンを逆算`,
      description: `r_${index}以降の報酬を割引率γ=${config.gamma}で合計します。`,
      explanation: '正のリターンは選択行動を強め、負のリターンは選択行動を弱める方向の勾配になります。',
      logMessage: `[RL] t=${index} 割引リターンG=${returns[index]}を計算`,
      details: {
        time: index,
        reward: experiences[index].reward,
        futureReturn: index + 1 < returns.length ? returns[index + 1] : 0,
        returnToGo: returns[index],
        gamma: config.gamma,
      },
      returns: revealedReturns,
      active: { experienceIndex: index },
    });
  }

  const aggregated = aggregatePolicyGradients(
    frozenNetwork,
    experiences,
    returns,
    config.temperature,
  );
  for (let index = 0; index < aggregated.individual.length; index += 1) {
    const gradient = aggregated.individual[index];
    pushStep({
      stage: 'rl-gradient',
      title: `時刻t=${index}：方策勾配を計算`,
      description: `G_${index}と選択行動の対数確率から全39パラメータの勾配を求めます。`,
      explanation: '行動生成時と同じ固定ネットワーク、入力、温度付き確率を使って連鎖律を適用します。',
      logMessage: `[RL] t=${index} 方策損失${gradient.policyLoss}の全39勾配を計算`,
      details: { time: index, gradient },
      returns,
      active: { experienceIndex: index, actionIndex: experiences[index].actionIndex },
    });
  }

  pushStep({
    stage: 'rl-aggregate',
    title: 'エピソード平均勾配を確定',
    description: `${experiences.length}遷移の39勾配をパラメータごとに平均します。`,
    explanation: 'エピソード長で平均するため、最大ステップ数だけで更新量が比例拡大しないようにします。',
    logMessage: `[RL] ${experiences.length}遷移の平均方策勾配を確定`,
    details: {
      experienceCount: experiences.length,
      gradientNorm: gradientNorm(aggregated.average),
      maximumGradient: maximumGradient(aggregated.average),
    },
    returns,
    aggregateGradient: aggregated.average,
  });

  const workingNetwork = cloneNetwork(frozenNetwork);
  const parameterInfo = {};
  for (let index = 0; index < PARAMETER_SPECS.length; index += 1) {
    const spec = PARAMETER_SPECS[index];
    const before = getParameterValue(workingNetwork, spec);
    const gradient = getGradientValue(aggregated.average, spec);
    const update = -config.learningRate * gradient;
    const after = before + update;
    setParameterValue(workingNetwork, spec, after);
    parameterInfo[spec.name] = {
      before,
      gradient,
      learningRate: config.learningRate,
      update,
      after,
    };
    if (index === PARAMETER_SPECS.length - 1) workingNetwork.learningCount += 1;
    pushStep({
      stage: 'rl-update',
      title: `${spec.name}を方策勾配で更新`,
      description: `更新前、平均勾配、強化学習率、更新量、更新後を同じ式で確認します。`,
      explanation: '正の勾配は値を減らし、負の勾配は値を増やします。前のステップで部分更新前へ戻れます。',
      logMessage: `[RL更新] ${spec.name}: ${before} → ${after}`,
      details: { spec, ...parameterInfo[spec.name] },
      stepNetwork: workingNetwork,
      returns,
      aggregateGradient: aggregated.average,
      parameterInfo,
      active: { parameter: spec.name },
    });
  }

  const firstExperience = experiences[0];
  const afterForward = forwardPass(workingNetwork, firstExperience.inputs);
  const afterPolicy = temperaturePolicy(afterForward.logits, config.temperature);
  const counters = {
    foods: world.counters.foods - startingWorld.counters.foods,
    collisions: world.counters.collisions - startingWorld.counters.collisions,
    dangerHits: world.counters.dangerHits - startingWorld.counters.dangerHits,
  };
  const changeValues = Object.values(parameterInfo).map((info) => Math.abs(info.update));
  const summaryBase = {
    episodeNumber: config.episodeNumber,
    rewardProfile: config.rewardProfile,
    steps: experiences.length,
    doneReason,
    cumulativeReward,
    ...counters,
    explorationCount,
    exploitationCount,
    averageEntropy: experiences.reduce((sum, experience) => sum + experience.entropy, 0) / experiences.length,
    gradientNorm: gradientNorm(aggregated.average),
    maximumGradient: maximumGradient(aggregated.average),
    changedParameterCount: changeValues.filter((value) => value > 0).length,
    maximumAbsoluteChange: changeValues.length ? Math.max(...changeValues) : 0,
    beforePolicy: [...firstExperience.probabilities],
    afterPolicy: [...afterPolicy.probabilities],
    firstActionIndex: firstExperience.actionIndex,
  };
  const diagnostic = diagnoseEpisode(summaryBase, experiences, config.rewardProfile);
  const summary = { ...summaryBase, diagnostic };

  pushStep({
    stage: 'rl-comparison',
    title: `エピソード${config.episodeNumber}の更新完了`,
    description: `累積報酬${cumulativeReward}、餌${counters.foods}、全39パラメータ更新を比較します。`,
    explanation: '一回のエピソード更新で累積報酬が必ず改善する保証はありません。次のエピソード履歴で傾向を観察します。',
    logMessage: `[RL] エピソード${config.episodeNumber}完了：累積報酬${cumulativeReward}、餌${counters.foods}`,
    details: { summary, afterForward, afterPolicy },
    stepNetwork: workingNetwork,
    returns,
    aggregateGradient: aggregated.average,
    parameterInfo,
  });

  return {
    config,
    profile,
    steps,
    experiences: experiences.map(cloneExperience),
    returns,
    averageGradient: cloneGradient(aggregated.average),
    summary,
    finalNetwork: cloneNetwork(workingNetwork),
    initialWorld: cloneGridWorld(startingWorld),
    finalWorld: cloneGridWorld(world),
  };
}

export class ReinforcementStepEngine {
  constructor(network, initialWorld, options = {}) {
    const built = buildReinforcementTimeline(network, initialWorld, options);
    Object.assign(this, built);
    this.index = 0;
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

  get isComplete() {
    return !this.canGoNext;
  }

  first() {
    this.index = 0;
    return this.current;
  }

  previous() {
    if (this.canGoPrevious) this.index -= 1;
    return this.current;
  }

  next() {
    if (this.canGoNext) this.index += 1;
    return this.current;
  }

  runToEnd() {
    this.index = this.steps.length - 1;
    return this.current;
  }
}
