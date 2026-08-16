import { applyGridAction } from './grid-world.js';

export const REWARD_PROFILES = Object.freeze({
  balanced: Object.freeze({
    key: 'balanced',
    label: '正常：目的と報酬を対応',
    shortLabel: '正常',
    description: '餌を正報酬、危険・衝突・無駄な移動を負報酬にします。',
    rewards: Object.freeze({ move: -0.01, turn: -0.02, collision: -0.25, danger: -1, food: 1 }),
  }),
  sparse: Object.freeze({
    key: 'sparse',
    label: '疎な報酬：餌だけを評価',
    shortLabel: '疎な報酬',
    description: '餌以外を0にし、成功前に学習信号が得られない停滞を観察します。',
    rewards: Object.freeze({ move: 0, turn: 0, collision: 0, danger: 0, food: 1 }),
  }),
  'collision-bonus': Object.freeze({
    key: 'collision-bonus',
    label: '設計失敗：衝突へボーナス',
    shortLabel: '衝突ボーナス',
    description: '壁への衝突を正報酬にし、目的と評価指標の不一致を観察します。',
    rewards: Object.freeze({ move: -0.01, turn: -0.02, collision: 0.3, danger: -1, food: 1 }),
  }),
  'turn-bonus': Object.freeze({
    key: 'turn-bonus',
    label: '報酬ハッキング：旋回へボーナス',
    shortLabel: '旋回ボーナス',
    description: '左右旋回へ正報酬を与え、移動せず得点する抜け道を観察します。',
    rewards: Object.freeze({ move: -0.01, turn: 0.08, collision: -0.25, danger: -1, food: 1 }),
  }),
});

export function getRewardProfile(profileKey) {
  const profile = REWARD_PROFILES[profileKey];
  if (!profile) throw new Error(`未知の報酬プリセットです: ${profileKey}`);
  return profile;
}

export function rewardCategory(transition) {
  if (transition.kind === 'turn-left' || transition.kind === 'turn-right') return 'turn';
  if (['move', 'collision', 'danger', 'food'].includes(transition.kind)) return transition.kind;
  throw new Error(`報酬へ変換できない環境遷移です: ${transition.kind}`);
}

export function applyActionWithRewardProfile(world, actionIndex, profileKey = 'balanced') {
  const profile = getRewardProfile(profileKey);
  const next = applyGridAction(world, actionIndex);
  const category = rewardCategory(next.lastTransition);
  const baseReward = next.lastTransition.reward;
  const reward = profile.rewards[category];
  next.lastTransition = {
    ...next.lastTransition,
    baseReward,
    reward,
    rewardCategory: category,
    rewardProfile: profile.key,
  };
  return next;
}

function ratio(part, whole) {
  return whole ? part / whole : 0;
}

function displayNumber(value) {
  return Number(value.toFixed(6));
}

export function diagnoseEpisode(summary, experiences, profileKey) {
  const profile = getRewardProfile(profileKey);
  const steps = experiences.length;
  if (!steps) {
    return { key: 'none', label: '未診断', level: 'neutral', evidence: '環境遷移がまだありません。' };
  }

  const turns = experiences.filter((experience) => experience.rewardCategory === 'turn').length;
  const collisions = experiences.filter((experience) => experience.rewardCategory === 'collision').length;
  const counts = new Map();
  for (const experience of experiences) {
    const key = `${experience.stateKey}|${experience.actionIndex}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const maximumRepeat = Math.max(...counts.values());
  const turnRatio = ratio(turns, steps);
  const collisionRatio = ratio(collisions, steps);
  const noGoal = summary.foods === 0;
  const positiveWithoutGoal = noGoal && summary.cumulativeReward > 0;

  if (profile.key === 'turn-bonus' && positiveWithoutGoal && turnRatio >= 0.6) {
    return {
      key: 'reward-hacking-turn',
      label: '報酬ハッキング候補：旋回反復',
      level: 'danger',
      evidence: `餌0、累積報酬${displayNumber(summary.cumulativeReward)}、旋回${turns}/${steps}です。`,
    };
  }
  if (profile.key === 'collision-bonus' && positiveWithoutGoal && collisionRatio >= 0.35) {
    return {
      key: 'reward-hacking-collision',
      label: '報酬設計失敗：衝突反復',
      level: 'danger',
      evidence: `餌0、累積報酬${displayNumber(summary.cumulativeReward)}、衝突${collisions}/${steps}です。`,
    };
  }
  if (profile.key === 'sparse' && noGoal && summary.cumulativeReward === 0) {
    return {
      key: 'sparse-stall',
      label: '局所最適・停滞候補：学習信号なし',
      level: 'warning',
      evidence: `餌0のため、全${steps}遷移の報酬とリターンが0です。`,
    };
  }
  if (noGoal && maximumRepeat >= 3) {
    return {
      key: 'local-loop',
      label: '局所最適候補：同じ状態と行動を反復',
      level: 'warning',
      evidence: `同一の状態・行動を最大${maximumRepeat}回反復し、餌を取得していません。`,
    };
  }
  if (positiveWithoutGoal) {
    return {
      key: 'reward-mismatch',
      label: '報酬設計の不一致候補',
      level: 'warning',
      evidence: `餌0でも累積報酬${displayNumber(summary.cumulativeReward)}を得ています。`,
    };
  }
  return {
    key: 'aligned-or-inconclusive',
    label: summary.foods > 0 ? '目的達成を確認' : '顕著な異常パターンなし',
    level: summary.foods > 0 ? 'success' : 'neutral',
    evidence: `餌${summary.foods}、累積報酬${displayNumber(summary.cumulativeReward)}、最大反復${maximumRepeat}回です。`,
  };
}
