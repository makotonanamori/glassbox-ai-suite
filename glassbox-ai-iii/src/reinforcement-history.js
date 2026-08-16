const FORMAT = 'glassbox-ai-reinforcement-history';
const VERSION = 1;

function cloneValue(value) {
  if (typeof value === 'number' && Object.is(value, -0)) return 0;
  if (Array.isArray(value)) return value.map(cloneValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, cloneValue(child)]));
  }
  return value;
}

function assertFiniteTree(value, path = 'history') {
  if (typeof value === 'number' && !Number.isFinite(value)) {
    throw new Error(`${path}に有限でない数値があります。`);
  }
  if (Array.isArray(value)) {
    value.forEach((child, index) => assertFiniteTree(child, `${path}[${index}]`));
  } else if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, child]) => assertFiniteTree(child, `${path}.${key}`));
  }
}

function validateWorld(world, path) {
  if (!world || world.format !== 'glassbox-grid-world' || world.version !== 1) {
    throw new Error(`${path}の世界形式が不正です。`);
  }
  if (!world.agent || !Number.isInteger(world.agent.row) || !Number.isInteger(world.agent.column)) {
    throw new Error(`${path}.agentが不正です。`);
  }
}

function validateEpisode(episode, index) {
  const path = `episodes[${index}]`;
  if (!episode || !episode.config || !episode.summary) throw new Error(`${path}の必須項目がありません。`);
  if (!Array.isArray(episode.experiences) || !Array.isArray(episode.returns)) {
    throw new Error(`${path}の経験またはリターンが配列ではありません。`);
  }
  if (episode.experiences.length !== episode.returns.length || episode.experiences.length < 1) {
    throw new Error(`${path}の経験数とリターン数が一致しません。`);
  }
  validateWorld(episode.initialWorld, `${path}.initialWorld`);
  validateWorld(episode.finalWorld, `${path}.finalWorld`);
  episode.experiences.forEach((experience, experienceIndex) => {
    if (experience.time !== experienceIndex) throw new Error(`${path}.experiencesの時刻が連続していません。`);
    validateWorld(experience.beforeWorld, `${path}.experiences[${experienceIndex}].beforeWorld`);
    validateWorld(experience.afterWorld, `${path}.experiences[${experienceIndex}].afterWorld`);
    if (!Number.isFinite(experience.reward) || !Number.isFinite(experience.cumulativeReward)) {
      throw new Error(`${path}.experiences[${experienceIndex}]の報酬が不正です。`);
    }
  });
  assertFiniteTree(episode, path);
}

export function createReinforcementHistoryDocument(episodes, exportedAt = new Date().toISOString()) {
  if (!Array.isArray(episodes) || episodes.length < 1) {
    throw new Error('完了したエピソードが1件以上必要です。');
  }
  episodes.forEach(validateEpisode);
  return {
    format: FORMAT,
    version: VERSION,
    exportedAt: String(exportedAt),
    episodeCount: episodes.length,
    episodes: cloneValue(episodes),
  };
}

export function serializeReinforcementHistory(episodes, exportedAt) {
  return JSON.stringify(createReinforcementHistoryDocument(episodes, exportedAt), null, 2);
}

export function parseReinforcementHistoryJson(text) {
  let document;
  try {
    document = JSON.parse(text);
  } catch (error) {
    throw new Error(`RL履歴JSONを解析できません：${error.message}`);
  }
  if (document?.format !== FORMAT || document.version !== VERSION) {
    throw new Error('対応していないRL履歴形式です。');
  }
  if (!Array.isArray(document.episodes) || document.episodeCount !== document.episodes.length) {
    throw new Error('RL履歴のエピソード件数が一致しません。');
  }
  document.episodes.forEach(validateEpisode);
  return cloneValue(document);
}
