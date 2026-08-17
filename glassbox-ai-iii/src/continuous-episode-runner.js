const VISUAL_STAGES = new Set(['rl-transition', 'rl-comparison']);

export const CONTINUOUS_EPISODE_DEFAULTS = Object.freeze({
  count: 10,
  speed: 2,
  transitionDelayMs: 360,
  episodeDelayMs: 720,
});

export function isReinforcementVisualBoundary(step) {
  return Boolean(step && VISUAL_STAGES.has(step.stage));
}

export function continuousEpisodeDelay(speed, boundary = 'transition') {
  const numericSpeed = Number(speed);
  if (!Number.isFinite(numericSpeed) || numericSpeed <= 0) {
    throw new RangeError('連続エピソードの速度は0より大きい有限数にしてください。');
  }
  const base = boundary === 'episode'
    ? CONTINUOUS_EPISODE_DEFAULTS.episodeDelayMs
    : CONTINUOUS_EPISODE_DEFAULTS.transitionDelayMs;
  return Math.max(16, Math.round(base / numericSpeed));
}

export function advanceToReinforcementVisualBoundary(engine, onStep = () => {}) {
  if (!engine || typeof engine.next !== 'function') {
    throw new TypeError('ReinforcementStepEngineが必要です。');
  }

  let advancedSteps = 0;
  while (engine.canGoNext) {
    const step = engine.next();
    advancedSteps += 1;
    onStep(step);
    if (isReinforcementVisualBoundary(step)) {
      return { step, advancedSteps, complete: engine.isComplete };
    }
  }

  return { step: engine.current, advancedSteps, complete: engine.isComplete };
}
