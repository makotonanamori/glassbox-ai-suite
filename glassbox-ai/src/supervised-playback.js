export const SUPERVISED_PLAYBACK_DEFAULTS = Object.freeze({
  totalSteps: 139,
  speed: 4,
  stepDelayMs: 480,
});

export function supervisedPlaybackDelay(speed) {
  const numericSpeed = Number(speed);
  if (!Number.isFinite(numericSpeed) || numericSpeed <= 0) {
    throw new RangeError('教師あり学習の再生速度は0より大きい有限数にしてください。');
  }
  return Math.max(16, Math.round(SUPERVISED_PLAYBACK_DEFAULTS.stepDelayMs / numericSpeed));
}

export function advanceSupervisedPlayback(
  engine,
  { targetIndex, learningRate, onStep = () => {} },
) {
  if (!engine || typeof engine.next !== 'function' || typeof engine.appendLearning !== 'function') {
    throw new TypeError('StepEngineが必要です。');
  }

  let appendedLearning = false;
  if (!engine.canGoNext && engine.canStartLearning) {
    engine.appendLearning(targetIndex, learningRate);
    appendedLearning = true;
  }

  if (!engine.canGoNext) {
    return { step: engine.current, advanced: false, appendedLearning, complete: true };
  }

  const step = engine.next();
  onStep(step);
  return {
    step,
    advanced: true,
    appendedLearning,
    complete: !engine.canGoNext && !engine.canStartLearning,
  };
}
