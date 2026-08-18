import { FORWARD_STAGES } from "../model/step-engine.js";

export const LANGUAGE_PLAYBACK_DEFAULTS = Object.freeze({
  speed: 2,
  stageDelayMs: 480,
});

const forwardStages = (phase) => FORWARD_STAGES.map((stage) => ({
  phase,
  key: `${phase}:${stage.key}`,
  forwardIndex: stage.index,
  label: stage.label,
}));

export const LANGUAGE_PLAYBACK_STAGES = Object.freeze([
  ...forwardStages("prediction"),
  { phase: "training", key: "training:forward", label: "Training Forward" },
  { phase: "training", key: "training:loss", label: "Loss" },
  { phase: "training", key: "training:backward", label: "Backward" },
  { phase: "training", key: "training:gradient", label: "Gradient" },
  { phase: "training", key: "training:update", label: "SGD Update" },
  ...forwardStages("generation"),
  { phase: "generation", key: "generation:select", label: "Generate One Token" },
]);

export function languagePlaybackDelay(speed) {
  const numericSpeed = Number(speed);
  if (!Number.isFinite(numericSpeed) || numericSpeed <= 0) {
    throw new RangeError("言語モデルの再生速度は0より大きい有限数にしてください。");
  }
  return Math.max(16, Math.round(LANGUAGE_PLAYBACK_DEFAULTS.stageDelayMs / numericSpeed));
}
