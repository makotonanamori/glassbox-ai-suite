import test from "node:test";
import assert from "node:assert/strict";
import { FORWARD_STAGES } from "../model/step-engine.js";
import { LANGUAGE_PLAYBACK_STAGES, languagePlaybackDelay } from "../js/auto-playback.js";

test("auto playback connects prediction, training, and generation in 38 real stages", () => {
  assert.equal(LANGUAGE_PLAYBACK_STAGES.length, 38);
  assert.deepEqual(
    LANGUAGE_PLAYBACK_STAGES.slice(0, FORWARD_STAGES.length).map((stage) => stage.forwardIndex),
    FORWARD_STAGES.map((stage) => stage.index),
  );
  assert.deepEqual(
    LANGUAGE_PLAYBACK_STAGES.slice(16, 21).map((stage) => stage.key),
    ["training:forward", "training:loss", "training:backward", "training:gradient", "training:update"],
  );
  assert.deepEqual(
    LANGUAGE_PLAYBACK_STAGES.slice(21, 37).map((stage) => stage.forwardIndex),
    FORWARD_STAGES.map((stage) => stage.index),
  );
  assert.equal(LANGUAGE_PLAYBACK_STAGES.at(-1).key, "generation:select");
});

test("playback speed changes display delay only", () => {
  assert.equal(languagePlaybackDelay(0.5), 960);
  assert.equal(languagePlaybackDelay(1), 480);
  assert.equal(languagePlaybackDelay(2), 240);
  assert.equal(languagePlaybackDelay(4), 120);
  assert.throws(() => languagePlaybackDelay(0), RangeError);
});
