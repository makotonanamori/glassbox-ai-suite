import test from "node:test";
import assert from "node:assert/strict";
import {
  GENERATION_PHASES, advanceGenerationContext, generationDistribution, languagePlaybackDelay,
  languagePlaybackStageCount, selectGenerationToken, topGenerationCandidates,
} from "../js/auto-playback.js";

test("beginner playback repeats the four visible generation phases", () => {
  assert.deepEqual(
    GENERATION_PHASES.map((phase) => phase.key),
    ["candidates", "selection", "append", "repeat"],
  );
  assert.equal(languagePlaybackStageCount(5), 20);
  assert.equal(languagePlaybackStageCount(1), 4);
  assert.throws(() => languagePlaybackStageCount(0), RangeError);
});

test("playback speed changes display delay only", () => {
  assert.equal(languagePlaybackDelay(0.5), 960);
  assert.equal(languagePlaybackDelay(1), 480);
  assert.equal(languagePlaybackDelay(2), 240);
  assert.equal(languagePlaybackDelay(4), 120);
  assert.throws(() => languagePlaybackDelay(0), RangeError);
});

test("generation distribution excludes BOS and remains normalized", () => {
  const probabilities = generationDistribution([4, 2, 1, -1], { bosId: 0, temperature: 1 });
  assert.equal(probabilities[0], 0);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(probabilities[1] > probabilities[2]);
  assert.ok(probabilities[2] > probabilities[3]);
});

test("greedy and temperature selection use the displayed distribution", () => {
  const probabilities = [0, 0.2, 0.3, 0.5];
  assert.equal(selectGenerationToken(probabilities, { mode: "greedy" }), 3);
  assert.equal(selectGenerationToken(probabilities, { mode: "temperature", draw: 0.1 }), 1);
  assert.equal(selectGenerationToken(probabilities, { mode: "temperature", draw: 0.25 }), 2);
  assert.equal(selectGenerationToken(probabilities, { mode: "temperature", draw: 0.75 }), 3);
  assert.deepEqual(topGenerationCandidates(probabilities, 2), [
    { id: 3, probability: 0.5 },
    { id: 2, probability: 0.3 },
  ]);
});

test("context window drops only the oldest IDs after it reaches the limit", () => {
  assert.deepEqual(advanceGenerationContext([0, 4, 7], 9, 4), {
    contextIds: [0, 4, 7, 9],
    droppedIds: [],
  });
  assert.deepEqual(advanceGenerationContext([0, 4, 7, 9], 12, 4), {
    contextIds: [4, 7, 9, 12],
    droppedIds: [0],
  });
});
