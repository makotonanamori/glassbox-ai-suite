import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("初回ガイドは生成ループを先に見せ、内部Traceを詳細へ分ける", async () => {
  const [html, app] = await Promise.all([
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../js/app.js", import.meta.url), "utf8"),
  ]);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /id="guide-forward"/);
  assert.match(html, /id="guide-training"/);
  assert.match(html, /data-experience-entry/);
  assert.match(html, /id="guide-forward"[^>]+data-experience-mode="auto"/);
  assert.match(html, /id="guide-training"[^>]+data-experience-mode="detail"/);
  assert.match(html, /文章が伸びる様子を見る/);
  assert.match(html, /1ステップずつ詳しく見る/);
  assert.match(html, /id="experience-pause"/);
  assert.match(html, /id="experience-speed"/);
  assert.match(html, /id="experience-progress"/);
  assert.match(html, /id="generation-observer"/);
  assert.match(html, /id="beginner-sentence"/);
  assert.match(html, /id="beginner-candidates"/);
  assert.match(html, /id="beginner-selected-token"/);
  assert.match(html, /id="token-name-bridge"/);
  assert.match(html, /このまとまりを「Token（トークン）」/);
  assert.match(html, /候補 → 選択 → 追加を5回/);
  assert.match(html, /候補の確率/);
  assert.match(html, /文末へ足す/);
  assert.match(html, /id="learning-observer"/);
  assert.match(html, /id="learning-start"/);
  assert.match(html, /id="learning-before-candidates"/);
  assert.match(html, /id="learning-after-candidates"/);
  assert.match(html, /練習前と500回後を見比べる/);
  assert.match(html, /お手本を見る[\s\S]+次の語を予測[\s\S]+外れた分を調整[\s\S]+もう一度比べる/);
  assert.match(html, /Token[\s\S]+Embedding[\s\S]+Attention[\s\S]+Logits[\s\S]+Softmax[\s\S]+Loss[\s\S]+Gradient/);
  assert.match(app, /function runBeginnerAutoObserve\(\)/);
  assert.match(app, /continueLanguagePlayback/);
  assert.match(app, /executeGenerationPhase/);
  assert.match(app, /generationDistribution/);
  assert.match(app, /run\.visibleTokens\.push/);
  assert.match(app, /function showBeginnerDetail\(\)/);
  assert.match(app, /function runLearningObserver\(\)/);
  assert.match(app, /continueLearningObserver/);
  assert.match(app, /captureLearningObservation/);
  assert.match(app, /#forward-next/);
});

test("ブラウザ成果物は外部script・stylesheetへ依存しない", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
});
