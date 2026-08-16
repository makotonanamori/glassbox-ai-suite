import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("初回ガイドのIDは一意でForwardとTrainingの実操作へ案内する", async () => {
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
  assert.match(html, /自動で見る（おすすめ）/);
  assert.match(html, /1ステップずつ詳しく見る/);
  assert.match(app, /async function runBeginnerAutoObserve\(\)/);
  assert.match(app, /await runTraining\(1\)/);
  assert.match(app, /function showBeginnerDetail\(\)/);
  assert.match(app, /#forward-next/);
});

test("ブラウザ成果物は外部script・stylesheetへ依存しない", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
});
