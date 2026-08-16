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
  assert.match(html, /Token → Attention → Probability/);
  assert.match(html, /Loss → Gradient → SGD Update → 前後比較/);
  assert.match(app, /selectTab\("playground"\)/);
  assert.match(app, /selectTab\("training"\)/);
  assert.match(app, /#forward-next/);
  assert.match(app, /#train-one/);
});

test("ブラウザ成果物は外部script・stylesheetへ依存しない", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
});
