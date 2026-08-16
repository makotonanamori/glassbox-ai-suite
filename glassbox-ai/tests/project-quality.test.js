import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('教材UIのIDは一意で、RL軸と139ステップ軸を明示する', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /139ステップ数学タイムラインとは別軸/);
  assert.match(html, /id="rl-causal-rail"|class="rl-causal-rail"/);
  assert.match(html, /id="rl-export-history"/);
  assert.match(html, /生成AIの事前学習は主にこちらの系統/);
  assert.match(html, /id="quick-supervised"/);
  assert.match(html, /id="quick-reinforcement"/);
  assert.match(html, /id="quick-explore"/);
  assert.match(html, /環境履歴 → 累積報酬 → 探索\/活用 → 方策更新/);
});

test('初回ガイドは既存の実計算操作へ案内する', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(app, /function renderQuickStartGuide\(\)/);
  assert.match(app, /elements\['run-to-end'\]/);
  assert.match(app, /elements\['start-learning'\]/);
  assert.match(app, /elements\['rl-start-episode'\]/);
  assert.match(app, /elements\['rl-run-end'\]/);
});

test('ブラウザ成果物に外部script・外部stylesheet依存がない', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
});

test('専用起動scriptが配信元とアプリ固有URLを固定する', async () => {
  const [html, script, launcher] = await Promise.all([
    readFile(new URL('../index.html', import.meta.url), 'utf8'),
    readFile(new URL('../start-glassbox-ai.ps1', import.meta.url), 'utf8'),
    readFile(new URL('../start-glassbox-ai.cmd', import.meta.url), 'utf8'),
  ]);
  assert.match(html, /<meta name="application-name" content="glassbox-ai">/);
  assert.match(script, /'--directory'/);
  assert.match(script, /index\.html\?app=glassbox-ai/);
  assert.match(script, /occupied-by-another-app/);
  assert.match(script, /\$existingPort/);
  assert.match(launcher, /start-glassbox-ai\.ps1/);
});
