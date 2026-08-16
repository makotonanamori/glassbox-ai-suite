import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('Glassbox AI IIIのcanonical UIは強化学習だけを通常表示する', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /139ステップ数学タイムラインとは別軸/);
  assert.match(html, /id="quick-supervised"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /id="quick-explore"[^>]+data-experience-mode="detail"/);
  assert.match(html, /class="panel step-panel"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /class="panel learning-panel"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /id="rl-causal-rail"|class="rl-causal-rail"/);
  assert.match(html, /id="rl-export-history"/);
  assert.match(html, /生成AIの事前学習は主にこちらの系統/);
  assert.match(html, /id="quick-reinforcement"/);
  assert.match(html, /小さな世界で行動し、結果を受け取る/);
  assert.match(html, /id="quick-reinforcement"[^>]+data-experience-mode="auto"/);
  assert.match(html, /自動で見る/);
});

test('初回ガイドは既存の実計算操作へ案内する', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(app, /function renderQuickStartGuide\(\)/);
  assert.match(app, /elements\['run-to-end'\]/);
  assert.match(app, /elements\['start-learning'\]/);
  assert.match(app, /elements\['rl-start-episode'\]/);
  assert.match(app, /elements\['rl-run-end'\]/);
  assert.match(app, /async function runBeginnerAutoObserve\(\)/);
  assert.match(app, /runRlToEnd\(\)/);
  assert.match(app, /function showBeginnerDetail\(\)/);
});

test('ブラウザ成果物に外部script・外部stylesheet依存がない', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
  assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
});

test('専用起動scriptが配信元とアプリ固有URLを固定する', async () => {
  const [html, script, launcher] = await Promise.all([
    readFile(new URL('../index.html', import.meta.url), 'utf8'),
    readFile(new URL('../start-glassbox-ai-iii.ps1', import.meta.url), 'utf8'),
    readFile(new URL('../start-glassbox-ai-iii.cmd', import.meta.url), 'utf8'),
  ]);
  assert.match(html, /<meta name="application-name" content="glassbox-ai-iii">/);
  assert.match(script, /'--directory'/);
  assert.match(script, /\$appName = 'glassbox-ai-iii'/);
  assert.match(script, /path=auto/);
  assert.match(script, /while \(-not \(Test-PortAvailable/);
  assert.match(launcher, /start-glassbox-ai-iii\.ps1/);
});
