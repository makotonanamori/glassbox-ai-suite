import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('Glassbox AI Iのcanonical UIは教師あり学習だけを通常表示する', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /<title>Glassbox AI I/);
  assert.match(html, /id="quick-supervised"/);
  assert.match(html, /id="quick-reinforcement"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /class="panel grid-world-panel span-full"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /class="panel rl-panel span-full"[^>]+hidden[^>]+data-legacy-compatibility/);
  assert.match(html, /id="rl-causal-rail"|class="rl-causal-rail"/);
  assert.match(html, /id="rl-export-history"/);
  assert.match(html, /生成AIの事前学習は主にこちらの系統/);
  assert.match(html, /id="quick-explore"/);
  assert.match(html, /data-experience-entry/);
  assert.match(html, /id="quick-supervised"[^>]+data-experience-mode="auto"/);
  assert.match(html, /id="quick-explore"[^>]+data-experience-mode="detail"/);
  assert.match(html, /id="experience-pause"/);
  assert.match(html, /id="experience-speed"/);
  assert.match(html, /id="experience-progress"/);
});

test('初回ガイドは既存の実計算操作へ案内する', async () => {
  const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  assert.match(app, /function renderQuickStartGuide\(\)/);
  assert.match(app, /elements\['run-to-end'\]/);
  assert.match(app, /elements\['start-learning'\]/);
  assert.match(app, /elements\['rl-start-episode'\]/);
  assert.match(app, /elements\['rl-run-end'\]/);
  assert.match(app, /function runBeginnerAutoObserve\(\)/);
  assert.match(app, /startSupervisedPlayback/);
  assert.match(app, /advanceSupervisedPlayback/);
  assert.match(app, /function pauseSupervisedPlayback\(\)/);
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
