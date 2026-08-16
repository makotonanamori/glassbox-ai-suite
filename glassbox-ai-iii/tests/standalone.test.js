import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';

const projectRoot = new URL('../', import.meta.url);

async function sourceFiles(directoryUrl) {
  const entries = await readdir(directoryUrl, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const child = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directoryUrl);
    if (entry.isDirectory()) files.push(...await sourceFiles(child));
    else if (/\.(?:html|css|js)$/.test(entry.name)) files.push(child);
  }
  return files;
}

test('Glassbox AI III has its own application identity and RL-first entry', async () => {
  const [html, app] = await Promise.all([
    readFile(new URL('index.html', projectRoot), 'utf8'),
    readFile(new URL('src/app.js', projectRoot), 'utf8'),
  ]);
  assert.match(html, /application-name" content="glassbox-ai-iii"/);
  assert.match(html, /<title>Glassbox AI III/);
  assert.match(app, /: 'reinforcement';/);
});

test('Glassbox AI III runtime assets do not import sibling applications', async () => {
  const files = await sourceFiles(projectRoot);
  for (const file of files) {
    const content = await readFile(file, 'utf8');
    assert.doesNotMatch(content, /(?:src|href)=["']\.\.\/(?:glassbox-ai|glassbox-ai-ii)/);
    assert.doesNotMatch(content, /from\s+["']\.\.\/(?:glassbox-ai|glassbox-ai-ii)/);
  }
});

test('Glassbox AI III launcher serves only its own directory and identity', async () => {
  const launcher = await readFile(new URL('start-glassbox-ai-iii.ps1', projectRoot), 'utf8');
  assert.match(launcher, /\$appName = 'glassbox-ai-iii'/);
  assert.match(launcher, /'--directory', \$appRoot/);
  assert.match(launcher, /\[int\]\$Port = 8103/);
  assert.doesNotMatch(launcher, /Stop-Process.+CandidatePort/);
});
