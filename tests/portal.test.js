import test from "node:test";
import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";

const root = new URL("../", import.meta.url);

test("series landing has unique IDs and three explicit learning paths", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /正解との誤差から学ぶ/);
  assert.match(html, /行動の結果から学ぶ/);
  assert.match(html, /次Tokenの誤差から学ぶ/);
  assert.match(html, /href="\.\/glassbox-ai\/\?path=auto"/);
  assert.match(html, /href="\.\/glassbox-ai-iii\/\?path=auto"/);
  assert.match(html, /href="\.\/glassbox-ai-ii\/\?path=auto"/);
  const supervised = html.indexOf("Glassbox AI I ·");
  const language = html.indexOf("Glassbox AI II ·");
  const reinforcement = html.indexOf("Glassbox AI III ·");
  assert.ok(supervised < language && language < reinforcement, "landing order must be I → II → III");
});

test("landing and applications have no external script or stylesheet dependency", async () => {
  const files = [
    "../index.html",
    "../glassbox-ai/index.html",
    "../glassbox-ai-ii/index.html",
    "../glassbox-ai-iii/index.html",
  ];
  for (const path of files) {
    const html = await readFile(new URL(path, import.meta.url), "utf8");
    assert.doesNotMatch(html, /<script[^>]+src="https?:\/\//i);
    assert.doesNotMatch(html, /<link[^>]+href="https?:\/\//i);
  }
});

test("all landing targets and application identities exist", async () => {
  await access(new URL("../glassbox-ai/index.html", import.meta.url));
  await access(new URL("../glassbox-ai-ii/index.html", import.meta.url));
  await access(new URL("../glassbox-ai-iii/index.html", import.meta.url));
  const [neural, language, reinforcement] = await Promise.all([
    readFile(new URL("../glassbox-ai/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-ii/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/index.html", import.meta.url), "utf8"),
  ]);
  assert.match(neural, /application-name" content="glassbox-ai"/);
  assert.match(neural, /<title>Glassbox AI I/);
  assert.match(language, /application-name" content="glassbox-ai-ii"/);
  assert.match(language, /<title>Glassbox AI II/);
  assert.match(reinforcement, /application-name" content="glassbox-ai-iii"/);
  assert.match(reinforcement, /<title>Glassbox AI III/);
});

test("direct path requests are handled by each first-run guide", async () => {
  const [neuralHtml, neuralApp, languageApp, reinforcementApp] = await Promise.all([
    readFile(new URL("../glassbox-ai/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai/src/app.js", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-ii/js/app.js", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/src/app.js", import.meta.url), "utf8"),
  ]);
  assert.match(neuralApp, /quickStartRequest/);
  assert.match(neuralHtml, /location\.replace\('\.\.\/glassbox-ai-iii\/\?path=reinforcement'\)/);
  assert.match(languageApp, /firstRunRequest/);
  assert.match(languageApp, /path/);
  assert.match(reinforcementApp, /quickStartRequest/);
  assert.match(reinforcementApp, /: 'reinforcement';/);
});

test("all applications share the real auto-observe and detail contract", async () => {
  const [contract, neuralHtml, neuralApp, languageHtml, languageApp, reinforcementHtml, reinforcementApp] = await Promise.all([
    readFile(new URL("../docs/AUTO_OBSERVE_UX.md", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai/src/app.js", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-ii/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-ii/js/app.js", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/index.html", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/src/app.js", import.meta.url), "utf8"),
  ]);
  assert.match(contract, /スタート → 動く → 結果/);
  assert.match(contract, /ready\|running\|complete\|detail/);
  for (const html of [neuralHtml, languageHtml, reinforcementHtml]) {
    assert.match(html, /data-experience-entry/);
    assert.match(html, /data-experience-mode="auto"/);
    assert.match(html, /data-experience-mode="detail"/);
    assert.match(html, /自動で見る/);
    assert.match(html, /1ステップずつ詳しく見る/);
  }
  assert.match(neuralApp, /function runBeginnerAutoObserve\(\)/);
  assert.match(neuralApp, /startSupervisedPlayback/);
  assert.match(neuralApp, /advanceSupervisedPlayback/);
  assert.match(languageApp, /continueLanguagePlayback/);
  assert.match(languageApp, /executeGenerationPhase/);
  assert.match(languageApp, /generationDistribution/);
  assert.match(languageApp, /run\.visibleTokens\.push/);
  assert.match(reinforcementApp, /runRlToEnd\(\)/);
});

test("canonical responsibilities are synchronized across README and AGENTS", async () => {
  const [readme, sharedAgents, repoAgents, neuralAgents, languageAgents, reinforcementAgents] = await Promise.all([
    readFile(new URL("../README.md", import.meta.url), "utf8"),
    readFile(new URL("../AGENTS-Glassbox.md", import.meta.url), "utf8"),
    readFile(new URL("../AGENTS.md", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai/AGENTS.md", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-ii/AGENTS.md", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/AGENTS.md", import.meta.url), "utf8"),
  ]);
  for (const text of [readme, sharedAgents, repoAgents]) {
    assert.match(text, /Glassbox AI I/);
    assert.match(text, /Glassbox AI II/);
    assert.match(text, /Glassbox AI III/);
  }
  assert.match(neuralAgents, /canonicalな責務は教師あり学習/);
  assert.match(languageAgents, /canonical実装/);
  assert.match(reinforcementAgents, /canonical実装/);
  assert.match(sharedAgents, /I → II → III/);
});

test("public suite contains only Glassbox project entries", async () => {
  const entries = await readdir(root);
  const forbidden = ["gpt-5.4", "gpt-5.5", "lucy-mcp-server", "crash_captures", "comfyui_easyuse_captures"];
  forbidden.forEach((name) => assert.equal(entries.includes(name), false));
  assert.equal(entries.includes("glassbox-ai"), true);
  assert.equal(entries.includes("glassbox-ai-ii"), true);
  assert.equal(entries.includes("glassbox-ai-iii"), true);
});

test("MIT license and unified project scripts are present", async () => {
  const [license, packageJson, workflow, pagesWorkflow] = await Promise.all([
    readFile(new URL("../LICENSE", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8"),
    readFile(new URL("../.github/workflows/pages.yml", import.meta.url), "utf8"),
  ]);
  assert.match(license, /MIT License/);
  assert.match(license, /Glassbox AI contributors/);
  const pkg = JSON.parse(packageJson);
  assert.equal(pkg.license, "MIT");
  assert.equal(pkg.homepage, "https://makotonanamori.github.io/glassbox-ai-suite/");
  assert.equal(pkg.repository.url, "git+https://github.com/makotonanamori/glassbox-ai-suite.git");
  assert.equal(pkg.engines.node, ">=20");
  assert.match(pkg.scripts.check, /test:portal/);
  assert.match(pkg.scripts.check, /test:neural/);
  assert.match(pkg.scripts.check, /test:language/);
  assert.match(pkg.scripts.check, /test:reinforcement/);
  assert.match(workflow, /npm run check/);
  assert.match(workflow, /actions\/checkout@v6/);
  assert.match(workflow, /actions\/setup-node@v6/);
  assert.match(workflow, /node-version: 24/);
  assert.match(pagesWorkflow, /branches: \[main\]/);
  assert.match(pagesWorkflow, /pages: write/);
  assert.match(pagesWorkflow, /id-token: write/);
  assert.match(pagesWorkflow, /npm run check/);
  assert.match(pagesWorkflow, /actions\/checkout@v6/);
  assert.match(pagesWorkflow, /actions\/setup-node@v6/);
  assert.match(pagesWorkflow, /node-version: 24/);
  assert.match(pagesWorkflow, /actions\/configure-pages@v5/);
  assert.match(pagesWorkflow, /actions\/upload-pages-artifact@v4/);
  assert.match(pagesWorkflow, /actions\/deploy-pages@v4/);
});

test("OSS roadmap, changelog, security workflow, and real media are published", async () => {
  const [readme, roadmap, changelog, security, bugTemplate, featureTemplate, pullRequestTemplate] = await Promise.all([
    readFile(new URL("../README.md", import.meta.url), "utf8"),
    readFile(new URL("../ROADMAP.md", import.meta.url), "utf8"),
    readFile(new URL("../CHANGELOG.md", import.meta.url), "utf8"),
    readFile(new URL("../SECURITY.md", import.meta.url), "utf8"),
    readFile(new URL("../.github/ISSUE_TEMPLATE/bug_report.yml", import.meta.url), "utf8"),
    readFile(new URL("../.github/ISSUE_TEMPLATE/feature_request.yml", import.meta.url), "utf8"),
    readFile(new URL("../.github/pull_request_template.md", import.meta.url), "utf8"),
  ]);
  assert.match(readme, /docs\/media\/glassbox-ai-overview\.gif/);
  assert.match(readme, /ROADMAP\.md/);
  assert.match(readme, /CHANGELOG\.md/);
  assert.match(
    readme,
    /## ブラウザですぐ試す[\s\S]*## ローカルで実行する[\s\S]*## 開発・テストする[\s\S]*## 実画面/,
  );
  assert.match(readme, /https:\/\/makotonanamori\.github\.io\/glassbox-ai-suite\/glassbox-ai-ii\//);
  assert.match(readme, /\.\\start-glassbox-ai-suite\.cmd/);
  assert.match(readme, /npm run check/);
  assert.match(roadmap, /Current baseline/);
  assert.match(roadmap, /Now — v1\.x OSS readiness/);
  assert.match(roadmap, /Non-goals/);
  assert.match(changelog, /## \[Unreleased\]/);
  assert.match(changelog, /## \[1\.0\.0\] - 2026-08-16/);
  assert.match(changelog, /6d0ad66/);
  assert.match(changelog, /Git tag.*未作成/);
  assert.match(security, /Private vulnerability reporting/);
  assert.match(bugTemplate, /Security Advisory/);
  assert.match(featureTemplate, /教材上の目的/);
  assert.match(pullRequestTemplate, /npm run check/);

  const pngPaths = ["landing.png", "glassbox-ai-i.png", "glassbox-ai-ii.png", "glassbox-ai-iii.png"];
  for (const name of pngPaths) {
    const image = await readFile(new URL(`../docs/media/${name}`, import.meta.url));
    assert.ok(image.length > 10000, `${name} must contain a real screenshot`);
    assert.deepEqual([...image.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  }
  const gif = await readFile(new URL("../docs/media/glassbox-ai-overview.gif", import.meta.url));
  assert.ok(gif.length > 10000, "overview GIF must contain real frames");
  assert.match(gif.subarray(0, 6).toString("ascii"), /^GIF8[79]a$/);
});

test("Glassbox AI II and III have independent Windows launchers and ports", async () => {
  const [languageLauncher, reinforcementLauncher] = await Promise.all([
    readFile(new URL("../glassbox-ai-ii/start-glassbox-ai-ii.ps1", import.meta.url), "utf8"),
    readFile(new URL("../glassbox-ai-iii/start-glassbox-ai-iii.ps1", import.meta.url), "utf8"),
  ]);
  assert.match(languageLauncher, /\$appName = 'glassbox-ai-ii'/);
  assert.match(languageLauncher, /\[int\]\$Port = 8102/);
  assert.match(languageLauncher, /path=auto/);
  assert.match(languageLauncher, /'--directory', \$appRoot/);
  assert.match(reinforcementLauncher, /\$appName = 'glassbox-ai-iii'/);
  assert.match(reinforcementLauncher, /\[int\]\$Port = 8103/);
  assert.match(reinforcementLauncher, /path=auto/);
  assert.match(reinforcementLauncher, /'--directory', \$appRoot/);
});

test("Windows launcher binds the dedicated suite root without stopping unrelated servers", async () => {
  const launcher = await readFile(new URL("../start-glassbox-ai-suite.ps1", import.meta.url), "utf8");
  assert.match(launcher, /glassbox-ai-suite/);
  assert.match(launcher, /'--directory', \$suiteRoot/);
  assert.match(launcher, /127\.0\.0\.1/);
  assert.match(launcher, /while \(-not \(Test-PortAvailable/);
  assert.doesNotMatch(launcher, /Stop-Process.+CandidatePort/);
});
