import { LanguageDataset, DEFAULT_CORPUS } from "../training/dataset.js";
import { TinyTransformer } from "../model/transformer.js";
import { Trainer } from "../training/trainer.js";
import { ForwardStepEngine, FORWARD_STAGES } from "../model/step-engine.js";
import {
  GENERATION_PHASES, LANGUAGE_PLAYBACK_DEFAULTS, advanceGenerationContext, generationDistribution,
  languagePlaybackDelay, selectGenerationToken, topGenerationCandidates,
} from "./auto-playback.js";
import {
  LEARNING_OBSERVER_DEFAULTS, captureLearningObservation, learningCheckpoints, learningObserverDelay,
} from "./learning-observer.js";
import { SeededRandom } from "../utils/rng.js";
import { relativeError, stats } from "../utils/math.js";
import { exportApplicationState, importApplicationState } from "../utils/serialization.js";
import {
  attentionCellDetails, escapeHtml, formatter, renderLossChart, renderMatrix,
  renderParameterInspector, renderParameterTable, renderPrediction,
  renderTensorRows, renderTokenTrace, renderVector,
} from "../visualization/renderers.js";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const firstRunRequest = new URLSearchParams(window.location.search).get("path");

let dataset;
let tokenizer;
let model;
let trainer;
let engine = new ForwardStepEngine();
let currentTrace = null;
let selectedToken = 0;
let selectedAttention = { row: 0, col: 0 };
let selectedParameter = "embeddings.token";
let lastTrainingResult = null;
let autoTraining = false;
let generationCounter = 0;
let firstRunMode = firstRunRequest === "training" ? "training" : "forward";
let experienceState = "ready";
let experienceMessage = "「自動で見る」を押してください。";
let experiencePlaybackRun = null;
let experiencePlaybackToken = 0;
let trainingPlaybackSession = null;
let trainingPlaybackPhase = "";
let lastGenerationTrace = null;
let lastGenerationPrompt = null;
let learningObserverRun = null;
let learningObserverToken = 0;

function precision() { return Number($("#precision").value); }
function fmt(value) { return formatter(precision())(value); }
function setNotice(message, error = false) {
  $("#notice").textContent = message;
  $("#model-status").textContent = error ? "STOPPED" : "READY";
  $("#model-status").className = error ? "status-error" : "status-ok";
}

function selectTab(name) {
  $$(".tab").forEach((item) => item.classList.toggle("active", item.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));
}

function renderFirstRunGuide() {
  const entry = $("[data-experience-entry]");
  entry.dataset.experienceState = experienceState;
  const playbackActive = Boolean(experiencePlaybackRun && !experiencePlaybackRun.complete);
  const playbackPaused = Boolean(playbackActive && experiencePlaybackRun.paused);
  const learningActive = Boolean(learningObserverRun && !learningObserverRun.complete);
  entry.dataset.playbackState = playbackPaused ? "paused" : (playbackActive ? "running" : "idle");
  const autoSelected = experienceState !== "detail";
  $("#guide-forward").classList.toggle("selected", autoSelected);
  $("#guide-forward").setAttribute("aria-pressed", String(autoSelected));
  $("#guide-training").classList.toggle("selected", !autoSelected);
  $("#guide-training").setAttribute("aria-pressed", String(!autoSelected));
  $("#guide-forward").disabled = learningActive || (playbackActive && !playbackPaused);
  $("#guide-forward strong").textContent = playbackPaused
    ? "▶ 続ける"
    : (playbackActive ? "文章を生成中…" : "▶ 文章が伸びる様子を見る");
  $("#guide-training").disabled = learningActive;
  $("#experience-pause").disabled = !playbackActive || playbackPaused;
  $("#experience-progress").textContent = experiencePlaybackRun
    ? `${experiencePlaybackRun.completedTokens} / ${experiencePlaybackRun.totalTokens}回`
    : `0 / ${LANGUAGE_PLAYBACK_DEFAULTS.tokens}回`;
  setExperienceControlsLocked(playbackActive || learningActive);

  const flowState = experienceState === "detail" ? "ready" : experienceState;
  const flowOrder = { ready: 0, running: 1, complete: 2 };
  const current = flowOrder[flowState] ?? 0;
  entry.querySelectorAll("[data-flow-step]").forEach((item, index) => {
    item.classList.toggle("done", index < current || flowState === "complete");
    item.classList.toggle("active", index === current);
  });
  $("#first-run-status").textContent = experienceMessage;
  renderGenerationObserver();
  renderLearningObserver();
}

function setExperienceControlsLocked(locked) {
  const controls = [
    "#prepare-forward", "#forward-previous", "#forward-next", "#forward-run", "#forward-reset",
    "#generate-one", "#generate-many", "#train-one", "#auto-train", "#pause-train", "#reinitialize",
    "#seed-input", "#learning-rate", "#clip-norm", "#prompt-input", "#sampling-mode", "#temperature", "#generation-count",
    "#detail-from-bridge",
  ];
  if (locked) {
    controls.forEach((selector) => { const element = $(selector); if (element) element.disabled = true; });
    $$('[data-train-count]').forEach((button) => { button.disabled = true; });
    $$(".tab").forEach((button) => { button.disabled = true; });
    return;
  }
  controls.forEach((selector) => { const element = $(selector); if (element) element.disabled = false; });
  $("#forward-previous").disabled = !currentTrace || engine.index <= 0;
  $("#forward-next").disabled = !currentTrace || engine.index >= FORWARD_STAGES.length - 1;
  $("#pause-train").disabled = !autoTraining;
  $$('[data-train-count]').forEach((button) => { button.disabled = false; });
  $$(".tab").forEach((button) => { button.disabled = false; });
}

function focusFirstRunTarget() {
  const target = firstRunMode === "forward" ? $("#forward-next") : $("#train-one");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.focus({ preventScroll: true });
}

function focusTrainingGuide() {
  const target = $("#guide-training");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.focus({ preventScroll: true });
}

function cancelLanguagePlayback() {
  experiencePlaybackToken += 1;
  if (trainingPlaybackSession && !trainingPlaybackSession.updateComplete) model?.zeroGrad();
  experiencePlaybackRun = null;
  trainingPlaybackSession = null;
  trainingPlaybackPhase = "";
}

function cancelLearningObserver({ discard = false } = {}) {
  learningObserverToken += 1;
  if (learningObserverRun && !learningObserverRun.complete) learningObserverRun.paused = true;
  if (discard) learningObserverRun = null;
}

function pauseLanguagePlayback() {
  if (!experiencePlaybackRun || experiencePlaybackRun.complete || experiencePlaybackRun.paused) return;
  experiencePlaybackToken += 1;
  experiencePlaybackRun.paused = true;
  experienceState = "running";
  const phase = GENERATION_PHASES[experiencePlaybackRun.displayPhaseIndex];
  experienceMessage = `${experiencePlaybackRun.completedTokens} / ${experiencePlaybackRun.totalTokens}回、` +
    `「${phase.label}」で一時停止しました。「続ける」で再開できます。`;
  renderFirstRunGuide();
}

function generationChoice(probabilities, mode) {
  const draw = mode === "temperature"
    ? new SeededRandom(`${trainer.seed}:generation:${generationCounter++}`).next()
    : null;
  return {
    id: selectGenerationToken(probabilities, { mode, draw: draw ?? 0 }),
    draw,
  };
}

function renderGenerationObserver() {
  if (!tokenizer) return;
  const run = experiencePlaybackRun;
  const phase = run ? GENERATION_PHASES[run.displayPhaseIndex]?.key ?? "repeat" : "ready";
  const observer = $("#generation-observer");
  observer.dataset.generationPhase = run?.complete ? "complete" : phase;
  const phaseIndex = GENERATION_PHASES.findIndex((item) => item.key === phase);
  $$("[data-loop-phase]").forEach((item, index) => {
    item.classList.toggle("active", Boolean(run && !run.complete && index === phaseIndex));
    item.classList.toggle("done", Boolean(run && (run.complete || index < phaseIndex)));
  });

  $("#generation-round").textContent = !run
    ? "START待ち"
    : (run.complete
      ? `${run.completedTokens} Token生成`
      : `${Math.min(run.completedTokens + 1, run.totalTokens)}回目 / ${run.totalTokens}`);

  const visibleTokens = run?.visibleTokens ?? tokenizer.tokenize($("#prompt-input").value);
  $("#beginner-sentence").innerHTML = visibleTokens.length
    ? visibleTokens.map((token, index) => `<span class="beginner-token ${index === run?.newTokenIndex ? "new" : ""}">${escapeHtml(token)}</span>`).join("")
    : '<span class="placeholder-token">文章を入力してください</span>';

  const candidates = run?.probabilities ? topGenerationCandidates(run.probabilities, 5) : [];
  if (run?.selectedId != null && !candidates.some((item) => item.id === run.selectedId)) {
    candidates.push({ id: run.selectedId, probability: run.probabilities[run.selectedId] });
  }
  $("#beginner-candidates").innerHTML = candidates.length
    ? candidates.map(({ id, probability }) => `<div class="candidate-row ${id === run.selectedId ? "selected" : ""}"><span class="candidate-token">${escapeHtml(tokenizer.vocabulary[id])}</span><span class="candidate-bar"><i style="width:${Math.max(0, Math.min(100, probability * 100))}%"></i></span><span class="candidate-probability">${(probability * 100).toFixed(2)}%</span></div>`).join("")
    : '<div class="candidate-placeholder">STARTを押すと、実際の確率が並びます</div>';

  const selectedToken = run?.selectedId == null ? null : tokenizer.vocabulary[run.selectedId];
  $("#beginner-selected-token").textContent = selectedToken ?? "?";
  $("#beginner-selected-token").classList.toggle("chosen", selectedToken != null);
  $("#token-name-bridge").hidden = !run || (run.completedTokens === 0 && selectedToken == null);
  if (!selectedToken) {
    $("#beginner-selection-note").textContent = "まだ選んでいません";
  } else if (run.mode === "greedy") {
    $("#beginner-selection-note").textContent = "いちばん確率が高い候補を選択";
  } else {
    $("#beginner-selection-note").textContent = `乱数 ${run.draw.toFixed(4)} が入った確率区間を選択`;
  }

  const contextIds = run?.contextIds ?? promptTokenIds();
  $("#beginner-context").innerHTML = tokenizer.decode(contextIds).map((token) =>
    `<span class="context-token ${token === "<BOS>" ? "bos" : ""}">${escapeHtml(token)}</span>`).join("");
  const dropped = run?.droppedContext ?? [];
  $("#beginner-dropped").hidden = dropped.length === 0;
  $("#beginner-dropped strong").textContent = dropped.length ? tokenizer.decode(dropped).join(" ") : "—";
}

function learningCandidatesHtml(observation) {
  if (!observation) return '<div class="candidate-placeholder">練習を始めると実際の確率が並びます</div>';
  const candidates = [...observation.candidates];
  if (!candidates.some((item) => item.id === observation.targetId)) {
    candidates.push({ id: observation.targetId, probability: observation.targetProbability });
  }
  return candidates.map(({ id, probability }) => {
    const classes = [id === observation.selectedId ? "selected" : "", id === observation.targetId ? "target" : ""].filter(Boolean).join(" ");
    const targetLabel = id === observation.targetId ? '<em>お手本</em>' : "";
    return `<div class="candidate-row ${classes}"><span class="candidate-token">${escapeHtml(tokenizer.vocabulary[id])}${targetLabel}</span><span class="candidate-bar"><i style="width:${Math.max(0, Math.min(100, probability * 100))}%"></i></span><span class="candidate-probability">${(probability * 100).toFixed(2)}%</span></div>`;
  }).join("");
}

function learningSentenceHtml(observation) {
  if (!observation) return '<span class="placeholder-token">—</span>';
  const promptLength = observation.promptTokens.length;
  const tokens = [...observation.promptTokens, ...observation.generatedTokens];
  return tokens.map((token, index) => `<span class="beginner-token ${index >= promptLength ? "learned-continuation" : ""}">${escapeHtml(token)}</span>`).join("");
}

function renderLearningResult(prefix, observation, loss, label) {
  $(`#learning-${prefix}-choice`).textContent = observation ? `選ぶ語：${observation.selectedToken}` : "—";
  $(`#learning-${prefix}-candidates`).innerHTML = learningCandidatesHtml(observation);
  $(`#learning-${prefix}-sentence`).innerHTML = learningSentenceHtml(observation);
  $(`#learning-${prefix}-target`).textContent = observation ? `${(observation.targetProbability * 100).toFixed(2)}%` : "—";
  $(`#learning-${prefix}-loss`).textContent = Number.isFinite(loss) ? fmt(loss) : "—";
  if (prefix === "after") $("#learning-after-label").textContent = label;
}

function renderLearningObserver() {
  if (!tokenizer) return;
  const run = learningObserverRun;
  const active = Boolean(run && !run.complete);
  const paused = Boolean(active && run.paused);
  const observer = $("#learning-observer");
  observer.dataset.learningState = run?.complete ? "complete" : (active ? (paused ? "paused" : "running") : "ready");
  $("#learning-badge").textContent = !run ? "練習前" : (run.complete ? `${run.completedSteps}回 完了` : `${run.completedSteps} / ${run.totalSteps}回`);
  $("#learning-progress").textContent = `${run?.completedSteps ?? 0} / ${run?.totalSteps ?? LEARNING_OBSERVER_DEFAULTS.totalSteps}回`;
  $("#learning-status").textContent = run?.message ?? "同じ問いを、練習前と後で比べます。";
  $("#learning-start").disabled = Boolean(experiencePlaybackRun && !experiencePlaybackRun.complete) || (active && !paused);
  $("#learning-start").textContent = paused
    ? "▶ 続ける"
    : (active ? "練習中…" : (run?.complete ? "▶ さらに500回練習して比べる" : "▶ 練習前と500回後を見比べる"));
  $("#learning-pause").disabled = !active || paused;
  $("#learning-speed").disabled = false;

  renderLearningResult("before", run?.beforeObservation, run?.lossBefore, "練習前");
  renderLearningResult(
    "after",
    run?.currentObservation,
    run?.currentLoss,
    run ? `${run.startStep + run.completedSteps}回後` : "いま",
  );

  const entries = run?.checkpoints ?? [];
  $("#learning-checkpoints").innerHTML = entries.length
    ? entries.map((item, index) => `<div class="learning-checkpoint ${index === entries.length - 1 && active ? "current" : "done"}"><span>${item.relativeStep}回</span><strong>${escapeHtml(item.observation.selectedToken)}</strong><small>${(item.observation.targetProbability * 100).toFixed(1)}% fish</small></div>`).join("")
    : '<span class="checkpoint-placeholder">途中では候補が揺れることも、そのまま表示します。</span>';
}

function pauseLearningObserver() {
  if (!learningObserverRun || learningObserverRun.complete || learningObserverRun.paused) return;
  learningObserverToken += 1;
  learningObserverRun.paused = true;
  learningObserverRun.message = `${learningObserverRun.completedSteps} / ${learningObserverRun.totalSteps}回で一時停止しました。同じParameterから再開できます。`;
  renderFirstRunGuide();
}

async function continueLearningObserver(run) {
  const token = ++learningObserverToken;
  try {
    while (token === learningObserverToken && !run.paused && !run.complete) {
      const checkpoint = run.checkpointSteps[run.nextCheckpointIndex];
      let result = null;
      for (let index = run.completedSteps; index < checkpoint; index += 1) result = trainer.trainOneStep();
      run.completedSteps = checkpoint;
      run.nextCheckpointIndex += 1;
      run.currentObservation = captureLearningObservation(model, tokenizer);
      run.currentLoss = trainer.evaluateLoss();
      run.lastTrainingResult = result;
      run.checkpoints.push({
        relativeStep: run.completedSteps,
        absoluteStep: trainer.step,
        observation: run.currentObservation,
        loss: run.currentLoss,
        example: result?.sample?.sentence ?? "",
      });
      run.message = `${run.completedSteps}回練習。いま見たお手本は「${result?.sample?.sentence ?? "例文"}」。同じ問いの先頭候補は「${run.currentObservation.selectedToken}」です。`;
      $("#training-step-global").textContent = trainer.step;
      renderLearningObserver();
      if (run.nextCheckpointIndex >= run.checkpointSteps.length) {
        run.complete = true;
        run.message = `同じ問いの生成が「${run.beforeObservation.generatedText}」から「${run.currentObservation.generatedText}」へ変わりました。`;
        break;
      }
      await new Promise((resolve) => window.setTimeout(resolve, learningObserverDelay($("#learning-speed").value)));
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    }
    if (token !== learningObserverToken || run.paused || !run.complete) return;
    lastTrainingResult = run.lastTrainingResult;
    $("#prompt-input").value = LEARNING_OBSERVER_DEFAULTS.prompt;
    currentTrace = model.forward(promptTokenIds()).trace;
    lastGenerationTrace = currentTrace;
    lastGenerationPrompt = LEARNING_OBSERVER_DEFAULTS.prompt;
    engine.load(currentTrace);
    engine.run();
    renderForward();
    renderAttention();
    renderTraining();
    renderParameters();
    renderSnapshots();
    renderFirstRunGuide();
    setNotice(`${run.totalSteps}回の実Trainingを完了。学習前後の候補と生成文を同じPromptで比較しました。`);
    $("#learning-observer").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    if (token !== learningObserverToken) return;
    run.complete = true;
    run.message = `練習を完了できませんでした：${error.message}`;
    renderFirstRunGuide();
    setNotice(run.message, true);
  }
}

function runLearningObserver() {
  if (learningObserverRun?.paused) {
    learningObserverRun.paused = false;
    learningObserverRun.message = `${learningObserverRun.completedSteps} / ${learningObserverRun.totalSteps}回から再開します。`;
    renderFirstRunGuide();
    void continueLearningObserver(learningObserverRun);
    return;
  }
  if (learningObserverRun && !learningObserverRun.complete) return;

  cancelLanguagePlayback();
  autoTraining = false;
  trainer.learningRate = Number($("#learning-rate").value);
  trainer.clipNorm = Number($("#clip-norm").value);
  if (!(trainer.learningRate > 0) || !(trainer.clipNorm > 0)) {
    setNotice("Learning rateとClip normは正の有限値にしてください。", true);
    return;
  }
  const totalSteps = LEARNING_OBSERVER_DEFAULTS.totalSteps;
  const beforeObservation = captureLearningObservation(model, tokenizer);
  learningObserverRun = {
    startStep: trainer.step,
    totalSteps,
    completedSteps: 0,
    checkpointSteps: learningCheckpoints(totalSteps, LEARNING_OBSERVER_DEFAULTS.checkpointEvery),
    nextCheckpointIndex: 0,
    beforeObservation,
    currentObservation: beforeObservation,
    lossBefore: trainer.evaluateLoss(),
    currentLoss: trainer.evaluateLoss(),
    checkpoints: [],
    lastTrainingResult: null,
    paused: false,
    complete: false,
    message: "お手本の例文を見ながら、次Tokenの候補を実際に更新しています。",
  };
  renderFirstRunGuide();
  $("#learning-observer").scrollIntoView({ behavior: "smooth", block: "center" });
  void continueLearningObserver(learningObserverRun);
}

function generationPlaybackMessage(phase, run) {
  if (phase.key === "candidates") return "次に来そうな候補と確率が出ました。棒の長さと数字は同じ確率です。";
  const token = tokenizer.vocabulary[run.selectedId];
  if (phase.key === "selection") return `候補から「${token}」を1個選びました。`;
  if (phase.key === "append") {
    return token === "<EOS>" ? "文章の終わりを表すものを選びました。" : `「${token}」を文末へ足しました。文章が1つ伸びました。`;
  }
  return run.ended ? "文章の終わりに到達しました。" : "伸びた文章を使って、次のToken予測へ戻ります。";
}

function executeGenerationPhase(phase, run) {
  if (phase.key === "candidates") {
    run.newTokenIndex = null;
    run.selectedId = null;
    run.draw = null;
    run.droppedContext = [];
    const forward = model.forward(run.contextIds);
    run.trace = forward.trace;
    currentTrace = forward.trace;
    selectedToken = Math.min(selectedToken, currentTrace.tokens.length - 1);
    selectedAttention = {
      row: Math.min(selectedAttention.row, currentTrace.tokens.length - 1),
      col: Math.min(selectedAttention.col, currentTrace.tokens.length - 1),
    };
    engine.load(currentTrace);
    engine.run();
    lastGenerationTrace = currentTrace;
    lastGenerationPrompt = tokenizer.detokenize(run.contextIds);
    const [rows, cols] = currentTrace.logits.shape;
    const logits = currentTrace.logits.data.slice((rows - 1) * cols, rows * cols);
    run.probabilities = generationDistribution(logits, {
      bosId: tokenizer.tokenToId.get("<BOS>"),
      temperature: run.mode === "temperature" ? run.temperature : 1,
    });
    renderForward();
    renderAttention();
    return;
  }
  if (phase.key === "selection") {
    const choice = generationChoice(run.probabilities, run.mode);
    run.selectedId = choice.id;
    run.draw = choice.draw;
    return;
  }
  if (phase.key === "append") {
    const eosId = tokenizer.tokenToId.get("<EOS>");
    run.completedTokens += 1;
    if (run.selectedId === eosId) {
      run.ended = true;
    } else {
      run.visibleTokens.push(tokenizer.vocabulary[run.selectedId]);
      run.newTokenIndex = run.visibleTokens.length - 1;
      const advanced = advanceGenerationContext(run.contextIds, run.selectedId, model.config.contextLength);
      run.droppedContext = advanced.droppedIds;
      run.contextIds = advanced.contextIds;
    }
    const text = tokenizer.detokenize(run.visibleTokens);
    $("#generation-output").textContent = run.ended ? `${text} → <EOS>` : text;
    $("#prompt-input").value = text;
  }
}

async function continueLanguagePlayback(run) {
  const token = ++experiencePlaybackToken;
  try {
    while (token === experiencePlaybackToken && !run.paused && !run.complete) {
      const phase = GENERATION_PHASES[run.nextPhaseIndex];
      executeGenerationPhase(phase, run);
      run.displayPhaseIndex = run.nextPhaseIndex;
      run.nextPhaseIndex = (run.nextPhaseIndex + 1) % GENERATION_PHASES.length;
      run.completedStages += 1;
      experienceMessage = generationPlaybackMessage(phase, run);
      const finished = phase.key === "repeat" && (run.ended || run.completedTokens >= run.totalTokens);
      if (finished) {
        run.complete = true;
        experienceState = "complete";
      experienceMessage = run.ended
          ? `文章の終わりを選び、${run.completedTokens}回で生成を止めました。`
          : `${run.completedTokens}回、候補から1つずつ選びました。文が「${tokenizer.detokenize(run.visibleTokens)}」まで伸びました。`;
      }
      renderFirstRunGuide();
      if (run.complete) break;
      await new Promise((resolve) => window.setTimeout(resolve, languagePlaybackDelay($("#experience-speed").value)));
    }
    if (token !== experiencePlaybackToken || run.paused || !run.complete) return;
    setNotice(`${run.completedTokens}回の生成ループを実際の値で完了しました。`);
    $("#generation-observer").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    if (token !== experiencePlaybackToken) return;
    run.complete = true;
    experienceState = "ready";
    experienceMessage = `自動再生を完了できませんでした：${error.message}`;
    setNotice(experienceMessage, true);
    renderFirstRunGuide();
  }
}

function runBeginnerAutoObserve() {
  if (experiencePlaybackRun?.paused) {
    experiencePlaybackRun.paused = false;
    experienceState = "running";
    experienceMessage = `${experiencePlaybackRun.completedTokens} / ${experiencePlaybackRun.totalTokens}回から再開します。`;
    renderFirstRunGuide();
    void continueLanguagePlayback(experiencePlaybackRun);
    return;
  }
  if (experiencePlaybackRun && !experiencePlaybackRun.complete) return;

  cancelLanguagePlayback();
  autoTraining = false;
  generationCounter = 0;
  const totalTokens = Math.max(1, Math.min(32, Number($("#generation-count").value) || LANGUAGE_PLAYBACK_DEFAULTS.tokens));
  const initialPrompt = $("#prompt-input").value;
  const visibleTokens = tokenizer.tokenize(initialPrompt);
  const contextIds = promptTokenIds();
  lastGenerationTrace = null;
  lastGenerationPrompt = null;
  $("#generation-output").textContent = "—";
  experiencePlaybackRun = {
    initialPrompt,
    visibleTokens,
    contextIds,
    probabilities: null,
    selectedId: null,
    draw: null,
    droppedContext: [],
    newTokenIndex: null,
    mode: $("#sampling-mode").value,
    temperature: Number($("#temperature").value),
    completedTokens: 0,
    totalTokens,
    completedStages: 0,
    displayPhaseIndex: 0,
    nextPhaseIndex: 0,
    ended: false,
    paused: false,
    complete: false,
  };
  firstRunMode = "forward";
  experienceState = "running";
  experienceMessage = "まず、次に来そうな候補の確率を計算します。";
  setNotice("候補の確率 → 1つ選ぶ → 文へ追加、を繰り返します。");
  renderFirstRunGuide();
  $("#generation-observer").scrollIntoView({ behavior: "smooth", block: "center" });
  void continueLanguagePlayback(experiencePlaybackRun);
}

function showBeginnerDetail() {
  cancelLanguagePlayback();
  firstRunMode = "forward";
  experienceState = "detail";
  experienceMessage = "いま見た候補確率の計算を先頭へ戻しました。「Next」で内部を1つずつ確認できます。";
  selectTab("playground");
  if (lastGenerationTrace) {
    if (lastGenerationPrompt != null) $("#prompt-input").value = lastGenerationPrompt;
    currentTrace = lastGenerationTrace;
    engine.load(currentTrace);
  } else if (!currentTrace) prepareForward();
  engine.reset();
  renderForward();
  renderFirstRunGuide();
  focusFirstRunTarget();
}

function initialize(seed = 42) {
  cancelLanguagePlayback();
  cancelLearningObserver({ discard: true });
  dataset = new LanguageDataset(DEFAULT_CORPUS);
  tokenizer = dataset.tokenizer;
  model = new TinyTransformer(tokenizer.vocabulary, { seed });
  trainer = new Trainer(model, dataset, {
    learningRate: Number($("#learning-rate").value),
    clipNorm: Number($("#clip-norm").value),
    seed,
  });
  engine = new ForwardStepEngine();
  currentTrace = null;
  selectedToken = 0;
  selectedAttention = { row: 0, col: 0 };
  selectedParameter = "embeddings.token";
  lastTrainingResult = null;
  generationCounter = 0;
  lastGenerationTrace = null;
  lastGenerationPrompt = null;
  experienceState = "ready";
  experienceMessage = "「文章が伸びる様子を見る」を押してください。";
  $("#seed-input").value = seed;
  renderStatic();
  prepareForward();
  renderFirstRunGuide();
  setNotice(`Seed ${seed}の未学習モデルを初期化しました。`);
}

function renderStatic() {
  $("#parameter-count").textContent = model.parameterCount().toLocaleString();
  $("#parameter-total").textContent = `${model.parameterCount().toLocaleString()} total`;
  $("#vocab-count").textContent = tokenizer.vocabulary.length;
  $("#corpus-view").innerHTML = dataset.corpus.map((line) => `<div class="corpus-line">${escapeHtml(line)}</div>`).join("");
  renderArchitecture();
  renderGlossary();
  renderGradientOptions();
  renderSnapshots();
  renderTraining();
  renderParameters();
}

function promptTokenIds() {
  const encoded = tokenizer.encode($("#prompt-input").value, { addBos: true });
  return (encoded.length ? encoded : [tokenizer.tokenToId.get("<BOS>")]).slice(-model.config.contextLength);
}

function prepareForward() {
  try {
    const ids = promptTokenIds();
    currentTrace = model.forward(ids).trace;
    selectedToken = Math.min(selectedToken, ids.length - 1);
    selectedAttention = { row: Math.min(selectedAttention.row, ids.length - 1), col: Math.min(selectedAttention.col, ids.length - 1) };
    engine.load(currentTrace);
    renderForward();
    renderAttention();
    renderArchitecture();
    renderSnapshotComparison();
    setNotice(`${ids.length} TokenのForward Traceを固定しました。Nextで段階表示します。`);
  } catch (error) {
    setNotice(error.message, true);
  }
}

function renderPipeline() {
  const index = engine.index;
  $("#pipeline").innerHTML = FORWARD_STAGES.map((stage) => `<div class="pipeline-step ${stage.index < index ? "done" : stage.index === index ? "current" : ""}"><strong>${String(stage.index + 1).padStart(2, "0")}</strong>${escapeHtml(stage.label)}</div>`).join("");
}

function renderTokens() {
  if (!currentTrace) return;
  $("#token-cells").classList.remove("empty");
  $("#token-cells").innerHTML = currentTrace.tokens.map((token, index) => `<button class="token-cell ${index === selectedToken ? "selected" : ""}" data-token-index="${index}"><strong>${escapeHtml(token.text)}</strong><span>ID ${token.id}</span><span>POS ${token.position}</span></button>`).join("");
  $("#selected-token-label").textContent = `Token ${currentTrace.tokens[selectedToken].text} / POS ${selectedToken}`;
}

function rowLabels() { return currentTrace.tokens.map((token) => `${token.text} · ${token.position}`); }

function selectedRow(tensor) {
  const cols = tensor.shape[1];
  return tensor.data.slice(selectedToken * cols, (selectedToken + 1) * cols);
}

function stageDataHtml() {
  if (!currentTrace || engine.index < 0) return "";
  const stage = FORWARD_STAGES[engine.index];
  const labels = rowLabels();
  const p = precision();
  const headIndex = Number($("#attention-head").value);
  const head = currentTrace.heads[headIndex];
  const token = currentTrace.tokens[selectedToken];
  switch (stage.key) {
    case "tokenizer":
      return `<div class="data-block"><h3>Tokenizer result</h3><pre class="formula">text → tokens → IDs
${currentTrace.tokens.map((item) => `${item.position}: ${escapeHtml(item.text)} → ${item.id}`).join("\n")}</pre></div>`;
    case "tokenEmbeddings":
      return `<div class="data-block">${renderTensorRows(currentTrace.tokenEmbeddings, labels, { precision: p, title: "E_token[token_id]" })}</div>`;
    case "positionEmbeddings":
      return `<div class="data-block">${renderTensorRows(currentTrace.positionEmbeddings, labels, { precision: p, title: "E_pos[position]" })}</div>`;
    case "initialRepresentation":
      return `<div class="data-grid"><div class="data-block">${renderVector(selectedRow(currentTrace.tokenEmbeddings), { precision: p, label: `${token.text}: token` })}${renderVector(selectedRow(currentTrace.positionEmbeddings), { precision: p, label: `position ${selectedToken}` })}${renderVector(selectedRow(currentTrace.initialRepresentation), { precision: p, label: "sum" })}</div><pre class="formula">x[${selectedToken}, d] = E_token[${token.id}, d] + E_pos[${selectedToken}, d]</pre></div>`;
    case "layerNorm1": {
      const values = selectedRow(currentTrace.initialRepresentation);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
      return `<div class="data-grid"><div class="data-block">${renderVector(values, { precision: p, label: "x" })}${renderVector(selectedRow(currentTrace.layerNorm1), { precision: p, label: "LN1(x)" })}</div><pre class="formula">μ = ${fmt(mean)}
variance = ${fmt(variance)}
LN(x_d) = γ_d × (x_d - μ) / √(variance + 1e-5) + β_d</pre></div>`;
    }
    case "qkv":
      return `<div class="data-grid">${currentTrace.heads.map((item, index) => `<div class="data-block"><h3>Head ${index + 1}</h3>${renderVector(selectedRow(item.q), { precision: p, label: "Q" })}${renderVector(selectedRow(item.k), { precision: p, label: "K" })}${renderVector(selectedRow(item.v), { precision: p, label: "V" })}<pre class="formula">Q = LN1(x) × Wq_head${index + 1} + bq_head${index + 1}</pre></div>`).join("")}</div>`;
    case "rawScores":
      return `<div class="data-block"><h3>Head ${headIndex + 1}: QKᵀ / √4</h3><div class="matrix-scroll">${renderMatrix(head.rawScores, labels, labels, { precision: p })}</div></div>`;
    case "maskedScores":
      return `<div class="data-block"><h3>Head ${headIndex + 1}: j &gt; i → −∞</h3><div class="matrix-scroll">${renderMatrix(head.maskedScores, labels, labels, { precision: p })}</div></div>`;
    case "attentionWeights":
      return `<div class="data-block"><h3>Head ${headIndex + 1}: row softmax</h3><div class="matrix-scroll">${renderMatrix(head.attentionWeights, labels, labels, { precision: p })}</div><pre class="formula">A[i,j] = exp(score[i,j] - max) / Σ exp(score[i,k] - max)
未来位置のA[i,j] = 0</pre></div>`;
    case "attentionOutput":
      return `<div class="data-grid">${currentTrace.heads.map((item, index) => `<div class="data-block">${renderVector(selectedRow(item.output), { precision: p, label: `Head ${index + 1} Σ A×V` })}</div>`).join("")}<div class="data-block">${renderVector(selectedRow(currentTrace.concatenatedHeads), { precision: p, label: "concat" })}${renderVector(selectedRow(currentTrace.attentionOutput), { precision: p, label: "concat × Wo + bo" })}</div></div>`;
    case "residual1":
      return `<div class="data-grid"><div class="data-block">${renderVector(selectedRow(currentTrace.initialRepresentation), { precision: p, label: "Before x" })}${renderVector(selectedRow(currentTrace.attentionOutput), { precision: p, label: "+ Attention" })}${renderVector(selectedRow(currentTrace.residual1), { precision: p, label: "After h1" })}</div><pre class="formula">h1 = x + attention_output</pre></div>`;
    case "mlp":
      return `<div class="data-grid"><div class="data-block">${renderVector(selectedRow(currentTrace.layerNorm2), { precision: p, label: "LN2(h1)" })}${renderVector(selectedRow(currentTrace.mlpPreActivation), { precision: p, label: "Linear 1 (16d)" })}${renderVector(selectedRow(currentTrace.mlpActivation), { precision: p, label: "GELU (16d)" })}${renderVector(selectedRow(currentTrace.mlpOutput), { precision: p, label: "Linear 2 (8d)" })}</div><pre class="formula">pre = LN2(h1) × W1 + b1
hidden = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
mlp = hidden × W2 + b2</pre></div>`;
    case "residual2":
      return `<div class="data-grid"><div class="data-block">${renderVector(selectedRow(currentTrace.residual1), { precision: p, label: "Before h1" })}${renderVector(selectedRow(currentTrace.mlpOutput), { precision: p, label: "+ MLP" })}${renderVector(selectedRow(currentTrace.residual2), { precision: p, label: "After h2" })}</div><pre class="formula">h2 = h1 + mlp_output</pre></div>`;
    case "finalNorm":
      return `<div class="data-block">${renderVector(selectedRow(currentTrace.residual2), { precision: p, label: "h2" })}${renderVector(selectedRow(currentTrace.finalNorm), { precision: p, label: "LN_final(h2)" })}</div>`;
    case "logits":
      return `<div class="data-grid"><div class="data-block">${renderVector(selectedRow(currentTrace.finalNorm), { precision: p, label: "y" })}</div><div class="data-block">${renderVector(selectedRow(currentTrace.logits), { precision: p, label: `logits (${tokenizer.vocabulary.length})`, columns: 8 })}</div><pre class="formula">logits = y × W_vocab + b_vocab</pre></div>`;
    case "probabilities":
      return `<div class="data-grid"><div class="data-block">${renderPrediction(currentTrace, tokenizer.vocabulary, p, tokenizer.vocabulary.length)}</div><pre class="formula">P(token_j) = exp(logit_j - max(logits))
             / Σ_k exp(logit_k - max(logits))</pre></div>`;
    default: return "";
  }
}

function renderForward() {
  if (!currentTrace) return;
  const stage = engine.current();
  $("#stage-title").textContent = stage.label;
  $("#stage-counter").textContent = `${stage.index + 1} / ${FORWARD_STAGES.length}`;
  $("#stage-description").textContent = stage.description;
  renderPipeline();
  renderTokens();
  $("#stage-view").classList.remove("empty");
  $("#stage-view").innerHTML = stageDataHtml();
  $("#prediction-view").innerHTML = renderPrediction(currentTrace, tokenizer.vocabulary, precision());
  $("#prediction-view").classList.remove("empty");
  $("#token-trace").innerHTML = renderTokenTrace(currentTrace, selectedToken, precision());
  $("#token-trace").classList.remove("empty");
  $("#forward-previous").disabled = engine.index <= 0;
  $("#forward-next").disabled = engine.index >= FORWARD_STAGES.length - 1;
  renderArchitecture();
  renderFirstRunGuide();
}

function renderAttention() {
  if (!currentTrace) return;
  const headIndex = Number($("#attention-head").value);
  const mode = $("#attention-mode").value;
  const head = currentTrace.heads[headIndex];
  const labels = rowLabels();
  $("#attention-matrix").classList.remove("empty");
  $("#attention-matrix").innerHTML = renderMatrix(head[mode], labels, labels, { precision: precision(), interactive: true, selected: selectedAttention });
  $("#attention-inspector").classList.remove("empty");
  $("#attention-inspector").innerHTML = attentionCellDetails(currentTrace, headIndex, selectedAttention.row, selectedAttention.col, precision());
  $("#head-comparison").innerHTML = currentTrace.heads.map((item, index) => `<div class="data-block"><h3>Head ${index + 1} · Softmax Weight</h3><div class="matrix-scroll">${renderMatrix(item.attentionWeights, labels, labels, { precision: precision() })}</div></div>`).join("");
}

function topPrediction(trace) {
  const [rows, cols] = trace.probabilities.shape;
  const values = trace.probabilities.data.slice((rows - 1) * cols, rows * cols);
  let id = 0;
  values.forEach((value, index) => { if (value > values[id]) id = index; });
  return { id, token: tokenizer.vocabulary[id], probability: values[id] };
}

function renderTrainingResult() {
  const result = lastTrainingResult ?? trainingPlaybackSession;
  if (!result) return;
  const tokens = tokenizer.decode(result.sample.inputIds);
  const targets = tokenizer.decode(result.sample.targetIds);
  const predictions = result.beforeTrace.probabilities;
  const cols = predictions.shape[1];
  const predictedTokens = result.sample.inputIds.map((_, row) => {
    const values = predictions.data.slice(row * cols, (row + 1) * cols);
    let id = 0; values.forEach((value, index) => { if (value > values[id]) id = index; }); return tokenizer.vocabulary[id];
  });
  $("#training-sample").classList.remove("empty");
  $("#training-sample").innerHTML = `<div class="data-grid"><div class="data-block"><h3>Input Tokens</h3><code>${tokens.map(escapeHtml).join(" → ")}</code></div><div class="data-block"><h3>Target Tokens</h3><code>${targets.map(escapeHtml).join(" → ")}</code></div><div class="data-block"><h3>Predicted Tokens</h3><code>${predictedTokens.map(escapeHtml).join(" → ")}</code></div></div><div class="table-scroll"><table><thead><tr><th>Position</th><th>Input</th><th>Target</th><th>Prediction</th><th>Loss</th></tr></thead><tbody>${tokens.map((token, index) => `<tr><td>${index}</td><td>${escapeHtml(token)}</td><td>${escapeHtml(targets[index])}</td><td>${escapeHtml(predictedTokens[index])}</td><td>${fmt(result.beforeTrace.lossByPosition[index])}</td></tr>`).join("")}</tbody></table></div>`;
  $("#training-comparison").classList.remove("empty");
  const beforeTop = topPrediction(result.beforeTrace);
  if (!result.afterTrace) {
    $("#training-comparison").innerHTML = `<div class="comparison-grid"><div class="metric"><span>Current phase</span><strong>${escapeHtml(trainingPlaybackPhase || "forward")}</strong></div><div class="metric"><span>Loss before</span><strong>${fmt(result.lossBefore)}</strong></div><div class="metric"><span>Top before</span><strong>${escapeHtml(beforeTop.token)} ${(beforeTop.probability * 100).toFixed(2)}%</strong></div><div class="metric"><span>Gradient norm</span><strong>${result.rawGradientNorm == null ? "未計算" : fmt(result.rawGradientNorm)}</strong></div><div class="metric"><span>Clip scale</span><strong>${result.clipScale == null ? "未計算" : fmt(result.clipScale)}</strong></div><div class="metric"><span>Learning rate</span><strong>${fmt(trainer.learningRate)}</strong></div></div>`;
    return;
  }
  const afterTop = topPrediction(result.afterTrace);
  $("#training-comparison").innerHTML = `<div class="comparison-grid"><div class="metric"><span>Loss before</span><strong>${fmt(result.lossBefore)}</strong></div><div class="metric"><span>Loss after</span><strong>${fmt(result.lossAfter)}</strong></div><div class="metric"><span>Top before</span><strong>${escapeHtml(beforeTop.token)} ${(beforeTop.probability * 100).toFixed(2)}%</strong></div><div class="metric"><span>Top after</span><strong>${escapeHtml(afterTop.token)} ${(afterTop.probability * 100).toFixed(2)}%</strong></div><div class="metric"><span>Gradient norm</span><strong>${fmt(result.rawGradientNorm)}</strong></div><div class="metric"><span>Clip scale</span><strong>${fmt(result.clipScale)}</strong></div></div>`;
}

function renderTraining() {
  $("#training-step").textContent = `STEP ${trainer.step}`;
  $("#training-step-global").textContent = trainer.step;
  $("#loss-chart").innerHTML = renderLossChart(trainer.lossHistory);
  const initialLoss = trainer.lossHistory[0]?.loss;
  const currentLoss = trainer.lossHistory.at(-1)?.loss;
  $("#loss-summary").textContent = currentLoss == null ? "—" : `${fmt(initialLoss)} → ${fmt(currentLoss)}`;
  let evalLoss;
  try { evalLoss = trainer.evaluateLoss(); } catch { evalLoss = NaN; }
  const paramValues = model.parameters().flatMap(({ tensor }) => tensor.data);
  const paramStats = stats(paramValues);
  $("#diagnostics").innerHTML = `<div class="metric-grid"><div class="metric"><span>Evaluation Loss</span><strong>${fmt(evalLoss)}</strong></div><div class="metric"><span>Gradient Norm</span><strong>${fmt(lastTrainingResult?.rawGradientNorm ?? trainingPlaybackSession?.rawGradientNorm ?? 0)}</strong></div><div class="metric"><span>Parameter Norm</span><strong>${fmt(paramStats.norm)}</strong></div><div class="metric"><span>Max |Activation|</span><strong>${fmt(currentTrace?.maxAbsoluteActivation ?? 0)}</strong></div><div class="metric"><span>NaN / Infinity</span><strong>${currentTrace?.nanCount ?? 0}</strong></div><div class="metric"><span>Training Step</span><strong>${trainer.step}</strong></div></div>`;
  const phaseOrder = ["forward", "loss", "backward", "gradient", "update"];
  const phaseIndex = phaseOrder.indexOf(trainingPlaybackPhase);
  $$("#training-flow [data-training-phase]").forEach((item, index) => {
    item.classList.toggle("active", index === phaseIndex);
    item.classList.toggle("done", phaseIndex >= 0 && index < phaseIndex);
  });
  renderTrainingResult();
  renderFirstRunGuide();
}

function renderParameters() {
  const params = model.parameters();
  $("#parameter-table").innerHTML = renderParameterTable(params, trainer.lastUpdate, precision(), selectedParameter);
  const found = model.parameterMap.get(selectedParameter) ?? params[0].tensor;
  $("#parameter-inspector").classList.remove("empty");
  $("#parameter-inspector").innerHTML = renderParameterInspector(selectedParameter, found, trainer.lastUpdate?.[selectedParameter], trainer.learningRate, precision());
}

function renderArchitecture() {
  const current = engine.current()?.key ?? "";
  const node = (key, label, detail = "") => `<div class="architecture-node ${current === key ? "current" : ""}"><strong>${label}</strong>${detail ? `<br><span class="microcopy">${detail}</span>` : ""}</div>`;
  $("#architecture-view").innerHTML = [
    node("tokenizer", "Tokens", "Context ≤ 8"), "<div class=\"architecture-arrow\">↓</div>",
    node("initialRepresentation", "Token Embedding + Position Embedding", "[T, 8] + [T, 8]"), "<div class=\"architecture-arrow\">↓</div>",
    node("layerNorm1", "Pre-LayerNorm 1", "ε = 1e-5"), "<div class=\"architecture-arrow\">↓</div>",
    `<div class="architecture-split">${node("qkv", "Head 1", "Q/K/V [T, 4]")}${node("qkv", "Head 2", "Q/K/V [T, 4]")}</div>`, "<div class=\"architecture-arrow\">↓ causal mask · softmax · concat</div>",
    node("attentionOutput", "Attention Projection", "[T, 8] × Wo"), "<div class=\"architecture-arrow\">↓ + residual</div>",
    node("mlp", "Pre-LayerNorm 2 → MLP", "8 → 16 GELU → 8"), "<div class=\"architecture-arrow\">↓ + residual</div>",
    node("finalNorm", "Final LayerNorm", "[T, 8]"), "<div class=\"architecture-arrow\">↓</div>",
    node("logits", "Vocabulary Projection", `[T, ${model?.config.vocabSize ?? "V"}]`), "<div class=\"architecture-arrow\">↓</div>",
    node("probabilities", "Softmax → Next Token", "ΣP = 1"),
  ].join("");
}

function renderGlossary() {
  const terms = {
    Token: "文字列をモデルが扱う最小単位へ分けたもの。この実装では単語と句読点。",
    Embedding: "離散的なToken IDや位置を、学習可能な8次元実数ベクトルへ写像する表。",
    LayerNorm: "Tokenごとに次元方向の平均と分散を整え、学習可能なγ・βを適用する。",
    Query: "現在のTokenが何を探すかを表すベクトル。",
    Key: "参照候補Tokenがどの特徴を持つかを表すベクトル。",
    Value: "Attention weightで実際に混合される情報ベクトル。",
    Attention: "QとKの類似度から、過去TokenのVをどの割合で混ぜるか決める計算。",
    "Causal Mask": "未来Tokenを参照できないよう、j > iのscoreをsoftmax対象外にする。",
    Residual: "変換前のベクトルを変換結果へ足し、元の情報経路を残す接続。",
    GELU: "MLP内の非線形活性化関数。本実装はtanh近似を使う。",
    Logit: "Vocabulary各Tokenに対するsoftmax前の制約なしscore。",
    Loss: "正解Tokenへ割り当てた確率の低さを−logで測る値。",
    Gradient: "Parameterを微小に増やしたときLossがどちらへどれだけ変わるか。",
    SGD: "Parameterからlearning rate×gradientを引く単純な更新法。",
  };
  $("#glossary").innerHTML = Object.entries(terms).map(([term, description]) => `<div><dl><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(description)}</dd></dl></div>`).join("");
}

function renderGradientOptions() {
  $("#gradient-parameter").innerHTML = model.parameters().map(({ name }) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  $("#gradient-parameter").value = selectedParameter;
  updateGradientIndexLimit();
}

function updateGradientIndexLimit() {
  const tensor = model.parameterMap.get($("#gradient-parameter").value);
  $("#gradient-index").max = Math.max(0, tensor.data.length - 1);
  if (Number($("#gradient-index").value) > tensor.data.length - 1) $("#gradient-index").value = 0;
}

function renderSnapshots() {
  const labels = [...trainer.snapshots.keys()];
  const selected = $("#snapshot-select").value;
  $("#snapshot-select").innerHTML = labels.map((label) => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`).join("");
  if (labels.includes(selected)) $("#snapshot-select").value = selected;
  renderSnapshotComparison();
}

function renderSnapshotComparison() {
  if (!currentTrace || !trainer.snapshots.size) return;
  const label = $("#snapshot-select").value || [...trainer.snapshots.keys()][0];
  try {
    const ids = promptTokenIds();
    const snapshotTrace = trainer.traceFromSnapshot(label, ids);
    const before = topPrediction(snapshotTrace);
    const now = topPrediction(currentTrace);
    const sample = dataset.samples[0];
    const snapshotLoss = trainer.traceFromSnapshot(label, sample.inputIds, sample.targetIds).loss;
    const currentLoss = model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss;
    const attentionBefore = snapshotTrace.heads[0].attentionWeights.data;
    const attentionNow = currentTrace.heads[0].attentionWeights.data;
    const meanAttentionDelta = attentionBefore.reduce((sum, value, index) => sum + Math.abs(value - attentionNow[index]), 0) / attentionBefore.length;
    $("#snapshot-comparison").classList.remove("empty");
    $("#snapshot-comparison").innerHTML = `<div class="comparison-grid"><div class="metric"><span>${escapeHtml(label)} prediction</span><strong>${escapeHtml(before.token)} ${(before.probability * 100).toFixed(2)}%</strong></div><div class="metric"><span>Current prediction</span><strong>${escapeHtml(now.token)} ${(now.probability * 100).toFixed(2)}%</strong></div><div class="metric"><span>${escapeHtml(label)} corpus sample loss</span><strong>${fmt(snapshotLoss)}</strong></div><div class="metric"><span>Current corpus sample loss</span><strong>${fmt(currentLoss)}</strong></div><div class="metric"><span>Head 1 mean |Δattention|</span><strong>${fmt(meanAttentionDelta)}</strong></div><div class="metric"><span>Probability Δ (current top)</span><strong>${fmt(now.probability - (snapshotTrace.probabilities.data.at(-snapshotTrace.probabilities.shape[1] + now.id)))}</strong></div></div>`;
  } catch (error) {
    $("#snapshot-comparison").textContent = error.message;
  }
}

async function runTraining(count) {
  if (autoTraining && count !== Infinity) return;
  try {
    if (experiencePlaybackRun) cancelLanguagePlayback();
    if (learningObserverRun && !learningObserverRun.complete) cancelLearningObserver();
    trainingPlaybackPhase = "";
    trainer.learningRate = Number($("#learning-rate").value);
    trainer.clipNorm = Number($("#clip-norm").value);
    if (!(trainer.learningRate > 0) || !(trainer.clipNorm > 0)) throw new Error("Learning rateとClip normは正の有限値にしてください。");
    let completed = 0;
    setTrainingButtons(true);
    while ((count === Infinity ? autoTraining : completed < count)) {
      lastTrainingResult = trainer.trainOneStep();
      completed += 1;
      if (completed % 10 === 0) {
        $("#training-step").textContent = `STEP ${trainer.step}`;
        $("#training-step-global").textContent = trainer.step;
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    currentTrace = model.forward(promptTokenIds()).trace;
    engine.load(currentTrace).index;
    engine.run();
    renderForward(); renderAttention(); renderTraining(); renderParameters(); renderSnapshots();
    setNotice(`${completed} Training Stepを完了。Loss・Gradient・Parameter更新を実値で反映しました。`);
  } catch (error) {
    autoTraining = false;
    setNotice(`Training停止: ${error.message}`, true);
  } finally {
    setTrainingButtons(false);
  }
}

function setTrainingButtons(running) {
  $("#train-one").disabled = running;
  $$('[data-train-count]').forEach((button) => { button.disabled = running; });
  $("#auto-train").disabled = running;
  $("#pause-train").disabled = !running || !autoTraining;
}

function chooseToken(logits, mode, temperature) {
  const probabilities = generationDistribution(logits, {
    bosId: tokenizer.tokenToId.get("<BOS>"),
    temperature: mode === "temperature" ? temperature : 1,
  });
  return generationChoice(probabilities, mode).id;
}

function generate(count) {
  try {
    if (experiencePlaybackRun) cancelLanguagePlayback();
    if (learningObserverRun && !learningObserverRun.complete) cancelLearningObserver();
    let ids = promptTokenIds();
    const visibleTokens = tokenizer.tokenize($("#prompt-input").value);
    const mode = $("#sampling-mode").value;
    const temperature = Number($("#temperature").value);
    for (let i = 0; i < count; i += 1) {
      const trace = model.forward(ids).trace;
      const [rows, cols] = trace.logits.shape;
      const logits = trace.logits.data.slice((rows - 1) * cols, rows * cols);
      const nextId = chooseToken(logits, mode, temperature);
      if (nextId === tokenizer.tokenToId.get("<EOS>")) break;
      visibleTokens.push(tokenizer.vocabulary[nextId]);
      ids.push(nextId);
      ids = ids.slice(-model.config.contextLength);
    }
    const text = tokenizer.detokenize(visibleTokens);
    $("#generation-output").textContent = text || "<EOS>";
    $("#prompt-input").value = text;
    prepareForward();
  } catch (error) { setNotice(error.message, true); }
}

function runGradientCheck() {
  try {
    const name = $("#gradient-parameter").value;
    const tensor = model.parameterMap.get(name);
    const index = Number($("#gradient-index").value);
    if (!Number.isInteger(index) || index < 0 || index >= tensor.data.length) throw new Error("Indexが範囲外です。");
    const sample = dataset.samples[0];
    model.zeroGrad();
    const forward = model.forward(sample.inputIds, { targets: sample.targetIds });
    model.backward(forward.lossTensor);
    const analytical = tensor.grad[index];
    const original = tensor.data[index];
    const epsilon = 1e-5;
    tensor.data[index] = original + epsilon;
    const plus = model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss;
    tensor.data[index] = original - epsilon;
    const minus = model.forward(sample.inputIds, { targets: sample.targetIds }).trace.loss;
    tensor.data[index] = original;
    const numerical = (plus - minus) / (2 * epsilon);
    const error = relativeError(analytical, numerical);
    $("#gradient-result").innerHTML = `<strong class="${error < 1e-3 ? "delta-positive" : "delta-negative"}">${error < 1e-3 ? "PASS" : "CHECK"}</strong> — ${escapeHtml(name)}[${index}]
analytical = ${fmt(analytical)}
numerical  = ${fmt(numerical)}
relative error = ${error.toExponential(3)}
ε = ${epsilon}`;
    renderParameters();
    setNotice(`Gradient Check ${error < 1e-3 ? "合格" : "要確認"}: relative error ${error.toExponential(3)}`);
  } catch (error) { $("#gradient-result").textContent = error.message; setNotice(error.message, true); }
}

function exportState() {
  const json = exportApplicationState(model, trainer, dataset);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = `glassbox-ai-ii-step-${trainer.step}.json`; link.click();
  URL.revokeObjectURL(url);
  $("#io-status").textContent = `${link.download} を出力しました (${json.length.toLocaleString()} chars)`;
}

async function importState(file) {
  try {
    cancelLanguagePlayback();
    cancelLearningObserver({ discard: true });
    const restored = importApplicationState(await file.text());
    ({ model, trainer, dataset, tokenizer } = restored);
    $("#seed-input").value = trainer.seed;
    $("#learning-rate").value = trainer.learningRate;
    $("#clip-norm").value = trainer.clipNorm;
    engine = new ForwardStepEngine(); currentTrace = null; lastTrainingResult = null;
    renderStatic(); prepareForward();
    $("#io-status").textContent = `${file.name} を検証し、STEP ${trainer.step}を復元しました。`;
    setNotice("JSON状態を安全に復元しました。");
  } catch (error) { $("#io-status").textContent = `拒否: ${error.message}`; setNotice(`Import拒否: ${error.message}`, true); }
}

function bindEvents() {
  $$(".tab").forEach((button) => button.addEventListener("click", () => selectTab(button.dataset.tab)));
  $("#guide-forward").addEventListener("click", runBeginnerAutoObserve);
  $("#guide-training").addEventListener("click", showBeginnerDetail);
  $("#detail-from-bridge").addEventListener("click", showBeginnerDetail);
  $("#experience-pause").addEventListener("click", pauseLanguagePlayback);
  $("#learning-start").addEventListener("click", runLearningObserver);
  $("#learning-pause").addEventListener("click", pauseLearningObserver);
  $("#prepare-forward").addEventListener("click", prepareForward);
  $("#forward-next").addEventListener("click", () => { engine.next(); renderForward(); });
  $("#forward-previous").addEventListener("click", () => { engine.previous(); renderForward(); });
  $("#forward-run").addEventListener("click", () => {
    engine.run();
    renderForward();
    if (firstRunMode === "forward") focusTrainingGuide();
  });
  $("#forward-reset").addEventListener("click", () => { engine.reset(); renderForward(); });
  $("#token-cells").addEventListener("click", (event) => {
    const target = event.target.closest("[data-token-index]"); if (!target) return;
    selectedToken = Number(target.dataset.tokenIndex); selectedAttention = { row: selectedToken, col: Math.min(selectedAttention.col, selectedToken) }; renderForward(); renderAttention();
  });
  $("#attention-head").addEventListener("change", () => { renderAttention(); renderForward(); });
  $("#attention-mode").addEventListener("change", renderAttention);
  $("#attention-matrix").addEventListener("click", (event) => {
    const target = event.target.closest("[data-attention-row]"); if (!target) return;
    selectedAttention = { row: Number(target.dataset.attentionRow), col: Number(target.dataset.attentionCol) }; renderAttention();
  });
  $("#precision").addEventListener("change", () => { renderForward(); renderAttention(); renderTraining(); renderParameters(); renderSnapshotComparison(); });
  $("#learning-rate").addEventListener("change", () => { trainer.learningRate = Number($("#learning-rate").value); renderParameters(); });
  $("#clip-norm").addEventListener("change", () => { trainer.clipNorm = Number($("#clip-norm").value); });
  $("#reinitialize").addEventListener("click", () => {
    if (confirm("現在の学習済みParameterと履歴を破棄し、指定Seedから再初期化しますか？")) initialize(Number($("#seed-input").value));
  });
  $("#generate-one").addEventListener("click", () => generate(1));
  $("#generate-many").addEventListener("click", () => generate(Math.max(1, Math.min(32, Number($("#generation-count").value) || 1))));
  $("#train-one").addEventListener("click", () => runTraining(1));
  $$('[data-train-count]').forEach((button) => button.addEventListener("click", () => runTraining(Number(button.dataset.trainCount))));
  $("#auto-train").addEventListener("click", () => { autoTraining = true; runTraining(Infinity); });
  $("#pause-train").addEventListener("click", () => { autoTraining = false; });
  $("#parameter-table").addEventListener("click", (event) => { const row = event.target.closest("[data-parameter]"); if (!row) return; selectedParameter = row.dataset.parameter; renderParameters(); });
  $("#parameter-inspector").addEventListener("click", (event) => { const cell = event.target.closest("[data-param-index]"); if (!cell) return; $("#gradient-parameter").value = selectedParameter; updateGradientIndexLimit(); $("#gradient-index").value = cell.dataset.paramIndex; setNotice(`${selectedParameter}[${cell.dataset.paramIndex}]をDebug Gradient Check対象に設定しました。`); });
  $("#gradient-parameter").addEventListener("change", updateGradientIndexLimit);
  $("#run-gradient-check").addEventListener("click", runGradientCheck);
  $("#capture-snapshot").addEventListener("click", () => { trainer.captureSnapshot(`STEP ${trainer.step} · ${new Date().toLocaleTimeString("ja-JP")}`); renderSnapshots(); setNotice("現在ParameterをSnapshotへ複製しました。"); });
  $("#snapshot-select").addEventListener("change", renderSnapshotComparison);
  $("#export-state").addEventListener("click", exportState);
  $("#import-state").addEventListener("change", (event) => { const [file] = event.target.files; if (file) importState(file); event.target.value = ""; });
}

bindEvents();
initialize(42);

if (firstRunRequest === "forward" || firstRunRequest === "training") {
  window.requestAnimationFrame(() => {
    selectTab(firstRunRequest === "training" ? "training" : "playground");
    renderFirstRunGuide();
    focusFirstRunTarget();
  });
}
