import {
  INPUT_NAMES,
  OUTPUT_NAMES,
  PARAMETER_SPECS,
  cloneNetwork,
  createNetwork,
  getParameterValue,
} from './neural-network.js';
import { StepEngine } from './step-engine.js';
import { getGradientValue } from './backpropagation.js';
import { EXPLANATIONS, GLOSSARY } from './explanations.js';
import { parseStateJson, serializeState } from './serializer.js';
import { OperationLog, downloadTextFile } from './state-manager.js';
import { renderNetwork } from './visualization.js';
import {
  GRID_ACTIONS,
  applyGridAction,
  cloneGridWorld,
  createGridWorld,
  getDirection,
  getTeacherAction,
  gridWorldSignature,
  senseGridWorld,
} from './grid-world.js';
import {
  actionDisplay,
  renderGridBoard,
  renderGridSensors,
} from './grid-world-visualization.js';
import { REWARD_PROFILES, getRewardProfile } from './reward-profiles.js';
import { ReinforcementStepEngine, RL_DEFAULTS } from './reinforcement-learning.js';
import { serializeReinforcementHistory } from './reinforcement-history.js';

const PRESETS = Object.freeze({
  simple: { label: 'プリセット1：単純な入力', inputs: [0.8, 0, 0, 0, 0], targetIndex: 0 },
  combined: { label: 'プリセット2：複数入力の組み合わせ', inputs: [0.7, -0.5, 0.3, 0, 0.8], targetIndex: 1 },
  negative: { label: 'プリセット3：負の値を含む入力', inputs: [-0.9, 0.4, -0.6, 0.2, 0.1], targetIndex: 2 },
});

const quickStartRequest = new URLSearchParams(window.location.search).get('path');
const requestedQuickStartModes = new Set(['supervised', 'reinforcement', 'explore']);

const elements = Object.fromEntries(
  [
    'input-controls', 'preset-select', 'apply-preset', 'seed-input', 'learning-rate', 'precision-select',
    'network-svg', 'connection-inspector', 'phase-value', 'step-value', 'learning-count', 'status-message',
    'quick-supervised', 'quick-reinforcement', 'quick-explore', 'quick-start-status',
    'timeline-first', 'step-previous', 'step-next', 'run-to-end', 'auto-play', 'auto-pause', 'speed-select',
    'step-progress', 'step-title', 'step-description', 'formula-display', 'natural-explanation',
    'target-controls', 'start-learning', 'comparison', 'parameter-body', 'parameter-count',
    'calculation-reset', 'undo-learning', 'full-reset', 'save-state', 'load-state-file',
    'export-log-text', 'export-log-json', 'clear-log', 'log-output', 'glossary', 'output-summary',
    'grid-board', 'grid-world-seed', 'grid-world-reset', 'grid-sync-sensors', 'grid-prepare-teacher',
    'grid-apply-action', 'grid-sync-state', 'grid-agent-direction', 'grid-world-step', 'grid-food-count',
    'grid-collision-count', 'grid-danger-count', 'grid-last-reward', 'grid-last-event',
    'grid-prediction', 'grid-teacher-action', 'grid-teacher-length', 'grid-sensor-list',
    'rl-reward-profile', 'rl-random-seed', 'rl-gamma', 'rl-temperature', 'rl-learning-rate',
    'rl-max-steps', 'rl-profile-description', 'rl-start-episode', 'rl-first', 'rl-previous',
    'rl-next', 'rl-run-end', 'rl-run-ten', 'rl-session-reset', 'rl-export-history', 'rl-step-progress',
    'rl-grid-board', 'rl-episode-number', 'rl-step-position', 'rl-cumulative-reward',
    'rl-environment-time', 'rl-foods', 'rl-exploration', 'rl-exploitation', 'rl-entropy', 'rl-policy-bars',
    'rl-diagnostic-label', 'rl-diagnostic-evidence', 'rl-phase', 'rl-current-title',
    'rl-current-description', 'rl-formula', 'rl-explanation', 'rl-experience-body',
    'rl-axis-note', 'rl-history-chart', 'rl-history-body', 'rl-parameter-body', 'rl-parameter-count',
  ].map((id) => [id, document.getElementById(id)]),
);

let inputs = [...PRESETS.simple.inputs];
let targetIndex = PRESETS.simple.targetIndex;
let learningRate = 0.1;
let precision = 4;
let engine = new StepEngine(createNetwork('glassbox-1'), inputs);
let lastLearningUndoNetwork = null;
let gridWorld = createGridWorld('grid-1');
let gridSensorsSynchronized = false;
let synchronizedGridSignature = null;
let rlEngine = null;
let rlEpisodeHistory = [];
let rlEpisodeRecords = [];
let rlNextEpisodeNumber = 1;
let rlSessionStartNetwork = null;
let rlSessionStartWorld = null;
let rlSessionStartInputs = null;
const operationLog = new OperationLog();
let autoTimer = null;
let quickStartMode = requestedQuickStartModes.has(quickStartRequest) ? quickStartRequest : 'supervised';
let experienceState = quickStartMode === 'explore' ? 'detail' : 'ready';
let experienceMessage = experienceState === 'detail'
  ? '「次のステップ」で、同じ計算を止めながら確認できます。'
  : '「自動で見る」を押してください。';

function formatNumber(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const rounded = Number(value).toFixed(precision);
  return Object.is(Number(rounded), -0) ? Number(0).toFixed(precision) : rounded;
}

function formatSigned(value, suffix = '') {
  const sign = value > 0 ? '+' : '';
  return `${sign}${formatNumber(value)}${suffix}`;
}

function addLog(message) {
  if (message) operationLog.add(message);
}

function clearRlSession(message = null) {
  const hadSession = Boolean(rlEngine || rlEpisodeHistory.length || rlSessionStartNetwork);
  rlEngine = null;
  rlEpisodeHistory = [];
  rlEpisodeRecords = [];
  rlNextEpisodeNumber = 1;
  rlSessionStartNetwork = null;
  rlSessionStartWorld = null;
  rlSessionStartInputs = null;
  if (hadSession && message) addLog(message);
}

function readRlConfig(episodeNumber = rlNextEpisodeNumber) {
  return {
    rewardProfile: elements['rl-reward-profile'].value || RL_DEFAULTS.rewardProfile,
    randomSeed: elements['rl-random-seed'].value.trim() || RL_DEFAULTS.randomSeed,
    gamma: Number(elements['rl-gamma'].value),
    temperature: Number(elements['rl-temperature'].value),
    learningRate: Number(elements['rl-learning-rate'].value),
    maxSteps: Number(elements['rl-max-steps'].value),
    episodeNumber,
  };
}

function createRlEpisodeWorld(episodeNumber) {
  const baseSeed = elements['grid-world-seed'].value.trim() || 'grid-1';
  return createGridWorld(`${baseSeed}:rl:${episodeNumber}`);
}

function rememberRlSessionStart() {
  if (rlSessionStartNetwork) return;
  rlSessionStartNetwork = cloneNetwork(engine.current.network);
  rlSessionStartWorld = cloneGridWorld(gridWorld);
  rlSessionStartInputs = [...inputs];
}

function invalidateGridSynchronization() {
  gridSensorsSynchronized = false;
  synchronizedGridSignature = null;
}

function isGridInputSynchronized() {
  if (!gridSensorsSynchronized) return false;
  if (synchronizedGridSignature !== gridWorldSignature(gridWorld)) return false;
  const sensed = senseGridWorld(gridWorld);
  return sensed.values.every((value, index) => Math.abs(value - inputs[index]) < 1e-15);
}

function synchronizeGridObservation({ includeTeacher = false } = {}) {
  pauseAuto();
  const sensed = senseGridWorld(gridWorld);
  const currentNetwork = cloneNetwork(engine.current.network);
  inputs = [...sensed.values];
  gridSensorsSynchronized = true;
  synchronizedGridSignature = gridWorldSignature(gridWorld);

  let teacher = null;
  if (includeTeacher) {
    teacher = getTeacherAction(gridWorld);
    if (!teacher.safePathFound || teacher.actionIndex === null) {
      invalidateGridSynchronization();
      elements['status-message'].textContent = '安全な教師経路を作成できませんでした。';
      addLog('[環境] 教師データ準備を中止：安全経路なし');
      render();
      return;
    }
    targetIndex = teacher.actionIndex;
  }

  engine = new StepEngine(currentNetwork, inputs);
  const values = sensed.values.map((value) => formatNumber(value)).join(', ');
  if (teacher) {
    addLog(
      `[環境] センサー [${values}] と教師 ${actionDisplay(teacher.actionIndex)} を学習データとして準備`,
    );
  } else {
    addLog(`[環境] 現在世界のセンサー [${values}] をネットワーク入力へ反映`);
  }
  render();
}

function syncInputsToControls() {
  inputs.forEach((value, index) => {
    const slider = document.getElementById(`input-range-${index}`);
    const number = document.getElementById(`input-number-${index}`);
    const current = document.getElementById(`input-current-${index}`);
    if (slider) slider.value = value;
    if (number) number.value = value;
    if (current) current.textContent = formatNumber(value);
  });
}

function resetCalculation(message = '[リセット] 計算だけリセット（現在のパラメータを維持）') {
  pauseAuto();
  const currentNetwork = cloneNetwork(engine.current.network);
  engine = new StepEngine(currentNetwork, inputs);
  addLog(message);
  render();
}

function updateInput(index, rawValue) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return;
  inputs[index] = Math.max(-1, Math.min(1, parsed));
  invalidateGridSynchronization();
  syncInputsToControls();
  resetCalculation(`[入力] ${INPUT_NAMES[index]}を${inputs[index]}へ変更し、計算状態をリセット`);
}

function createInputControls() {
  const fragment = document.createDocumentFragment();
  INPUT_NAMES.forEach((name, index) => {
    const row = document.createElement('div');
    row.className = 'input-row';
    row.innerHTML = `
      <label for="input-range-${index}">${name} <span class="symbol">x${index + 1}</span></label>
      <input id="input-range-${index}" type="range" min="-1" max="1" step="0.01" value="${inputs[index]}">
      <input id="input-number-${index}" type="number" min="-1" max="1" step="0.01" value="${inputs[index]}" aria-label="${name}の数値入力">
      <output id="input-current-${index}">${formatNumber(inputs[index])}</output>
    `;
    row.querySelector('input[type="range"]').addEventListener('input', (event) => updateInput(index, event.target.value));
    row.querySelector('input[type="number"]').addEventListener('input', (event) => updateInput(index, event.target.value));
    fragment.append(row);
  });
  elements['input-controls'].replaceChildren(fragment);
}

function createPresetOptions() {
  for (const [key, preset] of Object.entries(PRESETS)) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = preset.label;
    elements['preset-select'].append(option);
  }
}

function createRlRewardOptions() {
  for (const profile of Object.values(REWARD_PROFILES)) {
    const option = document.createElement('option');
    option.value = profile.key;
    option.textContent = profile.label;
    elements['rl-reward-profile'].append(option);
  }
  elements['rl-reward-profile'].value = RL_DEFAULTS.rewardProfile;
}

function createRlParameterRows() {
  const fragment = document.createDocumentFragment();
  for (const spec of PARAMETER_SPECS) {
    const row = document.createElement('tr');
    row.id = `rl-parameter-${spec.name}`;
    row.innerHTML = `
      <th scope="row">${spec.name}</th>
      <td data-field="current"></td>
      <td data-field="gradient"></td>
      <td data-field="update"></td>
      <td data-field="after"></td>
    `;
    fragment.append(row);
  }
  elements['rl-parameter-body'].replaceChildren(fragment);
  elements['rl-parameter-count'].textContent = `${PARAMETER_SPECS.length} / 39`;
}

function createTargetControls() {
  const fragment = document.createDocumentFragment();
  OUTPUT_NAMES.forEach((name, index) => {
    const label = document.createElement('label');
    label.className = 'target-option';
    label.innerHTML = `<input type="radio" name="target" value="${index}"${index === targetIndex ? ' checked' : ''}> 正解：${name}`;
    label.querySelector('input').addEventListener('change', () => {
      targetIndex = index;
      if (engine.hasLearningTimeline) {
        resetCalculation(`[学習] 正解クラスを${name}へ変更し、既存の学習ステップを破棄`);
      } else {
        addLog(`[学習] 正解クラスを${name}に設定`);
        render();
      }
    });
    fragment.append(label);
  });
  elements['target-controls'].replaceChildren(fragment);
}

function syncTargetControls() {
  document.querySelectorAll('input[name="target"]').forEach((radio) => {
    radio.checked = Number(radio.value) === targetIndex;
  });
}

function createGlossary() {
  const list = document.createElement('dl');
  for (const [term, definition] of GLOSSARY) {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = term;
    dd.textContent = definition;
    list.append(dt, dd);
  }
  elements.glossary.replaceChildren(list);
}

function createParameterRows() {
  const fragment = document.createDocumentFragment();
  for (const spec of PARAMETER_SPECS) {
    const row = document.createElement('tr');
    row.id = `parameter-${spec.name}`;
    row.innerHTML = `
      <th scope="row">${spec.name}</th>
      <td>${spec.from}</td>
      <td>${spec.to}</td>
      <td data-field="current"></td>
      <td data-field="before"></td>
      <td data-field="gradient"></td>
      <td data-field="update"></td>
      <td data-field="after"></td>
    `;
    fragment.append(row);
  }
  elements['parameter-body'].replaceChildren(fragment);
  elements['parameter-count'].textContent = `${PARAMETER_SPECS.length} / 39`;
}

function renderFormula(step) {
  const fragment = document.createDocumentFragment();
  if (!step.formulaTerms.length) {
    const placeholder = document.createElement('p');
    placeholder.className = 'formula-placeholder';
    placeholder.textContent = 'ステップを進めると、ここに実際の値を代入した数式が表示されます。';
    fragment.append(placeholder);
  }
  for (const term of step.formulaTerms) {
    const line = document.createElement('div');
    line.className = `formula-term ${term.status}`;
    const marker = document.createElement('span');
    marker.className = 'term-marker';
    marker.textContent = term.status === 'done' ? '✓' : term.status === 'current' ? '▶' : '○';
    line.append(marker);
    for (const part of term.parts) {
      const span = document.createElement('span');
      if (part.type === 'number') {
        span.className = 'formula-number';
        if (part.percent) span.textContent = `${formatNumber(part.value * 100)}%`;
        else if (part.percentagePoint) span.textContent = `${formatSigned(part.value * 100, 'ポイント')}`;
        else span.textContent = formatNumber(part.value);
      } else {
        span.textContent = part.value;
      }
      line.append(span);
    }
    fragment.append(line);
  }
  elements['formula-display'].replaceChildren(fragment);
}

function inspectParameter(spec) {
  const value = getParameterValue(engine.current.network, spec);
  const sign = value > 0 ? '正（同方向）' : value < 0 ? '負（反対方向）' : 'ゼロ';
  elements['connection-inspector'].innerHTML = `
    <strong>${spec.name}</strong>
    <span>${spec.from} → ${spec.to}</span>
    <span>現在値 ${formatNumber(value)}</span>
    <span>${sign} / |w| = ${formatNumber(Math.abs(value))}</span>
  `;
  document.querySelectorAll('#parameter-body tr.inspected').forEach((row) => row.classList.remove('inspected'));
  document.getElementById(`parameter-${spec.name}`)?.classList.add('inspected');
}

function renderOutputSummary(step) {
  const fragment = document.createDocumentFragment();
  step.forward.probabilities.forEach((probability, index) => {
    const item = document.createElement('div');
    item.className = `output-chip${step.forward.selectedIndex === index ? ' selected' : ''}`;
    item.innerHTML = `<span>${OUTPUT_NAMES[index]} <small>${GRID_ACTIONS[index].label}</small></span><strong>${probability === null ? '—' : `${formatNumber(probability * 100)}%`}</strong>`;
    fragment.append(item);
  });
  if (step.forward.selectedIndex !== null) {
    const decision = document.createElement('div');
    decision.className = 'decision-chip';
    decision.textContent = `選択された判断：${actionDisplay(step.forward.selectedIndex)}（argmax）`;
    fragment.append(decision);
  }
  elements['output-summary'].replaceChildren(fragment);
}

function renderGridWorldPanel(step) {
  const sensed = senseGridWorld(gridWorld);
  const teacher = getTeacherAction(gridWorld);
  const synchronized = isGridInputSynchronized();
  const direction = getDirection(gridWorld);
  const transition = gridWorld.lastTransition;

  renderGridBoard(elements['grid-board'], gridWorld);
  renderGridSensors(elements['grid-sensor-list'], sensed, formatNumber, synchronized);

  elements['grid-sync-state'].textContent = synchronized
    ? '同期済み'
    : '未同期：世界と入力が不一致';
  elements['grid-sync-state'].className = `sync-badge ${synchronized ? 'ready' : 'stale'}`;
  elements['grid-agent-direction'].textContent = `${direction.symbol} ${direction.label}`;
  elements['grid-world-step'].textContent = String(gridWorld.counters.steps);
  elements['grid-food-count'].textContent = String(gridWorld.counters.foods);
  elements['grid-collision-count'].textContent = String(gridWorld.counters.collisions);
  elements['grid-danger-count'].textContent = String(gridWorld.counters.dangerHits);
  elements['grid-last-reward'].textContent = formatSigned(transition?.reward ?? 0);
  elements['grid-last-reward'].className =
    (transition?.reward ?? 0) > 0 ? 'positive-reward' : (transition?.reward ?? 0) < 0 ? 'negative-reward' : '';
  elements['grid-last-event'].textContent = transition?.event ?? 'まだ環境遷移はありません。';

  const prediction = step.forward.selectedIndex;
  const decisionReady = ['argmax', 'forward-after-update', 'comparison'].includes(step.stage);
  elements['grid-prediction'].textContent =
    prediction === null
      ? '推論未完了'
      : `${actionDisplay(prediction)}${synchronized ? '' : '（現在世界とは未同期）'}`;
  elements['grid-teacher-action'].textContent = teacher.action
    ? actionDisplay(teacher.actionIndex)
    : '安全経路なし';
  elements['grid-teacher-length'].textContent = teacher.safePathFound
    ? `餌まで残り${teacher.pathLength}行動の安全最短経路`
    : '危険を避ける経路を発見できませんでした';

  elements['grid-prepare-teacher'].disabled = !teacher.safePathFound || teacher.actionIndex === null;
  elements['grid-apply-action'].disabled = !synchronized || prediction === null || !decisionReady;
}

function renderParameters(step) {
  for (const spec of PARAMETER_SPECS) {
    const row = document.getElementById(`parameter-${spec.name}`);
    const info = step.training.parameterInfo[spec.name] ?? {};
    row.classList.toggle('active', step.active?.parameter === spec.name);
    row.querySelector('[data-field="current"]').textContent = formatNumber(getParameterValue(step.network, spec));
    row.querySelector('[data-field="before"]').textContent = info.before === undefined ? '—' : formatNumber(info.before);
    row.querySelector('[data-field="gradient"]').textContent = info.gradient === undefined ? '—' : formatNumber(info.gradient);
    row.querySelector('[data-field="update"]').textContent = info.update === undefined ? '—' : formatSigned(info.update);
    row.querySelector('[data-field="after"]').textContent = info.after === undefined ? '—' : formatNumber(info.after);
  }
}

function renderComparison(step) {
  const comparison = step.training.comparison;
  if (!comparison) {
    elements.comparison.className = 'comparison empty';
    elements.comparison.textContent = '1回の学習が完了すると、学習前後の確率・損失・更新量をここで比較します。';
    return;
  }
  elements.comparison.className = `comparison ${comparison.lossDecreased ? 'success' : 'warning'}`;
  elements.comparison.innerHTML = `
    <div class="comparison-head"><strong>正解：${OUTPUT_NAMES[comparison.targetIndex]}</strong><span>${comparison.lossDecreased ? '損失は減少' : '損失は非減少'}</span></div>
    <div class="metric-grid">
      <div><span>正解確率（前）</span><strong>${formatNumber(comparison.targetProbabilityBefore * 100)}%</strong></div>
      <div><span>正解確率（後）</span><strong>${formatNumber(comparison.targetProbabilityAfter * 100)}%</strong></div>
      <div><span>確率変化</span><strong>${formatSigned(comparison.targetProbabilityChange * 100, 'ポイント')}</strong></div>
      <div><span>損失（前）</span><strong>${formatNumber(comparison.beforeLoss)}</strong></div>
      <div><span>損失（後）</span><strong>${formatNumber(comparison.afterLoss)}</strong></div>
      <div><span>損失差</span><strong>${formatSigned(comparison.afterLoss - comparison.beforeLoss)}</strong></div>
      <div><span>変更された重み</span><strong>${comparison.changedWeightCount} / 32</strong></div>
      <div><span>変更パラメータ</span><strong>${comparison.changedParameterCount} / 39</strong></div>
      <div><span>最大変更量</span><strong>${formatNumber(comparison.maximumAbsoluteChange)}</strong></div>
    </div>
    <div class="probability-comparison">
      ${OUTPUT_NAMES.map((name, index) => `<span>${name}: ${formatNumber(comparison.beforeProbabilities[index] * 100)}% → ${formatNumber(comparison.afterProbabilities[index] * 100)}%</span>`).join('')}
    </div>
  `;
}

function appendRlFormulaLine(fragment, text, status = 'done') {
  const line = document.createElement('div');
  line.className = `formula-term ${status}`;
  const marker = document.createElement('span');
  marker.className = 'term-marker';
  marker.textContent = status === 'current' ? '▶' : status === 'pending' ? '○' : '✓';
  const value = document.createElement('span');
  value.textContent = text;
  line.append(marker, value);
  fragment.append(line);
}

function renderRlFormula(step) {
  const fragment = document.createDocumentFragment();
  if (!step) {
    const placeholder = document.createElement('p');
    placeholder.className = 'formula-placeholder';
    placeholder.textContent = 'エピソードを開始すると、実際の状態、確率、乱数、報酬、勾配を表示します。';
    fragment.append(placeholder);
    elements['rl-formula'].replaceChildren(fragment);
    return;
  }

  const details = step.details;
  if (step.stage === 'rl-ready') {
    appendRlFormulaLine(fragment, 'q(a|s; τ) = softmax(logit / τ)', 'current');
    appendRlFormulaLine(fragment, 'G_t = r_t + γ × G_(t+1)', 'pending');
    appendRlFormulaLine(fragment, 'θ_new = θ_old − η_RL × 平均方策勾配', 'pending');
  } else if (step.stage === 'rl-observation') {
    details.inputs.forEach((value, index) => {
      appendRlFormulaLine(fragment, `x${index + 1} = ${formatNumber(value)}`, 'current');
    });
  } else if (step.stage === 'rl-policy') {
    details.policy.probabilities.forEach((probability, index) => {
      appendRlFormulaLine(
        fragment,
        `q(${OUTPUT_NAMES[index]}) = exp(${formatNumber(details.forward.logits[index])} / ${formatNumber(details.policy.temperature)}) / Σexp = ${formatNumber(probability)}`,
        'current',
      );
    });
    appendRlFormulaLine(fragment, `方策エントロピー H = ${formatNumber(details.policy.entropy)}`);
  } else if (step.stage === 'rl-sample') {
    details.policy.probabilities.forEach((_, index) => {
      let lower = 0;
      for (let offset = 0; offset < index; offset += 1) lower += details.policy.probabilities[offset];
      const upper = index === details.policy.probabilities.length - 1
        ? 1
        : lower + details.policy.probabilities[index];
      appendRlFormulaLine(
        fragment,
        `${formatNumber(lower)} ≤ u < ${formatNumber(upper)} : ${actionDisplay(index)}`,
        details.sample.actionIndex === index ? 'current' : 'done',
      );
    });
    appendRlFormulaLine(
      fragment,
      `u = ${formatNumber(details.sample.randomValue)} → ${actionDisplay(details.sample.actionIndex)}（${details.exploration ? '探索' : '活用'}）`,
      'current',
    );
  } else if (step.stage === 'rl-transition') {
    appendRlFormulaLine(
      fragment,
      `r_${details.time} = reward[${details.transition.rewardCategory}] = ${formatSigned(details.transition.reward)}`,
      'current',
    );
    appendRlFormulaLine(fragment, `基準報酬 = ${formatSigned(details.transition.baseReward)}`);
    appendRlFormulaLine(fragment, `累積報酬 = ${formatSigned(details.cumulativeReward)}`);
  } else if (step.stage === 'rl-return') {
    appendRlFormulaLine(
      fragment,
      `G_${details.time} = ${formatNumber(details.reward)} + ${formatNumber(details.gamma)} × ${formatNumber(details.futureReturn)} = ${formatNumber(details.returnToGo)}`,
      'current',
    );
  } else if (step.stage === 'rl-gradient') {
    const gradient = details.gradient;
    const chosenProbability = gradient.policy.probabilities[gradient.actionIndex];
    appendRlFormulaLine(
      fragment,
      `L_${details.time} = −${formatNumber(gradient.returnToGo)} × log(${formatNumber(chosenProbability)}) = ${formatNumber(gradient.policyLoss)}`,
      'current',
    );
    gradient.outputDeltas.forEach((delta, index) => {
      appendRlFormulaLine(
        fragment,
        `∂L/∂logit_${index + 1} = (G/τ)(q_${index + 1} − y_${index + 1}) = ${formatNumber(delta)}`,
      );
    });
    appendRlFormulaLine(fragment, 'このdeltaから連鎖律で39個すべての勾配を計算');
  } else if (step.stage === 'rl-aggregate') {
    appendRlFormulaLine(
      fragment,
      `平均勾配 g = (1 / ${details.experienceCount}) × Σ_t ∇L_t`,
      'current',
    );
    appendRlFormulaLine(fragment, `勾配L2ノルム = ${formatNumber(details.gradientNorm)}`);
    appendRlFormulaLine(fragment, `最大 |勾配| = ${formatNumber(details.maximumGradient)}`);
  } else if (step.stage === 'rl-update') {
    appendRlFormulaLine(fragment, `更新対象 ${details.spec.name}`, 'current');
    appendRlFormulaLine(
      fragment,
      `${formatNumber(details.before)} − ${formatNumber(details.learningRate)} × ${formatNumber(details.gradient)} = ${formatNumber(details.after)}`,
      'current',
    );
    appendRlFormulaLine(fragment, `更新量 = ${formatSigned(details.update)}`);
  } else if (step.stage === 'rl-comparison') {
    const summary = details.summary;
    appendRlFormulaLine(fragment, `累積報酬 = ${formatSigned(summary.cumulativeReward)}`, 'current');
    appendRlFormulaLine(fragment, `餌 = ${summary.foods} / 探索 = ${summary.explorationCount} / 活用 = ${summary.exploitationCount}`);
    OUTPUT_NAMES.forEach((name, index) => {
      appendRlFormulaLine(
        fragment,
        `初期状態 ${name}: ${formatNumber(summary.beforePolicy[index] * 100)}% → ${formatNumber(summary.afterPolicy[index] * 100)}%`,
      );
    });
    appendRlFormulaLine(fragment, `変更パラメータ = ${summary.changedParameterCount} / 39`);
  }
  elements['rl-formula'].replaceChildren(fragment);
}

function currentRlPolicy(step) {
  if (!step) return null;
  if (step.details.policy) return step.details.policy;
  if (step.details.gradient?.policy) return step.details.gradient.policy;
  if (step.details.afterPolicy) return step.details.afterPolicy;
  const activeIndex = step.active?.experienceIndex;
  const experience = Number.isInteger(activeIndex)
    ? step.experiences.find((item) => item.time === activeIndex)
    : step.experiences[step.experiences.length - 1];
  if (!experience) return null;
  return {
    probabilities: experience.probabilities,
    entropy: experience.entropy,
    greedyActionIndex: experience.greedyActionIndex,
  };
}

function renderRlPolicy(step) {
  const policy = currentRlPolicy(step);
  const selectedIndex = step?.active?.actionIndex ?? step?.details?.sample?.actionIndex ?? null;
  if (!policy) {
    const placeholder = document.createElement('p');
    placeholder.className = 'rl-policy-empty';
    placeholder.textContent = '温度付き方策は観測後に表示されます。';
    elements['rl-policy-bars'].replaceChildren(placeholder);
    elements['rl-entropy'].textContent = '—';
    return;
  }
  const fragment = document.createDocumentFragment();
  policy.probabilities.forEach((probability, index) => {
    const row = document.createElement('div');
    row.className = `rl-policy-row${policy.greedyActionIndex === index ? ' greedy' : ''}${selectedIndex === index ? ' selected' : ''}`;
    const label = document.createElement('span');
    label.textContent = `${OUTPUT_NAMES[index]} ${GRID_ACTIONS[index].label}${policy.greedyActionIndex === index ? ' / argmax' : ''}`;
    const track = document.createElement('span');
    track.className = 'rl-policy-track';
    const fill = document.createElement('span');
    fill.className = 'rl-policy-fill';
    fill.style.width = `${probability * 100}%`;
    track.append(fill);
    const value = document.createElement('strong');
    value.textContent = `${formatNumber(probability * 100)}%`;
    row.append(label, track, value);
    fragment.append(row);
  });
  elements['rl-policy-bars'].replaceChildren(fragment);
  elements['rl-entropy'].textContent = formatNumber(policy.entropy);
}

function renderRlExperiences(step) {
  const fragment = document.createDocumentFragment();
  const activeIndex = step?.active?.experienceIndex;
  for (const experience of step?.experiences ?? []) {
    const row = document.createElement('tr');
    row.classList.toggle('active', experience.time === activeIndex);
    const returnValue = step.returns[experience.time];
    const values = [
      experience.time,
      worldStateDisplay(experience.beforeWorld),
      actionDisplay(experience.actionIndex),
      experience.exploration ? '探索' : '活用',
      experience.event,
      worldStateDisplay(experience.afterWorld),
      formatSigned(experience.reward),
      formatSigned(experience.cumulativeReward),
      returnValue === null || returnValue === undefined ? '—' : formatNumber(returnValue),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? 'th' : 'td');
      if (index === 0) cell.scope = 'row';
      cell.textContent = value;
      row.append(cell);
    });
    fragment.append(row);
  }
  if (!fragment.childNodes.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 9;
    cell.textContent = '経験はまだありません。';
    row.append(cell);
    fragment.append(row);
  }
  elements['rl-experience-body'].replaceChildren(fragment);
}

function worldStateDisplay(world) {
  if (!world) return '—';
  const direction = getDirection(world);
  return `(${world.agent.row},${world.agent.column}) ${direction.label}${direction.symbol}`;
}

const RL_STAGE_GROUPS = Object.freeze({
  'rl-observation': 'observe',
  'rl-policy': 'policy',
  'rl-sample': 'sample',
  'rl-transition': 'transition',
  'rl-return': 'return',
  'rl-gradient': 'gradient',
  'rl-aggregate': 'gradient',
  'rl-update': 'update',
  'rl-comparison': 'compare',
});

function renderRlAxis(step) {
  const activeGroup = RL_STAGE_GROUPS[step?.stage] ?? null;
  document.querySelectorAll('[data-rl-axis]').forEach((item) => {
    item.classList.toggle('current', item.dataset.rlAxis === activeGroup);
  });
  const activeTime = step?.details?.time ?? step?.active?.experienceIndex;
  const collectionStage = ['rl-observation', 'rl-policy', 'rl-sample', 'rl-transition'].includes(step?.stage);
  const displayedTime = Number.isInteger(activeTime) ? activeTime : 0;
  elements['rl-environment-time'].textContent = step?.stage === 'rl-ready'
    ? '開始前'
    : Number.isInteger(activeTime)
      ? `t=${activeTime}`
      : collectionStage
        ? 't=0'
        : '終了';
  if (!step) {
    elements['rl-axis-note'].textContent = '139ステップの数学タイムラインとは独立しています。エピソード開始後、環境時刻と報酬の因果をこの軸で追います。';
  } else if (collectionStage) {
    elements['rl-axis-note'].textContent = `環境時刻t=${displayedTime}の経験を収集中です。ネットワークは固定され、まだParameter更新は行いません。`;
  } else if (step.stage === 'rl-ready') {
    elements['rl-axis-note'].textContent = 'エピソード開始前です。次に環境を5センサーとして観測します。';
  } else {
    elements['rl-axis-note'].textContent = `環境との相互作用は終了しています。保存済み経験へ報酬を割り当て、方策更新を計算中です。`;
  }
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function renderRlHistoryChart() {
  const svg = elements['rl-history-chart'];
  const fragment = document.createDocumentFragment();
  const title = svgElement('title');
  title.textContent = rlEpisodeHistory.length
    ? `${rlEpisodeHistory.length}エピソードの累積報酬。塗りつぶし点は餌取得あり。`
    : 'エピソード履歴はまだありません。';
  fragment.append(title);
  if (!rlEpisodeHistory.length) {
    const text = svgElement('text', { x: 380, y: 98, 'text-anchor': 'middle', class: 'rl-chart-empty' });
    text.textContent = 'エピソード完了後に累積報酬を表示';
    fragment.append(text);
    svg.replaceChildren(fragment);
    return;
  }

  const width = 760;
  const height = 190;
  const left = 55;
  const right = 24;
  const top = 20;
  const bottom = 34;
  const rewards = rlEpisodeHistory.map((summary) => summary.cumulativeReward);
  const minimum = Math.min(0, ...rewards);
  const maximum = Math.max(0, ...rewards);
  const span = maximum - minimum || 1;
  const xFor = (index) => rlEpisodeHistory.length === 1
    ? (left + width - right) / 2
    : left + (index / (rlEpisodeHistory.length - 1)) * (width - left - right);
  const yFor = (value) => top + ((maximum - value) / span) * (height - top - bottom);

  [minimum, 0, maximum].filter((value, index, values) => values.indexOf(value) === index).forEach((value) => {
    const y = yFor(value);
    fragment.append(svgElement('line', { x1: left, y1: y, x2: width - right, y2: y, class: 'rl-chart-grid' }));
    const label = svgElement('text', { x: left - 8, y: y + 4, 'text-anchor': 'end', class: 'rl-chart-label' });
    label.textContent = formatNumber(value);
    fragment.append(label);
  });
  fragment.append(svgElement('line', { x1: left, y1: top, x2: left, y2: height - bottom, class: 'rl-chart-axis' }));
  fragment.append(svgElement('line', { x1: left, y1: height - bottom, x2: width - right, y2: height - bottom, class: 'rl-chart-axis' }));

  const points = rlEpisodeHistory.map((summary, index) => `${xFor(index)},${yFor(summary.cumulativeReward)}`).join(' ');
  fragment.append(svgElement('polyline', { points, class: 'rl-chart-line' }));
  rlEpisodeHistory.forEach((summary, index) => {
    const point = svgElement('circle', {
      cx: xFor(index),
      cy: yFor(summary.cumulativeReward),
      r: 4,
      class: `rl-chart-point${summary.foods > 0 ? ' goal' : ''}`,
    });
    const pointTitle = svgElement('title');
    pointTitle.textContent = `エピソード${summary.episodeNumber}: 累積報酬${formatNumber(summary.cumulativeReward)}、餌${summary.foods}`;
    point.append(pointTitle);
    fragment.append(point);
  });
  const axisLabel = svgElement('text', { x: width - right, y: height - 10, 'text-anchor': 'end', class: 'rl-chart-label' });
  axisLabel.textContent = `エピソード 1–${rlEpisodeHistory.length} / ●は餌取得`;
  fragment.append(axisLabel);
  svg.replaceChildren(fragment);
}

function renderRlHistoryTable() {
  const fragment = document.createDocumentFragment();
  for (const summary of rlEpisodeHistory) {
    const row = document.createElement('tr');
    const profile = getRewardProfile(summary.rewardProfile);
    const values = [
      summary.episodeNumber,
      profile.shortLabel,
      formatSigned(summary.cumulativeReward),
      summary.foods,
      summary.collisions,
      summary.dangerHits,
      `${summary.explorationCount} / ${summary.exploitationCount}`,
      summary.diagnostic.label,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? 'th' : 'td');
      if (index === 0) cell.scope = 'row';
      cell.textContent = value;
      row.append(cell);
    });
    fragment.append(row);
  }
  if (!fragment.childNodes.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 8;
    cell.textContent = '完了したエピソードはまだありません。';
    row.append(cell);
    fragment.append(row);
  }
  elements['rl-history-body'].replaceChildren(fragment);
}

function renderRlParameters(step) {
  const network = step?.network ?? engine.current.network;
  for (const spec of PARAMETER_SPECS) {
    const row = document.getElementById(`rl-parameter-${spec.name}`);
    const info = step?.parameterInfo?.[spec.name];
    const gradient = step?.aggregateGradient ? getGradientValue(step.aggregateGradient, spec) : null;
    row.classList.toggle('active', step?.active?.parameter === spec.name);
    row.querySelector('[data-field="current"]').textContent = formatNumber(getParameterValue(network, spec));
    row.querySelector('[data-field="gradient"]').textContent = gradient === null ? '—' : formatNumber(gradient);
    row.querySelector('[data-field="update"]').textContent = info ? formatSigned(info.update) : '—';
    row.querySelector('[data-field="after"]').textContent = info ? formatNumber(info.after) : '—';
  }
}

function renderRlPanel() {
  const step = rlEngine?.current ?? null;
  const profile = getRewardProfile(elements['rl-reward-profile'].value || RL_DEFAULTS.rewardProfile);
  elements['rl-profile-description'].textContent = `${profile.description} 報酬: 餌${profile.rewards.food >= 0 ? '+' : ''}${profile.rewards.food} / 危険${profile.rewards.danger >= 0 ? '+' : ''}${profile.rewards.danger} / 衝突${profile.rewards.collision >= 0 ? '+' : ''}${profile.rewards.collision} / 前進${profile.rewards.move >= 0 ? '+' : ''}${profile.rewards.move} / 旋回${profile.rewards.turn >= 0 ? '+' : ''}${profile.rewards.turn}`;

  renderGridBoard(elements['rl-grid-board'], step?.world ?? gridWorld);
  elements['rl-episode-number'].textContent = rlEngine ? String(rlEngine.config.episodeNumber) : '—';
  elements['rl-step-position'].textContent = rlEngine ? `${rlEngine.index} / ${rlEngine.length - 1}` : '0 / 0';
  elements['rl-cumulative-reward'].textContent = formatSigned(step?.cumulativeReward ?? 0);
  const startingFoods = rlEngine?.steps[0].world.counters.foods ?? gridWorld.counters.foods;
  elements['rl-foods'].textContent = String((step?.world ?? gridWorld).counters.foods - startingFoods);
  elements['rl-exploration'].textContent = String(step?.explorationCount ?? 0);
  elements['rl-exploitation'].textContent = String(step?.exploitationCount ?? 0);
  elements['rl-phase'].textContent = step?.stage ?? '準備';
  elements['rl-current-title'].textContent = step?.title ?? '強化学習はまだ始まっていません';
  elements['rl-current-description'].textContent = step?.description ?? '設定を確認し、「新しいエピソード」を押してください。';
  elements['rl-explanation'].textContent = step?.explanation ?? 'エピソード中はネットワークを固定し、終了後に報酬を方策勾配へ変換します。';

  renderRlFormula(step);
  renderRlAxis(step);
  renderRlPolicy(step);
  renderRlExperiences(step);
  renderRlParameters(step);
  renderRlHistoryChart();
  renderRlHistoryTable();

  const diagnostic = step?.stage === 'rl-comparison' ? step.details.summary.diagnostic : null;
  const diagnosticBox = elements['rl-diagnostic-label'].parentElement;
  diagnosticBox.className = `rl-diagnostic${diagnostic ? ` ${diagnostic.level}` : ''}`;
  elements['rl-diagnostic-label'].textContent = diagnostic?.label ?? 'エピソード終了後に診断';
  elements['rl-diagnostic-evidence'].textContent = diagnostic?.evidence ?? '報酬の高さと餌取得を別々に検査します。';

  elements['rl-step-progress'].max = Math.max(1, (rlEngine?.length ?? 1) - 1);
  elements['rl-step-progress'].value = rlEngine?.index ?? 0;
  elements['rl-first'].disabled = !rlEngine || !rlEngine.canGoPrevious;
  elements['rl-previous'].disabled = !rlEngine || !rlEngine.canGoPrevious;
  elements['rl-next'].disabled = !rlEngine || !rlEngine.canGoNext;
  elements['rl-run-end'].disabled = !rlEngine || !rlEngine.canGoNext;
  elements['rl-session-reset'].disabled = !rlSessionStartNetwork;
  elements['rl-export-history'].disabled = rlEpisodeRecords.length === 0;
  const settingsLocked = Boolean(rlEngine && !rlEngine.isComplete);
  [
    'rl-reward-profile', 'rl-random-seed', 'rl-gamma', 'rl-temperature',
    'rl-learning-rate', 'rl-max-steps',
  ].forEach((id) => {
    elements[id].disabled = settingsLocked;
  });
  elements['rl-start-episode'].textContent = rlEngine && !rlEngine.isComplete
    ? '現在を破棄して新しいエピソード'
    : '新しいエピソード';
}

function renderLog() {
  elements['log-output'].textContent = operationLog.toText() || 'ログはまだありません。';
  elements['log-output'].scrollTop = elements['log-output'].scrollHeight;
}

function renderQuickStartGuide() {
  const entry = document.querySelector('[data-experience-entry]');
  entry.dataset.experienceState = experienceState;
  const autoSelected = experienceState !== 'detail';
  elements['quick-supervised'].classList.toggle('selected', autoSelected);
  elements['quick-supervised'].setAttribute('aria-pressed', String(autoSelected));
  elements['quick-explore'].classList.toggle('selected', !autoSelected);
  elements['quick-explore'].setAttribute('aria-pressed', String(!autoSelected));
  elements['quick-supervised'].disabled = experienceState === 'running';
  elements['quick-explore'].disabled = experienceState === 'running';

  const flowState = experienceState === 'detail' ? 'ready' : experienceState;
  const flowOrder = { ready: 0, running: 1, complete: 2 };
  const current = flowOrder[flowState] ?? 0;
  entry.querySelectorAll('[data-flow-step]').forEach((item, index) => {
    item.classList.toggle('done', index < current || flowState === 'complete');
    item.classList.toggle('active', index === current);
  });
  elements['quick-start-status'].textContent = experienceMessage;
}

async function runBeginnerAutoObserve() {
  if (experienceState === 'running') return;
  pauseAuto();
  quickStartMode = 'supervised';
  experienceState = 'running';
  experienceMessage = '動いています。予測して、1回学習し、もう一度結果を計算しています。';
  renderQuickStartGuide();
  await new Promise((resolve) => window.requestAnimationFrame(resolve));

  try {
    clearRlSession('[RL] 初心者向け自動実行により既存の強化学習セッションを失効');
    engine = new StepEngine(cloneNetwork(engine.current.network), inputs);
    addLog('[自動で見る] 教師あり学習を開始');
    while (engine.canGoNext) addLog(engine.next().logMessage);
    engine.appendLearning(targetIndex, learningRate);
    lastLearningUndoNetwork = engine.getUndoNetwork();
    while (engine.canGoNext) addLog(engine.next().logMessage);
    const comparison = engine.current.training.comparison;
    experienceState = 'complete';
    experienceMessage =
      `結果が出ました。正解「${OUTPUT_NAMES[targetIndex]}」の選ばれやすさが ` +
      `${formatNumber(comparison.targetProbabilityBefore * 100)}% → ` +
      `${formatNumber(comparison.targetProbabilityAfter * 100)}% に変わりました。`;
    addLog(`[自動で見る] 完了：${experienceMessage}`);
    render();
    elements.comparison.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (error) {
    experienceState = 'ready';
    experienceMessage = `自動実行を完了できませんでした：${error.message}`;
    render();
  }
}

function showBeginnerDetail() {
  pauseAuto();
  quickStartMode = 'explore';
  experienceState = 'detail';
  experienceMessage = '同じ計算を先頭へ戻しました。「次のステップ」で1つずつ確認できます。';
  engine.first();
  addLog('[詳しく見る] 現在のタイムラインを先頭へ戻す');
  render();
  moveQuickStartToCurrentAction();
}

function moveQuickStartToCurrentAction() {
  let target;
  if (quickStartMode === 'reinforcement') {
    target = !rlEngine ? elements['rl-start-episode'] : (rlEngine.isComplete ? elements['rl-start-episode'] : elements['rl-run-end']);
  } else if (quickStartMode === 'explore') {
    target = engine.canGoNext ? elements['step-next'] : elements['timeline-first'];
  } else if (engine.current.stage === 'comparison') {
    target = elements.comparison;
  } else if (engine.canStartLearning) {
    target = elements['start-learning'];
  } else {
    target = elements['run-to-end'];
  }
  target?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (target instanceof HTMLButtonElement) target.focus({ preventScroll: true });
}

function render() {
  const step = engine.current;
  elements['phase-value'].textContent = step.phase;
  elements['step-value'].textContent = `${engine.index} / ${engine.length - 1}`;
  elements['learning-count'].textContent = String(step.network.learningCount);
  elements['status-message'].textContent = step.title;
  elements['step-progress'].max = Math.max(1, engine.length - 1);
  elements['step-progress'].value = engine.index;
  elements['step-title'].textContent = step.title;
  elements['step-description'].textContent = step.description;
  elements['natural-explanation'].textContent = EXPLANATIONS[step.explanationKey] ?? EXPLANATIONS.learning;

  renderFormula(step);
  renderGridWorldPanel(step);
  renderNetwork(elements['network-svg'], step, formatNumber, inspectParameter);
  renderOutputSummary(step);
  renderParameters(step);
  renderComparison(step);
  renderRlPanel();
  renderLog();
  renderQuickStartGuide();

  elements['step-previous'].disabled = !engine.canGoPrevious;
  elements['step-next'].disabled = !engine.canGoNext;
  elements['run-to-end'].disabled = !engine.canGoNext;
  elements['start-learning'].disabled = !engine.canStartLearning;
  elements['undo-learning'].disabled = !lastLearningUndoNetwork;
  elements['auto-play'].disabled = Boolean(autoTimer) || !engine.canGoNext;
  elements['auto-pause'].disabled = !autoTimer;
  syncInputsToControls();
  syncTargetControls();
}

function advanceOne({ log = true } = {}) {
  if (!engine.canGoNext) {
    pauseAuto();
    return false;
  }
  const step = engine.next();
  if (log) addLog(step.logMessage);
  render();
  if (!engine.canGoNext) pauseAuto();
  return true;
}

function pauseAuto() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    addLog('[自動実行] 一時停止');
  }
  elements['auto-play'].disabled = !engine.canGoNext;
  elements['auto-pause'].disabled = true;
}

function startAuto() {
  pauseAuto();
  if (!engine.canGoNext) return;
  const speed = Number(elements['speed-select'].value);
  addLog(`[自動実行] 再生（${speed}倍）`);
  autoTimer = setInterval(() => advanceOne(), 1000 / speed);
  render();
}

function downloadState() {
  const json = serializeState({
    network: engine.current.network,
    learningRate,
    inputs,
    targetIndex,
  });
  downloadTextFile('glassbox-ai-state.json', json, 'application/json;charset=utf-8');
  addLog('[保存] 現在のネットワーク状態をJSONで保存');
  renderLog();
}

async function loadState(file) {
  const text = await file.text();
  const restored = parseStateJson(text);
  pauseAuto();
  inputs = restored.inputs;
  targetIndex = restored.targetIndex;
  learningRate = restored.learningRate;
  invalidateGridSynchronization();
  elements['learning-rate'].value = learningRate;
  elements['seed-input'].value = restored.network.seed;
  engine = new StepEngine(restored.network, inputs);
  lastLearningUndoNetwork = null;
  clearRlSession('[RL] 状態読込により既存の強化学習セッションを失効');
  addLog(`[読込] ${file.name}から検証済み状態を読み込み`);
  render();
}

function commitRlEpisodeIfComplete() {
  if (!rlEngine?.isComplete || rlEngine.committed) return false;
  const previousNetwork = cloneNetwork(rlEngine.steps[0].network);
  const finalNetwork = cloneNetwork(rlEngine.finalNetwork);
  const finalWorld = cloneGridWorld(rlEngine.finalWorld);
  const finalInputs = senseGridWorld(finalWorld).values;
  lastLearningUndoNetwork = previousNetwork;
  inputs = [...finalInputs];
  gridWorld = finalWorld;
  gridSensorsSynchronized = true;
  synchronizedGridSignature = gridWorldSignature(gridWorld);
  engine = new StepEngine(finalNetwork, inputs);
  rlEpisodeHistory.push(rlEngine.summary);
  rlEpisodeRecords.push({
    config: { ...rlEngine.config },
    profile: {
      key: rlEngine.profile.key,
      label: rlEngine.profile.label,
      shortLabel: rlEngine.profile.shortLabel,
      rewards: { ...rlEngine.profile.rewards },
    },
    initialWorld: cloneGridWorld(rlEngine.initialWorld),
    finalWorld: cloneGridWorld(rlEngine.finalWorld),
    experiences: rlEngine.experiences,
    returns: [...rlEngine.returns],
    summary: rlEngine.summary,
    initialNetwork: cloneNetwork(rlEngine.steps[0].network),
    finalNetwork: cloneNetwork(rlEngine.finalNetwork),
    averageGradient: rlEngine.averageGradient,
    parameterInfo: rlEngine.steps.at(-1).parameterInfo,
  });
  rlNextEpisodeNumber = rlEngine.config.episodeNumber + 1;
  rlEngine.committed = true;
  addLog(
    `[RL] エピソード${rlEngine.summary.episodeNumber}を主ネットワークへ反映：` +
      `累積報酬${rlEngine.summary.cumulativeReward}、餌${rlEngine.summary.foods}、` +
      `${rlEngine.summary.diagnostic.label}`,
  );
  return true;
}

function exportRlHistory() {
  try {
    const json = serializeReinforcementHistory(rlEpisodeRecords);
    downloadTextFile('glassbox-ai-rl-history.json', json, 'application/json;charset=utf-8');
    addLog(`[RL保存] ${rlEpisodeRecords.length}エピソードの環境・報酬・方策更新履歴をJSONで保存`);
    renderLog();
  } catch (error) {
    elements['status-message'].textContent = `RL履歴を保存できません：${error.message}`;
    addLog(`[RLエラー] 履歴保存を中止：${error.message}`);
    renderLog();
  }
}

function startRlEpisode({ confirmDiscard = true } = {}) {
  pauseAuto();
  if (
    confirmDiscard &&
    rlEngine &&
    !rlEngine.isComplete &&
    rlEngine.index > 0 &&
    !window.confirm('現在の未完了RLタイムラインを破棄し、新しいエピソードを作成しますか？')
  ) return false;

  try {
    rememberRlSessionStart();
    const config = readRlConfig();
    const episodeWorld = createRlEpisodeWorld(config.episodeNumber);
    rlEngine = new ReinforcementStepEngine(engine.current.network, episodeWorld, config);
    addLog(
      `[RL] エピソード${config.episodeNumber}を開始：${getRewardProfile(config.rewardProfile).shortLabel}、` +
        `γ=${config.gamma}、τ=${config.temperature}、最大${config.maxSteps}ステップ`,
    );
    render();
    return true;
  } catch (error) {
    elements['status-message'].textContent = `強化学習を開始できません：${error.message}`;
    addLog(`[RLエラー] ${error.message}`);
    renderLog();
    return false;
  }
}

function advanceRlOne() {
  if (!rlEngine?.canGoNext) return false;
  const step = rlEngine.next();
  addLog(step.logMessage);
  commitRlEpisodeIfComplete();
  render();
  return true;
}

function runRlToEnd() {
  if (!rlEngine) return;
  let count = 0;
  while (rlEngine.canGoNext) {
    const step = rlEngine.next();
    addLog(step.logMessage);
    count += 1;
  }
  commitRlEpisodeIfComplete();
  addLog(`[RL操作] エピソード最後まで実行（${count}計算ステップ）`);
  render();
}

function runTenRlEpisodes() {
  pauseAuto();
  if (
    rlEngine &&
    !rlEngine.isComplete &&
    rlEngine.index > 0 &&
    !window.confirm('現在の未完了RLタイムラインを破棄し、10エピソード連続実行へ進みますか？')
  ) return;
  rememberRlSessionStart();
  try {
    const firstEpisode = rlNextEpisodeNumber;
    for (let offset = 0; offset < 10; offset += 1) {
      const episodeNumber = rlNextEpisodeNumber;
      const config = readRlConfig(episodeNumber);
      const episodeWorld = createRlEpisodeWorld(episodeNumber);
      rlEngine = new ReinforcementStepEngine(engine.current.network, episodeWorld, config);
      rlEngine.runToEnd();
      commitRlEpisodeIfComplete();
    }
    addLog(`[RL一括] エピソード${firstEpisode}〜${rlNextEpisodeNumber - 1}を連続実行`);
    render();
  } catch (error) {
    elements['status-message'].textContent = `10エピソード連続実行を中止：${error.message}`;
    addLog(`[RLエラー] ${error.message}`);
    render();
  }
}

function resetRlSession() {
  if (!rlSessionStartNetwork) return;
  if (!window.confirm('このRLセッションで行った全エピソード更新と履歴を取り消しますか？')) return;
  const restoredNetwork = cloneNetwork(rlSessionStartNetwork);
  const restoredWorld = cloneGridWorld(rlSessionStartWorld);
  const restoredInputs = [...rlSessionStartInputs];
  clearRlSession();
  engine = new StepEngine(restoredNetwork, restoredInputs);
  gridWorld = restoredWorld;
  inputs = restoredInputs;
  const sensed = senseGridWorld(gridWorld).values;
  const matchesWorld = sensed.every((value, index) => Math.abs(value - inputs[index]) < 1e-15);
  gridSensorsSynchronized = matchesWorld;
  synchronizedGridSignature = matchesWorld ? gridWorldSignature(gridWorld) : null;
  lastLearningUndoNetwork = null;
  addLog('[RLリセット] セッション開始前のネットワーク、世界、入力を復元');
  render();
}

function bindEvents() {
  elements['quick-supervised'].addEventListener('click', runBeginnerAutoObserve);
  elements['quick-reinforcement'].addEventListener('click', () => {
    quickStartMode = 'reinforcement';
    renderQuickStartGuide();
    moveQuickStartToCurrentAction();
  });
  elements['quick-explore'].addEventListener('click', showBeginnerDetail);

  elements['apply-preset'].addEventListener('click', () => {
    const preset = PRESETS[elements['preset-select'].value];
    inputs = [...preset.inputs];
    targetIndex = preset.targetIndex;
    invalidateGridSynchronization();
    resetCalculation(`[プリセット] ${preset.label}を適用（重みは維持）`);
  });

  elements['grid-world-reset'].addEventListener('click', () => {
    const seed = elements['grid-world-seed'].value.trim() || 'grid-1';
    gridWorld = createGridWorld(seed);
    elements['grid-world-seed'].value = seed;
    invalidateGridSynchronization();
    addLog(`[環境] 世界シード「${seed}」からグリッドワールドをリセット`);
    render();
  });

  elements['grid-sync-sensors'].addEventListener('click', () => {
    synchronizeGridObservation();
  });

  elements['grid-prepare-teacher'].addEventListener('click', () => {
    synchronizeGridObservation({ includeTeacher: true });
  });

  elements['grid-apply-action'].addEventListener('click', () => {
    if (!isGridInputSynchronized()) {
      elements['status-message'].textContent = '現在世界のセンサーを入力へ送ってから判断を適用してください。';
      return;
    }
    const actionIndex = engine.current.forward.selectedIndex;
    if (actionIndex === null) {
      elements['status-message'].textContent = 'softmaxとargmaxが完了していません。';
      return;
    }
    gridWorld = applyGridAction(gridWorld, actionIndex);
    invalidateGridSynchronization();
    addLog(
      `[環境] AI判断 ${actionDisplay(actionIndex)} を適用：${gridWorld.lastTransition.event} ` +
        `単発報酬 ${gridWorld.lastTransition.reward}`,
    );
    render();
  });

  elements['rl-reward-profile'].addEventListener('change', () => {
    addLog(`[RL設定] 報酬プリセットを「${getRewardProfile(elements['rl-reward-profile'].value).label}」へ変更`);
    render();
  });
  elements['rl-start-episode'].addEventListener('click', () => {
    startRlEpisode();
    if (quickStartMode === 'reinforcement') moveQuickStartToCurrentAction();
  });
  elements['rl-first'].addEventListener('click', () => {
    if (!rlEngine) return;
    rlEngine.first();
    addLog('[RL操作] 現在のRLタイムラインの最初へ戻る');
    render();
  });
  elements['rl-previous'].addEventListener('click', () => {
    if (!rlEngine?.canGoPrevious) return;
    rlEngine.previous();
    addLog(`[RL操作] 前のRLステップへ戻る：${rlEngine.current.title}`);
    render();
  });
  elements['rl-next'].addEventListener('click', advanceRlOne);
  elements['rl-run-end'].addEventListener('click', runRlToEnd);
  elements['rl-run-ten'].addEventListener('click', runTenRlEpisodes);
  elements['rl-session-reset'].addEventListener('click', resetRlSession);
  elements['rl-export-history'].addEventListener('click', exportRlHistory);

  elements['learning-rate'].addEventListener('change', (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value) || value <= 0) {
      event.target.value = learningRate;
      elements['status-message'].textContent = '学習率は0より大きい有限数にしてください。';
      return;
    }
    learningRate = value;
    if (engine.hasLearningTimeline) resetCalculation(`[設定] 学習率を${learningRate}へ変更し、学習ステップを破棄`);
    else {
      addLog(`[設定] 学習率を${learningRate}へ変更`);
      render();
    }
  });

  elements['precision-select'].addEventListener('change', (event) => {
    precision = Number(event.target.value);
    addLog(`[表示] 小数点以下${precision}桁へ変更（内部値は不変）`);
    render();
  });

  elements['timeline-first'].addEventListener('click', () => {
    pauseAuto();
    engine.first();
    addLog('[操作] タイムラインの最初へ戻る（スナップショット復元）');
    render();
  });
  elements['step-previous'].addEventListener('click', () => {
    pauseAuto();
    engine.previous();
    addLog(`[操作] 前のステップへ戻る：${engine.current.title}`);
    render();
  });
  elements['step-next'].addEventListener('click', () => advanceOne());
  elements['run-to-end'].addEventListener('click', () => {
    pauseAuto();
    let count = 0;
    while (engine.canGoNext) {
      const step = engine.next();
      addLog(step.logMessage);
      count += 1;
    }
    addLog(`[操作] 最後まで実行（${count}ステップ）`);
    render();
    if (quickStartMode === 'supervised') moveQuickStartToCurrentAction();
  });
  elements['auto-play'].addEventListener('click', startAuto);
  elements['auto-pause'].addEventListener('click', pauseAuto);
  elements['speed-select'].addEventListener('change', () => {
    if (autoTimer) startAuto();
  });

  elements['start-learning'].addEventListener('click', () => {
    pauseAuto();
    try {
      clearRlSession('[RL] 教師あり学習開始により既存の強化学習セッションを失効');
      engine.appendLearning(targetIndex, learningRate);
      lastLearningUndoNetwork = engine.getUndoNetwork();
      addLog(`[学習] 学習ステップを開始：正解${OUTPUT_NAMES[targetIndex]}、学習率${learningRate}`);
      advanceOne();
      if (quickStartMode === 'supervised') moveQuickStartToCurrentAction();
    } catch (error) {
      elements['status-message'].textContent = error.message;
    }
  });

  elements['calculation-reset'].addEventListener('click', () => resetCalculation());
  elements['undo-learning'].addEventListener('click', () => {
    if (!lastLearningUndoNetwork) return;
    pauseAuto();
    clearRlSession('[RL] 学習取消により既存の強化学習セッションを失効');
    engine = new StepEngine(lastLearningUndoNetwork, inputs);
    lastLearningUndoNetwork = null;
    addLog('[リセット] 直前の学習前へ戻し、39パラメータを復元');
    render();
  });
  elements['full-reset'].addEventListener('click', () => {
    const seed = elements['seed-input'].value.trim() || 'glassbox-1';
    if (!window.confirm(`シード「${seed}」から39パラメータを再初期化し、ログを含む全履歴を削除します。続けますか？`)) return;
    pauseAuto();
    operationLog.clear();
    clearRlSession();
    engine = new StepEngine(createNetwork(seed), inputs);
    lastLearningUndoNetwork = null;
    addLog(`[リセット] 完全リセット：シード「${seed}」から再初期化`);
    render();
  });

  elements['save-state'].addEventListener('click', downloadState);
  elements['load-state-file'].addEventListener('change', async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    try {
      await loadState(file);
    } catch (error) {
      addLog(`[エラー] 状態読込を拒否：${error.message}`);
      elements['status-message'].textContent = `読込を拒否しました：${error.message}`;
      renderLog();
    } finally {
      event.target.value = '';
    }
  });
  elements['export-log-text'].addEventListener('click', () => {
    downloadTextFile('glassbox-ai-log.txt', operationLog.toText());
  });
  elements['export-log-json'].addEventListener('click', () => {
    downloadTextFile('glassbox-ai-log.json', operationLog.toJson(), 'application/json;charset=utf-8');
  });
  elements['clear-log'].addEventListener('click', () => {
    operationLog.clear();
    addLog('[ログ] 画面ログを消去');
    renderLog();
  });
}

function initialize() {
  createPresetOptions();
  createRlRewardOptions();
  createInputControls();
  createTargetControls();
  createGlossary();
  createParameterRows();
  createRlParameterRows();
  bindEvents();
  addLog('[初期化] シード「glassbox-1」から未学習ネットワークを生成');
  render();
}

initialize();

if (requestedQuickStartModes.has(quickStartRequest)) {
  window.requestAnimationFrame(() => moveQuickStartToCurrentAction());
}
