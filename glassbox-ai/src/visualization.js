import { OUTPUT_NAMES, PARAMETER_SPECS, getParameterValue } from './neural-network.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const POSITIONS = Object.freeze({
  input: Array.from({ length: 5 }, (_, index) => ({ x: 100, y: 74 + index * 94 })),
  hidden: Array.from({ length: 4 }, (_, index) => ({ x: 500, y: 102 + index * 118 })),
  output: Array.from({ length: 3 }, (_, index) => ({ x: 900, y: 145 + index * 148 })),
});

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function isActiveConnection(active, layer, from, to) {
  const connection = active?.connection;
  return connection?.layer === layer && connection.from === from && connection.to === to;
}

function isActiveNode(active, layer, index) {
  const node = active?.node;
  return node?.layer === layer && (node.index === null || node.index === index);
}

function appendText(group, x, y, text, className) {
  const element = svgElement('text', { x, y, 'text-anchor': 'middle', class: className });
  element.textContent = text;
  group.append(element);
}

function nodeFill(value, layer) {
  if (value === null || !Number.isFinite(value)) return '#17222c';
  if (layer === 'output') {
    const lightness = 17 + Math.min(1, Math.abs(value)) * 22;
    return `hsl(178 46% ${lightness}%)`;
  }
  const magnitude = Math.min(1, Math.abs(value));
  const hue = value >= 0 ? 205 : 29;
  return `hsl(${hue} 42% ${17 + magnitude * 20}%)`;
}

function appendConnection({ svg, layer, from, to, source, target, weight, active, spec, onInspect, formatNumber }) {
  const line = svgElement('line', {
    x1: source.x + 44,
    y1: source.y,
    x2: target.x - 44,
    y2: target.y,
    class: `network-connection ${weight >= 0 ? 'positive' : 'negative'}${active ? ' active' : ''}`,
    'stroke-width': Math.min(8, 0.7 + Math.abs(weight) * 9),
    'stroke-opacity': Math.min(0.92, 0.18 + Math.abs(weight) * 1.45),
    tabindex: 0,
    role: 'button',
    'aria-label': `${spec.name}、${formatNumber(weight)}`,
  });
  line.dataset.parameter = spec.name;
  const title = svgElement('title');
  title.textContent = `${spec.name}: ${formatNumber(weight)}（${spec.from} → ${spec.to}）`;
  line.append(title);
  line.addEventListener('mouseenter', () => onInspect(spec));
  line.addEventListener('focus', () => onInspect(spec));
  line.addEventListener('click', () => onInspect(spec));
  line.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onInspect(spec);
    }
  });
  svg.append(line);
}

function appendNode(svg, position, options) {
  const { layer, index, label, primary, secondary, value, active } = options;
  const group = svgElement('g', {
    class: `network-node ${layer}${active ? ' active' : ''}`,
    transform: `translate(${position.x} ${position.y})`,
    'data-node': `${layer}-${index}`,
  });
  const circle = svgElement('circle', {
    r: 43,
    fill: nodeFill(value, layer),
  });
  group.append(circle);
  appendText(group, 0, -13, label, 'node-label');
  appendText(group, 0, 7, primary, 'node-value');
  appendText(group, 0, 25, secondary, 'node-secondary');
  svg.append(group);
}

export function renderNetwork(svg, snapshot, formatNumber, onInspect) {
  svg.replaceChildren();
  const { network, forward, active } = snapshot;

  appendText(svg, POSITIONS.input[0].x, 19, '入力層', 'layer-label');
  appendText(svg, POSITIONS.hidden[0].x, 19, '中間層（tanh）', 'layer-label');
  appendText(svg, POSITIONS.output[0].x, 19, '出力層（softmax）', 'layer-label');

  for (let hidden = 0; hidden < 4; hidden += 1) {
    for (let input = 0; input < 5; input += 1) {
      const spec = PARAMETER_SPECS.find(
        (candidate) => candidate.layer === 'IH' && candidate.row === hidden && candidate.column === input,
      );
      appendConnection({
        svg,
        layer: 'IH',
        from: input,
        to: hidden,
        source: POSITIONS.input[input],
        target: POSITIONS.hidden[hidden],
        weight: getParameterValue(network, spec),
        active: isActiveConnection(active, 'IH', input, hidden),
        spec,
        onInspect,
        formatNumber,
      });
    }
  }

  for (let output = 0; output < 3; output += 1) {
    for (let hidden = 0; hidden < 4; hidden += 1) {
      const spec = PARAMETER_SPECS.find(
        (candidate) => candidate.layer === 'HO' && candidate.row === output && candidate.column === hidden,
      );
      appendConnection({
        svg,
        layer: 'HO',
        from: hidden,
        to: output,
        source: POSITIONS.hidden[hidden],
        target: POSITIONS.output[output],
        weight: getParameterValue(network, spec),
        active: isActiveConnection(active, 'HO', hidden, output),
        spec,
        onInspect,
        formatNumber,
      });
    }
  }

  for (let input = 0; input < 5; input += 1) {
    const value = forward.inputsCommitted ? forward.inputs[input] : null;
    appendNode(svg, POSITIONS.input[input], {
      layer: 'input',
      index: input,
      label: `I${input + 1}`,
      primary: value === null ? 'x = —' : `x = ${formatNumber(value)}`,
      secondary: `入力${input + 1}`,
      value,
      active: isActiveNode(active, 'input', input),
    });
  }

  for (let hidden = 0; hidden < 4; hidden += 1) {
    const pre = forward.hiddenPreActivations[hidden];
    const activation = forward.hiddenActivations[hidden];
    appendNode(svg, POSITIONS.hidden[hidden], {
      layer: 'hidden',
      index: hidden,
      label: `H${hidden + 1}`,
      primary: pre === null ? 'z = —' : `z = ${formatNumber(pre)}`,
      secondary: activation === null ? 'a = —' : `a = ${formatNumber(activation)}`,
      value: activation ?? pre,
      active: isActiveNode(active, 'hidden', hidden),
    });
  }

  for (let output = 0; output < 3; output += 1) {
    const logit = forward.logits[output];
    const probability = forward.probabilities[output];
    appendNode(svg, POSITIONS.output[output], {
      layer: 'output',
      index: output,
      label: `${OUTPUT_NAMES[output]} / O${output + 1}`,
      primary: logit === null ? 'logit = —' : `logit = ${formatNumber(logit)}`,
      secondary: probability === null ? 'p = —' : `p = ${formatNumber(probability * 100)}%`,
      value: probability,
      active: isActiveNode(active, 'output', output),
    });
  }
}
