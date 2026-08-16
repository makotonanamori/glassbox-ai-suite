import {
  GRID_ACTIONS,
  GRID_DIRECTIONS,
  GRID_SENSOR_DEFINITIONS,
  getGridCellType,
} from './grid-world.js';

const CELL_PRESENTATION = Object.freeze({
  empty: { symbol: '', label: '空きセル' },
  wall: { symbol: '▦', label: '壁' },
  danger: { symbol: '!', label: '危険セル' },
  food: { symbol: '●', label: '餌' },
  agent: { symbol: '', label: 'エージェント' },
  'agent-danger': { symbol: '', label: '危険セル上のエージェント' },
});

export function renderGridBoard(container, world) {
  const fragment = document.createDocumentFragment();
  container.style.setProperty('--grid-size', world.size);

  for (let row = 0; row < world.size; row += 1) {
    for (let column = 0; column < world.size; column += 1) {
      const type = getGridCellType(world, row, column);
      const presentation = CELL_PRESENTATION[type];
      const cell = document.createElement('div');
      cell.className = `grid-cell ${type}`;
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-rowindex', String(row + 1));
      cell.setAttribute('aria-colindex', String(column + 1));
      cell.setAttribute('aria-label', `${row + 1}行${column + 1}列：${presentation.label}`);

      if (type === 'agent' || type === 'agent-danger') {
        const direction = GRID_DIRECTIONS[world.agent.direction];
        const agent = document.createElement('span');
        agent.className = 'grid-agent';
        agent.textContent = direction.symbol;
        agent.setAttribute('aria-hidden', 'true');
        cell.title = `エージェント：${direction.label}向き`;
        cell.append(agent);
      } else {
        cell.textContent = presentation.symbol;
        cell.title = presentation.label;
      }
      fragment.append(cell);
    }
  }
  container.replaceChildren(fragment);
}

export function renderGridSensors(container, sensed, formatNumber, synchronized) {
  const fragment = document.createDocumentFragment();
  GRID_SENSOR_DEFINITIONS.forEach((definition) => {
    const sensor = sensed.sensors[definition.index];
    const row = document.createElement('div');
    row.className = 'grid-sensor-row';
    row.innerHTML = `
      <span class="grid-sensor-symbol">${definition.symbol}</span>
      <strong>${definition.label}</strong>
      <output>${formatNumber(sensor.value)}</output>
      <small>${definition.explanation}</small>
    `;
    fragment.append(row);
  });
  container.classList.toggle('synchronized', synchronized);
  container.replaceChildren(fragment);
}

export function actionDisplay(actionIndex) {
  if (actionIndex === null || actionIndex === undefined) return '—';
  const action = GRID_ACTIONS[actionIndex];
  return `${action.outputName} / ${action.label}`;
}
