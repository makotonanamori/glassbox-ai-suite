import { createSeededRandom } from './seeded-random.js';

export const GRID_SIZE = 7;

export const GRID_DIRECTIONS = Object.freeze([
  Object.freeze({ key: 'N', label: '北', symbol: '↑', rowDelta: -1, columnDelta: 0 }),
  Object.freeze({ key: 'E', label: '東', symbol: '→', rowDelta: 0, columnDelta: 1 }),
  Object.freeze({ key: 'S', label: '南', symbol: '↓', rowDelta: 1, columnDelta: 0 }),
  Object.freeze({ key: 'W', label: '西', symbol: '←', rowDelta: 0, columnDelta: -1 }),
]);

export const GRID_ACTIONS = Object.freeze([
  Object.freeze({ index: 0, key: 'forward', label: '前進', outputName: '行動A' }),
  Object.freeze({ index: 1, key: 'turn-left', label: '左折', outputName: '行動B' }),
  Object.freeze({ index: 2, key: 'turn-right', label: '右折', outputName: '行動C' }),
]);

export const GRID_SENSOR_DEFINITIONS = Object.freeze([
  Object.freeze({
    index: 0,
    key: 'front-clear',
    symbol: 'x1',
    label: '前方通行可能',
    explanation: '前が空きなら+1、壁または境界なら-1',
  }),
  Object.freeze({
    index: 1,
    key: 'food-forward',
    symbol: 'x2',
    label: '餌方向・前後',
    explanation: '餌が正面ほど+1、背面ほど-1',
  }),
  Object.freeze({
    index: 2,
    key: 'food-side',
    symbol: 'x3',
    label: '餌方向・左右',
    explanation: '餌が左なら負、右なら正',
  }),
  Object.freeze({
    index: 3,
    key: 'danger-ahead',
    symbol: 'x4',
    label: '前方危険度',
    explanation: '直前+1、2マス先+0.5、検出なし-1',
  }),
  Object.freeze({
    index: 4,
    key: 'food-proximity',
    symbol: 'x5',
    label: '餌への近さ',
    explanation: 'マンハッタン距離を-1～+1へ正規化',
  }),
]);

const WALLS = Object.freeze([
  Object.freeze({ row: 1, column: 3 }),
  Object.freeze({ row: 2, column: 3 }),
  Object.freeze({ row: 3, column: 3 }),
  Object.freeze({ row: 5, column: 2 }),
]);

const DANGERS = Object.freeze([
  Object.freeze({ row: 2, column: 5 }),
  Object.freeze({ row: 4, column: 4 }),
  Object.freeze({ row: 5, column: 5 }),
]);

const FOOD_CANDIDATES = Object.freeze([
  Object.freeze({ row: 1, column: 5 }),
  Object.freeze({ row: 5, column: 4 }),
  Object.freeze({ row: 5, column: 1 }),
  Object.freeze({ row: 1, column: 1 }),
]);

const AGENT_START = Object.freeze({ row: 3, column: 1 });

function clonePoint(point) {
  return { row: point.row, column: point.column };
}

function samePoint(first, second) {
  return first.row === second.row && first.column === second.column;
}

function pointKey(point) {
  return `${point.row},${point.column}`;
}

function isInside(point) {
  return point.row >= 0 && point.row < GRID_SIZE && point.column >= 0 && point.column < GRID_SIZE;
}

function containsPoint(points, point) {
  return points.some((candidate) => samePoint(candidate, point));
}

function selectFood(seed, foodCount, agent, previousFood = null) {
  const random = createSeededRandom(`${seed}:food:${foodCount}`);
  const startIndex = Math.floor(random.next() * FOOD_CANDIDATES.length);
  for (let offset = 0; offset < FOOD_CANDIDATES.length; offset += 1) {
    const candidate = FOOD_CANDIDATES[(startIndex + offset) % FOOD_CANDIDATES.length];
    if (!samePoint(candidate, agent) && (!previousFood || !samePoint(candidate, previousFood))) {
      return clonePoint(candidate);
    }
  }
  return clonePoint(FOOD_CANDIDATES[startIndex]);
}

export function cloneGridWorld(world) {
  return {
    format: world.format,
    version: world.version,
    size: world.size,
    seed: String(world.seed),
    walls: world.walls.map(clonePoint),
    dangers: world.dangers.map(clonePoint),
    food: clonePoint(world.food),
    agent: { ...world.agent },
    counters: { ...world.counters },
    lastTransition: world.lastTransition ? { ...world.lastTransition } : null,
  };
}

export function createGridWorld(seed = 'grid-1') {
  const directionRandom = createSeededRandom(`${seed}:direction`);
  const direction = Math.floor(directionRandom.next() * GRID_DIRECTIONS.length);
  const agent = { ...AGENT_START, direction };
  return {
    format: 'glassbox-grid-world',
    version: 1,
    size: GRID_SIZE,
    seed: String(seed),
    walls: WALLS.map(clonePoint),
    dangers: DANGERS.map(clonePoint),
    food: selectFood(seed, 0, agent),
    agent,
    counters: {
      steps: 0,
      foods: 0,
      collisions: 0,
      dangerHits: 0,
    },
    lastTransition: {
      actionIndex: null,
      kind: 'initial',
      reward: 0,
      event: '世界を初期化しました。センサーはまだネットワークへ送られていません。',
      moved: false,
    },
  };
}

export function getDirection(world) {
  return GRID_DIRECTIONS[world.agent.direction];
}

export function getForwardPoint(world, distance = 1) {
  const direction = getDirection(world);
  return {
    row: world.agent.row + direction.rowDelta * distance,
    column: world.agent.column + direction.columnDelta * distance,
  };
}

export function isWallOrBoundary(world, point) {
  return !isInside(point) || containsPoint(world.walls, point);
}

export function isDanger(world, point) {
  return isInside(point) && containsPoint(world.dangers, point);
}

export function getGridCellType(world, row, column) {
  const point = { row, column };
  if (samePoint(world.agent, point)) return isDanger(world, point) ? 'agent-danger' : 'agent';
  if (containsPoint(world.walls, point)) return 'wall';
  if (containsPoint(world.dangers, point)) return 'danger';
  if (samePoint(world.food, point)) return 'food';
  return 'empty';
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function senseGridWorld(world) {
  const direction = getDirection(world);
  const front = getForwardPoint(world, 1);
  const secondFront = getForwardPoint(world, 2);
  const frontClear = isWallOrBoundary(world, front) ? -1 : 1;

  const foodRowDelta = world.food.row - world.agent.row;
  const foodColumnDelta = world.food.column - world.agent.column;
  const euclideanDistance = Math.hypot(foodRowDelta, foodColumnDelta) || 1;
  const unitFoodRow = foodRowDelta / euclideanDistance;
  const unitFoodColumn = foodColumnDelta / euclideanDistance;
  const foodForward = clamp(
    unitFoodRow * direction.rowDelta + unitFoodColumn * direction.columnDelta,
    -1,
    1,
  );
  const rightRowDelta = direction.columnDelta;
  const rightColumnDelta = -direction.rowDelta;
  const foodSide = clamp(
    unitFoodRow * rightRowDelta + unitFoodColumn * rightColumnDelta,
    -1,
    1,
  );

  let dangerAhead = -1;
  if (isDanger(world, front)) dangerAhead = 1;
  else if (!isWallOrBoundary(world, front) && isDanger(world, secondFront)) dangerAhead = 0.5;

  const manhattanDistance = Math.abs(foodRowDelta) + Math.abs(foodColumnDelta);
  const maximumDistance = (GRID_SIZE - 1) * 2;
  const foodProximity = clamp(1 - (2 * manhattanDistance) / maximumDistance, -1, 1);
  const values = [frontClear, foodForward, foodSide, dangerAhead, foodProximity];

  return {
    values,
    front,
    foodDistance: manhattanDistance,
    sensors: GRID_SENSOR_DEFINITIONS.map((definition) => ({
      ...definition,
      value: values[definition.index],
    })),
  };
}

function turnDirection(direction, delta) {
  return (direction + delta + GRID_DIRECTIONS.length) % GRID_DIRECTIONS.length;
}

export function applyGridAction(world, actionIndex) {
  if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= GRID_ACTIONS.length) {
    throw new Error('グリッド行動は0（前進）、1（左折）、2（右折）のいずれかです。');
  }

  const next = cloneGridWorld(world);
  const action = GRID_ACTIONS[actionIndex];
  let reward;
  let event;
  let moved = false;

  if (action.key === 'turn-left') {
    next.agent.direction = turnDirection(next.agent.direction, -1);
    reward = -0.02;
    event = `左へ90度旋回し、${getDirection(next).label}を向きました。`;
    next.lastTransition = { actionIndex, kind: 'turn-left', reward, event, moved };
  } else if (action.key === 'turn-right') {
    next.agent.direction = turnDirection(next.agent.direction, 1);
    reward = -0.02;
    event = `右へ90度旋回し、${getDirection(next).label}を向きました。`;
    next.lastTransition = { actionIndex, kind: 'turn-right', reward, event, moved };
  } else {
    const destination = getForwardPoint(next);
    if (isWallOrBoundary(next, destination)) {
      next.counters.collisions += 1;
      reward = -0.25;
      event = '前方の壁または境界に衝突し、位置は変わりませんでした。';
      next.lastTransition = { actionIndex, kind: 'collision', reward, event, moved };
    } else {
      next.agent.row = destination.row;
      next.agent.column = destination.column;
      moved = true;
      if (isDanger(next, destination)) {
        next.counters.dangerHits += 1;
        reward = -1;
        event = '危険セルへ進入しました。単発報酬は大きな負値です。';
        next.lastTransition = { actionIndex, kind: 'danger', reward, event, moved };
      } else if (samePoint(next.food, destination)) {
        const previousFood = clonePoint(next.food);
        next.counters.foods += 1;
        reward = 1;
        next.food = selectFood(next.seed, next.counters.foods, next.agent, previousFood);
        event = '餌を取得しました。次の餌を同じ世界シードから再配置しました。';
        next.lastTransition = { actionIndex, kind: 'food', reward, event, moved };
      } else {
        reward = -0.01;
        event = '前方の空きセルへ1マス進みました。';
        next.lastTransition = { actionIndex, kind: 'move', reward, event, moved };
      }
    }
  }

  next.counters.steps += 1;
  if (!next.lastTransition || next.lastTransition.actionIndex !== actionIndex) {
    next.lastTransition = { actionIndex, kind: action.key, reward, event, moved };
  }
  return next;
}

function teacherStateKey(row, column, direction) {
  return `${row},${column},${direction}`;
}

function teacherNextState(world, state, actionIndex) {
  if (actionIndex === 1) {
    return { ...state, direction: turnDirection(state.direction, -1) };
  }
  if (actionIndex === 2) {
    return { ...state, direction: turnDirection(state.direction, 1) };
  }
  const direction = GRID_DIRECTIONS[state.direction];
  const destination = {
    row: state.row + direction.rowDelta,
    column: state.column + direction.columnDelta,
  };
  if (isWallOrBoundary(world, destination) || isDanger(world, destination)) return null;
  return { ...state, row: destination.row, column: destination.column };
}

export function getTeacherPlan(world) {
  const start = {
    row: world.agent.row,
    column: world.agent.column,
    direction: world.agent.direction,
    plan: [],
  };
  const queue = [start];
  const visited = new Set([teacherStateKey(start.row, start.column, start.direction)]);
  let queueIndex = 0;

  while (queueIndex < queue.length) {
    const state = queue[queueIndex];
    queueIndex += 1;
    if (state.row === world.food.row && state.column === world.food.column) {
      return {
        safePathFound: true,
        actionIndex: state.plan[0] ?? null,
        plan: [...state.plan],
        pathLength: state.plan.length,
      };
    }

    for (const action of GRID_ACTIONS) {
      const next = teacherNextState(world, state, action.index);
      if (!next) continue;
      const key = teacherStateKey(next.row, next.column, next.direction);
      if (visited.has(key)) continue;
      visited.add(key);
      queue.push({ ...next, plan: [...state.plan, action.index] });
    }
  }

  return {
    safePathFound: false,
    actionIndex: null,
    plan: [],
    pathLength: null,
  };
}

export function getTeacherAction(world) {
  const plan = getTeacherPlan(world);
  const action = plan.actionIndex === null ? null : GRID_ACTIONS[plan.actionIndex];
  return { ...plan, action };
}

export function gridWorldSignature(world) {
  return [
    pointKey(world.agent),
    world.agent.direction,
    pointKey(world.food),
    world.counters.steps,
  ].join('|');
}
