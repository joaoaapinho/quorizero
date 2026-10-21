// Quoridor rules and network inputs for the browser, ported from src/engine and
// src/jaxgame/features.py. Checked against Python by tests/frontend/test_js.py.

export const N_PAWN = 12;
export const ACTION_SIZE = 140;
export const MAX_PLY = 200; // draw, as in jaxgame.rules
export const GOAL_ROWS = [8, 0];
export const N_PLANES = 9;
// N S E W | straight jumps | diagonal jumps (same order as engine.actions)
export const PAWN_OFFSETS = [
  [-1, 0], [1, 0], [0, 1], [0, -1],
  [-2, 0], [2, 0], [0, 2], [0, -2],
  [-1, 1], [-1, -1], [1, 1], [1, -1],
];
const DIRS = PAWN_OFFSETS.slice(0, 4);
const FAR = 99;

export function initialState() {
  return {
    pawns: [[0, 4], [8, 4]],
    h: new Uint8Array(64), // H wall anchors, index r * 8 + c
    v: new Uint8Array(64),
    wallsLeft: [10, 10],
    toMove: 0,
    ply: 0,
    winner: -1,
    done: false,
  };
}

const onBoard = (r, c) => r >= 0 && r < 9 && c >= 0 && c < 9;
const wallAt = (walls, r, c) => r >= 0 && r < 8 && c >= 0 && c < 8 && walls[r * 8 + c] === 1;

// True if a wall blocks the step between orthogonally adjacent cells.
function blocked(s, r, c, nr, nc) {
  if (nr === r - 1) return wallAt(s.h, r - 1, c) || wallAt(s.h, r - 1, c - 1);
  if (nr === r + 1) return wallAt(s.h, r, c) || wallAt(s.h, r, c - 1);
  if (nc === c + 1) return wallAt(s.v, r, c) || wallAt(s.v, r - 1, c);
  return wallAt(s.v, r, c - 1) || wallAt(s.v, r - 1, c - 1);
}

const canStep = (s, r, c, dr, dc) =>
  onBoard(r + dr, c + dc) && !blocked(s, r, c, r + dr, c + dc);

// Steps from every cell (r * 9 + c) to goalRow, FAR if walled off.
export function distanceMap(s, goalRow) {
  const dist = new Int32Array(81).fill(FAR);
  const queue = [];
  for (let c = 0; c < 9; c++) {
    dist[goalRow * 9 + c] = 0;
    queue.push([goalRow, c]);
  }
  for (let i = 0; i < queue.length; i++) {
    const [r, c] = queue[i];
    for (const [dr, dc] of DIRS) {
      const nr = r + dr, nc = c + dc;
      if (canStep(s, r, c, dr, dc) && dist[nr * 9 + nc] === FAR) {
        dist[nr * 9 + nc] = dist[r * 9 + c] + 1;
        queue.push([nr, nc]);
      }
    }
  }
  return dist;
}

export function pathLength(s, player) {
  const [r, c] = s.pawns[player];
  return distanceMap(s, GOAL_ROWS[player])[r * 9 + c];
}

function reaches(s, player) {
  const [r0, c0] = s.pawns[player];
  const seen = new Uint8Array(81);
  const stack = [[r0, c0]];
  seen[r0 * 9 + c0] = 1;
  while (stack.length) {
    const [r, c] = stack.pop();
    if (r === GOAL_ROWS[player]) return true;
    for (const [dr, dc] of DIRS) {
      const nr = r + dr, nc = c + dc;
      if (canStep(s, r, c, dr, dc) && !seen[nr * 9 + nc]) {
        seen[nr * 9 + nc] = 1;
        stack.push([nr, nc]);
      }
    }
  }
  return false;
}

const offsetIndex = (dr, dc) => PAWN_OFFSETS.findIndex(([a, b]) => a === dr && b === dc);

function pawnActions(s) {
  const [r, c] = s.pawns[s.toMove];
  const [or, oc] = s.pawns[1 - s.toMove];
  const actions = [];
  DIRS.forEach(([dr, dc], i) => {
    if (!canStep(s, r, c, dr, dc)) return;
    const nr = r + dr, nc = c + dc;
    if (nr !== or || nc !== oc) return actions.push(i);
    if (canStep(s, nr, nc, dr, dc)) return actions.push(4 + i); // straight jump
    const sides = dc === 0 ? [[0, 1], [0, -1]] : [[1, 0], [-1, 0]];
    for (const [sr, sc] of sides) {
      if (canStep(s, nr, nc, sr, sc)) actions.push(offsetIndex(dr + sr, dc + sc));
    }
  });
  return actions;
}

function slotFree(s, horizontal, r, c) {
  if (horizontal) {
    return !wallAt(s.h, r, c) && !wallAt(s.h, r, c - 1) && !wallAt(s.h, r, c + 1) && !wallAt(s.v, r, c);
  }
  return !wallAt(s.v, r, c) && !wallAt(s.v, r - 1, c) && !wallAt(s.v, r + 1, c) && !wallAt(s.h, r, c);
}

export function legalMask(s) {
  const mask = new Uint8Array(ACTION_SIZE);
  if (s.done) return mask;
  for (const a of pawnActions(s)) mask[a] = 1;
  if (s.wallsLeft[s.toMove] === 0) return mask;
  for (let k = 0; k < 128; k++) {
    const horizontal = k < 64;
    const walls = horizontal ? s.h : s.v;
    const r = Math.floor((k % 64) / 8), c = k % 8;
    if (!slotFree(s, horizontal, r, c)) continue;
    walls[k % 64] = 1; // try it, check both paths, undo
    mask[N_PAWN + k] = reaches(s, 0) && reaches(s, 1) ? 1 : 0;
    walls[k % 64] = 0;
  }
  return mask;
}

export function step(s, action) {
  const next = { ...s, pawns: s.pawns.map((p) => [...p]), h: s.h.slice(), v: s.v.slice(), wallsLeft: [...s.wallsLeft] };
  const player = s.toMove;
  if (action < N_PAWN) {
    const [dr, dc] = PAWN_OFFSETS[action];
    next.pawns[player] = [s.pawns[player][0] + dr, s.pawns[player][1] + dc];
    if (next.pawns[player][0] === GOAL_ROWS[player]) next.winner = player;
  } else {
    const k = action - N_PAWN;
    (k < 64 ? next.h : next.v)[k % 64] = 1;
    next.wallsLeft[player] -= 1;
  }
  next.toMove = 1 - player;
  next.ply = s.ply + 1;
  next.done = next.winner !== -1 || next.ply >= MAX_PLY;
  return next;
}

// Network inputs (src/jaxgame/features.py)

function permutation(flipRows) {
  const perm = PAWN_OFFSETS.map(([dr, dc]) => (flipRows ? offsetIndex(-dr, dc) : offsetIndex(dr, -dc)));
  for (let orient = 0; orient < 2; orient++) {
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const [rr, cc] = flipRows ? [7 - r, c] : [r, 7 - c];
        perm.push(N_PAWN + orient * 64 + rr * 8 + cc);
      }
    }
  }
  return perm;
}
export const FLIP = permutation(true);

// Swap a 140-vector between the board's frame and the mover's canonical frame.
export function orient(vec, toMove) {
  return toMove === 0 ? FLIP.map((i) => vec[i]) : Array.from(vec);
}

// Board seen from the mover, who always heads to row 0 (player 0 is flipped).
function canonical(s) {
  if (s.toMove === 1) return s;
  const flipWalls = (w) => {
    const out = new Uint8Array(64);
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) out[(7 - r) * 8 + c] = w[r * 8 + c];
    return out;
  };
  return { ...s, pawns: s.pawns.map(([r, c]) => [8 - r, c]), h: flipWalls(s.h), v: flipWalls(s.v) };
}

// Planes (9, 9, N_PLANES) channels-last, flattened: index (r * 9 + c) * N_PLANES + plane.
export function observe(state) {
  const s = canonical(state);
  const me = s.toMove, opp = 1 - me;
  const planes = new Float32Array(81 * N_PLANES);
  const set = (r, c, plane, x) => (planes[(r * 9 + c) * N_PLANES + plane] = x);
  set(...s.pawns[me], 0, 1);
  set(...s.pawns[opp], 1, 1);
  const mine = distanceMap(s, 0), theirs = distanceMap(s, 8);
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      if (wallAt(s.h, r, c)) set(r, c, 2, 1);
      if (wallAt(s.v, r, c)) set(r, c, 3, 1);
      set(r, c, 4, s.wallsLeft[me] / 10);
      set(r, c, 5, s.wallsLeft[opp] / 10);
      set(r, c, 6, Math.min(mine[r * 9 + c], 32) / 16);
      set(r, c, 7, Math.min(theirs[r * 9 + c], 32) / 16);
      set(r, c, 8, 1);
    }
  }
  return planes;
}
