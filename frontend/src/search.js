// Browser move choice: PUCT as in src/mcts/search.py (no root noise, most-visited
// move), so the site plays like the agent measured offline.

import { ACTION_SIZE, N_PAWN, legalMask, pathLength, step } from './game.js';

const C_INIT = 1.25;
const C_BASE = 19652;

// Difficulty presets. Easy samples the raw policy and plays a random pawn move 20%
// of the time (~50% against the greedy bot). Medium and Hard search with the network.
export const LEVELS = {
  easy: { sims: 0, temperature: 1, blunder: 0.2 },
  medium: { sims: 64 },
  hard: { sims: 400 },
};

export function softmaxOverLegal(logits, legal) {
  let max = -Infinity;
  for (let a = 0; a < ACTION_SIZE; a++) if (legal[a]) max = Math.max(max, logits[a]);
  const probs = new Float32Array(ACTION_SIZE);
  let sum = 0;
  for (let a = 0; a < ACTION_SIZE; a++) if (legal[a]) sum += probs[a] = Math.exp(logits[a] - max);
  return probs.map((p) => p / sum);
}

// Used when no model is deployed: uniform priors, shortest-path value.
export function heuristicEvaluate(state, legal) {
  const count = legal.reduce((n, x) => n + x, 0);
  const priors = Float32Array.from(legal, (x) => x / count);
  const advantage = pathLength(state, 1 - state.toMove) - pathLength(state, state.toMove);
  return { priors, value: Math.tanh(advantage / 2) };
}

async function expand(state, evaluate) {
  const node = { state, visits: 1, valueSum: 0, children: new Map() };
  if (state.done) return node; // game over: value 0, as in mctx
  node.legal = legalMask(state);
  const { priors, value } = await evaluate(state, node.legal);
  node.priors = priors;
  node.valueSum = value;
  return node;
}

// Value of `child` for the player who moved into it: reward + discount * value.
function qValue(parent, child) {
  const reward = child.state.winner === parent.state.toMove ? 1 : 0;
  const discount = child.state.done ? 0 : -1;
  return reward + discount * (child.valueSum / child.visits);
}

// PUCT with Q min-max normalised over the node and its children, unvisited moves
// count as the minimum (mctx.qtransform_by_parent_and_siblings).
function select(node) {
  const nodeValue = node.valueSum / node.visits;
  let lo = nodeValue, hi = nodeValue;
  for (const child of node.children.values()) {
    const q = qValue(node, child);
    lo = Math.min(lo, q);
    hi = Math.max(hi, q);
  }
  const scale = Math.max(hi - lo, 1e-8);
  const pbC = C_INIT + Math.log((node.visits + C_BASE + 1) / C_BASE);
  const explore = Math.sqrt(node.visits) * pbC;
  let best = -1, bestScore = -Infinity;
  for (let a = 0; a < ACTION_SIZE; a++) {
    if (!node.legal[a]) continue;
    const child = node.children.get(a);
    const q = child ? (qValue(node, child) - lo) / scale : 0;
    const score = q + (explore * node.priors[a]) / ((child ? child.visits : 0) + 1);
    if (score > bestScore) [best, bestScore] = [a, score];
  }
  return best;
}

async function simulate(root, evaluate) {
  const path = [root];
  let node = root;
  let value = 0; // a finished game is worth 0 to the side to move (as in mctx)
  while (!node.state.done) {
    const action = select(node);
    const child = node.children.get(action);
    if (!child) {
      const leaf = await expand(step(node.state, action), evaluate);
      node.children.set(action, leaf);
      path.push(leaf);
      value = leaf.valueSum; // the network's value of the new position
      break;
    }
    path.push((node = child));
  }
  if (node.state.done) node.visits += 1; // revisited a finished game
  for (let i = path.length - 1; i > 0; i--) {
    const child = path[i], parent = path[i - 1];
    const reward = child.state.winner === parent.state.toMove ? 1 : 0;
    value = reward + (child.state.done ? 0 : -1) * value;
    parent.valueSum += value;
    parent.visits += 1;
  }
}

function sample(probs, temperature) {
  const weights = Array.from(probs, (p) => (p > 0 ? p ** (1 / temperature) : 0));
  let x = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let a = 0; a < weights.length; a++) if ((x -= weights[a]) <= 0 && weights[a] > 0) return a;
  return weights.findIndex((w) => w > 0);
}

// options: { sims, temperature (0 = best move), blunder (chance of a random pawn move) }
export async function chooseMove(state, { sims = 0, temperature = 0, blunder = 0 }, evaluate) {
  const root = await expand(state, evaluate);
  const legal = [...root.legal.keys()].filter((a) => root.legal[a]);
  const instinct = (action) => ({ action, value: root.valueSum, sims: 0 });
  if (blunder && Math.random() < blunder) {
    const pawnMoves = legal.filter((a) => a < N_PAWN);
    return instinct(pawnMoves[Math.floor(Math.random() * pawnMoves.length)]);
  }
  if (legal.length === 1) return instinct(legal[0]);
  if (sims === 0) {
    const best = legal.reduce((a, b) => (root.priors[b] > root.priors[a] ? b : a));
    return instinct(temperature > 0 ? sample(root.priors, temperature) : best);
  }
  // Fixed budget, never a time limit: plays identically on every device, so a
  // game's bot moves can be re-checked later.
  let done = 0;
  while (done < sims) {
    await simulate(root, evaluate);
    done++;
  }
  let action = legal[0];
  for (const a of legal) {
    const visits = (c) => (c ? c.visits : 0);
    const [va, vb] = [visits(root.children.get(a)), visits(root.children.get(action))];
    if (va > vb || (va === vb && root.priors[a] > root.priors[action])) action = a;
  }
  return { action, value: root.valueSum / root.visits, sims: done };
}
