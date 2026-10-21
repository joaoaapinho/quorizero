// node frontend/test/check.mjs fixture.json (written by tests/frontend/test_js.py)
// Browser rules and planes must match the Python engine, and search must take a one-move win.

import { readFileSync } from 'node:fs';
import { initialState, legalMask, observe, step } from '../src/game.js';
import { chooseMove, heuristicEvaluate } from '../src/search.js';

const fail = (msg) => {
  console.error(msg);
  process.exit(1);
};

const fixture = JSON.parse(readFileSync(process.argv[2], 'utf8'));
for (const [g, game] of fixture.games.entries()) {
  let state = initialState();
  game.actions.forEach((action, ply) => {
    const mask = Array.from(legalMask(state));
    if (mask.join() !== game.legal[ply].join()) fail(`game ${g} ply ${ply}: legal mask differs`);
    const planes = game.planes[ply];
    if (planes) {
      const mine = observe(state);
      const worst = Math.max(...planes.map((x, i) => Math.abs(x - mine[i])));
      if (worst > 1e-6) fail(`game ${g} ply ${ply}: planes differ by ${worst}`);
    }
    state = step(state, action);
  });
  if (state.winner !== game.winner) fail(`game ${g}: winner ${state.winner} != ${game.winner}`);
}

// Player 0 one step from its goal row: every level must take the win.
const nearWin = { ...initialState(), pawns: [[7, 2], [8, 6]], wallsLeft: [0, 0] };
for (const sims of [64, 400]) {
  const { action } = await chooseMove(nearWin, { sims }, heuristicEvaluate);
  if (action !== 1) fail(`${sims} sims: missed the winning move (played ${action})`);
}
const easy = await chooseMove(nearWin, { sims: 0, temperature: 1, blunder: 1 }, heuristicEvaluate);
if (easy.action >= 12) fail('a blunder must be a pawn move');
console.log(`ok: ${fixture.games.length} games match, search takes the win`);
