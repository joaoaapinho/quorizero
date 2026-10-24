// Page logic: board, controls, the bot (a Web Worker), the win bar and game
// recording for retraining (one POST per game).

import { API_URL, CHALLENGE } from './config.js';
import { GOAL_ROWS, N_PAWN, PAWN_OFFSETS, initialState, legalMask, step } from './game.js';
import { LEVELS } from './search.js';

const CELL = 52, GAP = 12, PITCH = CELL + GAP, PAD = 22, LABEL = 18; // LABEL: room for coordinates
const BOARD = 9 * CELL + 8 * GAP;
const SIM_STEPS = [16, 32, 64, 128, 256, 400, 800, 1600];
const levelFor = (sims) => (sims >= 400 ? 'hard' : 'medium'); // Medium 16-256, Hard 400-1600
const SVG_NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);

const bot = new Worker(new URL('./bot.js', import.meta.url), { type: 'module' });
const settings = { mode: 'ai', humanSide: 0, level: 'easy', sims: 64, ...load('qz-settings', {}) };
if (!SIM_STEPS.includes(settings.sims)) settings.sims = 64; // saves from before the slider change
if (!['easy', 'medium', 'hard'].includes(settings.level)) settings.level = levelFor(settings.sims);
let state, legal, gameId, record, over = false, thinking = false, preview = null;
let ghost; // layer for the wall preview: hovering must not rebuild the click targets
let game = null; // the bot's level for this game: fixed once a move is made
let modelVersion = null, botHasModel = false;
const wallOwner = new Map();
// Challenge: open only while the owner keeps it open (GET /challenge).
const challengeInfo = { ...CHALLENGE, open: !API_URL }; // no API (local dev): always shown
let challenge = false, claimGameId = null;

// Who is who

const vsBot = () => challenge || settings.mode === 'ai';
const side = () => (challenge ? challengeInfo.humanSide : settings.humanSide);
const name = (seat) => (vsBot() ? (seat === side() ? 'You' : 'QuoriZero') : `Player ${seat + 1}`);
const humanTurn = () => !over && !thinking && (!vsBot() || state.toMove === side());
// The bottom player always plays upward: you vs the bot, player 1 in 2-player mode.
const flipped = () => (vsBot() ? side() : 0) === 0;
const viewRow = (r) => (flipped() ? 8 - r : r);
const viewWallRow = (r) => (flipped() ? 7 - r : r);

const chosen = () =>
  challenge
    ? { level: 'challenge', sims: challengeInfo.sims }
    : { level: settings.level, sims: settings.level === 'easy' ? 0 : settings.sims };
const pendingChange = () =>
  vsBot() && !over && state?.ply > 0 && (game.level !== settings.level || game.sims !== chosen().sims);

function botOptions() {
  return game.level === 'easy' ? LEVELS.easy : { sims: game.sims };
}

// Settings reach the bot only before the first move, later ones wait for the next
// game, so nobody can weaken the bot halfway and still claim a win at that level.
function applySettings() {
  if (state?.ply === 0 && !thinking && !over) {
    game = chosen();
    if (record) Object.assign(record, { difficulty: game.level, sims: game.sims });
  }
  updateControls();
}

// Drawing

function el(tag, attrs, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent?.appendChild(node);
  return node;
}

function wallRect(k) {
  const horizontal = k < 64, c = k % 8, y = viewWallRow(Math.floor((k % 64) / 8)) * PITCH;
  return horizontal
    ? { x: c * PITCH, y: y + CELL + 2, width: 2 * CELL + GAP, height: GAP - 4 }
    : { x: c * PITCH + CELL + 2, y, width: GAP - 4, height: 2 * CELL + GAP };
}

function render() {
  const svg = $('board');
  svg.replaceChildren();
  el('rect', { x: -PAD, y: -PAD, width: BOARD + 2 * PAD, height: BOARD + 2 * PAD, rx: 16, class: 'board-bg' }, svg);
  // Each goal row is tinted in the colour of the player racing towards it.
  for (const seat of [0, 1]) {
    const atTop = viewRow(GOAL_ROWS[seat]) === 0;
    el('rect', { x: 0, y: atTop ? -PAD + 7 : BOARD + PAD - 11, width: BOARD, height: 4, rx: 2, class: `edge-p${seat + 1}` }, svg);
  }
  for (let i = 0; i < 9; i++) {
    const row = el('text', { x: -PAD - LABEL / 2, y: viewRow(i) * PITCH + CELL / 2, class: 'coord' }, svg);
    row.textContent = i;
    const col = el('text', { x: i * PITCH + CELL / 2, y: BOARD + PAD + LABEL / 2, class: 'coord' }, svg);
    col.textContent = 'abcdefghi'[i];
  }
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const goal = r === GOAL_ROWS[0] ? ' goal-p1' : r === GOAL_ROWS[1] ? ' goal-p2' : '';
      el('rect', { x: c * PITCH, y: viewRow(r) * PITCH, width: CELL, height: CELL, rx: 7, class: `cell${goal}` }, svg);
    }
  }
  for (let k = 0; k < 128; k++) {
    if (!(k < 64 ? state.h : state.v)[k % 64]) continue;
    el('rect', { ...wallRect(k), rx: 3, class: `wall-p${(wallOwner.get(k) ?? 0) + 1}` }, svg);
  }
  state.pawns.forEach(([r, c], seat) => {
    const cx = c * PITCH + CELL / 2, cy = viewRow(r) * PITCH + CELL / 2;
    el('circle', { cx, cy, r: 17, class: `pawn pawn-p${seat + 1}` }, svg);
  });
  if (humanTurn()) drawTargets(svg);
  ghost = el('g', { class: 'ghost-layer' }, svg);
  drawPreview();
  for (const seat of [0, 1]) {
    $(`name-p${seat + 1}`).textContent = name(seat);
    $(`walls-p${seat + 1}`).textContent = `${state.wallsLeft[seat]} walls`;
    $(`name-p${seat + 1}`).parentElement.classList.toggle('to-move', !over && state.toMove === seat);
  }
}

function drawTargets(svg) {
  const seat = state.toMove;
  if (state.wallsLeft[seat] > 0) {
    for (let k = 0; k < 128; k++) {
      const horizontal = k < 64, wr = viewWallRow(Math.floor((k % 64) / 8)), c = k % 8;
      const box = horizontal
        ? { x: c * PITCH + CELL / 2, y: wr * PITCH + CELL - 4, width: PITCH, height: GAP + 8 }
        : { x: c * PITCH + CELL - 4, y: wr * PITCH + CELL / 2, width: GAP + 8, height: PITCH };
      const action = N_PAWN + k;
      if (!legal[action]) continue; // crossing, overlapping or path-blocking: not clickable
      const target = el('rect', { ...box, class: 'target' }, svg);
      target.addEventListener('pointerenter', (e) => e.pointerType === 'mouse' && showPreview(action));
      target.addEventListener('pointerleave', (e) => e.pointerType === 'mouse' && showPreview(null));
      // Mouse: click places. Touch: first tap previews, second tap places.
      target.addEventListener('click', () =>
        preview === action || !matchMedia('(pointer: coarse)').matches ? play(action) : showPreview(action),
      );
    }
  }
  const [r, c] = state.pawns[seat];
  for (let a = 0; a < N_PAWN; a++) {
    if (!legal[a]) continue;
    const [dr, dc] = PAWN_OFFSETS[a];
    const cx = (c + dc) * PITCH + CELL / 2, cy = viewRow(r + dr) * PITCH + CELL / 2;
    el('circle', { cx, cy, r: 8, class: `move-dot move-p${seat + 1}` }, svg);
    el('circle', { cx, cy, r: 22, class: 'target' }, svg).addEventListener('click', () => play(a));
  }
}

function showPreview(action) {
  preview = action !== null && legal[action] ? action : null;
  drawPreview();
}

function drawPreview() {
  ghost.replaceChildren();
  if (preview === null || !humanTurn()) return;
  el('rect', { ...wallRect(preview - N_PAWN), rx: 3, class: `wall-p${state.toMove + 1} ghost` }, ghost);
}

// Win probability

function setOdds(p1) {
  $('odds-fill').style.width = `${(p1 * 100).toFixed(1)}%`;
  $('odds-p1').textContent = `${name(0)} ${Math.round(p1 * 100)}%`;
  $('odds-p2').textContent = `${Math.round((1 - p1) * 100)}% ${name(1)}`;
}

function requestOdds() {
  bot.postMessage({ type: 'evaluate', id: gameId, state });
}

// Game flow

function describe(s, action) {
  if (action < N_PAWN) {
    const [dr, dc] = PAWN_OFFSETS[action];
    const [r, c] = s.pawns[s.toMove];
    return 'abcdefghi'[c + dc] + (r + dr);
  }
  const k = action - N_PAWN;
  return 'abcdefgh'[k % 8] + Math.floor((k % 64) / 8) + (k < 64 ? 'h' : 'v');
}

function play(action) {
  if (over || !legal[action]) return; // the rules engine has the final word
  const mover = state.toMove;
  const item = document.createElement('li');
  item.className = `p${mover + 1}`;
  item.textContent = `${state.ply + 1}. ${describe(state, action)}`;
  $('history').append(item);
  $('history').scrollTop = $('history').scrollHeight; // scroll the list, never the page
  if (action >= N_PAWN) wallOwner.set(action - N_PAWN, mover);
  state = step(state, action);
  legal = legalMask(state);
  preview = null;
  if (record) {
    const now = Date.now();
    record.moves.push(action);
    record.move_ms.push(now - record.last);
    record.last = now;
    save(CURRENT, record);
  }
  if (state.done) return finish(state.winner, 'completed');
  render();
  updateControls();
  requestOdds();
  if (vsBot() && state.toMove !== side()) askBot();
  else showStatus();
}

function askBot() {
  thinking = true;
  showStatus();
  render();
  bot.postMessage({ type: 'move', id: gameId, state, options: botOptions() });
}

function showStatus() {
  const text = thinking
    ? 'QuoriZero is thinking…'
    : vsBot()
      ? 'Your move. Tap a dot to step, or a groove to place a wall.'
      : `${name(state.toMove)} to move.`;
  $('status').textContent = challenge ? `Challenge · ${text}` : text;
}

// randomUUID only exists on secure pages (https, localhost), so a LAN address
// or 0.0.0.0 falls back to random bytes.
function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function start() {
  if (record?.status === 'playing' && humanMoved(record)) {
    submit({ ...record, status: 'abandoned', ended_at: Date.now() });
  }
  gameId = newId();
  game = chosen();
  state = initialState();
  legal = legalMask(state);
  over = thinking = false;
  preview = null;
  wallOwner.clear();
  record = vsBot()
    ? {
        game_id: gameId, model_version: versionLabel(), difficulty: game.level, sims: game.sims,
        human_side: side(), status: 'playing', winner: -1, moves: [], move_ms: [],
        started_at: Date.now(), ended_at: null, last: Date.now(),
      }
    : null;
  $('history').replaceChildren();
  updateControls();
  render();
  setOdds(0.5); // the opening counts as even, estimates start after the first move
  if (vsBot() && side() === 1) askBot();
  else showStatus();
}

function finish(winner, how) {
  over = true;
  thinking = false;
  if (record && humanMoved(record)) {
    Object.assign(record, { status: how, winner, ended_at: Date.now() });
    submit(record);
  }
  render();
  updateControls(); // a finished challenge unlocks the settings
  setOdds(winner === 0 ? 1 : winner === 1 ? 0 : 0.5);
  const loserName = name(1 - winner);
  const beatChallenge = challenge && how === 'completed' && winner === side();
  const title = beatChallenge ? "Congratulations, you've beaten the QuoriZero challenge."
    : winner === -1 ? 'Draw' : vsBot() ? (winner === side() ? 'You win!' : 'QuoriZero wins') : `${name(winner)} wins!`;
  const detail = beatChallenge ? ''
    : winner === -1 ? 'The 200-move limit was reached.'
    : how === 'resigned' ? `${loserName === 'You' ? 'You' : loserName} resigned.`
    : `Reached the goal row in ${Math.ceil(state.ply / 2)} moves.`;
  $('status').textContent = title;
  $('result-text').textContent = title;
  $('result-detail').textContent = detail;
  claimGameId = beatChallenge ? gameId : null;
  $('claim').hidden = !beatChallenge;
  $('claim-status').textContent = '';
  $('claim-email').disabled = $('claim-submit').disabled = false;
  $('again').textContent = challenge && !beatChallenge ? 'Try again' : 'Play again';
  $('leave').hidden = !challenge;
  $('result').showModal();
}

bot.onmessage = ({ data }) => {
  if (data.type === 'ready') {
    botHasModel = data.model;
    return showVersion();
  }
  if (data.id !== gameId) return; // a reply about a game that's over
  if (data.type === 'value') {
    if (data.ply !== state.ply || over) return;
    const p = (1 + data.value) / 2; // for the side to move
    return setOdds(state.toMove === 0 ? p : 1 - p);
  }
  thinking = false;
  play(data.action);
};

// Controls

function segmented(id, current, onPick) {
  for (const button of $(id).querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.value === String(current)));
    button.onclick = () => onPick(button.dataset.value);
  }
}

const LEVEL_HINTS = {
  easy: 'Relaxed: plays on instinct and makes the odd mistake.',
  medium: 'Thinks a little before every move.',
  hard: 'Thinks deeply. A real challenge.',
};

function updateControls() {
  save('qz-settings', { mode: settings.mode, humanSide: settings.humanSide, level: settings.level, sims: settings.sims });
  segmented('mode', challenge ? 'ai' : settings.mode, (v) => { settings.mode = v; start(); });
  segmented('side', side(), (v) => { settings.humanSide = Number(v); start(); });
  segmented('level', challenge ? 'challenge' : settings.level, (v) => {
    settings.level = v;
    if (v !== 'easy') settings.sims = LEVELS[v].sims;
    applySettings();
  });
  $('side-group').hidden = $('level-group').hidden = !vsBot();
  // A challenge game is locked: nothing about the bot can change until it ends.
  const locked = challenge; // until "Leave challenge"
  $('lock-note').hidden = !locked;
  $('lock-sims').textContent = challengeInfo.sims;
  document.querySelector('.panel').classList.toggle('locked', locked);
  for (const button of document.querySelectorAll('#mode button, #side button, #level button')) {
    button.disabled = locked;
  }
  const easy = settings.level === 'easy' && !challenge;
  const sims = challenge ? challengeInfo.sims : settings.sims;
  $('sims').disabled = easy || locked;
  $('sims').value = SIM_STEPS.indexOf(sims) >= 0 ? SIM_STEPS.indexOf(sims) : SIM_STEPS.indexOf(64);
  $('sims-out').textContent = easy ? 'instinct' : sims;
  $('level-hint').textContent = challenge
    ? `QuoriZero searches ${sims} positions per move.`
    : LEVEL_HINTS[settings.level] + (pendingChange() ? ' Takes effect next game.' : '');
  showChallenge();
}

function showChallenge() {
  $('challenge').hidden = !(challengeInfo.open || challenge); // a way out even if it closed
  $('challenge').classList.toggle('active', challenge);
  $('challenge-start').textContent = challenge ? 'Leave challenge' : 'Take the challenge';
  $('challenge-start').classList.toggle('primary', !challenge);
  $('prize').textContent = challengeInfo.prize;
}

function enterChallenge() {
  challenge = true;
  start();
  $('board').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// The only way out of the challenge: back to a normal game with the saved settings.
function leaveChallenge() {
  if ($('result').open) $('result').close();
  challenge = false;
  start();
}

$('sims').addEventListener('input', (e) => {
  settings.sims = SIM_STEPS[Number(e.target.value)];
  settings.level = levelFor(settings.sims);
  applySettings();
});
$('new-game').addEventListener('click', start); // in the challenge: a new challenge game
$('challenge-start').addEventListener('click', () => (challenge ? leaveChallenge() : enterChallenge()));
$('lock-leave').addEventListener('click', leaveChallenge);
$('leave').addEventListener('click', leaveChallenge);
$('claim').addEventListener('submit', async (e) => {
  e.preventDefault();
  const status = $('claim-status');
  if (!API_URL) {
    status.textContent = 'The challenge only counts on the live site.';
    return;
  }
  status.textContent = 'Sending…';
  await flush(); // the game itself must reach the server before the claim
  const unsent = () => (status.textContent = "Couldn't reach the server. Please try again in a moment.");
  if (load(PENDING, []).some((g) => g.game_id === claimGameId)) return unsent();
  try {
    const body = JSON.stringify({ game_id: claimGameId, email: $('claim-email').value.trim() });
    const res = await fetch(`${API_URL}/claims`, { method: 'POST', body });
    if (!res.ok) {
      status.textContent = 'This claim could not be accepted.';
      return;
    }
    status.textContent = "Thank you. I'll be in touch.";
    $('claim-email').disabled = $('claim-submit').disabled = true;
  } catch {
    unsent();
  }
});
function showBoardStyle() {
  $('board-style').setAttribute('aria-pressed', String(document.documentElement.dataset.board === 'classic'));
}
$('board-style').addEventListener('click', () => {
  const board = document.documentElement.dataset.board === 'classic' ? 'modern' : 'classic';
  document.documentElement.dataset.board = board;
  try { localStorage.setItem('qz-board', board); } catch {}
  showBoardStyle();
});
showBoardStyle();
$('theme').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('qz-theme', theme); } catch {}
});
$('resign').addEventListener('click', () => {
  if (over) return;
  finish(vsBot() ? 1 - side() : 1 - state.toMove, 'resigned');
});
$('again').addEventListener('click', () => {
  $('result').close();
  start();
});

const versionLabel = () => (botHasModel ? (modelVersion ?? 'unknown') : 'heuristic');
function showVersion() {
  $('version').textContent = botHasModel ? `Model ${versionLabel()}` : 'No trained model loaded: a simple heuristic bot is playing.';
}

// Recording: localStorage first, then one POST per finished game

const PENDING = 'qz-pending', CURRENT = 'qz-current';
function save(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }
function load(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
const clean = ({ last, ...game }) => game;
// Only games the player took part in: QuoriZero's opening move alone is not a game.
const humanMoved = (game) => game.moves.length > game.human_side;

async function upload(game) {
  if (!API_URL) return false; // no API yet: keep it queued locally
  try {
    // text/plain keeps it a "simple" request: no CORS preflight, works as a beacon
    const res = await fetch(`${API_URL}/games`, { method: 'POST', body: JSON.stringify(game) });
    return res.ok || res.status === 400; // 400: rejected for good, don't retry
  } catch {
    return false;
  }
}

async function flush() {
  const left = [];
  for (const game of load(PENDING, [])) if (!(await upload(game))) left.push(game);
  save(PENDING, left);
}

function submit(game) {
  save(PENDING, [...load(PENDING, []), clean(game)].slice(-50));
  save(CURRENT, null);
  flush();
}

addEventListener('pagehide', () => {
  if (record?.status !== 'playing' || !humanMoved(record)) return;
  record.status = 'abandoned';
  record.ended_at = Date.now();
  if (API_URL) navigator.sendBeacon(`${API_URL}/games`, JSON.stringify(clean(record)));
  save(CURRENT, null);
});

// Start-up

const interrupted = load(CURRENT, null); // the tab closed mid-game last time
if (interrupted && humanMoved(interrupted)) submit({ ...interrupted, status: 'abandoned', ended_at: Date.now() });
flush();

if (API_URL) {
  fetch(`${API_URL}/challenge`)
    .then((r) => r.json())
    .then((c) => Object.assign(challengeInfo, { open: c.open, sims: c.sims, humanSide: c.human_side, prize: c.prize }))
    .catch(() => (challengeInfo.open = false))
    .then(showChallenge);
}
showChallenge();

fetch(new URL('../model/version.json', import.meta.url))
  .then((r) => (r.ok ? r.json() : null))
  .then((v) => v && (modelVersion = v.version))
  .catch(() => {})
  .then(showVersion);

$('board').setAttribute('viewBox', `${-PAD - LABEL} ${-PAD} ${BOARD + 2 * PAD + LABEL} ${BOARD + 2 * PAD + LABEL}`);
start();
