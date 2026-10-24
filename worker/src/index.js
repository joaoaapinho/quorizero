// Games API on Cloudflare Workers + D1
//   POST /games            store one finished game (from the website)
//   GET  /games?since=ms   export games for retraining (Bearer EXPORT_TOKEN)
//   GET  /games?id=...     one game, for reviewing a challenge claim (Bearer)
//   GET  /challenge        is the challenge open, and on what terms
//   POST /challenge        {"open": true|false}: only the owner opens/closes it (Bearer)
//   POST /claims           {"game_id", "email"}: a challenge win, emails the owner
//   GET  /retrain-status   weighted losses since the last retrain, and whether one is due (Bearer)
//   POST /retrains         record a retraining run, which restarts the count (Bearer)
// Only the shape is checked here, the rules are checked offline by replaying
// every game through the Python engine (src/ingestion/games.py).

const MAX_BODY = 20_000;
const GAMES_PER_HOUR = 60; // per IP
const STATUSES = ['completed', 'resigned', 'abandoned'];
const LEVELS = ['easy', 'medium', 'hard', 'challenge'];
// The challenge's terms live here, so no browser can claim a win on easier ones.
const CHALLENGE = { sims: 800, humanSide: 1, prize: '$20 gift card' };
const CLAIMS_PER_DAY = 5; // per IP
const EMAIL = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,}$/;
// A QuoriZero loss counts by level: Hard and challenge 1, Medium ½, Easy 0 (a casual
// level loses on purpose). Same weights in src/ingestion/retrain.py.
const LOSS_WEIGHT = "CASE difficulty WHEN 'hard' THEN 1.0 WHEN 'challenge' THEN 1.0 WHEN 'medium' THEN 0.5 ELSE 0 END";
const RETRAIN_AT = 5; // weighted losses, unless the RETRAIN_AT variable says otherwise
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Authorization, Content-Type',
};

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...CORS } });

const isInt = (x, lo, hi) => Number.isInteger(x) && x >= lo && x <= hi;

function valid(g) {
  return (
    typeof g === 'object' && g !== null &&
    typeof g.game_id === 'string' && g.game_id.length <= 64 &&
    STATUSES.includes(g.status) && LEVELS.includes(g.difficulty) &&
    isInt(g.human_side, 0, 1) && isInt(g.winner, -1, 1) &&
    (g.sims === undefined || isInt(g.sims, 0, 5000)) &&
    (g.difficulty !== 'challenge' || (g.sims === CHALLENGE.sims && g.human_side === CHALLENGE.humanSide)) &&
    Array.isArray(g.moves) && g.moves.length <= 200 && g.moves.every((a) => isInt(a, 0, 139)) &&
    Array.isArray(g.move_ms) && g.move_ms.length === g.moves.length &&
    g.move_ms.every((ms) => isInt(ms, 0, 86_400_000))
  );
}

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function saveGame(request, env) {
  const body = await request.text(); // sent as text/plain: no CORS preflight
  if (body.length > MAX_BODY) return json({ error: 'too large' }, 413);
  let game;
  try {
    game = JSON.parse(body);
  } catch {
    return json({ error: 'bad json' }, 400);
  }
  if (!valid(game)) return json({ error: 'bad game' }, 400);

  const now = Date.now();
  const ipHash = await sha256(`${request.headers.get('CF-Connecting-IP') ?? ''}:${env.EXPORT_TOKEN}`);
  const recent = await env.DB.prepare('SELECT COUNT(*) AS n FROM games WHERE ip_hash = ? AND created_at > ?')
    .bind(ipHash, now - 3_600_000)
    .first();
  if (recent.n >= GAMES_PER_HOUR) return json({ error: 'rate limited' }, 429);

  // created_at never repeats, so "games after X" cursors can't skip a same-millisecond game
  await env.DB.prepare(
    `INSERT OR IGNORE INTO games (id, created_at, ip_hash, model_version, difficulty, sims,
       human_side, status, winner, moves, move_ms, started_at, ended_at)
     VALUES (?, MAX(?, COALESCE((SELECT MAX(created_at) FROM games), 0) + 1),
       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      game.game_id, now, ipHash, String(game.model_version ?? '').slice(0, 100), game.difficulty,
      game.sims ?? null, game.human_side, game.status, game.winner, JSON.stringify(game.moves),
      JSON.stringify(game.move_ms), Number(game.started_at) || null, Number(game.ended_at) || null,
    )
    .run();
  return json({ ok: true });
}

const isOwner = (request, env) =>
  Boolean(env.EXPORT_TOKEN) && request.headers.get('Authorization') === `Bearer ${env.EXPORT_TOKEN}`;

const parseGame = ({ ip_hash, moves, move_ms, ...row }) => ({
  ...row,
  moves: JSON.parse(moves),
  move_ms: JSON.parse(move_ms),
});

async function exportGames(url, request, env) {
  if (!isOwner(request, env)) return json({ error: 'unauthorized' }, 401);
  const id = url.searchParams.get('id');
  if (id) {
    const game = await env.DB.prepare('SELECT * FROM games WHERE id = ?').bind(id).first();
    return game ? json({ games: [parseGame(game)] }) : json({ error: 'not found' }, 404);
  }
  const since = Number(url.searchParams.get('since')) || 0;
  const limit = Math.min(Number(url.searchParams.get('limit')) || 1000, 1000);
  const { results } = await env.DB.prepare(
    'SELECT * FROM games WHERE created_at > ? ORDER BY created_at LIMIT ?',
  )
    .bind(since, limit)
    .all();
  return json({ games: results.map(parseGame) });
}

async function challengeOpen(env) {
  const row = await env.DB.prepare("SELECT value FROM settings WHERE key = 'challenge_open'").first();
  return row?.value === 'true';
}

async function getChallenge(env) {
  const { sims, humanSide, prize } = CHALLENGE;
  return json({ open: await challengeOpen(env), sims, human_side: humanSide, prize });
}

// Only the owner opens or closes the challenge; a win never closes it by itself.
async function setChallenge(request, env) {
  if (!isOwner(request, env)) return json({ error: 'unauthorized' }, 401);
  let open;
  try {
    open = JSON.parse(await request.text()).open;
  } catch {}
  if (typeof open !== 'boolean') return json({ error: 'send {"open": true|false}' }, 400);
  await env.DB.prepare(
    "INSERT INTO settings (key, value) VALUES ('challenge_open', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
  )
    .bind(String(open))
    .run();
  return getChallenge(env);
}

// Losses since the last retrain: finished games QuoriZero lost with a real model loaded.
async function retrainStatus(env) {
  const last = await env.DB.prepare('SELECT MAX(games_until) AS since FROM retrains').first();
  const since = last?.since ?? 0;
  const row = await env.DB.prepare(
    `SELECT COALESCE(SUM(${LOSS_WEIGHT}), 0) AS losses, COUNT(*) AS games, MAX(created_at) AS until
     FROM games
     WHERE created_at > ? AND status = 'completed' AND winner = human_side
       AND model_version != 'heuristic' AND ${LOSS_WEIGHT} > 0`,
  )
    .bind(since)
    .first();
  const retrainAt = Number(env.RETRAIN_AT) || RETRAIN_AT;
  return {
    since, until: row.until ?? since, losses: row.losses, games: row.games,
    retrain_at: retrainAt, due: row.losses >= retrainAt,
  };
}

async function getRetrainStatus(request, env) {
  if (!isOwner(request, env)) return json({ error: 'unauthorized' }, 401);
  return json(await retrainStatus(env));
}

async function recordRetrain(request, env) {
  if (!isOwner(request, env)) return json({ error: 'unauthorized' }, 401);
  let run;
  try {
    run = JSON.parse(await request.text());
  } catch {}
  const ok =
    isInt(run?.games_until, 0, Number.MAX_SAFE_INTEGER) && typeof run.losses === 'number' && run.losses >= 0 &&
    typeof run.promoted === 'boolean' && (run.gate_score == null || typeof run.gate_score === 'number') &&
    (run.model_version == null || (typeof run.model_version === 'string' && run.model_version.length <= 100));
  if (!ok) return json({ error: 'send {games_until, losses, promoted, gate_score?, model_version?}' }, 400);
  await env.DB.prepare(
    'INSERT INTO retrains (created_at, games_until, losses, promoted, gate_score, model_version) VALUES (?, ?, ?, ?, ?, ?)',
  )
    .bind(Date.now(), run.games_until, run.losses, run.promoted ? 1 : 0, run.gate_score ?? null, run.model_version ?? null)
    .run();
  return json(await retrainStatus(env));
}

async function claimPrize(request, env, ctx) {
  let claim;
  try {
    claim = JSON.parse(await request.text());
  } catch {}
  const { game_id: gameId, email } = claim ?? {};
  if (typeof gameId !== 'string' || typeof email !== 'string' || email.length > 254 || !EMAIL.test(email)) {
    return json({ error: 'bad claim' }, 400);
  }
  if (!(await challengeOpen(env))) return json({ error: 'the challenge is closed' }, 403);
  const game = await env.DB.prepare('SELECT * FROM games WHERE id = ?').bind(gameId).first();
  const won =
    game?.difficulty === 'challenge' && game.sims === CHALLENGE.sims && game.human_side === CHALLENGE.humanSide &&
    game.status === 'completed' && game.winner === game.human_side;
  if (!won) return json({ error: 'not a challenge win' }, 400);

  const now = Date.now();
  const ipHash = await sha256(`${request.headers.get('CF-Connecting-IP') ?? ''}:${env.EXPORT_TOKEN}`);
  const recent = await env.DB.prepare('SELECT COUNT(*) AS n FROM claims WHERE ip_hash = ? AND created_at > ?')
    .bind(ipHash, now - 86_400_000)
    .first();
  if (recent.n >= CLAIMS_PER_DAY) return json({ error: 'rate limited' }, 429);
  const saved = await env.DB.prepare(
    'INSERT OR IGNORE INTO claims (game_id, email, created_at, ip_hash) VALUES (?, ?, ?, ?)',
  )
    .bind(gameId, email, now, ipHash)
    .run();
  if (saved.meta?.changes) ctx.waitUntil(notifyOwner(env, parseGame(game), email));
  return json({ ok: true });
}

// Email the owner through Resend (free tier) when RESEND_API_KEY and OWNER_EMAIL are set.
async function notifyOwner(env, game, email) {
  if (!env.RESEND_API_KEY || !env.OWNER_EMAIL) return;
  const text = [
    'Someone claims to have beaten the QuoriZero challenge.',
    '',
    `Claimant: ${email}`,
    `Game: ${game.id}`,
    `Model: ${game.model_version}`,
    `Moves: ${game.moves.length}`,
    '',
    'Review it before sending anything:',
    `  QZ_EXPORT_TOKEN=... python -m ingestion.verify_claim ${game.id} --api <worker-url>`,
  ].join('\n');
  await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      from: env.EMAIL_FROM || 'QuoriZero <onboarding@resend.dev>',
      to: [env.OWNER_EMAIL],
      subject: 'QuoriZero challenge claim to review',
      text,
    }),
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const route = `${request.method} ${url.pathname}`;
    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });
    if (route === 'POST /games') return saveGame(request, env);
    if (route === 'GET /games') return exportGames(url, request, env);
    if (route === 'GET /challenge') return getChallenge(env);
    if (route === 'POST /challenge') return setChallenge(request, env);
    if (route === 'POST /claims') return claimPrize(request, env, ctx);
    if (route === 'GET /retrain-status') return getRetrainStatus(request, env);
    if (route === 'POST /retrains') return recordRetrain(request, env);
    return json({ error: 'not found' }, 404);
  },
};
