// node --no-warnings worker/test.mjs : the Worker against a local SQLite standing in for D1.

import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import worker from './src/index.js';

const sqlite = new DatabaseSync(':memory:');
sqlite.exec(readFileSync(new URL('./schema.sql', import.meta.url), 'utf8'));
// D1-shaped wrapper: prepare(sql)[.bind(...args)].first() / .all() / .run()
const statement = (sql, args = []) => ({
  bind: (...bound) => statement(sql, bound),
  first: async () => sqlite.prepare(sql).get(...args),
  all: async () => ({ results: sqlite.prepare(sql).all(...args) }),
  run: async () => ({ meta: sqlite.prepare(sql).run(...args) }), // D1 nests results in meta
});
const DB = { prepare: (sql) => statement(sql) };
const env = { DB, EXPORT_TOKEN: 'secret', RESEND_API_KEY: 'key', OWNER_EMAIL: 'owner@test.dev' };
const background = [];
const ctx = { waitUntil: (promise) => background.push(promise) };
const emails = [];
globalThis.fetch = async (url, init) => { emails.push({ url, body: JSON.parse(init.body) }); return new Response('{}'); };
const call = (method, path, body, headers = {}) =>
  worker.fetch(new Request(`https://api.test${path}`, { method, body, headers }), env, ctx);
const owner = { Authorization: 'Bearer secret' };
const check = (ok, msg) => {
  if (!ok) {
    console.error('FAIL', msg);
    process.exit(1);
  }
};

const game = {
  game_id: 'g1', model_version: 'main/iter_0100', difficulty: 'hard', sims: 400, human_side: 0,
  status: 'completed', winner: 1, moves: [1, 0, 1], move_ms: [900, 1200, 700],
  started_at: 1, ended_at: 2,
};
check((await call('POST', '/games', JSON.stringify(game))).status === 200, 'store a game');
check((await call('POST', '/games', JSON.stringify(game))).status === 200, 'resend is fine');
check((await call('POST', '/games', '{nope')).status === 400, 'bad json');
check((await call('POST', '/games', JSON.stringify({ ...game, moves: [999] }))).status === 400, 'bad move');
check((await call('GET', '/games?since=0')).status === 401, 'export needs the token');

const res = await call('GET', '/games?since=0', undefined, { Authorization: 'Bearer secret' });
const { games } = await res.json();
check(games.length === 1, 'duplicate ignored');
check(games[0].id === 'g1' && games[0].moves.join() === '1,0,1' && !('ip_hash' in games[0]), 'export shape');
check(games[0].sims === 400, 'sims stored');
check((await call('POST', '/games', JSON.stringify({ ...game, game_id: 'c1', difficulty: 'hard', sims: 9999 }))).status === 400, 'sims capped');

// Challenge
const challengeWin = { ...game, game_id: 'w1', difficulty: 'challenge', sims: 800, human_side: 1, winner: 1 };
const claim = (gameId, email = 'player@mail.com') => call('POST', '/claims', JSON.stringify({ game_id: gameId, email }));
check((await (await call('GET', '/challenge')).json()).open === false, 'closed until the owner opens it');
check((await call('POST', '/challenge', '{"open": true}')).status === 401, 'only the owner opens it');
check((await (await call('POST', '/challenge', '{"open": true}', owner)).json()).open === true, 'owner opens it');
check((await call('POST', '/games', JSON.stringify({ ...challengeWin, game_id: 'w0', sims: 400 }))).status === 400, 'challenge terms enforced');
check((await call('POST', '/games', JSON.stringify(challengeWin))).status === 200, 'challenge game stored');
check((await claim('w1', 'not-an-email')).status === 400, 'email checked');
check((await claim('g1')).status === 400, 'a hard-mode game is not a challenge win');
check((await claim('w1')).status === 200, 'challenge win claimed');
check((await claim('w1')).status === 200, 'claiming twice is harmless');
await Promise.all(background);
check(emails.length === 1 && emails[0].url.includes('resend') && emails[0].body.to[0] === 'owner@test.dev', 'owner emailed once');
check(emails[0].body.text.includes('player@mail.com') && emails[0].body.text.includes('w1'), 'email names claimant and game');
const one = await call('GET', '/games?id=w1', undefined, owner);
check((await one.json()).games[0].id === 'w1', 'owner can fetch one game');
await call('POST', '/challenge', '{"open": false}', owner);
check((await claim('w1')).status === 403, 'closed: no claims');

for (let i = 0; i < 70; i++) await call('POST', '/games', JSON.stringify({ ...game, game_id: `x${i}` }));
check((await call('POST', '/games', JSON.stringify({ ...game, game_id: 'late' }))).status === 429, 'rate limit');
console.log('ok: worker stores, dedups, validates, rate-limits, exports, and runs the challenge');
