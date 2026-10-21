// Games API (the Cloudflare Worker in worker/). Empty: finished games wait in
// localStorage (last 50) until it is set.
export const API_URL = 'https://quorizero-games.joaoapinho.workers.dev';

// Challenge defaults. The live site reads the real values from GET /challenge.
export const CHALLENGE = { sims: 800, humanSide: 1, prize: '$20 gift card' };
