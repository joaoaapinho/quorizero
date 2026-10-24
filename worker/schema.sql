-- One row per game played on the website. Moves are engine action indices (0..139).
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,           -- generated in the browser: re-sends are ignored
  created_at INTEGER NOT NULL,   -- server time, epoch ms, never repeats (cursor)
  ip_hash TEXT NOT NULL,         -- salted hash, only for rate limiting
  model_version TEXT,
  difficulty TEXT,               -- easy | medium | hard | challenge
  sims INTEGER,                  -- the bot's search budget (0 = no search)
  human_side INTEGER,
  status TEXT NOT NULL,          -- completed | resigned | abandoned
  winner INTEGER,                -- 0, 1, or -1 (none)
  moves TEXT NOT NULL,           -- JSON array of ints
  move_ms TEXT NOT NULL,         -- JSON array: thinking time before each move
  started_at INTEGER,
  ended_at INTEGER
);
CREATE INDEX IF NOT EXISTS games_by_time ON games (created_at);

-- Owner switches, e.g. challenge_open = 'true' | 'false' (POST /challenge).
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Challenge prize claims, one per winning game - reviewed by hand before paying.
CREATE TABLE IF NOT EXISTS claims (
  game_id TEXT PRIMARY KEY REFERENCES games (id),
  email TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  ip_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'  -- pending | paid | rejected (set by hand)
);
CREATE INDEX IF NOT EXISTS claims_by_ip ON claims (ip_hash, created_at);
CREATE INDEX IF NOT EXISTS games_by_ip ON games (ip_hash, created_at);

-- One row per retraining run, promoted or not. Losses are counted from the newest
-- games_until onwards, so recording a run restarts the count (GET /retrain-status).
CREATE TABLE IF NOT EXISTS retrains (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at INTEGER NOT NULL,     -- server time, epoch ms
  games_until INTEGER NOT NULL,    -- created_at of the newest game this run covered
  losses REAL NOT NULL,            -- weighted losses that triggered it
  promoted INTEGER NOT NULL,       -- 1 if the new model passed the gate
  gate_score REAL,
  model_version TEXT               -- the promoted model, if any
);
