CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date_iso TEXT NOT NULL,
    items_json TEXT,
    kcal INTEGER,
    protein_g INTEGER,
    carbs_g INTEGER,
    fat_g INTEGER,
    source TEXT,
    idempotency_key TEXT UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date_iso TEXT NOT NULL,
    plan_json TEXT,
    status TEXT,
    idempotency_key TEXT UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS wellness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date_iso TEXT NOT NULL,
    sleep_h REAL,
    mood TEXT,
    notes TEXT,
    idempotency_key TEXT UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    target_wxid TEXT,
    context_token TEXT,
    remind_at_iso TEXT NOT NULL,
    remind_at_epoch INTEGER NOT NULL,
    text TEXT NOT NULL,
    priority TEXT DEFAULT 'normal',
    delivered INTEGER DEFAULT 0,
    delivered_at INTEGER,
    idempotency_key TEXT UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_meals_user_date ON meals(user_id, date_iso);
CREATE INDEX IF NOT EXISTS ix_workouts_user_date ON workouts(user_id, date_iso);
CREATE INDEX IF NOT EXISTS ix_wellness_user_date ON wellness(user_id, date_iso);
CREATE INDEX IF NOT EXISTS ix_reminders_due ON reminders(delivered, remind_at_epoch);
