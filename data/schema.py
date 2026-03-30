"""SQLite schema definitions. Single source of truth for all table DDL."""

SCHEMA_VERSION = 3

SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT    PRIMARY KEY,
    session_id      TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    total_ticks     INTEGER,
    winner          TEXT,
    player_hp_final INTEGER,
    ai_hp_final     INTEGER,
    rng_seed        INTEGER NOT NULL,
    model_version   TEXT,
    config_hash     TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'human'
);
"""

SEMANTIC_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS semantic_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type          TEXT    NOT NULL,
    match_id            TEXT    NOT NULL,
    tick_id             INTEGER NOT NULL,
    actor               TEXT    NOT NULL,
    commitment          TEXT,
    opponent_fsm_state  TEXT,
    opponent_commitment TEXT,
    spacing_zone        TEXT,
    actor_hp            INTEGER NOT NULL,
    opponent_hp         INTEGER NOT NULL,
    actor_stamina       INTEGER NOT NULL,
    opponent_stamina    INTEGER NOT NULL,
    damage_dealt        INTEGER,
    reaction_ticks      INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);
"""

SEMANTIC_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_match_tick ON semantic_events(match_id, tick_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_actor_type ON semantic_events(actor, event_type);",
]

PLAYER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS player_profiles (
    profile_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id               TEXT    NOT NULL DEFAULT 'player_1',
    session_count           INTEGER NOT NULL DEFAULT 0,
    total_ticks_observed    INTEGER NOT NULL DEFAULT 0,
    action_frequencies      TEXT    NOT NULL DEFAULT '{}',
    recent_action_frequencies TEXT  NOT NULL DEFAULT '{}',
    bigrams                 TEXT    NOT NULL DEFAULT '{}',
    trigrams                TEXT    NOT NULL DEFAULT '{}',
    spacing_distribution    TEXT    NOT NULL DEFAULT '{}',
    movement_direction_bias REAL    NOT NULL DEFAULT 0.0,
    dodge_left_pct          REAL    NOT NULL DEFAULT 0.5,
    dodge_right_pct         REAL    NOT NULL DEFAULT 0.5,
    dodge_frequency         REAL    NOT NULL DEFAULT 0.0,
    aggression_index        REAL    NOT NULL DEFAULT 0.0,
    initiative_rate         REAL    NOT NULL DEFAULT 0.0,
    punish_conversion_rate  REAL    NOT NULL DEFAULT 0.0,
    low_hp_action_dist      TEXT    NOT NULL DEFAULT '{}',
    cornered_action_dist    TEXT    NOT NULL DEFAULT '{}',
    combo_sequences         TEXT    NOT NULL DEFAULT '[]',
    avg_reaction_time_ms    REAL    NOT NULL DEFAULT 0.0,
    reaction_time_stddev    REAL    NOT NULL DEFAULT 0.0,
    win_rate_vs_ai          REAL    NOT NULL DEFAULT 0.0,
    avg_match_duration      REAL    NOT NULL DEFAULT 0.0,
    last_updated            TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(player_id)
);
"""

AI_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_decisions (
    decision_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    match_id            TEXT    NOT NULL,
    tick_id             INTEGER NOT NULL,
    predicted_top       TEXT,
    pred_confidence     REAL,
    pred_probs          TEXT,
    tactical_mode       TEXT    NOT NULL,
    ai_action           TEXT    NOT NULL,
    positioning_bias    REAL,
    commit_delay        INTEGER,
    reason_tags         TEXT,
    outcome             TEXT,
    outcome_tick        INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);
"""

AI_DECISIONS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_decisions_match ON ai_decisions(match_id, tick_id);",
]

MODEL_REGISTRY_TABLE = """
CREATE TABLE IF NOT EXISTS model_registry (
    version         TEXT    PRIMARY KEY,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    model_path      TEXT    NOT NULL,
    model_type      TEXT    NOT NULL,
    eval_accuracy   REAL,
    eval_top2_acc   REAL,
    dataset_size    INTEGER,
    is_active       INTEGER NOT NULL DEFAULT 0,
    metadata        TEXT    NOT NULL DEFAULT '{}'
);
"""

ALL_TABLES = [
    SCHEMA_VERSION_TABLE,
    MATCHES_TABLE,
    SEMANTIC_EVENTS_TABLE,
    PLAYER_PROFILES_TABLE,
    AI_DECISIONS_TABLE,
    MODEL_REGISTRY_TABLE,
]

ALL_INDEXES = SEMANTIC_EVENTS_INDEXES + AI_DECISIONS_INDEXES
