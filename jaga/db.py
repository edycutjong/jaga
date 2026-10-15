"""SQLite state in ~/.jaga/jaga.db: tripwires, compiled specs, windows, events, thresholds, labels.

Local only. The per-window 256-d vectors live beside it in vectors-<session>.npz, keyed by embed.id.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tripwire (
  id INTEGER PRIMARY KEY, slot TEXT UNIQUE,
  sentence TEXT NOT NULL, spec_json TEXT NOT NULL,
  tf_request_id TEXT, model TEXT, compiled_at TEXT,
  armed INTEGER DEFAULT 0, unarmed_reason TEXT);
CREATE TABLE IF NOT EXISTS caption_vec (tripwire_id INTEGER, kind TEXT CHECK(kind IN ('pos','neg')),
  text TEXT, vec BLOB);
CREATE TABLE IF NOT EXISTS embed (id INTEGER PRIMARY KEY, ts REAL, session TEXT, scored INTEGER, luma REAL,
  motion_pct REAL, still_s REAL, person_in_zone INTEGER, ref_diff_pct REAL, camera_pos TEXT,
  embed_ms REAL, source TEXT);
CREATE TABLE IF NOT EXISTS win (id INTEGER PRIMARY KEY, embed_id INTEGER REFERENCES embed(id), ts REAL,
  tripwire_id INTEGER, margin REAL, z REAL, mu0 REAL, sigma0 REAL, theta_version INTEGER, embed_ms REAL,
  fired INTEGER, state TEXT, above INTEGER);
CREATE TABLE IF NOT EXISTS event (id INTEGER PRIMARY KEY, ts REAL, tripwire_id INTEGER, kind TEXT,
  detail TEXT, latency_ms REAL, session TEXT, wrong INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS threshold (tripwire_id INTEGER, version INTEGER, mode TEXT,
  theta REAL, mu0 REAL, sigma0 REAL, source TEXT, frozen_for TEXT, recall REAL, set_at TEXT);
CREATE TABLE IF NOT EXISTS label (id INTEGER PRIMARY KEY, ts REAL, session TEXT, kind TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS session (name TEXT PRIMARY KEY, started REAL, ended REAL, scored INTEGER,
  camera_pos TEXT, source TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS compile_log (id INTEGER PRIMARY KEY, ts REAL, sentence TEXT, tf_request_id TEXT,
  tf_response_id TEXT, model TEXT, temperature REAL, shape TEXT, latency_s REAL, ok INTEGER, detail TEXT,
  spec_json TEXT);
CREATE INDEX IF NOT EXISTS embed_session ON embed(session, ts);
CREATE INDEX IF NOT EXISTS win_tripwire ON win(tripwire_id, ts);
"""

EVENT_KINDS = {
    "trip", "action", "action_failed", "suspend", "camera_moved", "watcher_stopped", "watcher_ok", "too_dark",
    "operator_on", "rearmed", "armed", "disarmed", "notify", "compile", "compile_failed", "unarmed", "source_ended",
}


class DB:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else config.jaga_home() / "jaga.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # --- tripwires --------------------------------------------------------------------------------
    def upsert_tripwire(self, slot: str, sentence: str, spec: dict, tf_request_id: str | None, model: str | None,
                        compiled_at: str | None) -> int:
        with self._lock:
            row = self.conn.execute("SELECT id FROM tripwire WHERE slot=?", (slot,)).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE tripwire SET sentence=?, spec_json=?, tf_request_id=?, model=?, compiled_at=?, armed=0,"
                    " unarmed_reason=NULL WHERE id=?",
                    (sentence, json.dumps(spec), tf_request_id, model, compiled_at, row["id"]))
                self.conn.execute("DELETE FROM caption_vec WHERE tripwire_id=?", (row["id"],))
                return int(row["id"])
            cur = self.conn.execute(
                "INSERT INTO tripwire(slot, sentence, spec_json, tf_request_id, model, compiled_at)"
                " VALUES(?,?,?,?,?,?)",
                (slot, sentence, json.dumps(spec), tf_request_id, model, compiled_at))
            return int(cur.lastrowid)

    def tripwire(self, slot: str) -> dict | None:
        rows = self.query("SELECT * FROM tripwire WHERE slot=?", (slot,))
        return rows[0] if rows else None

    def tripwires(self) -> list[dict]:
        return self.query("SELECT * FROM tripwire ORDER BY slot")

    def set_armed(self, tripwire_id: int, armed: bool, reason: str | None = None) -> None:
        self.execute("UPDATE tripwire SET armed=?, unarmed_reason=? WHERE id=?", (int(armed), reason, tripwire_id))

    # --- windows and events -----------------------------------------------------------------------
    def add_embed(self, ts: float, session: str, scored: bool, luma: float | None, motion_pct: float | None,
                  still_s: float | None, person_in_zone: int | None, ref_diff_pct: float | None,
                  camera_pos: str | None, embed_ms: float | None, source: str) -> int:
        cur = self.execute(
            "INSERT INTO embed(ts, session, scored, luma, motion_pct, still_s, person_in_zone, ref_diff_pct,"
            " camera_pos, embed_ms, source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ts, session, int(scored), luma, motion_pct, still_s, person_in_zone, ref_diff_pct, camera_pos,
             embed_ms, source))
        return int(cur.lastrowid)

    def add_win(self, embed_id: int, ts: float, tripwire_id: int, margin: float, z: float | None,
                mu0: float | None, sigma0: float | None, theta_version: int | None, embed_ms: float | None,
                fired: bool, state: str, above: bool | None) -> int:
        cur = self.execute(
            "INSERT INTO win(embed_id, ts, tripwire_id, margin, z, mu0, sigma0, theta_version, embed_ms, fired,"
            " state, above) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (embed_id, ts, tripwire_id, margin, z, mu0, sigma0, theta_version, embed_ms, int(fired), state,
             None if above is None else int(above)))
        return int(cur.lastrowid)

    def add_event(self, kind: str, detail: str = "", tripwire_id: int | None = None,
                  latency_ms: float | None = None, session: str | None = None, ts: float | None = None) -> int:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        cur = self.execute(
            "INSERT INTO event(ts, tripwire_id, kind, detail, latency_ms, session) VALUES(?,?,?,?,?,?)",
            (ts or time.time(), tripwire_id, kind, detail, latency_ms, session))
        return int(cur.lastrowid)

    def mark_wrong(self, event_id: int) -> bool:
        cur = self.execute("UPDATE event SET wrong=1 WHERE id=?", (event_id,))
        return cur.rowcount == 1

    def recent_events(self, n: int = 5, session: str | None = None) -> list[dict]:
        if session is None:
            return self.query("SELECT e.*, t.slot FROM event e LEFT JOIN tripwire t ON t.id=e.tripwire_id"
                              " ORDER BY e.id DESC LIMIT ?", (n,))
        return self.query("SELECT e.*, t.slot FROM event e LEFT JOIN tripwire t ON t.id=e.tripwire_id"
                          " WHERE e.session=? ORDER BY e.id DESC LIMIT ?", (session, n))

    def add_threshold(self, tripwire_id: int, version: int, mode: str, theta: float | None, mu0: float | None,
                      sigma0: float | None, source: str, frozen_for: str | None = None,
                      recall: float | None = None) -> None:
        self.execute(
            "INSERT INTO threshold VALUES(?,?,?,?,?,?,?,?,?,?)",
            (tripwire_id, version, mode, theta, mu0, sigma0, source, frozen_for, recall,
             time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    def latest_threshold(self, tripwire_id: int) -> dict | None:
        rows = self.query("SELECT * FROM threshold WHERE tripwire_id=? ORDER BY version DESC, set_at DESC LIMIT 1",
                          (tripwire_id,))
        return rows[0] if rows else None

    def latest_job_threshold(self, slot: str) -> dict | None:
        rows = self.query("SELECT th.* FROM threshold th JOIN tripwire t ON t.id = th.tripwire_id WHERE t.slot=?"
                          " AND th.source LIKE 'nebius-job:%' ORDER BY th.set_at DESC LIMIT 1", (slot,))
        return rows[0] if rows else None

    def open_session(self) -> str | None:
        rows = self.query("SELECT name FROM session WHERE ended IS NULL ORDER BY started DESC LIMIT 1")
        return rows[0]["name"] if rows else None

    def add_label(self, session: str, kind: str, note: str = "", ts: float | None = None) -> int:
        cur = self.execute("INSERT INTO label(ts, session, kind, note) VALUES(?,?,?,?)",
                           (ts or time.time(), session, kind, note))
        return int(cur.lastrowid)

    def start_session(self, name: str, scored: bool, camera_pos: str | None, source: str, note: str = "") -> None:
        self.execute("INSERT OR REPLACE INTO session(name, started, ended, scored, camera_pos, source, note)"
                     " VALUES(?,?,NULL,?,?,?,?)", (name, time.time(), int(scored), camera_pos, source, note))

    def end_session(self, name: str) -> None:
        self.execute("UPDATE session SET ended=? WHERE name=?", (time.time(), name))

    def log_compile(self, sentence: str, ok: bool, detail: str = "", result: dict | None = None) -> None:
        r = result or {}
        self.execute(
            "INSERT INTO compile_log(ts, sentence, tf_request_id, tf_response_id, model, temperature, shape,"
            " latency_s, ok, detail, spec_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), sentence, r.get("tf_request_id"), r.get("tf_response_id"), r.get("model"),
             r.get("temperature"), r.get("shape"), r.get("latency_s"), int(ok), detail,
             json.dumps(r["spec"]) if ok and r.get("spec") else None))

    def cached_spec(self, sentence: str) -> dict | None:
        """The latest successful live compile of this exact sentence (Token Factory down: reuse it)."""
        rows = self.query("SELECT * FROM compile_log WHERE sentence=? AND ok=1 AND spec_json IS NOT NULL"
                          " ORDER BY id DESC LIMIT 1", (" ".join(sentence.split()),))
        if not rows:
            return None
        r = rows[0]
        return {"spec": json.loads(r["spec_json"]), "tf_request_id": r["tf_request_id"],
                "tf_response_id": r["tf_response_id"], "model": r["model"], "temperature": r["temperature"],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(r["ts"])), "cached": True}

    def recent_compiles(self, n: int = 10) -> list[dict]:
        return self.query("SELECT id, ts, sentence, tf_request_id, model, shape, latency_s, ok, detail"
                          " FROM compile_log ORDER BY id DESC LIMIT ?", (n,))

    def close(self) -> None:
        with self._lock:
            self.conn.close()
