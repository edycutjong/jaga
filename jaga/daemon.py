"""The watcher: the one flow, live. sense (Cosmos-Embed1) → decide (score.py) → act (act.py).

One thread embeds windows from the ring, runs the guards, steps every tripwire and logs one `embed` row
plus one `win` row per tripwire. Actions run on their own worker so a slow plug never stalls the camera;
every plug call (a trip's OFF, a status read, the operator test) holds one driver lock, so they never
interleave. The plug's state is read every few seconds (read only); an observed OFF → ON after an
acknowledged OFF re-arms a tripped tripwire, through the same gate every arm path uses (not in the dark,
not with a moved camera). A watchdog reports WATCHER STOPPED when windows stop, fail, or a thread dies.
Nothing here can switch power ON (invariant I1); the overlay shows OFF only for this trip's
acknowledgement (I3).
"""
from __future__ import annotations

import base64
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict

import numpy as np

from . import act as act_mod
from .capture import Ring, Window
from .config import TUNING, Tuning
from .db import DB
from .frames import to_jpeg
from .guards import GuardResult, Guards
from .score import Threshold, Tripwire, window_period
from .vectors import VectorStore

TRACE_S = 120.0  # the overlay's margin trace covers the last two minutes
FAILED_WINDOWS_STOP = 3  # this many failed windows in a row = WATCHER STOPPED
MIN_DWELL_S = 2.0
SLOW_RETRY_S = 60.0  # after the fast OFF retries, a failed OFF is retried this often, for as long as it stays tripped
SLOW_RETRY_NOTIFY_EVERY = 10  # … and pushed again every this many tries
DARK_REMINDER_S = 600.0  # while a plug_off tripwire is unarmed in the dark and the plug is ON, remind this often
FIRST_WINDOW_S = 60.0  # no window at all this long after start = WATCHER STOPPED


class Watcher:
    def __init__(self, *, db: DB, embedder, ring: Ring, driver: act_mod.PlugDriver, notifier: act_mod.Notifier,
                 session: str, camera_pos: str | None = None, scored: bool = False, tuning: Tuning = TUNING,
                 guards: Guards | None = None, publish=None, source=None, frozen_for: str | None = None,
                 offline: bool = False, labels: bool = False):
        self.db, self.embedder, self.ring, self.driver, self.notifier = db, embedder, ring, driver, notifier
        self.session, self.camera_pos, self.scored, self.t = session, camera_pos, scored, tuning
        self.frozen_for = frozen_for
        self.offline = offline  # media-time windows (recorded file, no pacing)
        self.labels = labels or scored  # the label keys (exits, still episodes) are on in soaks and when asked for
        self.started_mono = time.monotonic()
        self.dark_reminded_mono = 0.0
        self.compiler_down = False  # set by the server when Token Factory is unreachable (header: degraded)
        self.guards = guards or Guards(tuning)
        self.publish = publish or (lambda kind, payload: None)
        self.source = source
        self.lock = threading.RLock()
        self.driver_lock = threading.Lock()  # one plug conversation at a time
        self.tripwires: dict[str, Tripwire] = {}
        self.meta: dict[str, dict] = {}
        self.capture_slots: set[str] = set()
        self.dark_unarmed: set[str] = set()
        self.dark = False
        self.light_streak = 0
        self.ts_hist: deque[float] = deque(maxlen=51)
        self.embed_s_hist: deque[float] = deque(maxlen=50)
        self.last_seq = 0
        self.last_window_mono: float | None = None
        self.fail_streak = 0
        self.watcher_ok = True
        self.windows = 0
        self.last_guard: GuardResult | None = None
        self.plug = {**driver.describe(), "state": "unknown", "last_action": None, "latency_ms": None,
                     "failed": False, "detail": ""}
        self.plug_seen_off = False
        self.last_off_ack_mono = 0.0
        self._poll_now = False
        self._next_poll = 0.0
        self.retry: dict | None = None
        self.trace: dict[str, deque] = {}
        self.frame_jpeg: str | None = None
        self.filmstrip: list[str] = []
        self.stop_evt = threading.Event()
        self.actions: queue.Queue = queue.Queue()
        self.notify_q: queue.Queue = queue.Queue(maxsize=64)
        self.vstore = VectorStore(session)
        self.threads: list[threading.Thread] = []
        db.start_session(session, scored, camera_pos, "offline" if offline else "live",
                         note=f"frozen_for={frozen_for}" if frozen_for else "")

    # ------------------------------------------------------------------ tripwire management (API thread)
    def load_tripwire(self, slot: str, sentence: str, spec: dict, pos: np.ndarray, neg: np.ndarray,
                      threshold: Threshold, meta: dict | None = None) -> Tripwire:
        from .compile import may_cut
        from .tripwires import build

        meta = dict(meta or {})
        tid = self.db.upsert_tripwire(slot, sentence, spec, meta.get("tf_request_id"), meta.get("model"),
                                      meta.get("created"))
        self.db.execute("DELETE FROM caption_vec WHERE tripwire_id=?", (tid,))
        for kind, texts, vecs in (("pos", spec["positives"], pos), ("neg", spec["hard_negatives"], neg)):
            for text, v in zip(texts, vecs, strict=True):
                self.db.execute("INSERT INTO caption_vec(tripwire_id, kind, text, vec) VALUES(?,?,?,?)",
                                (tid, kind, text, np.asarray(v, np.float32).tobytes()))
        op = meta.get("operating")
        if op and op.get("action") == "plug_off" and not may_cut(sentence, spec.get("name", "")):
            op = {**op, "action": "notify"}  # cutting power is only for heating/cooking appliances
        tw = build(slot, sentence, spec, pos, neg, threshold, self.t, tid, op)
        with self.lock:
            self.tripwires[slot] = tw
            self.meta[slot] = {**meta, "sentence": sentence, "spec": spec, "tripwire_id": tid}
            self.trace[slot] = deque(maxlen=int(TRACE_S * 2) + 8)
        if threshold.version is not None:
            self.db.add_threshold(tid, threshold.version, threshold.mode, threshold.theta, threshold.mu0,
                                  threshold.sigma0, threshold.source, self.frozen_for)
        self._event("compile", tw, f"{tw.name}: loaded ({threshold.mode}"
                    + (f", θ v{threshold.version}" if threshold.version is not None else "") + ")")
        return tw

    def set_operating(self, slot: str, action: str | None = None, dwell_s: float | None = None) -> dict:
        """The user's choice in the compile drawer, before arming. Power is cut only for heating/cooking appliances."""
        from .compile import may_cut

        with self.lock:
            tw = self._get(slot)
            if tw.state not in ("unarmed",):
                raise ValueError("disarm before changing what the tripwire does")
            if action is not None:
                if action not in ("plug_off", "notify"):
                    raise ValueError("action must be plug_off or notify")
                if action == "plug_off" and not may_cut(tw.sentence, tw.name):
                    raise ValueError("cutting power is only for heating or cooking appliances (an iron, a stove, "
                                     "a heater, a kettle…): notify only")
                tw.action = action
            if dwell_s is not None:
                if not MIN_DWELL_S <= float(dwell_s) <= 600:
                    raise ValueError("dwell must be 2–600 s")
                tw.dwell_s = float(dwell_s)
                tw.rule.dwell_s = float(dwell_s)
            self.meta[slot]["operating"] = {"action": tw.action, "dwell_s": tw.dwell_s, "cooldown_s": tw.cooldown_s,
                                            "set_by": "user, before arming"}
            return self.meta[slot]["operating"]

    def override_dwell(self, slot: str, dwell_s: float, note: str) -> None:
        """A labelled override (the video's demo dwell): shown on screen next to the spec's own dwell."""
        if float(dwell_s) < MIN_DWELL_S:
            raise ValueError("dwell must be at least 2 s")
        with self.lock:
            tw = self._get(slot)
            spec_dwell = tw.dwell_s
            tw.dwell_s = float(dwell_s)
            tw.rule.dwell_s = float(dwell_s)
            self.meta[slot]["dwell_note"] = f"{note} · default {spec_dwell:.0f} s"

    def _arm_blocker(self, tw: Tripwire) -> str | None:
        """The one gate every arm without a new capture goes through (call under self.lock)."""
        if self.dark:
            return "too dark to see"
        if tw.stale_ref:
            return "camera moved — re-arm to capture a new reference"
        if not self.guards.has_reference:
            return "no camera reference — re-arm to capture the normal state"
        if tw.threshold.mode == "fresh-z" and (tw.threshold.mu0 is None or tw.threshold.sigma0 is None):
            return "no normal state yet — arm with a 2-minute capture"
        return None

    def arm(self, slot: str, capture_s: float | None = None) -> None:
        with self.lock:
            tw = self._get(slot)
            if self.dark:
                raise ValueError("too dark to see: turn a light on, then arm")
            capture_s = self.t.capture_s if capture_s is None else float(capture_s)
            if capture_s > 0:
                if not self.capture_slots:
                    self.guards.begin_reference()
                self.capture_slots.add(slot)
                tw.start_capture(self._now(), capture_s)
                self._event("armed", tw, f"{tw.name}: capturing the normal state for {capture_s:.0f} s")
            else:
                why = self._arm_blocker(tw)
                if why:
                    raise ValueError(why)
                tw.arm()
                self.db.set_armed(tw.tripwire_id, True)
                self._event("armed", tw, f"{tw.name}: armed with its stored reference")
            self.dark_unarmed.discard(slot)

    def disarm(self, slot: str) -> None:
        with self.lock:
            tw = self._get(slot)
            tw.unarm("disarmed")
            self.capture_slots.discard(slot)
            self.dark_unarmed.discard(slot)
            self.db.set_armed(tw.tripwire_id, False, "disarmed")
            self._event("disarmed", tw, f"{tw.name}: disarmed")

    def add_example(self, slot: str) -> dict:
        """The optional teaching step: 'it is happening now'. After 3 examples, the noise floor is checked."""
        with self.lock:
            tw = self._get(slot)
            if tw.last is None or tw.threshold.mu0 is None:
                return {"ok": False, "detail": "capture the normal state first"}
            ex = self.meta[slot].setdefault("examples", [])
            ex.append(tw.last.m)
            if len(ex) < 3:
                return {"ok": True, "examples": len(ex)}
            ok = tw.noise_floor_ok(ex[-3:])
            if not ok:
                tw.unarm("can't see that: name an object and where it is")
                self.db.set_armed(tw.tripwire_id, False, tw.unarmed_reason)
                self._event("unarmed", tw, f"{tw.name}: can't see that (examples did not clear the normal state)")
            return {"ok": ok, "examples": len(ex)}

    def label(self, kind: str, note: str = "") -> int:
        lid = self.db.add_label(self.session, kind, note, ts=self._now())
        self.publish("label", {"id": lid, "kind": kind, "note": note, "ts": self._now()})
        return lid

    def _get(self, slot: str) -> Tripwire:
        if slot not in self.tripwires:
            raise KeyError(f"no tripwire {slot}")
        return self.tripwires[slot]

    def _now(self) -> float:
        return self.ts_hist[-1] if (self.offline and self.ts_hist) else time.time()

    # ------------------------------------------------------------------ threads
    def start(self) -> None:
        for target, name in ((self._loop, "jaga-loop"), (self._actions_worker, "jaga-act"),
                             (self._watchdog, "jaga-watchdog"), (self._plug_poll, "jaga-plug"),
                             (self._notify_loop, "jaga-notify")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self.threads.append(th)
        self._poll_now = True  # read the plug's real state at once

    def stop(self) -> None:
        self.stop_evt.set()
        self.ring.close()
        self.actions.put(None)  # queued OFFs before it still run: power is cut before the watcher exits
        for th in self.threads:
            th.join(timeout=5)
        self.vstore.flush()
        self.db.end_session(self.session)

    def _loop(self) -> None:
        while not self.stop_evt.is_set():
            win = self.ring.wait_window(self.last_seq, timeout=0.5)
            if win is None:
                if self.ring.closed:
                    break
                continue
            try:
                self.process(win)
            except Exception as e:  # never die silently: three failed windows in a row = WATCHER STOPPED
                print(f"jaga: window failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                with self.lock:
                    self.fail_streak += 1
                    n = self.fail_streak
                    stop = n >= FAILED_WINDOWS_STOP and self.watcher_ok
                    if stop:
                        self.watcher_ok = False
                if stop:
                    self._event("watcher_stopped", None,
                                f"WATCHER STOPPED — {n} windows failed in a row ({type(e).__name__})", notify=True)
            finally:
                self.last_seq = win.seq
                self.ring.mark_consumed(win.seq)

    # ------------------------------------------------------------------ one window
    def process(self, win: Window) -> list:
        t0 = time.perf_counter()
        vec = self.embedder.embed_video(win.frames)
        if not np.all(np.isfinite(vec)):
            raise ValueError("the model returned a non-finite embedding")
        embed_s = time.perf_counter() - t0
        g = self.guards.update(win.ts[-1], win.newest, win.previous)
        ts = win.ts[-1] if self.offline else time.time()
        results = []
        with self.lock:
            self.ts_hist.append(ts)
            self.embed_s_hist.append(embed_s)
            self.windows += 1
            self.last_guard = g
            period_p50, period_p95 = window_period(self.ts_hist)
            stride = period_p50 or 1.0
            eid = self.db.add_embed(ts, self.session, self.scored, g.luma, g.motion_pct, g.still_s,
                                    g.person_in_zone, g.ref_diff_pct, self.camera_pos, embed_s * 1000,
                                    "offline" if self.offline else "live")
            self.vstore.add(eid, vec)
            self._guards_act(ts, g)
            if self.capture_slots:
                self.guards.add_reference(win.newest)
            blocked = self.dark or g.too_dark  # level-triggered: no window in the dark counts as above
            for slot in sorted(self.tripwires):
                tw = self.tripwires[slot]
                was = tw.state
                r = tw.step(ts, vec, stride_s=stride, period_p95=period_p95, blocked=blocked)
                th = tw.threshold
                self.db.add_win(eid, ts, tw.tripwire_id, r.m, r.z, th.mu0, th.sigma0, th.version, embed_s * 1000,
                                r.fired, r.state, r.above)
                score = r.m if th.mode == "labelled-m" else r.z
                self.trace[slot].append((ts, score, th.theta, r.above))
                if was == "capturing" and tw.state != "capturing":
                    self._capture_done(tw)
                if r.fired:
                    self._on_trip(tw, ts, r)
                results.append(r)
            if self.capture_slots and not any(self.tripwires[s].state == "capturing" for s in self.capture_slots):
                self.capture_slots.clear()
            self.frame_jpeg = base64.b64encode(to_jpeg(win.newest, 960, 78)).decode()
            self.filmstrip = [base64.b64encode(to_jpeg(f, 152, 70)).decode() for f in win.frames]
            # the heartbeat moves only after a complete window
            self.last_window_mono = time.monotonic()
            self.fail_streak = 0
            back = not self.watcher_ok
            self.watcher_ok = True
            snap = self.snapshot(include_frames=True)
        if back:
            self._event("watcher_ok", None, "windows are flowing again")
        self.publish("window", snap)
        return results

    def _capture_done(self, tw: Tripwire) -> None:
        th = tw.threshold
        self.meta.setdefault(tw.slot, {})["armed_at"] = self._now()  # opens the optional "show me" check
        self.meta[tw.slot]["examples"] = []
        if self.guards.end_reference():
            self.guards.begin_reference()  # another tripwire may still be capturing
        self.db.set_armed(tw.tripwire_id, True)
        if th.mode == "fresh-z":
            self.db.add_threshold(tw.tripwire_id, 0, th.mode, th.theta, th.mu0, th.sigma0, "normal-state",
                                  self.frozen_for)
        self._event("armed", tw, f"{tw.name}: armed · μ₀ {th.mu0:+.4f} · σ₀ {th.sigma0:.4f} over "
                    f"{len(tw.capture_margins)} windows")

    def _rearm_or_hold(self, tw: Tripwire, why_text: str) -> None:
        """Re-arm with the stored reference if the gate allows (call under self.lock)."""
        why = self._arm_blocker(tw)
        if why:
            tw.unarm(why)
            self.db.set_armed(tw.tripwire_id, False, why)
            if why == "too dark to see":
                self.dark_unarmed.add(tw.slot)
            self._event("unarmed", tw, f"{tw.name}: {why_text}, but {why}")
            return
        tw.rearm()
        self.db.set_armed(tw.tripwire_id, True)
        self._event("rearmed", tw, f"RESET · {why_text} — {tw.name} re-armed with its stored reference")

    def _guards_act(self, ts: float, g: GuardResult) -> None:
        live = [tw for tw in self.tripwires.values() if tw.state in ("armed", "holding", "capturing")]
        if g.too_dark and not self.dark:
            self.dark, self.light_streak = True, 0
            self.dark_reminded_mono = time.monotonic()  # the entry push below counts as the first reminder
            for tw in live:
                tw.unarm("too dark to see")
                self.capture_slots.discard(tw.slot)
                self.dark_unarmed.add(tw.slot)
                self.db.set_armed(tw.tripwire_id, False, "too dark to see")
            if live:
                self._event("too_dark", None, f"too dark to see (luma {g.luma:.2f}) — {len(live)} unarmed", notify=True)
        elif self.dark:
            if g.luma >= self.t.dark_floor * self.t.light_return_factor:
                self.light_streak += 1
                if self.light_streak >= self.t.light_return_windows:
                    self.dark = False
                    for slot in sorted(self.dark_unarmed):
                        tw = self.tripwires[slot]
                        if tw.threshold.theta is not None and tw.threshold.mu0 is not None:
                            self._rearm_or_hold(tw, "light returned")
                    self.dark_unarmed.clear()
            else:
                self.light_streak = 0
        if g.luma_jump and not g.too_dark:
            hit = [tw for tw in self.tripwires.values() if tw.state in ("armed", "holding")]
            for tw in hit:
                tw.suspend(ts, self.t.suspend_s)
            if hit:
                self._event("suspend", None, f"sudden light change — trips suspended for {self.t.suspend_s:.0f} s")
        if g.camera_moved:
            moved = [tw for tw in self.tripwires.values() if tw.state in ("armed", "holding", "capturing")]
            for tw in self.tripwires.values():
                tw.stale_ref = True  # every stored reference predates the move, tripped tripwires' too
            for tw in moved:
                tw.unarm("camera moved — re-arm")
                self.capture_slots.discard(tw.slot)
                self.db.set_armed(tw.tripwire_id, False, "camera moved")
            self.guards.clear_reference()
            if moved:
                self._event("camera_moved", None,
                            f"camera moved — {len(moved)} unarmed; re-arm to capture a new reference", notify=True)

    # ------------------------------------------------------------------ act
    def _on_trip(self, tw: Tripwire, ts: float, r) -> None:
        held = r.dwell_s
        self._event("trip", tw, f"{tw.name}: tripped · {r.count}/{r.k} windows above in {held:.0f} s"
                    + (f" · margin {r.m:+.4f}" if tw.threshold.mode == "labelled-m" else f" · z {r.z:+.2f}"), ts=ts)
        if tw.action == "plug_off":
            tw.off_acked = False
            self.plug["state"] = "unknown"  # a cached "off" must never stand in for THIS trip's acknowledgement (I3)
            self.actions.put(("off", tw.slot, ts, held))
        else:
            self.actions.put(("notify", tw.slot, ts, held))

    def _actions_worker(self) -> None:
        while True:
            job = self.actions.get()
            if job is None:
                return
            kind, slot, ts, held = job
            tw = self.tripwires.get(slot)
            if tw is None or (kind == "notify" and self.stop_evt.is_set()):
                continue
            try:
                if kind == "notify":
                    clock = time.strftime("%H:%M:%S", time.localtime(ts))
                    text = f"jaga · {slot} {tw.name} · tripped {clock} · held {held:.0f} s"
                    self._push(text)
                    self._event("notify", tw, text)
                else:
                    self._turn_off(tw, ts, held, first=True)
            except Exception as e:  # the cut path must never die: show it, push it, retry the OFF
                print(f"jaga: action worker error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                if kind == "off":
                    with self.lock:
                        self.plug.update(failed=True, detail=f"unexpected error ({type(e).__name__})",
                                         last_action="off-failed")
                        self.retry = {"slot": slot, "left": self.t.action_retries, "next": time.monotonic()
                                      + self.t.action_retry_s, "ts": ts, "held": held, "tries": 1}
                self._event("action_failed", tw, f"ACTION FAILED: unexpected error ({type(e).__name__}) · retrying",
                            notify=True)

    def _turn_off(self, tw: Tripwire, ts: float, held: float, first: bool) -> bool:
        clock = time.strftime("%H:%M:%S", time.localtime(ts))
        try:
            with self.driver_lock:
                ack = self.driver.turn_off()  # returns only after the plug acknowledges OFF (I3)
        except Exception as e:  # ActionFailed, or anything a driver did not expect: never "done"
            reason = str(e) if isinstance(e, act_mod.ActionFailed) else f"unexpected error ({type(e).__name__})"
            with self.lock:
                self.plug.update(failed=True, detail=reason[:200], last_action="off-failed")
                if first:
                    self.retry = {"slot": tw.slot, "left": self.t.action_retries, "next": time.monotonic()
                                  + self.t.action_retry_s, "ts": ts, "held": held, "tries": 1}
            if first:
                self._push(f"jaga · {tw.slot} {tw.name} · tripped {clock} · ACTION FAILED — plug did not switch off")
                self._event("action_failed", tw, f"ACTION FAILED: {reason[:120]} · push sent · retrying OFF")
            self.publish("window", self.snapshot())
            return False
        with self.lock:
            tw.off_acked = True
            self.last_off_ack_mono = time.monotonic()
            self.plug.update(state="off", failed=False, detail=ack.detail, last_action="off",
                             latency_ms=round(ack.latency_ms, 1))
            self.plug_seen_off = True
            tries = (self.retry or {}).get("tries", 0)
            self.retry = None
        note = f" after {tries} retries" if not first else ""
        self._event("action", tw, f"plug OFF acknowledged in {ack.latency_ms:.0f} ms{note}", latency_ms=ack.latency_ms)
        self._push(f"jaga · {tw.slot} {tw.name} · tripped {clock} · plug OFF · held {held:.0f} s")
        self.publish("tripped", {"slot": tw.slot, "ts": ts, "plug": dict(self.plug)})
        self.publish("window", self.snapshot())
        return True

    def _plug_poll(self) -> None:
        """Read the plug's state every few seconds (read only); retry a failed OFF (never ON); re-arm on OFF → ON."""
        while not self.stop_evt.wait(0.5):
            try:
                self._poll_once()
            except Exception as e:  # a poll error must not end polling
                print(f"jaga: plug poll error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def _poll_once(self) -> None:
        with self.lock:
            retry = dict(self.retry) if self.retry else None
        if retry and time.monotonic() >= retry["next"]:
            tw = self.tripwires.get(retry["slot"])
            if tw is None or tw.state != "tripped":
                with self.lock:
                    self.retry = None
            elif not self._turn_off(tw, retry["ts"], retry["held"], first=False):
                with self.lock:
                    if self.retry:
                        self.retry["tries"] += 1
                        self.retry["left"] -= 1
                        self.retry["next"] = time.monotonic() + self.t.action_retry_s
                        if self.retry["left"] <= 0:  # never give up on an OFF: slow down, keep telling people
                            self.retry["left"] = 1
                            self.retry["next"] = time.monotonic() + SLOW_RETRY_S
                            if self.retry["tries"] % SLOW_RETRY_NOTIFY_EVERY == 0 or not self.retry.get("slow"):
                                self.retry["slow"] = True
                                self._event("action_failed", tw,
                                            f"plug still not OFF after {self.retry['tries']} tries — retrying every "
                                            f"{SLOW_RETRY_S:.0f} s; check the plug", notify=True)
            return
        with self.lock:
            force, self._poll_now = self._poll_now, False
            if not force and time.monotonic() < self._next_poll:
                return
            self._next_poll = time.monotonic() + self.t.rearm_poll_s
        with self.driver_lock:
            started = time.monotonic()
            st = self.driver.status()
        if st is None:
            return
        with self.lock:
            if started < self.last_off_ack_mono:
                return  # this read began before the latest OFF landed: stale, never acted on
            tripped = [tw for tw in self.tripwires.values()
                       if tw.action == "plug_off" and tw.state == "tripped" and tw.off_acked]
            if st is False:
                self.plug["state"] = "off"
                if tripped:
                    self.plug_seen_off = True
            else:
                self.plug["state"] = "on"
                if tripped and self.plug_seen_off:  # an observed OFF → ON: someone switched it back on
                    self.plug_seen_off = False
                    for tw in tripped:
                        self._rearm_or_hold(tw, "plug switched back on")
        self.publish("window", self.snapshot())

    def plug_test(self) -> dict:
        """Operator action (POST /plug/test): OFF then ON, timed. ON exists only in operator_on.py (I1).

        It holds the driver lock across both calls, so a trip's OFF can never land between them and be undone:
        a trip during the test cuts power right after it."""
        from . import operator_on

        with self.driver_lock:
            res = operator_on.plug_test(self.driver)
        with self.lock:
            self.plug.update(state="on", failed=False, last_action="operator-on", latency_ms=res["on_ms"],
                             detail=f"test: off {res['off_ms']:.0f} ms · on {res['on_ms']:.0f} ms")
            self.plug_seen_off = True  # the test passed through OFF: a tripped tripwire re-arms on the next read
            self.last_off_ack_mono = time.monotonic()
            self._poll_now = True
        self._event("operator_on", None, f"operator test: OFF {res['off_ms']:.0f} ms → ON {res['on_ms']:.0f} ms",
                    latency_ms=res["on_ms"])
        return res

    # ------------------------------------------------------------------ watchdog + notifications
    def _watchdog(self) -> None:
        while not self.stop_evt.wait(1.0):
            reason = None
            with self.lock:
                dead = [t.name for t in self.threads if not t.is_alive()]
                if dead and self.watcher_ok:
                    reason = f"WATCHER STOPPED — a watcher thread ended: {', '.join(dead)}"
                elif self.last_window_mono is not None and any(
                        tw.state in ("armed", "holding", "capturing", "tripped") for tw in self.tripwires.values()):
                    _, p95 = window_period(self.ts_hist)
                    limit = max(self.t.watchdog_factor * (p95 or 1.0), self.t.watchdog_floor_s)
                    stale = time.monotonic() - self.last_window_mono
                    if stale > limit and self.watcher_ok:
                        reason = f"WATCHER STOPPED — no window for {stale:.0f} s"
                elif (self.last_window_mono is None and self.watcher_ok
                      and time.monotonic() - self.started_mono > FIRST_WINDOW_S):
                    reason = f"WATCHER STOPPED — no window since start ({FIRST_WINDOW_S:.0f} s): check the camera"
                if reason:
                    self.watcher_ok = False
                remind = None
                dark_cut = [self.tripwires[s] for s in self.dark_unarmed if self.tripwires[s].action == "plug_off"]
                if (self.dark and dark_cut and self.plug.get("state") == "on"
                        and time.monotonic() - self.dark_reminded_mono > DARK_REMINDER_S):
                    self.dark_reminded_mono = time.monotonic()
                    remind = dark_cut[0]
            if reason:
                self._event("watcher_stopped", None, reason, notify=True)
            if remind is not None:
                self._event("too_dark", remind, f"still too dark to see — {remind.name} is unarmed and its plug is ON",
                            notify=True)
            if not self.watcher_ok:
                self.publish("window", self.snapshot())

    def _push(self, text: str) -> None:
        """Queue a notification; the OS banner and ntfy run on their own thread, never under the lock."""
        try:
            self.notify_q.put_nowait(text)
        except queue.Full:
            pass

    def _notify_loop(self) -> None:
        while not self.stop_evt.is_set() or not self.notify_q.empty():
            try:
                text = self.notify_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.notifier.notify(text)
            except Exception as e:
                print(f"jaga: notify failed: {type(e).__name__}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ events and state
    def _event(self, kind: str, tw: Tripwire | None, detail: str, latency_ms: float | None = None,
               notify: bool = False, ts: float | None = None) -> int:
        """Log, publish and (optionally) push one event. Never raises: an event must not kill a watcher thread."""
        ts = self._now() if ts is None else ts
        try:
            eid = self.db.add_event(kind, detail, tw.tripwire_id if tw else None, latency_ms, self.session, ts=ts)
        except Exception as e:
            print(f"jaga: event not logged ({type(e).__name__}): {kind} {detail}", file=sys.stderr, flush=True)
            eid = -1
        if notify:
            self._push(f"jaga · {detail}")
        try:
            self.publish("event", {"id": eid, "kind": kind, "slot": tw.slot if tw else None, "detail": detail,
                                   "ts": ts, "latency_ms": latency_ms})
        except Exception:
            pass
        return eid

    def snapshot(self, include_frames: bool = False) -> dict:
        with self.lock:
            p50, p95 = window_period(self.ts_hist)
            emb = sorted(self.embed_s_hist)
            g = self.last_guard
            rows = []
            for slot in sorted(self.tripwires):
                tw = self.tripwires[slot]
                r = tw.last
                th = tw.threshold
                left = None
                if tw.state == "capturing":
                    left = tw.capture_s if tw.capture_until is None else max(0.0, tw.capture_until - self._now())
                rows.append({
                    "slot": slot, "name": tw.name, "sentence": tw.sentence, "action": tw.action,
                    "dwell_s": tw.dwell_s, "effective_dwell_s": tw.effective_dwell(p95), "state": tw.state,
                    "unarmed_reason": tw.unarmed_reason, "mode": th.mode, "theta": th.theta,
                    "theta_version": th.version, "theta_source": th.source, "mu0": th.mu0, "sigma0": th.sigma0,
                    "m": None if r is None else r.m, "z": None if r is None else r.z,
                    "above": None if r is None else r.above, "progress": 0.0 if r is None else r.progress,
                    "count": 0 if r is None else r.count, "k": 0 if r is None else r.k, "n": 0 if r is None else r.n,
                    "capture_left_s": left, "capture_windows": len(tw.capture_margins),
                    "off_acked": tw.off_acked, "stale_ref": tw.stale_ref,
                    "tf_request_id": self.meta.get(slot, {}).get("tf_request_id"),
                    "positives": self.meta.get(slot, {}).get("spec", {}).get("positives", []),
                    "hard_negatives": self.meta.get(slot, {}).get("spec", {}).get("hard_negatives", []),
                    "cooldown_s": tw.cooldown_s, "dwell_note": self.meta.get(slot, {}).get("dwell_note"),
                    "armed_at": self.meta.get(slot, {}).get("armed_at"),
                    "examples": len(self.meta.get(slot, {}).get("examples", [])),
                    "proposed": {k: self.meta.get(slot, {}).get("spec", {}).get(k) for k in ("action", "dwell_s")},
                    "operating_set_by": (self.meta.get(slot, {}).get("operating") or {}).get("set_by"),
                })
            snap = {
                "ts": self._now(), "session": self.session, "scored": self.scored, "camera_pos": self.camera_pos,
                "windows": rows, "watcher_ok": self.watcher_ok, "n_windows": self.windows,
                "motion_pct": None if g is None else g.motion_pct, "still_s": None if g is None else g.still_s,
                "luma": None if g is None else g.luma, "dark": self.dark,
                "s_per_clip_p50": None if not emb else float(np.percentile(emb, 50)),
                "s_per_clip_p95": None if not emb else float(np.percentile(emb, 95)),
                "period_p50": p50, "period_p95": p95, "plug": dict(self.plug),
                "device": getattr(self.embedder, "device", None), "frozen_for": self.frozen_for,
                "labels": self.labels, "compiler_down": self.compiler_down,
                "source": None if self.source is None else asdict(self.source.status),
            }
            if include_frames:
                snap["frame_jpeg_b64"] = self.frame_jpeg
                snap["filmstrip_jpeg_b64"] = self.filmstrip
            return snap

    def traces(self) -> dict:
        with self.lock:
            return {slot: [list(x) for x in d] for slot, d in self.trace.items()}
