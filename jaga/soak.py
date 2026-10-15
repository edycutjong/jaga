"""Scored sessions (Soak A and Soak B) follow a schedule committed before they start (docs/PROTOCOL.md).

The runner announces each pre-registered step when it is due (spoken with macOS `say`, and as a line on
the overlay) and logs it as a `plan:<kind>` label. The person's own key presses (1–5, 6–9, 0) log what
actually happened, so a deviation is recorded, not hidden. A scored session never feeds calibration.
"""
from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step:
    step: int
    start_min: float
    end_min: float
    kind: str
    text: str


def load_schedule(path: str | Path) -> list[Step]:
    with Path(path).open() as f:
        return [Step(int(r["step"]), float(r["start_min"]), float(r["end_min"]), r["kind"], r["text"])
                for r in csv.DictReader(f)]


class ScheduleRunner(threading.Thread):
    """Starts the clock with the first window (step 0 is the capture); announces each step 30 s ahead and when due."""

    def __init__(self, watcher, steps: list[Step], *, say: bool = True, warn_s: float = 30.0,
                 started: threading.Event | None = None, on_complete=None):
        super().__init__(daemon=True, name="jaga-schedule")
        self.w, self.steps, self.say_enabled, self.warn_s = watcher, steps, say, warn_s
        self.on_complete = on_complete
        self.started = started or threading.Event()
        self.t0: float | None = None
        self._halt = threading.Event()  # not `_stop`: Thread._stop is CPython's

    def stop(self) -> None:
        self._halt.set()

    def _say(self, text: str) -> None:
        self.w.publish("schedule", {"text": text, "ts": time.time()})
        if self.say_enabled:
            try:
                subprocess.Popen(["say", "-r", "185", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass

    def run(self) -> None:
        while not self._halt.is_set() and not self.started.wait(1.0):
            pass
        if self._halt.is_set():
            return
        self.t0 = time.time()
        self.w.label("plan:session_start", f"schedule clock starts; {len(self.steps)} steps")
        warned: set[int] = set()
        done: set[int] = set()
        ended: set[int] = set()
        while not self._halt.wait(0.5):
            el = (time.time() - self.t0) / 60.0
            for s in self.steps:
                if s.step not in warned and el >= s.start_min - self.warn_s / 60.0 and s.step not in done:
                    warned.add(s.step)
                    self._say(f"In thirty seconds: {s.text}.")
                if s.step not in done and el >= s.start_min:
                    done.add(s.step)
                    self.w.label(f"plan:{s.kind}", f"step {s.step}: {s.text} ({s.start_min:.1f}–{s.end_min:.1f} min)")
                    self._say(f"Step {s.step}. {s.text}. Now.")
                if s.step in done and s.step not in ended and el >= s.end_min:
                    ended.add(s.step)
                    self.w.label(f"plan_end:{s.kind}", f"step {s.step} ends")
            if len(ended) == len(self.steps):
                self.w.label("plan:session_end", "schedule complete")
                self._say("The scheduled session is complete. Thank you. jaga stops in one minute.")
                if self.on_complete:
                    self.on_complete()
                return
