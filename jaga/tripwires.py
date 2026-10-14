"""Frozen tripwires: one Nemotron compile per sentence, saved with its Token Factory request ID.

`fixtures/tripwires/<slot>.json` holds the sentence, the TripwireSpec exactly as Super returned it, and
the request metadata. `<slot>.vec.npz` holds the caption vectors (`pos`, `neg`, from Cosmos-Embed1
`text_proj`). Once labelled clips exist, `seed.py freeze` adds the frozen threshold block (`frozen`:
θ_m, version, recall, source) — after the spec, never before, and the spec is never edited (leakage guard).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import TUNING, Tuning
from .score import Threshold, Tripwire

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures" / "tripwires"


def fixture_path(slot_or_path: str | Path) -> Path:
    p = Path(slot_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    cand = FIXTURES / f"{str(slot_or_path).zfill(2)}.json"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no frozen tripwire {slot_or_path!r} (looked in {FIXTURES})")


def load(slot_or_path: str | Path) -> dict:
    p = fixture_path(slot_or_path)
    d = json.loads(p.read_text())
    d["_path"] = str(p)
    return d


def vec_path(fixture: dict) -> Path:
    return Path(fixture["_path"]).with_suffix("").with_suffix(".vec.npz")


def load_vectors(fixture: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(vec_path(fixture)) as z:
        return z["pos"].astype(np.float32), z["neg"].astype(np.float32)


def save_vectors(fixture: dict, pos: np.ndarray, neg: np.ndarray) -> Path:
    p = vec_path(fixture)
    np.savez(p, pos=np.asarray(pos, np.float32), neg=np.asarray(neg, np.float32))
    return p


def frozen_threshold(fixture: dict) -> Threshold | None:
    f = fixture.get("frozen")
    if not f or f.get("theta_m") is None:
        return None
    return Threshold(mode="labelled-m", theta=float(f["theta_m"]), mu0=f.get("mu0"), sigma0=f.get("sigma0"),
                     version=int(f["version"]), source=f.get("source", "loo-clips"),
                     frozen_for=f.get("frozen_for"))


def fresh_threshold(tuning: Tuning = TUNING) -> Threshold:
    """Fresh sentence: θ_z after the 2-min normal-state capture sets μ₀/σ₀."""
    return Threshold(mode="fresh-z", theta=tuning.theta_z, source="default")


def operating(spec: dict, op: dict | None = None) -> dict:
    """Action, dwell and cool-down actually used: an explicit operating choice, else Super's proposal."""
    op = op or {}
    return {"action": op.get("action", spec["action"]), "dwell_s": float(op.get("dwell_s", spec["dwell_s"])),
            "cooldown_s": float(op.get("cooldown_s", spec["cooldown_s"])),
            "source": "operating choice" if op else "as compiled"}


def build(slot: str, sentence: str, spec: dict, pos: np.ndarray, neg: np.ndarray, threshold: Threshold,
          tuning: Tuning = TUNING, tripwire_id: int | None = None, op: dict | None = None) -> Tripwire:
    eff = operating(spec, op)
    return Tripwire(slot=slot, name=spec["name"], sentence=sentence, action=eff["action"],
                    dwell_s=eff["dwell_s"], cooldown_s=eff["cooldown_s"], pos=pos, neg=neg,
                    threshold=threshold, tripwire_id=tripwire_id, tuning=tuning)
