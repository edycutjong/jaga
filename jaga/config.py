"""Paths, model IDs and the tuned constants P1–P8.

Every constant marked "placeholder" is a starting value, not a result. Day-0 and the unscored
session set them; `docs/PROTOCOL.md` records the values frozen before a scored session.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- models -------------------------------------------------------------------------------------
COSMOS_MODEL = "nvidia/Cosmos-Embed1-224p"
COSMOS_REVISION = "787e0b996f5260a71ad474a283c90539a2e12986"  # the snapshot day-0 stage COSMOS measured
EMBED_DIM = 256
WINDOW_FRAMES = 8  # the card: optimised for 8 frames sampled at 1–2 fps
CAPTURE_FPS = 1.0

TF_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
TF_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# The Super model card recommends temperature 1.0 and top_p 0.95 "across all tasks and serving backends".
TF_TEMPERATURE = 1.0
TF_TOP_P = 0.95
# Super is a reasoning model: a small max_tokens can return reasoning and no content.
TF_MAX_TOKENS = 4096

# --- server -------------------------------------------------------------------------------------
HOST = "127.0.0.1"  # never the LAN
PORT = 8765


def jaga_home() -> Path:
    """Local state: SQLite DB, per-session vectors, labels. Never uploaded."""
    p = Path(os.environ.get("JAGA_HOME", Path.home() / ".jaga"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_dir() -> Path:
    """Secrets and device config live outside the repo."""
    return Path(os.environ.get("JAGA_CONFIG_DIR", Path.home() / ".config" / "jaga"))


@dataclass
class Tuning:
    """The tuned constants (architecture: P1–P8). Defaults are placeholders until measured."""

    luma_jump: float = 0.25  # P1: relative jump in mean luma between windows → suspend
    suspend_s: float = 60.0  # P2: suspend length after a jump
    dark_floor: float = 0.12  # P3: mean luma (0–1) below which the camera cannot see
    phase_response_min: float = 0.3  # P4: border-mask phase-correlation response floor
    camera_shift_frac: float = 0.04  # shift above 4 % of the width = camera moved
    camera_moved_windows: int = 3  # consecutive windows the camera-moved test must fail (one noisy frame never unarms)
    sigma0_floor: float = 0.01  # P5: floor on σ₀ (margin units)
    theta_z: float = 3.0  # P6: default θ_z for a fresh sentence
    k_fraction: float = 0.8  # P7: k = ceil(k_fraction · n)
    still_motion_pct: float = 0.5  # P8: stillness-timer motion threshold (% of pixels changed)
    motion_pixel_delta: int = 25  # grey-level change that counts a pixel as "changed"
    hog_min_weight: float = 0.0  # HOG pedestrian detector: OpenCV's default SVM score floor (baseline input only)
    watchdog_factor: float = 3.0  # WATCHER STOPPED after this many × the p95 window period
    watchdog_floor_s: float = 10.0  # …but never sooner than this
    rearm_poll_s: float = 5.0  # plug status poll while a plug_off tripwire is tripped
    light_return_factor: float = 1.2  # "too dark" clears once luma ≥ 1.2 × P3 …
    light_return_windows: int = 5  # … for this many consecutive windows
    action_retry_s: float = 5.0  # a failed turn_off is retried (OFF only) every this many seconds …
    action_retries: int = 12  # … at most this many times
    capture_s: float = 120.0  # the 2-min "normal state" capture at arming
    extra: dict = field(default_factory=dict)


TUNING = Tuning()
