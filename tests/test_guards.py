"""Environment guards and baseline inputs (jaga/guards.py) on synthetic frames."""
from dataclasses import replace

import cv2
import numpy as np

from jaga.config import TUNING
from jaga.guards import Guards


def room(seed: int = 0, w: int = 640, h: int = 360) -> np.ndarray:
    """A textured 'room': a wall with shelves and frames, so phase correlation has structure."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 150, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def with_person(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (250, 120), (360, 360), (35, 30, 30), -1)  # a person-sized blob in the middle
    cv2.circle(out, (305, 100), 28, (60, 50, 45), -1)
    return out


def pan(img: np.ndarray, frac: float) -> np.ndarray:
    m = np.float32([[1, 0, frac * img.shape[1]], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_camera_moved_not_triggered_by_a_person_leaving():
    """A person at the board in the reference, then the empty room (they left): not a camera move."""
    g = Guards(TUNING, hog=False)
    ref = with_person(room())
    g.set_reference([ref] * 10)
    results = [g.update(float(t), room(), room()) for t in range(10)]
    assert not any(r.camera_moved for r in results)
    assert all(r.phase_shift_frac < TUNING.camera_shift_frac for r in results)


def test_camera_moved_triggered_by_a_pan_after_three_windows():
    g = Guards(TUNING, hog=False)
    g.set_reference([room()] * 10)
    moved = [g.update(float(t), pan(room(), 0.08), room()).camera_moved for t in range(4)]
    assert moved == [False, False, True, True]  # debounced: 3 windows in a row


def test_luma_jump_flags_suspend():
    g = Guards(TUNING, hog=False)
    a = g.update(0.0, room(), room())
    b = g.update(1.0, (room().astype(np.float32) * 0.5).astype(np.uint8), room())
    assert not a.luma_jump and b.luma_jump


def test_too_dark_below_floor():
    g = Guards(TUNING, hog=False)
    dark = np.full((360, 640, 3), 12, np.uint8)
    assert g.update(0.0, dark, dark).too_dark
    assert not g.update(1.0, room(), room()).too_dark


def test_motion_and_stillness_timer_inputs():
    g = Guards(replace(TUNING, still_motion_pct=0.5), hog=False)
    still = room()
    r0 = g.update(0.0, still, still)
    assert r0.motion_pct == 0.0
    moving = with_person(still)
    r1 = g.update(1.0, moving, still)
    assert r1.motion_pct > 1.0 and r1.still_s == 0.0
    r2 = g.update(5.0, moving, moving)
    assert r2.still_s == 4.0  # seconds since the last frame-difference above P8


def test_reference_difference_against_the_arming_median():
    g = Guards(TUNING, hog=False)
    closed = room(3)
    g.set_reference([closed] * 5)
    same = g.update(0.0, closed, closed).ref_diff_pct
    changed = g.update(1.0, with_person(closed), closed).ref_diff_pct
    assert same < 0.5 < changed


def test_person_in_zone_returns_a_flag():
    g = Guards(TUNING, hog=True)
    r = g.update(0.0, room(), room())
    assert r.person_in_zone in (0, 1)


def test_flat_top_strip_falls_back_to_full_frame():
    g = Guards(TUNING, hog=False)
    img = room()
    img[:60, :] = 150  # a blank wall above the scene
    g.set_reference([img] * 5)
    assert g.phase_region == "full-frame"
