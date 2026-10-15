"""Safety invariants (architecture: Protocol Invariants). It switches mains power, so these are tested, not assumed.

I1 — a trip can only cut power or notify. ON exists only in jaga/operator_on.py, reached by operator actions:
POST /plug/test and the hand-run scripts/day0_plug.py.
I2 — a trip needs k-of-n windows over a dwell ≥ 2 × p95 window period (tests/test_score.py).
I3 — the UI shows OFF only after the driver acknowledges (tests/test_watcher.py).
"""
import ast
import re
from pathlib import Path

from conftest import make_watcher, push_windows, wait_for

PKG = Path(__file__).resolve().parents[1] / "jaga"
ON_PATTERNS = re.compile(r"turn_on\(|switch_on\(|['\"]on['\"]\s*:\s*['\"]true['\"]|set_status\(\s*True")


def test_i1_no_on_call_outside_the_operator_module():
    offenders = []
    for f in PKG.rglob("*.py"):
        if f.name == "operator_on.py":
            continue
        code = "\n".join(ln for ln in f.read_text().splitlines() if not ln.lstrip().startswith("#"))
        for m in ON_PATTERNS.finditer(code):
            offenders.append(f"{f.name}: {m.group(0)}")
    assert offenders == [], offenders


def test_i1_operator_module_only_reachable_from_plug_test():
    importers = []
    for f in PKG.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "operator_on" in node.module:
                importers.append(f.name)
            if isinstance(node, ast.ImportFrom) and any(a.name == "operator_on" for a in node.names):
                importers.append(f.name)
    assert importers == ["daemon.py"], importers
    daemon = ast.parse((PKG / "daemon.py").read_text())
    def imports_operator(fn):
        return any(isinstance(m, ast.ImportFrom) and any(a.name == "operator_on" for a in m.names)
                   for m in ast.walk(fn))

    owner = [n.name for n in ast.walk(daemon) if isinstance(n, ast.FunctionDef) and imports_operator(n)]
    assert owner == ["plug_test"]
    callers = [f.name for f in PKG.rglob("*.py")
               if re.search(r"\.plug_test\b", f.read_text()) and f.name != "daemon.py"]
    assert callers == ["server.py"]
    server = (PKG / "server.py").read_text()
    assert '@app.post("/plug/test")' in server
    scripts = [f.name for f in (PKG.parent / "scripts").glob("*.py") if "operator_on" in f.read_text()]
    assert scripts == ["day0_plug.py"], scripts  # the hand-run plug check is the only other operator


def test_i1_no_trip_path_imports_the_operator_module():
    for name in ("score.py", "act.py", "guards.py", "capture.py", "offline.py", "soak.py", "calibrate.py"):
        tree = ast.parse((PKG / name).read_text())
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names = [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
        assert not any("operator_on" in m for m in mods) and "operator_on" not in names, name


def test_i1_drivers_expose_off_and_status_only():
    from jaga.act import NullPlug, PlugDriver, ShellyPlug, TuyaPlug

    for cls in (PlugDriver, NullPlug, ShellyPlug, TuyaPlug):
        public = {n for n in vars(cls) if not n.startswith("_") and callable(getattr(cls, n))}
        assert not {n for n in public if "on" in n.split("_") or n in ("switch", "set", "toggle")}, (cls, public)


def test_i1_ten_trips_and_rearms_never_send_on(home, tuning, fakeplug):
    """Dynamic check: ten trips and re-arms through the real actuator never send an ON."""
    w, emb = make_watcher(home, tuning, fakeplug.url, dwell_s=3)
    w.start()
    try:
        w.arm("01", 0)
        t = 100.0
        for _ in range(10):
            push_windows(w, emb, "pos", 5, t)
            t += 5
            assert wait_for(lambda: w.plug["state"] == "off")
            fakeplug.press_button()
            assert wait_for(lambda: w.tripwires["01"].state == "armed")
            push_windows(w, emb, "neg", 2, t)
            t += 2
        assert fakeplug.off_calls == 10 and fakeplug.on_calls == 0
    finally:
        w.stop()


def test_never_cut_policy_coerces_fridge_to_notify(home, tuning):
    from jaga.compile import apply_action_policy

    spec, note = apply_action_policy("the fridge door is left open", {"action": "plug_off", "name": "FRIDGE"})
    assert spec["action"] == "notify" and note
    spec, note = apply_action_policy("nobody is at the ironing board", {"action": "plug_off", "name": "BOARD"})
    assert spec["action"] == "plug_off" and note is None
    w, _ = make_watcher(home, tuning, None, action="notify")
    w.tripwires["01"].sentence = "the kulkas door is open"
    try:
        w.set_operating("01", action="plug_off")
        raised = False
    except ValueError:
        raised = True
    assert raised
