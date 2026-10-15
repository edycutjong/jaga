"""The only code path that switches a plug ON. Operator-only (invariant I1).

Reachable only from operator actions: inside the package, from `POST /plug/test` in `jaga/server.py` (which
requires a localhost Host, an absent or same-origin Origin and the per-run session token); outside it, from the
day-0 plug check a person runs by hand (`scripts/day0_plug.py`). No trip, re-arm or guard imports this module;
`tests/test_invariants.py` enforces that. A trip only ever cuts power or notifies.
"""
from __future__ import annotations

import time

from .act import Ack, ActionFailed, PlugDriver, ShellyPlug, TuyaPlug, why


def switch_on(driver: PlugDriver) -> Ack:
    t0 = time.perf_counter()
    try:
        if isinstance(driver, TuyaPlug):
            r = driver._device().turn_on(switch=int(driver.dps))
            if isinstance(r, dict) and r.get("Error"):
                raise ActionFailed(f"tuya: {r.get('Error')}")
        elif isinstance(driver, ShellyPlug):
            driver._get("/rpc/Switch.Set", {"id": driver.switch_id, "on": "true"})
        else:
            raise ActionFailed("no plug configured")
    except ActionFailed:
        raise
    except Exception as e:
        raise ActionFailed(why(e)) from e
    st = driver.status()
    if st is not True:
        raise ActionFailed("the plug did not confirm ON")
    return Ack(off=False, latency_ms=(time.perf_counter() - t0) * 1000, detail="operator ON acknowledged")


def plug_test(driver: PlugDriver) -> dict:
    """Operator setup check and video pre-roll: OFF, then ON, with both latencies measured."""
    off = driver.turn_off()
    on = switch_on(driver)
    return {"off_ms": round(off.latency_ms, 1), "on_ms": round(on.latency_ms, 1)}
