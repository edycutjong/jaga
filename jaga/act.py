"""Act: cut a plug's power, or notify. Nothing in this module can switch power ON (invariant I1).

A driver exposes exactly two calls: `turn_off()` and `status()`. Switching ON is an operator action that
lives in `jaga/operator_on.py`, reachable only from `POST /plug/test` (localhost + Host/Origin allowlist +
session token) — never from a trip. `tests/test_invariants.py` checks both facts.

`turn_off()` returns only after the plug acknowledges: the UI shows OFF only after that (invariant I3).

Drivers, all LAN-only at runtime:
* TuyaPlug   — a Tuya "Smart Life" Wi-Fi plug through `tinytuya` with its local key (one-time wizard)
* ShellyPlug — Shelly Gen3 local HTTP RPC `/rpc/Switch.Set?id=0&on=false`, no account
* NullPlug   — nothing configured: every action fails visibly as ACTION FAILED, never as "done"
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from . import config


class ActionFailed(Exception):
    pass


def why(e: Exception) -> str:
    """A short reason a person can act on (the overlay is filmed; no stack traces or URLs)."""
    name = type(e).__name__
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return f"the plug answered HTTP {status}"
    if "Timeout" in name or isinstance(e, TimeoutError):
        return "the plug did not answer within 3 s"
    if "Connect" in name or isinstance(e, (ConnectionError, OSError)):
        return "the plug is not reachable on the home network"
    return f"{name}"


@dataclass
class Ack:
    off: bool
    latency_ms: float
    detail: str = ""


class PlugDriver:
    kind = "none"
    label = "Wi-Fi plug"  # no third-party brand on the filmed UI (rules:160)
    address = ""

    def turn_off(self) -> Ack:  # pragma: no cover - interface
        raise NotImplementedError

    def status(self) -> bool | None:  # True = ON, False = OFF, None = unknown
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind, "label": self.label, "address": self.address}


class NullPlug(PlugDriver):
    kind = "none"

    def __init__(self, reason: str = "no plug configured (~/.config/jaga/devices.json or shelly.json)"):
        self.reason = reason
        self.address = "not configured"

    def turn_off(self) -> Ack:
        raise ActionFailed(self.reason)

    def status(self) -> bool | None:
        return None


class TuyaPlug(PlugDriver):
    kind = "tuya"

    def __init__(self, dev_id: str, ip: str, local_key: str, version: float | str = 3.3, dps: str = "1"):
        self.dev_id, self.ip, self.local_key = dev_id, ip or "Auto", local_key
        self.version = float(version)
        self.dps = str(dps)
        self.address = self.ip

    def _device(self):
        import tinytuya

        d = tinytuya.OutletDevice(self.dev_id, self.ip, self.local_key, version=self.version)
        d.set_socketTimeout(3)
        d.set_socketRetryLimit(1)
        return d

    def status(self) -> bool | None:
        try:
            r = self._device().status()
        except Exception:
            return None
        dps = (r or {}).get("dps") or {}
        return None if self.dps not in dps else bool(dps[self.dps])

    def turn_off(self) -> Ack:
        t0 = time.perf_counter()
        try:
            r = self._device().turn_off(switch=int(self.dps))
        except Exception as e:  # socket errors, decryption errors
            raise ActionFailed(why(e)) from e
        if isinstance(r, dict) and r.get("Error"):
            raise ActionFailed(f"the plug refused: {r.get('Error')}"[:120])
        dps = (r or {}).get("dps") or {}
        if dps.get(self.dps) is False:
            return Ack(True, (time.perf_counter() - t0) * 1000, "acknowledged in the set response")
        st = self.status()  # some firmware answer the set call with no dps: read it back
        if st is False:
            return Ack(True, (time.perf_counter() - t0) * 1000, "acknowledged by a status read")
        raise ActionFailed("the plug did not confirm OFF")


class ShellyPlug(PlugDriver):
    kind = "shelly"

    def __init__(self, base_url: str, switch_id: int = 0, timeout_s: float = 3.0):
        self.base = base_url.rstrip("/")
        if not self.base.startswith("http"):
            self.base = "http://" + self.base
        self.switch_id = switch_id
        self.timeout_s = timeout_s
        self.address = self.base.split("://", 1)[1]

    def _get(self, path: str, params: dict) -> dict:
        import httpx

        r = httpx.get(f"{self.base}{path}", params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def status(self) -> bool | None:
        try:
            return bool(self._get("/rpc/Switch.GetStatus", {"id": self.switch_id})["output"])
        except Exception:
            return None

    def turn_off(self) -> Ack:
        t0 = time.perf_counter()
        try:
            self._get("/rpc/Switch.Set", {"id": self.switch_id, "on": "false"})
            out = self._get("/rpc/Switch.GetStatus", {"id": self.switch_id})["output"]
        except Exception as e:
            raise ActionFailed(why(e)) from e
        if out is False:
            return Ack(True, (time.perf_counter() - t0) * 1000, "acknowledged by a status read")
        raise ActionFailed("the plug did not confirm OFF")


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_driver(cfg_dir: Path | None = None) -> PlugDriver:
    """Pick the plug from config outside the repo. Order: JAGA_PLUG_URL, plug.json, devices.json, shelly.json."""
    cfg_dir = cfg_dir or config.config_dir()
    if os.environ.get("JAGA_PLUG_URL"):
        return ShellyPlug(os.environ["JAGA_PLUG_URL"])
    choice = _read_json(cfg_dir / "plug.json") or {}
    if choice.get("driver") == "shelly" or (not choice and (cfg_dir / "shelly.json").exists()
                                             and not (cfg_dir / "devices.json").exists()):
        sh = _read_json(cfg_dir / "shelly.json") or choice
        url = sh.get("url") or sh.get("ip")
        return ShellyPlug(url) if url else NullPlug("shelly.json has no ip/url")
    devices = _read_json(cfg_dir / "devices.json")
    if devices:
        if isinstance(devices, dict):
            devices = devices.get("devices", [devices])
        want = choice.get("name") or choice.get("id")
        dev = next((d for d in devices if want in (d.get("name"), d.get("id"))), None) if want else devices[0]
        if not dev:
            return NullPlug(f"plug {want!r} not in devices.json")
        ip = choice.get("ip") or dev.get("ip") or "Auto"
        return TuyaPlug(dev["id"], ip, dev.get("key") or dev.get("local_key", ""),
                        choice.get("version") or dev.get("version") or 3.3, choice.get("dps", "1"))
    return NullPlug()


class Notifier:
    """The overlay event row is the notification on screen. Optional extras: a macOS banner (off with
    --no-os-banner while filming) and one ntfy text line (off unless JAGA_NTFY_TOPIC is set)."""

    def __init__(self, os_banner: bool = True, ntfy_topic: str | None = None):
        self.os_banner = os_banner
        self.ntfy_topic = ntfy_topic if ntfy_topic is not None else os.environ.get("JAGA_NTFY_TOPIC")
        self.sent: list[str] = []

    def notify(self, text: str, title: str = "jaga") -> None:
        self.sent.append(text)
        if self.os_banner and os.uname().sysname == "Darwin":
            script = f"display notification {json.dumps(text)} with title {json.dumps(title)}"
            try:
                subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
        if self.ntfy_topic:
            try:
                import httpx

                httpx.post(f"https://ntfy.sh/{quote(self.ntfy_topic, safe='')}", content=text.encode(), timeout=5)
            except Exception:
                pass
