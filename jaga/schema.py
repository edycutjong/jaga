"""TripwireSpec: what Nemotron 3 Super returns for one sentence, and what the code enforces.

The model proposes; the code decides. The JSON schema sent to Token Factory is deliberately loose where
guided decoding engines differ (array lengths when `visual` is false); pydantic enforces the rest.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CAPTION_WORDS = 12

TRIPWIRE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "visual": {"type": "boolean"},
        "reason": {"type": "string"},
        "name": {"type": "string", "maxLength": 24},
        "positives": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "hard_negatives": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "dwell_s": {"type": "integer", "minimum": 1, "maximum": 600},
        "action": {"type": "string", "enum": ["plug_off", "notify"]},
        "cooldown_s": {"type": "integer", "minimum": 0, "maximum": 86400},
        "language_in": {"type": "string"},
    },
    "required": [
        "visual",
        "reason",
        "name",
        "positives",
        "hard_negatives",
        "dwell_s",
        "action",
        "cooldown_s",
        "language_in",
    ],
    "additionalProperties": False,
}


class TripwireSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual: bool
    reason: str = ""
    name: str = Field(min_length=1, max_length=24)
    positives: list[str] = Field(default_factory=list, max_length=5)
    hard_negatives: list[str] = Field(default_factory=list, max_length=6)
    dwell_s: int = Field(ge=1, le=600)
    action: Literal["plug_off", "notify"]
    cooldown_s: int = Field(ge=0, le=86400)
    language_in: str = Field(min_length=1, max_length=16)

    @field_validator("positives", "hard_negatives")
    @classmethod
    def _short_captions(cls, v: list[str]) -> list[str]:
        out = []
        for c in v:
            c = " ".join(c.split())
            if not c:
                raise ValueError("empty caption")
            if len(c.split()) > MAX_CAPTION_WORDS:
                raise ValueError(f"caption longer than {MAX_CAPTION_WORDS} words: {c!r}")
            out.append(c)
        return out

    @field_validator("name")
    @classmethod
    def _upper_name(cls, v: str) -> str:
        return " ".join(v.replace("_", " ").split()).upper()

    @model_validator(mode="after")
    def _visual_needs_captions(self) -> "TripwireSpec":
        if self.visual:
            if not 3 <= len(self.positives) <= 5:
                raise ValueError("a visual tripwire needs 3–5 positive captions")
            if not 3 <= len(self.hard_negatives) <= 6:
                raise ValueError("a visual tripwire needs 3–6 hard negatives")
        elif not self.reason.strip():
            raise ValueError("visual=false needs a reason")
        return self


def clamp_dwell(dwell_s: float, window_period_p95_s: float | None) -> float:
    """Invariant I2: a trip needs a dwell of at least 2 × the measured p95 window period.

    The window period is the p95 of diff(embed.ts) in the live loop — embed plus baselines, guards
    and SSE — not the embed time alone. Before any window is measured the spec's dwell stands.
    """
    if window_period_p95_s is None or window_period_p95_s <= 0:
        return float(dwell_s)
    return float(max(dwell_s, 2.0 * window_period_p95_s))
