"""Sentence → TripwireSpec, compiled by NVIDIA Nemotron 3 Super on Nebius Token Factory.

Why an LLM step at all: the Cosmos-Embed1 card warns that because its training captions are "predominantly
composed of short action-focused English captions, different text prompts may not be properly aligned to
video data". People type long, negated or Indonesian sentences. Super rewrites one sentence into that caption
register and adds hard negatives that turn a negation ("nobody") into a visual contrast.

Super is a reasoning model. With a small `max_tokens` it can spend everything on reasoning and return no
content, so the call leaves generous room, reads `content` only, strips any think block, and treats empty
content as a compile error. It never invents a spec.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import config
from .schema import TRIPWIRE_JSON_SCHEMA, TripwireSpec

SYSTEM_TEXT = """You compile one household sentence into a tripwire for NVIDIA Cosmos-Embed1, a video-text \
embedding model. It was trained on short, action-focused English video captions; other phrasings may not align with video.
Return one JSON object only, matching the schema.

Rules:
- visual: true only if one fixed home camera could see the situation. If it cannot (sounds, smells, heat, feelings, \
or details too small for a 224x224 frame such as an LED or a knob position), set visual to false, explain why in \
reason, and return empty positives and hard_negatives.
- positives: 3 to 5 short English captions, at most 12 words each, written like video captions, describing what one \
fixed camera SEES while the sentence is true. Every positive names the main object or place of the sentence. For \
"nobody is at X", describe X with no person at or near it (for example "an empty ironing board with an iron on it").
- hard_negatives: 3 to 6 short English captions, at most 12 words each, of the SAME place and object while the \
sentence is FALSE. Every negative also names the main object or place; change only the deciding detail (for example \
"a person ironing a shirt at the ironing board" instead of "an empty ironing board").
- Never describe what a camera cannot see (heat, sound, smell, intentions) in any caption.
- name: an upper-case label of at most 24 characters, words separated by spaces.
- action: "plug_off" only when the sentence is about a heating or cooking appliance left on (an iron, a stove, a \
heater, a kettle), or the place where one is used (an ironing board means an iron), so that cutting its power is the \
safe response; "notify" for everything else, including any appliance that must stay powered, such as a fridge or a \
freezer. Nothing may ever switch power on.
- dwell_s: seconds the situation must hold before acting: 120 when the action is plug_off, 30 for pets, 60 for \
doors, unless the sentence says otherwise; at most 600.
- cooldown_s: seconds before the tripwire may act again; 600 unless the sentence says otherwise.
- language_in: the ISO 639-1 code of the sentence's language. Always write the captions in English.
- reason: one short sentence; empty string when visual is true."""

# Code-side action policy (the model proposes, the code decides): appliances that must keep their power are never
# cut, whatever the compile says. A trip only ever cuts power or notifies; here cutting would be the harm.
NEVER_CUT = re.compile(r"\b(fridge|refrigerator|freezer|kulkas|lemari es|aquarium|akuarium|oxygen|medical|router|modem)\b",
                       re.I)


# A trip may cut power only for an appliance that is dangerous to leave on: an allow-list, with NEVER_CUT as a veto.
HEATS = re.compile(r"\b(iron(?:ing)?|stove|hob|cooktop|oven|heater|kettle|toaster|hot ?plate|grill|fryer|rice cooker"
                   r"|straightener|curling iron|hair ?dryer|electric blanket|soldering|setrika|kompor|pemanas|catok"
                   r"|magic ?com|penanak nasi)\b", re.I)


def may_cut(sentence: str, name: str = "") -> bool:
    text = f"{sentence} {name}"
    return bool(HEATS.search(text)) and not NEVER_CUT.search(text)


def apply_action_policy(sentence: str, spec: dict) -> tuple[dict, str | None]:
    if spec.get("action") == "plug_off" and not may_cut(sentence, spec.get("name", "")):
        if NEVER_CUT.search(sentence + " " + spec.get("name", "")):
            why = "plug_off coerced to notify: this appliance must keep its power"
        else:
            why = "plug_off coerced to notify: cutting power is only for heating or cooking appliances"
        return {**spec, "action": "notify"}, why
    return spec, None

# Order tried by compile_sentence(). Day-0 stage T measured that Token Factory rejects the docs' example shape
# ("tf": schema directly under json_schema) with HTTP 422 "json_schema.name: Field required"
# (evidence/day0/tf-stage-t.json), so the product tries the OpenAI shape first. "tf" stays testable.
ALL_SHAPES = ("tf", "openai", "json_object")
SHAPES = ("openai", "json_object")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class CompileError(Exception):
    def __init__(self, kind: str, detail: str, attempts: list | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.attempts = attempts or []


@dataclass
class CompileResult:
    sentence: str
    spec: dict
    tf_request_id: str | None
    tf_response_id: str | None
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    created: str
    shape: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    policy_note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_tf_key() -> str | None:
    """TOKENFACTORY_API_KEY from the environment, else from an env file outside the repo."""
    if os.environ.get("TOKENFACTORY_API_KEY"):
        return os.environ["TOKENFACTORY_API_KEY"]
    candidates = [os.environ.get("JAGA_TF_ENV_FILE"), str(config.config_dir() / "tokenfactory.env")]
    for c in candidates:
        if c and Path(c).expanduser().exists():
            for line in Path(c).expanduser().read_text().splitlines():
                line = line.strip()
                if line.startswith("TOKENFACTORY_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def make_client(timeout_s: float = 120.0):
    from openai import OpenAI

    key = load_tf_key()
    if not key:
        raise CompileError("no_key", "TOKENFACTORY_API_KEY is not set (env or ~/.config/jaga/tokenfactory.env)")
    return OpenAI(base_url=config.TF_BASE_URL, api_key=key, timeout=timeout_s, max_retries=1)


def strip_think(text: str | None) -> str:
    """Remove any reasoning block. If only a closing tag survived, keep what follows it."""
    if not text:
        return ""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise
        return json.loads(text[a : b + 1])


def _response_format(shape: str) -> dict:
    if shape == "tf":  # the Token Factory docs' example: the schema directly under json_schema
        return {"type": "json_schema", "json_schema": TRIPWIRE_JSON_SCHEMA}
    if shape == "openai":  # the OpenAI shape
        return {
            "type": "json_schema",
            "json_schema": {"name": "TripwireSpec", "schema": TRIPWIRE_JSON_SCHEMA, "strict": True},
        }
    if shape == "json_object":
        return {"type": "json_object"}
    raise ValueError(shape)


def _messages(sentence: str, feedback: str | None = None) -> list[dict]:
    user = f"Sentence: {sentence}\n\nJSON schema:\n{json.dumps(TRIPWIRE_JSON_SCHEMA)}"
    if feedback:
        user += f"\n\nYour previous answer was rejected: {feedback}\nReturn a corrected JSON object only."
    return [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": user}]


def _call(client, sentence: str, shape: str, temperature: float, feedback: str | None = None) -> tuple:
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=config.TF_MODEL,
        messages=_messages(sentence, feedback),
        response_format=_response_format(shape),
        temperature=temperature,
        top_p=config.TF_TOP_P,
        max_tokens=config.TF_MAX_TOKENS,
        reasoning_effort="low",
    )
    completion = raw.parse()
    latency = time.perf_counter() - t0
    request_id = raw.headers.get("x-request-id") or raw.headers.get("x-trace-id")
    return completion, request_id, latency


def compile_sentence(
    sentence: str,
    *,
    client=None,
    shapes: tuple[str, ...] = SHAPES,
    temperature: float = config.TF_TEMPERATURE,
) -> CompileResult:
    """Compile one sentence. Tries the OpenAI json_schema shape, then json_object with one retry."""
    sentence = " ".join(sentence.split())
    if not sentence:
        raise CompileError("empty_sentence", "type a sentence first")
    client = client or make_client()
    attempts: list[dict] = []
    for shape in shapes:
        tries = 2 if shape == "json_object" else 1
        feedback = None
        for _ in range(tries):
            try:
                completion, request_id, latency = _call(client, sentence, shape, temperature, feedback)
            except CompileError:
                raise
            except Exception as e:  # HTTP error / timeout / rejected request shape
                attempts.append({"shape": shape, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                break
            msg = completion.choices[0].message
            content = strip_think(msg.content)
            usage = completion.usage.model_dump() if completion.usage else {}
            base = {"shape": shape, "response_id": completion.id, "request_id": request_id,
                    "finish_reason": completion.choices[0].finish_reason, "latency_s": round(latency, 3)}
            if not content:
                attempts.append({**base, "ok": False, "error": "empty content (reasoning only)"})
                feedback = "the answer had no JSON content"
                continue
            try:
                spec = TripwireSpec.model_validate(extract_json(content))
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                attempts.append({**base, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                 "content_head": content[:200]})
                feedback = str(e)[:500]
                continue
            attempts.append({**base, "ok": True})
            final, policy = apply_action_policy(sentence, spec.model_dump())
            return CompileResult(
                sentence=sentence,
                spec=final,
                policy_note=policy,
                tf_request_id=request_id,
                tf_response_id=completion.id,
                model=completion.model or config.TF_MODEL,
                temperature=temperature,
                top_p=config.TF_TOP_P,
                max_tokens=config.TF_MAX_TOKENS,
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                shape=shape,
                latency_s=round(latency, 3),
                usage=usage,
                attempts=attempts,
            )
    raise CompileError("compile_failed", "no request shape produced a valid TripwireSpec", attempts)
