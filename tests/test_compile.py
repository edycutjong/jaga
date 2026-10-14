"""The compiler (jaga/compile.py) and the frozen tripwires. No Token Factory call: a fake client stands in."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import SPEC
from pydantic import ValidationError

from jaga.compile import SYSTEM_TEXT, CompileError, compile_sentence, extract_json, strip_think
from jaga.schema import TripwireSpec

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tripwires"


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)  # each: Exception, or content string
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        msg = SimpleNamespace(content=r)
        comp = SimpleNamespace(id="chatcmpl-test", model=kw["model"], usage=None,
                               choices=[SimpleNamespace(message=msg, finish_reason="stop")])
        return SimpleNamespace(parse=lambda: comp, headers={"x-request-id": f"req-{len(self.calls)}"})


def fake_client(replies):
    comp = FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=comp.create)))), comp


def test_strip_think_removes_reasoning():
    assert strip_think('<think>hmm</think>\n{"a": 1}') == '{"a": 1}'
    assert strip_think('reasoning…</think>{"a": 1}') == '{"a": 1}'
    assert strip_think(None) == ""


def test_extract_json_from_fenced_or_wrapped_text():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure: {"a": 2} done') == {"a": 2}


def test_compile_uses_openai_shape_first_and_records_request_id():
    client, comp = fake_client([json.dumps(SPEC)])
    r = compile_sentence("nobody is at the ironing board", client=client)
    assert r.shape == "openai" and r.tf_request_id == "req-1" and r.spec["name"] == "BOARD UNATTENDED"
    rf = comp.calls[0]["response_format"]
    assert rf["json_schema"]["name"] == "TripwireSpec" and comp.calls[0]["max_tokens"] >= 2048


def test_reasoning_only_reply_is_an_error_never_an_invented_spec():
    client, _ = fake_client(["<think>only thinking</think>", "", ""])
    with pytest.raises(CompileError) as e:
        compile_sentence("nobody is at the ironing board", client=client)
    assert e.value.kind == "compile_failed"
    assert any("empty content" in a.get("error", "") for a in e.value.attempts)


def test_falls_back_to_json_object_with_one_retry():
    bad = dict(SPEC, positives=["x"])  # too few positives → rejected → feedback retry
    client, comp = fake_client([RuntimeError("422 shape rejected"), json.dumps(bad), json.dumps(SPEC)])
    r = compile_sentence("nobody is at the ironing board", client=client)
    assert r.shape == "json_object" and len(comp.calls) == 3
    assert "rejected" in comp.calls[2]["messages"][-1]["content"]


def test_schema_enforces_the_contract():
    TripwireSpec.model_validate(SPEC)
    with pytest.raises(ValidationError):
        TripwireSpec.model_validate({**SPEC, "positives": ["one two three four five six seven eight nine ten eleven "
                                                           "twelve thirteen", "b", "c"]})
    with pytest.raises(ValidationError):
        TripwireSpec.model_validate({**SPEC, "action": "plug_on"})
    with pytest.raises(ValidationError):
        TripwireSpec.model_validate({**SPEC, "visual": False, "reason": ""})
    with pytest.raises(ValidationError):
        TripwireSpec.model_validate({**SPEC, "extra": 1})
    with pytest.raises(ValidationError):
        TripwireSpec.model_validate({**SPEC, "dwell_s": 601})
    ok = TripwireSpec.model_validate({**SPEC, "visual": False, "reason": "a smell", "positives": [],
                                      "hard_negatives": []})
    assert ok.visual is False


def test_fridge_sentence_never_cuts_power():
    client, _ = fake_client([json.dumps({**SPEC, "name": "FRIDGE OPEN"})])  # model proposes plug_off
    r = compile_sentence("the fridge door is left open and nobody is there", client=client)
    assert r.spec["action"] == "notify" and r.policy_note


@pytest.mark.parametrize("slot", ["01", "02", "03"])
def test_frozen_tripwires_are_intact(slot):
    """Each frozen tripwire: a valid spec verbatim from one compile, its request ID, the prompt hash it was
    compiled with, caption vectors for exactly its captions, and an explicit operating block."""
    import numpy as np

    d = json.loads((FIX / f"{slot}.json").read_text())
    TripwireSpec.model_validate(d["spec"])
    assert len(d["tf_request_id"]) >= 16 and d["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert d["compiler_prompt_sha256"] == hashlib.sha256(SYSTEM_TEXT.encode()).hexdigest(), \
        "the compiler prompt changed after the freeze: re-freezing needs a recorded decision"
    with np.load(FIX / f"{slot}.vec.npz") as z:
        assert z["pos"].shape == (len(d["spec"]["positives"]), 256)
        assert z["neg"].shape == (len(d["spec"]["hard_negatives"]), 256)
        assert np.allclose(np.linalg.norm(z["pos"], axis=1), 1, atol=1e-4)
    assert d["operating"]["action"] in ("plug_off", "notify") and d["operating"]["super_proposed"]
    assert d["superseded"] and d["superseded"][0]["tf_request_id"] != d["tf_request_id"]
