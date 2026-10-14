#!/usr/bin/env python3
"""Day-0 stage T: does Nemotron 3 Super on Token Factory return a valid TripwireSpec?

Five sentences (two Indonesian), each sent once per request shape:
  tf          — the Token Factory docs' example: the schema directly under `json_schema`
  openai      — the OpenAI shape `{name, schema, strict}`
  json_object — `{"type": "json_object"}` + pydantic validation + one retry
PASS = every sentence yields a valid TripwireSpec on the first shape in that order that works (5/5).
Indonesian is claimed only if both Indonesian sentences pass.
One extra non-visual sentence checks the `visual: false` refusal; it is not part of the 5/5.

Usage:  python scripts/day0_tf.py --out evidence/day0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaga import config  # noqa: E402
from jaga.compile import ALL_SHAPES as SHAPES, CompileError, compile_sentence, make_client  # noqa: E402

SENTENCES = [
    ("en", "nobody is at the ironing board"),
    ("en", "the cat has been on the kitchen counter"),
    ("en", "the door is left open and nobody is there"),
    ("id", "nggak ada orang di meja setrika"),
    ("id", "pintu dibiarkan terbuka dan tidak ada orang"),
]
NON_VISUAL = ("en", "tell me if something is wrong")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/day0")
    args = ap.parse_args()
    client = make_client()
    rows = []
    for lang, sentence in SENTENCES + [NON_VISUAL]:
        per_shape = {}
        for shape in SHAPES:
            try:
                r = compile_sentence(sentence, client=client, shapes=(shape,))
                per_shape[shape] = {"ok": True, **r.to_dict()}
            except CompileError as e:
                per_shape[shape] = {"ok": False, "kind": e.kind, "detail": e.detail, "attempts": e.attempts}
            ok = per_shape[shape]["ok"]
            spec = per_shape[shape].get("spec", {})
            print(f"{lang} {shape:11s} {'OK ' if ok else 'ERR'} {sentence!r} "
                  f"visual={spec.get('visual')} name={spec.get('name')!r}"
                  + ("" if ok else f" — {per_shape[shape]['kind']}: {per_shape[shape]['detail'][:120]}"))
        first_ok = next((s for s in SHAPES if per_shape[s]["ok"]), None)
        rows.append({"lang": lang, "sentence": sentence, "non_visual_check": (lang, sentence) == NON_VISUAL,
                     "first_working_shape": first_ok, "shapes": per_shape})

    scored = [r for r in rows if not r["non_visual_check"]]
    valid = [r for r in scored if r["first_working_shape"] and r["shapes"][r["first_working_shape"]]["spec"]["visual"]]
    indonesian = [r for r in scored if r["lang"] == "id"]
    id_pass = all(r in valid for r in indonesian)
    nv = rows[-1]
    nv_shape = nv["first_working_shape"]
    summary = {
        "stage": "T",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": config.TF_MODEL,
        "base_url": config.TF_BASE_URL,
        "temperature": config.TF_TEMPERATURE,
        "top_p": config.TF_TOP_P,
        "max_tokens": config.TF_MAX_TOKENS,
        "reasoning_effort": "low",
        "valid": f"{len(valid)}/{len(scored)}",
        "pass": len(valid) == len(scored),
        "indonesian": f"{sum(r in valid for r in indonesian)}/{len(indonesian)}",
        "indonesian_claimed": id_pass,
        "shape_ok_counts": {s: sum(r["shapes"][s]["ok"] for r in rows) for s in SHAPES},
        "non_visual_refused": bool(nv_shape) and nv["shapes"][nv_shape]["spec"]["visual"] is False,
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tf-stage-t.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("valid", "pass", "indonesian", "indonesian_claimed",
                                                "shape_ok_counts", "non_visual_refused")}))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
