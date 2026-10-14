#!/usr/bin/env python3
"""Day-0 stage COSMOS: the Cosmos-Embed1-224p model card's javelin smoke test, run on this Mac in fp32.

The card's "Example inference" sample hard-codes `.to("cuda", dtype=torch.bfloat16)` and reads the video with
decord. The card says it was only tested with BF16 on Ampere/Hopper GPUs under Linux. This script runs the same
six captions and the same 8-frame `np.linspace` sampling on the CPU and on Apple MPS in fp32, reads the video
with PyAV instead of decord, and measures seconds per clip.

PASS = the top-1 caption is one of the three javelin captions, not the motorcycle, car or knight captions.

Usage:
    python scripts/day0_cosmos.py --device cpu mps --out evidence/day0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

MODEL_ID = "nvidia/Cosmos-Embed1-224p"
MODEL_REVISION = "787e0b996f5260a71ad474a283c90539a2e12986"
VIDEO_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Branko_Paukovic%2C_javelin_throw.webm"
VIDEO_PATH = Path.home() / ".cache" / "jaga" / "javelin_throw.webm"

# Verbatim from the model card's sample, in its order.
CAPTIONS = [
    "a person riding a motorcycle in the night",
    "a car overtaking a white truck",
    "a video of a knight fighting with a sword",
    "a man wearing red spandex throwing a javelin",
    "a young man javelin throwing during the evening",  # distractor (card comment)
    "a man throwing a javelin with both hands",  # distractor (card comment)
]
JAVELIN = {3, 4, 5}


def fetch_video() -> Path:
    if not VIDEO_PATH.exists():
        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "jaga-day0/0.1 (hackathon smoke test)"})
        with urllib.request.urlopen(req, timeout=60) as r, open(VIDEO_PATH, "wb") as f:
            f.write(r.read())
    return VIDEO_PATH


def read_frames(path: Path, n: int = 8) -> np.ndarray:
    """Decode every frame with PyAV, then take n evenly spaced ones, like the card's np.linspace sampling."""
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    ids = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
    return np.stack([frames[i] for i in ids]), len(frames), ids


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run(device: str, runs: int, warmup: int) -> dict:
    if device == "mps" and not torch.backends.mps.is_available():
        return {"device": device, "status": "unavailable"}
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    model = model.to(device, dtype=torch.float32).eval()
    preprocess = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    load_s = time.perf_counter() - t0

    video = fetch_video()
    frames, n_decoded, ids = read_frames(video)
    batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))  # BTCHW, as the card does

    with torch.no_grad():
        text_inputs = preprocess(text=CAPTIONS).to(device, dtype=torch.float32)
        text_out = model.get_text_embeddings(**text_inputs)
        video_inputs = preprocess(videos=batch).to(device, dtype=torch.float32)
        video_out = model.get_video_embeddings(**video_inputs)
        v = video_out.visual_proj
        t = text_out.text_proj
        logits = model.logit_scale.exp() * v @ t.T
        probs = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()
        cos = (v @ t.T)[0].float().cpu().numpy()
        top1 = int(probs.argmax())

        # Timing: preprocess + embed (what the live loop pays per window), and embed only.
        for _ in range(warmup):
            model.get_video_embeddings(**preprocess(videos=batch).to(device, dtype=torch.float32))
        sync(device)
        clip_s, embed_s = [], []
        for _ in range(runs):
            a = time.perf_counter()
            vi = preprocess(videos=batch).to(device, dtype=torch.float32)
            sync(device)
            b = time.perf_counter()
            model.get_video_embeddings(**vi)
            sync(device)
            c = time.perf_counter()
            clip_s.append(c - a)
            embed_s.append(c - b)
        a = time.perf_counter()
        for _ in range(runs):
            model.get_text_embeddings(**preprocess(text=CAPTIONS[:1]).to(device, dtype=torch.float32))
        sync(device)
        text_s = (time.perf_counter() - a) / runs

    return {
        "device": device,
        "dtype": "float32",
        "status": "PASS" if top1 in JAVELIN else "FAIL",
        "top1_caption": CAPTIONS[top1],
        "top1_index": top1,
        "probs": {c: round(float(p), 6) for c, p in zip(CAPTIONS, probs)},
        "cosine": {c: round(float(x), 6) for c, x in zip(CAPTIONS, cos)},
        "visual_proj_shape": list(v.shape),
        "visual_proj_l2_norm": round(float(v.norm(dim=-1)[0]), 6),
        "text_proj_shape": list(t.shape),
        "model_load_s": round(load_s, 2),
        "runs": runs,
        "warmup": warmup,
        "s_per_clip_preprocess_plus_embed": {
            "p50": round(statistics.median(clip_s), 4),
            "p95": round(pct(clip_s, 0.95), 4),
            "min": round(min(clip_s), 4),
            "max": round(max(clip_s), 4),
        },
        "s_embed_only": {"p50": round(statistics.median(embed_s), 4), "p95": round(pct(embed_s, 0.95), 4)},
        "s_text_embed_one_caption_mean": round(text_s, 4),
        "frames_decoded": n_decoded,
        "frame_ids": ids,
        "frame_shape": list(frames.shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="evidence/day0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = fetch_video()
    env = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "video_url": VIDEO_URL,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": os.popen("sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip() or platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "av": av.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    results = []
    for d in args.device:
        try:
            r = run(d, args.runs, args.warmup)
        except Exception as e:  # a failure is a result, recorded as such
            r = {"device": d, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        (out / f"cosmos-{d}.json").write_text(json.dumps({**env, **r}, indent=2) + "\n")
        s = r.get("s_per_clip_preprocess_plus_embed", {})
        print(f"[{d}] {r['status']}  top1={r.get('top1_caption')!r}  s/clip p50={s.get('p50')} p95={s.get('p95')}")
    passed = [r for r in results if r["status"] == "PASS"]
    summary = {
        **env,
        "stage": "COSMOS",
        "pass": bool(passed),
        "devices": {r["device"]: r["status"] for r in results},
        "best_device": min(passed, key=lambda r: r["s_per_clip_preprocess_plus_embed"]["p50"])["device"] if passed else None,
    }
    (out / "cosmos-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("pass", "devices", "best_device")}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
