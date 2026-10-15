# DEMO — what jaga has measured, and how to reproduce it

This file is written by `scripts/bench.py` from `bench/out/summary.json`. Do not edit it by hand.
A section that has not run says so; nothing here is estimated.

## 1. Speed — Cosmos-Embed1-224p on this laptop

| device | precision | windows | p50 s/clip | p95 s/clip | torch | threads | machine load (1-min avg) |
|---|---|---|---|---|---|---|---|
| MPS | fp32 | 200 | 0.8616 | 0.9182 | 2.14.1 | 8 | not recorded |
| CPU | fp32 | 200 | 3.5283 | 7.3793 | 2.14.1 | 8 | not recorded |

One window = 8 frames → `preprocess` + `get_video_embeddings`, after 5 warm-up calls; input `javelin_throw.webm` (the model card's own sample video). *ran 2026-10-15T07:22:02+0700 at commit `4697826` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average 37.1*

Reproduce: `python scripts/bench.py speed --n 200`

### Live window period N (every dwell clamp and the watchdog use it)

p50 **1.11 s**, p95 2.44 s over 114 windows of session `p1-liveloop-file` (source: live; a recorded file played in real time through the live loop (the autopilot's process has no camera permission); machine shared with two other build sessions). *ran 2026-10-15T07:29:47+0700 at commit `e3e02bf` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average 12.16*

## 2. Labelled clips (one 8-s clip = one window)

Not yet run: no labelled clips have been recorded yet (`fixtures/clips.csv`).

## 3. Fresh sentence, no labelled clips (the zero-training path)

Not yet run: needs the 2-minute normal-state captures (`normal/<tripwire>`).

## 4. Soak B — someone standing still at the board (pre-registered: docs/PROTOCOL.md)

Not yet run: Soak B1/B2 have not been recorded. Protocol and schedules: `docs/PROTOCOL.md`.

## 5. Self-ablation

Not yet run: needs the labelled clips.

## 6. Outside people

Not yet run: no outside people's clips received yet.

## Reproduce

```bash
pip install -e .
jaga score --replay data/clips/iron/pos_03.mp4 --tripwire fixtures/tripwires/01.json   # no model download
python scripts/bench.py clips      # AUC / precision from the committed vectors
python scripts/bench.py speed      # needs the gated Cosmos-Embed1 weights (accept the licence on HF)
```
