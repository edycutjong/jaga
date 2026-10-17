# DEMO — what jaga has measured, and how to reproduce it

This file is written by `scripts/bench.py` from `bench/out/summary.json`. Do not edit it by hand.
A section that has not run says so; nothing here is estimated.

## 1. Speed — Cosmos-Embed1-224p on this laptop

| device | precision | timed calls | p50 s/clip | p95 s/clip | torch | threads | 1-min load average before → after |
|---|---|---|---|---|---|---|---|
| MPS | fp32 | 200 | 1.2378 | 2.9058 | 2.14.1 | 8 | 4.55 → 9.82 |
| CPU | fp32 | 200 | 3.0935 | 3.717 | 2.14.1 | 8 | 9.42 → 5.95 |

200 timed calls on one window (8 frames → `preprocess` + `get_video_embeddings`), after 5 warm-up calls; input `javelin_throw.webm` (the model card's own sample video). Other processes on the machine were not stopped; the load column shows how busy it was. *ran 2026-10-17T08:10:00+0700 at commit `7735988` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average when the result was written: 5.95*

Reproduce: `python scripts/bench.py speed --n 200`

### Live window period N (every dwell clamp and the watchdog use it)

p50 **1.27 s**, p95 1.79 s over 240 windows of session `p1-period-4` (source: live; a recorded clip looped in real time through the live loop (no camera in this run); other processes were running on the machine). *ran 2026-10-15T14:45:07+0700 at commit `76f2ca6` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average when the result was written: 6.85*

## 2. Labelled clips (one 8-s clip = one window)

Not yet run: no labelled clips have been recorded yet (`fixtures/clips.csv`).

## 3. Fresh sentence, no labelled clips (the zero-training path)

Not yet run: needs the 2-minute normal-state captures (`normal/<tripwire>`).

## 4. Soak B — someone standing still at the board (pre-registered: docs/PROTOCOL.md)

Not yet run: Soak B1/B2 have not been recorded. Protocol and schedules: `docs/PROTOCOL.md`.

## 4b. Soak A — one night, tripwires 02 and 03 (supporting)

Not yet run: the Soak A night has not been recorded.

## 5. Self-ablation

Not yet run: needs the labelled clips.

## 6. Outside people

Not yet run: no outside people's clips received yet.

## Reproduce

```bash
pip install -e .
python scripts/bench.py speed      # downloads Cosmos-Embed1 once (public on Hugging Face on 2026-10-15)
python scripts/bench.py period --session <name>   # from any logged session
```

Once the labelled clips are recorded and their vectors committed, these run without the model:

```bash
python scripts/bench.py clips      # AUC / precision from the committed vectors
jaga score --replay iron/pos_03 --tripwire fixtures/tripwires/01.json   # a clip id from fixtures/clips.csv
```
