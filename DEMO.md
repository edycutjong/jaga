# DEMO — what jaga has measured, and how to reproduce it

This file is written by `scripts/bench.py` from `bench/out/summary.json`. Do not edit it by hand.
A section that has not run says so; nothing here is estimated.

## 1. Speed — Cosmos-Embed1-224p on this laptop

| device | precision | timed calls | p50 s/clip | p95 s/clip | torch | threads | 1-min load average before → after |
|---|---|---|---|---|---|---|---|
| MPS | fp32 | 200 | 1.0904 | 1.3371 | 2.14.1 | 8 | 5.26 → 6.23 |
| CPU | fp32 | 200 | 3.5211 | 5.1541 | 2.14.1 | 8 | 8.42 → 15.74 |

200 timed calls on one window (8 frames → `preprocess` + `get_video_embeddings`), after 5 warm-up calls; input `javelin_throw.webm` (the model card's own sample video). Other processes on the machine were not stopped; the load column shows how busy it was. *ran 2026-10-15T12:09:15+0700 at commit `6cfef48` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average when the result was written: 15.74*

**The CPU row ran while the 1-min load average exceeded the machine's 10 cores** — other processes were competing for it, so read it as a loaded machine, not an idle one. At the same commit the live loop's own embed time was p50 0.89 s over 328 windows (below). Day-0 stage COSMOS measured MPS p50 0.8617 s over 20 calls (`evidence/day0/cosmos-mps.json`).

Reproduce: `python scripts/bench.py speed --n 200`

### Live window period N (every dwell clamp and the watchdog use it)

p50 **1.03 s**, p95 1.12 s over 328 windows of session `p1-period-2` (source: live; a recorded clip looped in real time through the live loop (no camera in this run); other processes were running on the machine). *ran 2026-10-15T11:51:34+0700 at commit `6cfef48` on Apple M1 Max (macOS-26.5.2-arm64-arm-64bit); 1-min load average when the result was written: 3.66*

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
