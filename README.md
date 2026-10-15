# jaga

**Type what to watch for. NVIDIA Cosmos-Embed1 watches a webcam on your laptop, and a cheap smart plug cuts power.
No model training, no cloud video.**

*jaga* is Indonesian for "to guard, to keep watch". Built on NVIDIA Cosmos.

> **jaga is an extra guard, not a fire-safety device.** It can miss an event; a trip only ever cuts power or sends a
> notification, and switching power back on is always a person's job.

*A scenario (a composite, not a real person):* Bu Sari leaves the iron on when she runs to the warung (the corner
shop), and her CCTV stays silent — an empty room with a hot iron in it has no motion.

## The one flow

1. You type a sentence: *"nobody is at the ironing board"*. The iron is plugged into a Wi-Fi smart plug.
2. **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) on **Nebius Token Factory** compiles it, under a
   JSON schema, into a *TripwireSpec*: 3–5 short action-style English captions of what the camera sees when the
   sentence is true, 3–6 hard negatives of the same place when it is false, a dwell and an action. You see the
   captions and negatives before arming, and choose what happens (cut the plug, or notify) and the dwell.
3. **NVIDIA Cosmos-Embed1-224p** runs on the laptop. Every window of the last 8 frames, sampled at 1 frame per second,
   becomes one 256-d vector in the same space as the captions.
4. Deterministic code decides: margin = best positive cosine − best negative cosine; a trip needs most windows over the
   dwell above the threshold, including the newest. A 2-minute "normal state" capture at arming (someone ironing as
   usual) sets the reference.
5. The plug switches **off** over the home network, and the overlay shows OFF only after the plug confirms it.
   A nightly **Nebius Serverless Job** recalibrates each tripwire's threshold from labelled margins.

The model chooses where to look; the code decides when to act.

## Status

The software is complete; the measurements are being made. Everything measured so far is in [`DEMO.md`](DEMO.md),
written by `scripts/bench.py` — no number appears anywhere before that script writes it. The labelled clips and the
pre-registered soak sessions (protocol committed before any data: [`docs/PROTOCOL.md`](docs/PROTOCOL.md)) are
recorded in the builder's own flat; their results are added to `DEMO.md` as they are measured.

## Quick start

```bash
git clone https://github.com/edycutjong/jaga && cd jaga
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                                   # the safety invariants and the decision rule, no model or key needed
```

**The model (gated):** accept the licence at <https://huggingface.co/nvidia/Cosmos-Embed1-224p> and log in with a
Hugging Face read token (`huggingface-cli login`). The first run downloads the weights; after that jaga loads them from
the local cache with no network call.

**The compiler:** a Nebius Token Factory key in `~/.config/jaga/tokenfactory.env` (`TOKENFACTORY_API_KEY=…`).

**The plug (optional for trying it):** a Tuya "Smart Life" Wi-Fi plug, paired in its app; run
`python -m tinytuya wizard` once and save its output as `~/.config/jaga/devices.json` (never inside the repo).
A Shelly Plug S Gen3 works too (`~/.config/jaga/shelly.json` with `{"ip": "…"}`). With no plug configured, a trip
shows **ACTION FAILED** — never "done".

```bash
jaga serve --load 01 --arm 01                 # watch the webcam; open http://127.0.0.1:8765/
jaga serve --source recording.mp4 --load 01   # or play a recording through the same loop
jaga compile "the door is left open and nobody is there"
```

On macOS, allow camera access for your terminal (System Settings → Privacy & Security → Camera).

## Testing it without the hardware

```bash
jaga score --replay data/clips/iron/pos_03.mp4 --tripwire fixtures/tripwires/01.json
```

`--replay` reads the clip's committed window vector (`fixtures/vecs/`), so it needs no model download, no key and no
camera; it prints the clip's margin and whether it is above the frozen threshold, and checks it against
`fixtures/decisions_frozen.csv`. The vectors and published decisions are committed with the labelled clips. Without
`--replay`, the same command embeds the clip with Cosmos-Embed1 on your machine.

## How the NVIDIA models are used

| Model | Where | What it does in the flow |
|---|---|---|
| `nvidia/Cosmos-Embed1-224p` | on the laptop, fp32 on Apple MPS or CPU (`jaga/embed.py`) | `get_video_embeddings` turns each 8-frame window into a 256-d vector; `get_text_embeddings` puts the compiled captions in the same space, so a sentence becomes a classifier without training |
| `nvidia/nemotron-3-super-120b-a12b` | Nebius Token Factory (`jaga/compile.py`) | rewrites the sentence into the short English caption register the Cosmos-Embed1 card says aligns with video, and invents the hard negatives that turn "nobody" into a visual contrast; structured JSON output, so the code can enforce it |

The Cosmos-Embed1 card documents Linux, CUDA and BF16 only; jaga runs it on macOS in fp32, and the speeds in
`DEMO.md` are ours, measured on this laptop.

## Where Nebius is used

* **Token Factory** compiles every sentence (one request per tripwire; the request ID is stored with the tripwire,
  shown in the compile drawer and listed at `/receipts`). The three measured tripwires were each compiled once and
  frozen before any clip was recorded (`fixtures/tripwires/`).
* **Serverless Jobs** (`cpu-d3`, `4vcpu-16gb`, stock `python:3.11`) run the nightly calibration
  (`jobs/calibrate.py`, standard library only): a per-tripwire ROC over labelled margins → the threshold at
  recall 0.9. The work is light; running it off the laptop keeps the watcher from pausing and leaves a receipt for
  every run ([`docs/RUNS.md`](docs/RUNS.md)).
* **Object Storage** carries the Job's input and result through presigned URLs, so the Job never holds a credential.

## What leaves the laptop

| What | To | Contains video frames? |
|---|---|---|
| the typed sentence | Token Factory | no |
| labelled margins (numbers and labels; no vectors) | Object Storage → the calibration Job | no |
| thresholds | back from Object Storage | no |
| one text line, if you turn on the optional push | ntfy | no |

**Video frames never leave this laptop.** The overlay is served on `127.0.0.1` only. The Tuya plug's own firmware
may talk to Tuya's cloud by itself; that is outside jaga's control, and the Shelly path can be fully local.

## Safety (it switches mains power)

* **I1** — a trip can only cut power or notify. The only code that can switch a plug on is `jaga/operator_on.py`,
  reachable only from the operator's `POST /plug/test` (localhost + same-origin + a per-run token). Tests check
  this statically and by running repeated trips against a fake plug.
* **I2** — one bad window cannot cut power: a trip needs most windows over the dwell above the threshold, including
  the newest, and the dwell is never shorter than twice the measured window period.
* **I3** — the overlay shows OFF only after the plug acknowledges; a failed call is shown and pushed as
  **ACTION FAILED** and retried (off only).
* Guards: a sudden light change suspends trips; too dark unarms ("too dark to see"); a moved camera unarms; a
  watchdog shows **WATCHER STOPPED** if windows stop. A fridge or freezer can never be set to cut power.

## Repository

| Path | What |
|---|---|
| `jaga/` | the package: embedder, capture, guards, decision rule, actuator, watcher, server, overlay, CLI |
| `jobs/calibrate.py` | the Serverless Job (stdlib) |
| `scripts/` | `bench.py` (every number), `seed.py` (fixtures), `protocol.py`, `verify_offline.py`, `check_submission_readiness.py`, day-0 stages |
| `fixtures/tripwires/` | the frozen tripwires with their Token Factory request IDs and caption vectors |
| `evidence/` | day-0 measurements and Job receipts |
| `docs/` | `PROTOCOL.md` (pre-registered), `RUNS.md` (Job receipts), `DX-REPORT.md` (friction log) |

## Limits

* Three events are measured, one room, one person — not "any event".
* 224p is small: jaga watches scene states (a person present or not, a door open), not an iron's power light.
* Captions are English; other languages are only as good as Super's rewrite.
* If the laptop sleeps or the camera is unplugged, nothing trips (the watchdog alerts while the process runs).

## License

Apache-2.0 ([`LICENSE`](LICENSE)); see [`NOTICE`](NOTICE). Built on NVIDIA Cosmos.
