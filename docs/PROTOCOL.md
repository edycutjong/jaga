# Soak B — pre-registered protocol

This file is committed **before any Soak B data exists**. The definitions, the decision rule, the baselines, the
statistics, the targets and the sentences that will report the result are fixed here. The few values that can
only come from unscored data (θ v1, P7, P8, R, the zone box and the HOG detection rate) are fixed by the
**rules** below and written into [Frozen values](#frozen-values-filled-before-b1) in a dated commit before B1 starts.
B1 does not start until that section is filled. Every run counts; an aborted run and its reason are reported.

*Amended 2026-10-15, still before any clip or soak data existed (see this file's git history): the frame-difference
orientation per tripwire, the fresh-mode hold-out, the Soak A scoring rules, the secondary 30-s row and the P7
fallback were written down explicitly, and the Soak B outputs per exit type were wired into `scripts/bench.py`.*

## 1. The question

At an ironing board, a person is often present but still (reading, scrolling a phone, standing). A stillness timer
cannot tell that apart from an empty board. **Of the still episodes in Soak B1 + B2, how many does jaga cut, and how
many does a stillness timer cut — judged by one decision rule — and how many real exits does each catch, how fast?**

Both halves matter. A system that never cuts is useless; a system that cuts on every still moment is a nuisance.

## 2. Set-up

| Item | Value |
|---|---|
| People / rooms | one person (the builder), one room; disclosed on every surface |
| Camera position | named `board` (recorded on every window as `camera_pos`) |
| Tripwire | **01 · "nobody is at the ironing board"** — `fixtures/tripwires/01.json`, frozen from one Nemotron 3 Super compile (Token Factory request `9727b666068a1e42a836b303ab8fd65c`) made before any clip existed; the earlier compile it replaced is listed under `superseded` in the fixture |
| Action | `plug_off` on the real Wi-Fi plug; a desk lamp on the same plug is the visible load; the iron is an **unplugged prop** |
| Threshold | θ_m **v1**, frozen before B1 (rule in §6); the same θ for B1 and B2; never refitted on a scored session |
| Dwell | **120 s primary** (the default); the 30-s demo dwell is a secondary row recomputed offline from the same logs for every system (`bench.py soak --dwell 30 --secondary`; jaga's live trips count only at 120 s) |
| k-of-n | k = max(2, ⌈P7 · n⌉), n = max(2, dwell ÷ window period) — never fewer than two windows (I2) |
| Cool-down | 600 s; a re-arm (the plug switched back on) ends it |
| Suspend | a jump in mean luma > P1 (25 %) suspends trips for P2 (60 s) |
| Sessions | **B1** and **B2**, each ≥ 2 h, on different days; a session ends at its schedule's last step (jaga stops itself a minute later, and nothing after the `plan:session_end` label is scored) |

Command (the overlay shows each step; the laptop speaks it 30 s ahead and when it is due):

```bash
jaga soak B --session B1 --schedule docs/protocol/B1.csv --load 01 --camera-pos board --zone <x0,y0,x1,y1>
```

## 3. The schedule (random, fixed now)

`scripts/protocol.py` draws each session's order and durations from a fixed seed (`jaga-soak-B1-v1`,
`jaga-soak-B2-v1`) and writes [`protocol/B1.csv`](protocol/B1.csv) and [`protocol/B2.csv`](protocol/B2.csv).
Re-running it reproduces both files exactly (a test checks this).

| | B1 | B2 |
|---|---|---|
| length | 154 min | 143 min |
| still episodes (3–5 min; stand or sit still at the board, hands still) | 10 | 10 |
| exits: leave the room (away 4–6 min) | 4 | 4 |
| exits: step ≥ 2 m away, stay in view (4–6 min) | 2 | 2 |
| activities between them (ironing the prop, folding, phone, reading; 3–6 min) | yes | yes |

Step 0 of each session is the 2-minute normal-state capture ("iron as usual").

**Logging.** The runner logs every planned step as a `plan:<kind>` label at the moment it is due. The person also
presses a key when a step actually starts or ends (1 still starts · 2 still ends · 3 leaves the room · 4 steps away,
in view · 5 back at the board). Scoring uses the key press nearest each planned time (within ±60 s) when there is
one, else the planned time. A step the person could not follow is logged with key 0 and a note, and stays in the count.
After a real cut the person presses the plug's own button when back at the board; the watcher re-arms on that
OFF → ON (it never switches anything on).

## 4. Systems compared — one decision rule for all

Every system is a per-window flag, run through the **same** `DecisionRule` as the live tripwire (`jaga/score.py`):
the same dwell, k-of-n, luma suspend and cool-down. The inputs were logged live on every window (soaks keep no frames,
so no baseline can be added afterwards).

| System | Per-window flag ("nobody is at the board") | Notes |
|---|---|---|
| **jaga** | margin m ≥ θ_m v1 | the live trips are the ones counted; an offline recomputation from the logged margins must agree (reported) |
| **stillness timer** | `motion_pct` < P8 | P8 = its best operating point (§6) |
| **HOG person-in-zone** | `person_in_zone` = 0 | OpenCV's built-in HOG pedestrian detector inside the zone box; labelled with its measured detection rate |
| **reference difference ∧ no person** | `ref_diff_pct` ≥ R and `person_in_zone` = 0 | difference from the median frame of the 2-min capture (held in memory, never stored) |
| **plug countdown timer** | — | computed from the log for T = 15, 30 and 60 min: it cuts T minutes after each switch-on; all three are shown |

After a computed baseline trip, that baseline is paused until the next scheduled step starts (the counterfactual
"someone switches it back on"); jaga's real cuts pause jaga until the real re-arm.

## 5. What counts

* **Still episode cut** — the system trips at a time t with episode start ≤ t ≤ episode end + one window period.
* **Exit caught** — the system trips at t with exit start < t ≤ return + one window period.
  **Time-to-trip** = t − exit start. Reported per exit type (leaves the room / steps away in view).
* **Other false cut** — a trip outside every exit and every still episode (during an activity): reported per
  attended hour (attended hours = the logged length of B1 + B2), secondary.
* **Plug countdown timer** — it cuts T minutes after each switch-on and, after its own cut, is switched back on at
  the next scheduled step (as for the other baselines). Reported: mid-session cuts while the person is present, and
  the minutes the power stays on after each exit begins (until the countdown cuts or the person returns).

## 6. Rules that fix the remaining values before B1

| Value | Rule (applied to unscored data only) |
|---|---|
| θ_m v1 | the highest margin that keeps recall ≥ 0.9 on the labelled **iron** clips (`scripts/seed.py freeze-theta --version 1`) |
| P7 (k fraction) | 0.8, unless the unscored session shows a logged exit missed at 0.8; then the largest of {0.7, 0.6} that catches every logged exit; if neither does, 0.6 and the miss is disclosed (`bench.py tune`) |
| P8 (stillness timer) | the strictest threshold that still flags every empty-board window of the unscored session as still: max(`motion_pct`) over those windows, rounded up to 0.01 % — the timer's best operating point |
| R (reference difference) | the 99th percentile of `ref_diff_pct` over the 2-min capture windows of the unscored session |
| Zone box | the board's box in the `board` camera view, as fractions of the frame |
| HOG detection rate | the share of the 10 person-present iron clips (the hard negatives) on which `person_in_zone` = 1, read on the newest frame of each clip's window as the live guard reads it (`bench.py hog --zone …`) |

## 7. Statistics

* Still episodes cut: k of N per system, **Clopper–Pearson** exact 95 % interval.
* jaga vs the stillness timer, paired on the same episodes: an **exact sign test** on the discordant episodes only.
  No claim is drawn from overlapping intervals.
* Exits caught: k of E per system and per exit type, Clopper–Pearson; median time-to-trip on caught exits.
* Other false cuts per attended hour: exact Poisson interval.
* Scoring command, with the frozen values passed explicitly (they are written into the result):
  `python scripts/bench.py soak --session B1 --session B2 --k-fraction <P7> --p8 <P8> --r <R>`, then the same with
  `--dwell 30 --secondary`.
* n is small (≥ 20 episodes, ≥ 12 exits); intervals will be wide, and every surface says so.

## 8. Targets (pre-registered) and the sentences that report them

**Targets:** jaga cuts **fewer than half as many** still episodes as the stillness timer (paired, the same episodes),
**and** jaga's recall on exits is **≥ 0.8**.

* If **both** hold, the video and README may say:
  "In N still episodes, a stillness timer cut the power X times and jaga cut Y (95 % CI a–b); jaga caught Z of E exits,
  median time-to-trip M s."
* Otherwise they read the counts aloud, with no claim of a win: "N still episodes: stillness timer X cuts, jaga Y;
  exits caught: jaga Z of E." A tie is called a tie.
* **Real but bad:** if jaga cuts as many still episodes as the stillness timer, the live path stays, the measured
  number is published, and the claim is re-scoped. No replay, no hand-tuned θ and no staged trip stands in for the
  live path.
* A baseline whose input was missing in B1 is reported on B2 only.

## Soak A (supporting; tripwires 02 and 03)

One 8-hour, mostly unattended night, θ frozen at v1, a room lamp left on all night (otherwise the too-dark guard
unarms the tripwires and the night measures the guard). A second lamp on a mechanical wall timer switches at known
times. The builder walks through every 2 h; on alternate passes the door (or fridge) is left open for at least the
dwell plus a minute, a logged positive (keys 6 = left open, 7 = closed). Only the tripwires whose area is in the one
camera view are armed.

* **Armed hours** — the time a tripwire could trip: each window in the `armed` or `holding` state counts until the
  next window, a gap counting at most twice the p95 window period.
* **False alarm** — a trip outside every logged positive (key 6 → key 7, plus one window period). Reported per armed
  hour with an exact Poisson interval, armed hours stated. **Positive caught** — a trip inside it; k of n,
  Clopper–Pearson. A key 6 with no key 7 counts as closed at the next key 6 or 15 minutes later, whichever comes
  first; such keys are reported.
* **Baselines, through the same rule** (dwell, k-of-n, suspend, cool-down) over the same armed windows: frame
  difference at the P8 frozen for Soak B, in **both** orientations — motion (`motion_pct` ≥ P8) and stillness
  (`motion_pct` < P8) — since an open door in an empty room is neither clearly one nor the other.
* **Fresh mode, offline** — the stored window vectors re-scored with μ₀/σ₀ from the first 60 s of the session's
  2-minute capture and θ_z = P6 (3.0), through the same rule.

```bash
jaga soak A --session A1 --load 02,03 --camera-pos kitchen
python scripts/bench.py soak-a --session A1
```

## Clips and fresh mode (supporting; fixed before any clip is scored)

* **Clips** — one 8-s clip = one window. Per tripwire: AUC with a Hanley–McNeil interval (reported as undefined at
  AUC 0 or 1); each clip judged at a leave-one-out θ (recall 0.9 on the other positives); precision from those
  decisions. The `iron-inview` clips are scored only at the frozen θ (a separate recall) and never fit it;
  `iron-day2` is held out.
* **Frame-difference baseline on clips** — orientation fixed per tripwire: 01, 02 and 03 describe a still scene
  ("nobody … there"), so frame difference is scored as stillness (−`motion_pct`); 04 (a cat on the counter) as motion.
  jaga vs frame difference is paired on the same clips (each at its own leave-one-out θ): exact sign test on the
  discordant clips.
* **Fresh mode** (no labelled clips) — μ₀/σ₀ from the first 60 s of a 2-minute normal-state recording; the second
  60 s is a held-out false-positive test scored at a **30-s dwell** (a longer dwell could never trip inside 60 s),
  and the share of its windows above θ_z is reported beside the trip count. The window stride is the measured live
  window period (p50).

## Frozen values (filled before B1)

*Not yet filled. B1 must not start until every row below has a value and a commit hash.*

| Value | Frozen | Source | Commit |
|---|---|---|---|
| θ_m v1 (tripwire 01) | — | `fixtures/tripwires/01.json` → `frozen` | — |
| P7 | — | unscored session | — |
| P8 | — | unscored session | — |
| R | — | unscored session | — |
| zone box (`board`) | — | measured on the camera view | — |
| HOG detection rate | — | 10 person-present iron clips | — |
