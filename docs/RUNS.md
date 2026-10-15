# Calibration runs — Nebius AI Cloud Serverless Jobs

Every row is written by `jaga calibrate` (`jaga/calibrate.py`) from the Job's own status and output; the
evidence file for each run is `evidence/jobs/<job-id>.json`. Compute: `cpu-d3` · `4vcpu-16gb` · stock
`python:3.11` · `--disk-size 16Gi` (the smallest size we probed that validated; the API's minimum for the image is
10 GiB) · region eu-west1. Input: labelled margins only (no vectors, no frames). Cost = container wall clock × hourly
price for 4 vCPU + 16 GiB: at the listed price for this platform (AMD Epyc Genoa, $0.012 per vCPU-hour + $0.0032 per
GiB-hour = $0.0992/h, <https://docs.nebius.com/compute/resources/pricing>) and, as a margin, at a higher rate of
$0.132/h (4 × $0.015 + 16 × $0.0045; not a listed price). Both are computed, not billed amounts.

A **smoke** run checks the plumbing on a committed synthetic test fixture. It is not a calibration and its
numbers mean nothing about any tripwire.

Provisioning = from the create call to the container start (queue, scheduling and the image pull); the cost
columns price the container's own wall clock only.

| created (UTC) | job ID | kind | state | provisioning (s) | wall clock (s) | cost @ $0.0992/h | cost @ $0.132/h | input | result | result read via |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-15 00:08:27 | `aijob-e01e9n3qk4qq8zzk13` | **smoke, not a calibration** | COMPLETED | 259 | 0.6 | $0.000017 | $0.000022 | 10 rows · sha256 `ce159f720b0b` | plumbing check: 10 rows in, result out (synthetic numbers, not reported) | object storage (presigned PUT by the Job) |
