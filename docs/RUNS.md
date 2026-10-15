# Calibration runs — Nebius AI Cloud Serverless Jobs

Every row is written by `jaga calibrate` (`jaga/calibrate.py`) from the Job's own status and output; the
evidence file for each run is `evidence/jobs/<job-id>.json`. Compute: `cpu-d3` · `4vcpu-16gb` · stock
`python:3.11` · `--disk-size 16Gi` (the smallest size the API accepted on day 0) · region eu-west1. Input: labelled margins
only (no vectors, no frames). Cost = container wall clock × hourly price for 4 vCPU + 16 GiB, at the docs list
price ($0.0992/h) and at a higher price seen earlier ($0.132/h); the billed amount is read from the console.

A **smoke** run checks the plumbing on a committed synthetic test fixture. It is not a calibration and its
numbers mean nothing about any tripwire.

Provisioning = from the create call to the container start (queue, scheduling and the image pull); the cost
columns price the container's own wall clock only.

| created (UTC) | job ID | kind | state | provisioning (s) | wall clock (s) | cost @ $0.0992/h | cost @ $0.132/h | input | result | result read via |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-15 00:08:27 | `aijob-e01e9n3qk4qq8zzk13` | **smoke, not a calibration** | COMPLETED | 259 | 0.6 | $0.000017 | $0.000022 | 10 rows · sha256 `ce159f720b0b` | plumbing check: 10 rows in, result out (synthetic numbers, not reported) | object storage (presigned PUT by the Job) |
