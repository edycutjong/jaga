# Test fixtures (synthetic — not data)

`smoke_margins.csv` is a hand-written, synthetic list of ten margins for a pretend tripwire `00`. It exists only to
check the calibration Job's plumbing end to end (inject → Job → presigned PUT → result back). A run on it is
labelled **smoke, not a calibration** in `docs/RUNS.md`, and its numbers mean nothing about any real tripwire.
