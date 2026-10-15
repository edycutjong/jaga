# Developer-experience report — building jaga on NVIDIA Cosmos-Embed1, Nemotron 3 Super and Nebius

A friction log, written as it happened (first entries 2026-10-15). Each item says what we hit, what we did, and
where the evidence is. It feeds the hackathon's feedback form.

## NVIDIA Cosmos-Embed1-224p

1. **The model card documents Linux + CUDA + BF16 only.** Its example hard-codes `.to("cuda", dtype=torch.bfloat16)`
   and reads video with `decord`; the card says it was tested only with BF16 on Ampere/Hopper under Linux.
   *What we did:* ran the card's own javelin example unchanged except for device and precision: fp32 on the Apple M1
   Max, on CPU and on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Both pick a javelin caption as top-1; between CPU and MPS the six
   caption cosines agree to 6 decimals and the probabilities differ by at most 2 × 10⁻⁶.
   *Evidence:* `evidence/day0/cosmos-cpu.json`, `cosmos-mps.json`, `scripts/day0_cosmos.py`.
   *Ask:* a one-line "runs on CPU/MPS in fp32" note (and a non-CUDA snippet) on the card would have saved the spike.
2. **`decord` has no macOS arm64 wheel.** *What we did:* read frames with PyAV, which the card's sampling maps onto
   directly (`np.linspace` over decoded frames). *Evidence:* `evidence/day0/decord-check.txt`.
3. **`trust_remote_code` + `transformers` version.** The remote code ran with `transformers` 4.51.3; we pin the model
   revision and the library version so a new upstream release cannot change the code that runs (`requirements.lock`,
   `jaga/config.py` `COSMOS_REVISION`).
4. **The processor squashes, it does not crop.** `CosmosEmbed1Processor` resizes any BTCHW input straight to 224×224
   (bilinear). We resize every frame (camera or file) to one canonical size first, so a live window and a recorded
   clip reach the model through identical pixels (`jaga/frames.py`).
5. **The weights are gated.** Accepting the licence on Hugging Face is a one-time click, but a first-time user meets a
   401 with no hint of why. jaga's loader names the fix in its error (`jaga/embed.py`).

## NVIDIA Nemotron 3 Super on Nebius Token Factory

6. **The docs' structured-output example is rejected.** Sending the JSON schema directly under `json_schema` (the
   Token Factory docs' shape) returns HTTP 422 `json_schema.name: Field required`; the OpenAI shape
   `{name, schema, strict}` works for every sentence we tried. *Evidence:* `evidence/day0/tf-stage-t.json`
   (`shape_ok_counts`: tf 0, openai 6, json_object 6).
7. **Super is a reasoning model.** With a small `max_tokens` it can spend the budget on reasoning and return no
   content. We send `max_tokens` 4096 with `reasoning_effort: "low"`, strip any think block, and treat empty content
   as a compile error — never as a spec (`jaga/compile.py`; test `test_reasoning_only_reply_is_an_error_never_an_invented_spec`).
8. **At the recommended temperature (1.0) the proposed action varies.** The first frozen compile proposed `plug_off` for
   a fridge (cutting a fridge's power is the harm) and, after a prompt fix, `notify` for the ironing board. We made the
   action and dwell an explicit choice shown before arming, kept a code-side "never cut" list (fridge, freezer,
   medical…), and measure caption variance with K = 5 extra compiles per sentence
   (`fixtures/tripwires/variance/`). The superseded compiles are recorded in each fixture.
9. **Indonesian works.** Both Indonesian test sentences compiled into valid English-caption specs
   (`evidence/day0/tf-stage-t.json`: Indonesian 2/2).

## Nebius AI Cloud — Serverless Jobs and Object Storage

10. **`--dry-run` validates, but does not show quota.** A GPU preset in a region where our vCPU quota is 0 still
    "validated" (`evidence/day0/g-stage.json`). Quota appears only on a real create.
11. **The minimum disk size is not stated.** `--disk-size 1Gi` and `8Gi` are rejected with `spec.disk.size_bytes is
    invalid` and no minimum; `16Gi` is accepted (`evidence/day0/g-stage.json`). The default is 250Gi.
12. **Reading a Job's output.** `nebius ai job logs <id>` returns a finished Job's stdout, but its window defaults to
    the last hour (`--since`). jaga's Job prints one `JAGA_RESULT {json}` line and also PUTs the result to a presigned
    URL, so either path works (`evidence/day0/logs-check.json`, `jaga/calibrate.py`).
13. **Presigned PUT for Object Storage.** The docs show `aws s3 presign` for GET only; boto3
    `generate_presigned_url('put_object')` against `storage.eu-west1.nebius.cloud` works (`evidence/day0/s3-stage.json`).
14. **Quota is per region and shared.** Two CPU instances per region are shared by every project on the tenant;
    the dispatcher waits on `ResourceExhausted` and never runs two jaga Jobs at once (`jaga/calibrate.py`).

## Local platform

15. **OpenCV 5.0 removed `cv2.HOGDescriptor` from the main wheel.** The person-in-zone baseline needs OpenCV's built-in
    HOG pedestrian detector (no download), so `opencv-python` is pinned below 5 (`pyproject.toml`).
16. **OpenCV and PyAV both bundle `libavdevice`.** macOS prints an objc duplicate-class warning when both load. The
    duplicated classes belong to ffmpeg's avfoundation *input device*, which neither OpenCV's own camera backend nor
    PyAV's file decoding uses; we note it rather than hide it.
17. **macOS camera permission.** A process started outside a GUI terminal (a launch agent, a detached job) cannot get
    camera permission: `VideoCapture(0)` fails. The live loop therefore also accepts a recorded file through the same
    capture path (`jaga serve --source <file>`), and the error message names the fix.
