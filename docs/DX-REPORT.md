# Developer-experience report — building jaga on NVIDIA Cosmos-Embed1, Nemotron 3 Super and Nebius

A friction log, written as it happened (first entries 2026-10-15). Each item says what we hit, what we did, and
where the evidence is. It feeds the hackathon's feedback form.

## NVIDIA Cosmos-Embed1-224p

1. **The model card's example is CUDA + BF16 only.** It hard-codes `.to("cuda", dtype=torch.bfloat16)` and reads
   video with `decord`; the card says it was tested only with BF16 on Ampere and Hopper GPUs, under Linux.
   *What we did:* ran the card's own javelin example unchanged except for device, precision and the video reader
   (PyAV instead of `decord`, item 2): fp32 on the Apple M1 Max, on CPU and on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Both pick a javelin caption as top-1; between CPU and MPS the six
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
5. **Gating is in flux.** The card carries a licence prompt (`extra_gated_prompt`), but on 2026-10-15 the Hugging Face
   API reported `gated: false`, and a file request without a token was redirected to the download (HTTP 307) rather
   than refused (`evidence/day0/hf-access.json`). Because gating may return, jaga's loader names the fix (accept the
   licence, `huggingface-cli login`) in its error if a download is refused (`jaga/embed.py`).

## NVIDIA Nemotron 3 Super on Nebius Token Factory

6. **The docs' structured-output example is rejected.** Sending the JSON schema directly under `json_schema` (the
   Token Factory docs' shape) returns HTTP 422 `json_schema.name: Field required`; the OpenAI shape
   `{name, schema, strict}` works for every sentence we tried. *Evidence:* `evidence/day0/tf-stage-t.json`
   (`shape_ok_counts`: tf 0, openai 6, json_object 6).
7. **Super is a reasoning model.** We send `max_tokens` 4096 with `reasoning_effort: "low"` so reasoning leaves room
   for the answer, strip any think block, and treat empty content as a compile error — never as a spec (`jaga/compile.py`; test `test_reasoning_only_reply_is_an_error_never_an_invented_spec`).
8. **At the recommended temperature (1.0) the proposed action and dwell vary.** Five repeat compiles of the same
   sentence ("nobody is at the ironing board", same prompt) proposed `notify` four times and `plug_off` once, with
   dwells of 30, 60, 60, 120 and 60 s (`fixtures/tripwires/variance/01-k*.json`). Separately, the very first compile of
   the fridge sentence (before a compiler-prompt revision, so not a temperature effect) proposed `plug_off` for a fridge
   (cutting a fridge's power is the harm). We made the action and dwell an explicit choice shown before arming, kept a
   code-side "never cut" list (fridge, freezer, medical…), and recorded K = 5 extra compiles per sentence
   (`fixtures/tripwires/variance/`); what that variance does to accuracy is not measured yet. The superseded compiles
   are recorded in each fixture.
9. **Indonesian works.** Both Indonesian test sentences compiled into valid English-caption specs
   (`evidence/day0/tf-stage-t.json`: Indonesian 2/2).

## Nebius AI Cloud — Serverless Jobs and Object Storage

10. **`--dry-run` validates the request, not availability.** Two GPU presets (`gpu-l40s-a` in eu-north1,
    `gpu-h200-sxm` in eu-west1) "validated" with no word on whether the tenant has quota for them
    (`evidence/day0/g-stage.json`). jaga needs no GPU; we note that a dry run is not an availability check.
11. **The minimum disk size is in the error's detail, not its first line.** `--disk-size 1Gi` and `8Gi` fail with
    `spec.disk.size_bytes is invalid`; only the detail lines further down say `Disk size must be >= 10737418240`
    (10 GiB, for the stock image). `16Gi` is the smallest size we probed that validated (`evidence/day0/g-stage.json`).
12. **Reading a Job's output.** jaga's Job PUTs its result to a presigned URL, and that is how the dispatcher reads
    it (`evidence/jobs/`). It also prints one `JAGA_RESULT {json}` line as a fallback: in a separate check,
    `nebius ai job logs <id>` returned a finished Job's stdout, with a window that defaults to the last hour
    (`--since`) (`evidence/day0/logs-check.json`).
13. **Presigned PUT for Object Storage.** The docs show `aws s3 presign` for GET only; boto3
    `generate_presigned_url('put_object')` against `storage.eu-west1.nebius.cloud` works (`evidence/day0/s3-stage.json`).
14. **Quota is per region** (the Nebius docs: quotas can be set per project within a region). The dispatcher waits
    and retries on `ResourceExhausted`, and one `jaga calibrate` run creates one Job (`jaga/calibrate.py`).

## Local platform

15. **OpenCV 5.0 removed `cv2.HOGDescriptor` from the main wheel.** The person-in-zone baseline needs OpenCV's built-in
    HOG pedestrian detector (no download), so `opencv-python` is pinned below 5 (`pyproject.toml`).
16. **OpenCV and PyAV both bundle `libavdevice`.** macOS prints an objc duplicate-class warning when both load. The
    duplicated classes belong to ffmpeg's avfoundation *input device*; it has caused no failure we could see, and we
    note it rather than hide it.
17. **macOS camera permission.** A process started outside a GUI terminal (a launch agent, a detached job) is not
    given the camera: `VideoCapture(0)` does not open, and OpenCV prints "not authorized to capture video"
    (`evidence/day0/camera-permission.json`). The live loop therefore also accepts a recorded file through the same
    capture path (`jaga serve --source <file>`), and jaga's error message names the fix.
