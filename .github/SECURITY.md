# Security Policy

jaga switches a mains plug OFF from a camera-driven decision, so safety defects matter as much as security ones.

## Supported versions
| Version | Supported |
|---|---|
| latest (`main`) | ✅ |

## Reporting a vulnerability
Please **do not** open a public issue for security vulnerabilities. Report them privately:

- Email **edy.cu@live.com**, or
- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) (Security → Report a vulnerability).

You will get an acknowledgment within 48 hours and a resolution timeline after triage. Please allow a reasonable
window to patch before public disclosure.

## What is in scope
- A path by which a trip, or any remote request, can switch a plug **ON** (the project's invariant I1).
- The overlay server (`127.0.0.1:8765`): cross-origin or DNS-rebinding access to the camera stream or to the plug controls.
- Video frames, vectors or device keys leaving the machine, or being written into the repository.

## What the tests assert (not just claim)
`tests/test_invariants.py` pins that no trip path can switch a plug ON (I1). `tests/test_server.py` pins the
cross-origin POST refusal, the session-token requirement, the DNS-rebinding Host refusal and the Fetch-Metadata check.

## Known dependency advisories
`transformers` is pinned to `4.51.3` (`pyproject.toml`), the version the model's remote code was measured with, and the
model revision is pinned too (`jaga/config.py`). `pip-audit -r requirements.lock` reports 28 known vulnerabilities (its own count) against that
version as of 2026-10-15 (CI reports them in an informational step). jaga loads exactly one pinned model revision
(`nvidia/Cosmos-Embed1-224p`) and no user-supplied checkpoint, but moving to a fixed `transformers` release needs the
speed and accuracy numbers in `DEMO.md` measured again, so it is a deliberate change, not a Dependabot bump.
