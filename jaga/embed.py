"""NVIDIA Cosmos-Embed1-224p: an 8-frame clip and a caption land in the same 256-d space.

The model card's sample hard-codes CUDA + BF16 and says it was only tested on Ampere/Hopper under Linux.
jaga runs it on Apple silicon in fp32: MPS when available (with PYTORCH_ENABLE_MPS_FALLBACK=1), else CPU.
Day-0 stage COSMOS ran the card's own javelin smoke test both ways (evidence/day0/cosmos-*.json).

The weights are gated on Hugging Face (accept the licence once). Once cached, the model loads with
`local_files_only=True`, so a running watcher makes no network call to load it.
"""
from __future__ import annotations

import os
import platform
import threading
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .frames import window_batch  # noqa: E402


class ModelUnavailable(RuntimeError):
    """The gated weights could not be loaded (licence not accepted, no token, or offline with no cache)."""


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested in ("cpu", "mps"):
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine")
        return requested
    if requested != "auto":
        raise ValueError(f"device must be auto|cpu|mps, got {requested!r}")
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Embedder:
    """Thread-safe wrapper: one model, one lock (the live loop and arming both use it)."""

    def __init__(self, device: str = "auto", model_id: str = config.COSMOS_MODEL,
                 revision: str = config.COSMOS_REVISION):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = pick_device(device)
        self.model_id, self.revision = model_id, revision
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        kw = {"revision": revision, "trust_remote_code": True}
        try:
            try:
                model = AutoModel.from_pretrained(model_id, local_files_only=True, **kw)
                proc = AutoProcessor.from_pretrained(model_id, local_files_only=True, **kw)
            except OSError:  # not cached yet: one download (needs the accepted licence + an HF read token)
                model = AutoModel.from_pretrained(model_id, **kw)
                proc = AutoProcessor.from_pretrained(model_id, **kw)
        except OSError as e:
            raise ModelUnavailable(
                f"could not load {model_id}: accept the licence at https://huggingface.co/{model_id} and set an "
                f"HF read token ({type(e).__name__}: {str(e)[:200]})") from e
        self.model = model.to(self.device, dtype=torch.float32).eval()
        self.processor = proc
        self.load_s = time.perf_counter() - t0
        self._torch = torch

    @property
    def info(self) -> dict:
        import torch
        import transformers

        return {"model": self.model_id, "revision": self.revision, "device": self.device, "dtype": "float32",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_threads": torch.get_num_threads(), "machine": platform.machine(),
                "load_s": round(self.load_s, 2)}

    def _sync(self) -> None:
        if self.device == "mps":
            self._torch.mps.synchronize()

    def embed_video(self, frames: np.ndarray) -> np.ndarray:
        """[8,H,W,3] uint8 RGB → [256] float32, L2-normalised (`visual_proj`)."""
        batch = window_batch(frames)
        with self._lock, self._torch.no_grad():
            x = self.processor(videos=batch).to(self.device, dtype=self._torch.float32)
            out = self.model.get_video_embeddings(**x)
            self._sync()
            v = out.visual_proj[0].float().cpu().numpy()
        return v.astype(np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """[n] captions → [n,256] float32, L2-normalised (`text_proj`)."""
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        with self._lock, self._torch.no_grad():
            x = self.processor(text=list(texts)).to(self.device, dtype=self._torch.float32)
            out = self.model.get_text_embeddings(**x)
            self._sync()
            t = out.text_proj.float().cpu().numpy()
        return t.astype(np.float32)


_shared: Embedder | None = None
_shared_lock = threading.Lock()


def shared(device: str = "auto") -> Embedder:
    """One Embedder per process (loading takes ~25–30 s on the M1 Max: evidence/day0/cosmos-*.json)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = Embedder(device)
        return _shared
