"""Model loading and inference.

Wraps `laya.Router` so the game can:
  * load only the checkpoints the hardware can hold (env LAYA_CHECKPOINTS),
  * build them with a low peak-memory loader (meta-device init, no random init),
  * fall back to a loaded checkpoint when the router picks one that is not resident,
  * report latency and the routing decision for every call.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("sway.engine")

VALID_CHECKPOINTS = ("english", "multilingual", "typed-decisions")
DEFAULT_CHECKPOINTS = "english,multilingual"


def parse_checkpoints(raw: Optional[str]) -> List[str]:
    names: List[str] = []
    for part in (raw if raw is not None else DEFAULT_CHECKPOINTS).split(","):
        name = part.strip().lower().replace("_", "-")
        if not name:
            continue
        if name not in VALID_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint {name!r}. Valid: {', '.join(VALID_CHECKPOINTS)}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("LAYA_CHECKPOINTS is empty")
    return names


def _low_memory_build(orig_build):
    """Return a build_model replacement that skips random init.

    The stock loader builds the encoder with random weights and then overwrites them,
    which peaks at several times the model size. Building on the meta device and
    assigning the checkpoint tensors directly keeps the peak close to one copy.
    """
    import torch

    def build(cfg, encoder_dir=None):
        with torch.device("meta"):
            model = orig_build(cfg, encoder_dir=encoder_dir)
        original_load = model.load_state_dict

        def load_state_dict(self, state_dict, strict=True):
            converted = {k: v.to(torch.float32) for k, v in state_dict.items()}
            result = original_load(converted, strict=strict, assign=True)
            # Non-persistent buffers (rotary frequencies) are not stored in the checkpoint
            # and would stay on the meta device, so that module is rebuilt for real.
            rotary = getattr(getattr(self, "encoder", None), "rotary_emb", None)
            if rotary is not None:
                self.encoder.rotary_emb = type(rotary)(config=self.encoder.config)
            leftovers = [n for n, b in self.named_buffers() if b.is_meta]
            leftovers += [n for n, p in self.named_parameters() if p.is_meta]
            if leftovers:
                raise RuntimeError("tensors left on meta device: " + ", ".join(leftovers[:5]))
            return result

        model.load_state_dict = types.MethodType(load_state_dict, model)
        return model

    return build


def load_agent(name: str, device: Optional[str] = None, low_memory: bool = True):
    """Load one Laya checkpoint by router name."""
    import laya
    import laya.agent as agent_mod
    from laya.router import DEFAULT_MODELS, _split

    repo, sub = _split(DEFAULT_MODELS[name])
    if not low_memory:
        return laya.load(repo, device=device, subfolder=sub)

    original = agent_mod.build_model
    agent_mod.build_model = _low_memory_build(original)
    try:
        return laya.load(repo, device=device, subfolder=sub)
    except Exception:
        log.exception("low-memory load failed for %s, retrying with the stock loader", name)
        agent_mod.build_model = original
        return laya.load(repo, device=device, subfolder=sub)
    finally:
        agent_mod.build_model = original


@dataclass
class Verdict:
    answers: Dict[str, Any]
    routing: Dict[str, Any]
    latency_ms: float
    input_tokens: int
    detection: Dict[str, Any] = field(default_factory=dict)


class EngineNotReady(RuntimeError):
    pass


class Engine:
    """Thread-safe facade over laya.Router with partial checkpoint loading."""

    def __init__(self, checkpoints: Optional[List[str]] = None, device: Optional[str] = None,
                 low_memory: Optional[bool] = None):
        self.checkpoints = checkpoints or parse_checkpoints(os.environ.get("LAYA_CHECKPOINTS"))
        self.device = device or os.environ.get("LAYA_DEVICE") or None
        if low_memory is None:
            low_memory = os.environ.get("LAYA_LOW_MEMORY", "1") != "0"
        self.low_memory = low_memory
        self._router = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None
        self.load_seconds: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self._router is not None

    def start(self) -> "Engine":
        """Load every configured checkpoint. Safe to call more than once."""
        with self._lock:
            if self._router is not None:
                return self
            from laya import Router

            t0 = time.perf_counter()
            try:
                default = "english" if "english" in self.checkpoints else self.checkpoints[0]
                router = Router(max_loaded=len(self.checkpoints), default=default)
                for name in self.checkpoints:
                    log.info("loading checkpoint %s", name)
                    router.attach(name, load_agent(name, self.device, self.low_memory))
                self._router = router
                self.error = None
            except Exception as exc:  # surfaced in the UI, never crashes the app
                log.exception("engine failed to start")
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            self.load_seconds = time.perf_counter() - t0
        self._warmup()
        return self

    def _warmup(self):
        try:
            self.predict("warm up", {"ok": {"type": "noul", "instructions": "Is this text a greeting?"}})
        except Exception:
            log.exception("warmup failed")

    def route(self, state):
        """Routing decision only (pure Python, no model needed)."""
        from laya import Router

        return (self._router or Router()).route(state)

    def predict(self, state, questions: Dict[str, Any]) -> Verdict:
        if self._router is None:
            raise EngineNotReady(self.error or "The engine is still loading.")
        decision = self._router.route(state, questions)
        wanted = decision["model"]
        model = wanted
        reason = decision["reason"]
        if wanted not in self.checkpoints:
            model = "multilingual" if "multilingual" in self.checkpoints else self.checkpoints[0]
            reason += f" (the {wanted} checkpoint is not loaded here, so {model} answered)"
        t0 = time.perf_counter()
        result = self._router.load(model).system_one(state, questions)
        latency = (time.perf_counter() - t0) * 1000
        return Verdict(
            answers=result["answers"],
            routing={"model": model, "wanted": wanted, "reason": reason},
            latency_ms=latency,
            input_tokens=int(result.get("usage", {}).get("input_tokens", 0)),
            detection=dict(decision.get("detection") or {}),
        )


_ENGINE: Optional[Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()
        return _ENGINE
