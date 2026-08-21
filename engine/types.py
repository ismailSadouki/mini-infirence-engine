from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time



@dataclass
class SamplingConfig():
    """
    Controls how the next token is selected.
    """
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    seed: Optional[int] = 42
    greedy: bool = True



@dataclass
class GenerationRequest():
    """
        input+generation constraints+metadata
        "What does the user want?"
    """
    request_id: str
    prompt_text: Optional[str] = None
    prompt_ids: Optional[list[int]] = None
    max_new_tokens: int = 128
    sampling: SamplingConfig = field(
        default_factory=SamplingConfig
    )
    arrival_time: float = field(
        default_factory=time.perf_counter
    )



@dataclass
class RequestState():
    """
    Mutable state of a request while it is being processed.
    "Where is this request right now?"
    """
    request_id: str
    prompt_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    position: int = 0
    finished: bool = False
    arrival_time: float = field(
        default_factory=time.perf_counter
    )
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_timestamps: list[float] = field(
        default_factory=list
    )
    cache_handle: Optional[object] = None


@dataclass
class GenerationOutput():
    """
    Final result returned by the inference engine.
    "What did the engine produce?"
    """
    request_id: str
    output_ids: list[int]
    text: str
    ttft: Optional[float] = None
    itl: list[float] = field(
        default_factory=list
    )
    finish_reason: str = "length"

    prefill_tokens: int = 0
    decode_tokens: int = 0