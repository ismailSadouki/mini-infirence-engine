from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


from engine.kv_cache import KVCache



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

    # Sequence / decode state
    prompt_len: int = field(init=False)
    current_pos: int = field(init=False)
    generated_count: int = field(default=0)


    # KV-cache ownership
    cache_id: Optional[object] = None
    cache_handle: Optional[KVCache] = None



    # Lifecycle
    finished: bool = False
    finished_reason: Optional[str] = None


    # Timing
    arrival_time: float = field(
        default_factory=time.perf_counter
    )
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_timestamps: list[float] = field(
        default_factory=list
    )
    
    def __post_init__(self):
        self.prompt_len = len(self.prompt_ids)
        self.current_pos = self.prompt_len

    


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