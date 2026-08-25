from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


from engine.request_queue import ActiveSet, WaitingQueue
from engine.types import RequestState



@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_size: int
    max_total_active_tokens: int
    max_waiting: int


class ContinuousScheduler:
    def __init__(
            self,
            config: SchedulerConfig
    ):
        """
        ├── waiting: WaitingQueue
        │      └── FCFS requests waiting for admission
        │
        └── active: ActiveSet
            └── requests currently generating
        """
        self.config = config

        self.waiting = WaitingQueue(
            max_waiting=config.max_waiting
        )

        self.active = ActiveSet(
            max_batch_size=config.max_batch_size
        )
    def submit(self, request: RequestState) -> None:
        self.waiting.enqueue(request)


    def admit_waiting(self) -> None:
        while(
            self.waiting
            and len(self.active) < self.config.max_batch_size
        ):


            request = self.waiting.peek()

            current_active_tokens = sum(
                r.current_pos
                for r in self.active
            )

            if (
                current_active_tokens + request.prompt_len
                > self.config.max_total_active_tokens
            ):
                break

            
            request = self.waiting.pop() 

            self.active.admit(request)


    def evict_finished(self) -> list[RequestState]:
        """Remove and return all finished requests."""

        return self.active.evict_finished()

    def step(self) -> list[RequestState]:
        """
        Perform one continuous-batching scheduling step.
        Order:
            1. Evict finished requests
            2. Admit waiting requests
            3. Return active requests that should decode one token
        """

        finished = self.evict_finished()
        self.admit_waiting()

        return finished, list(self.active) # active requests

    
            
