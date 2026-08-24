from dataclasses import dataclass



@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_size: int
    max_total_active_tokens: int
    max_waiting: int