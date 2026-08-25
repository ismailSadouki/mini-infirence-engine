from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)




from engine.types import RequestState


def make_request(
    request_id: str,
    prompt_len: int = 4,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_ids=list(range(prompt_len)),
    )
