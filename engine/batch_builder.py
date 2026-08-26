from dataclasses import dataclass
import torch


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


from engine.types import RequestState


@dataclass
class RaggedDecodeBatch:
    """
    One decode token per active request.
    input_ids: [B, 1]
    positions: [B]
    requests: The requests corresponding to each batch row.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    requests: list[RequestState]


def build_decode_batch(
        requests: list[RequestState],
        device: torch.device
) -> RaggedDecodeBatch:

    if not requests:
        raise ValueError("Cannot build decode batch from empty request list")

    input_ids = torch.tensor(
        [
            [request.generated_ids[-1]]
            for request in requests
        ],
        dtype=torch.long,
        device=device
    )

    positions = torch.tensor(
        [
            request.current_pos
            for request in requests
        ],
        dtype=torch.long,
        device=device
    )

    return RaggedDecodeBatch(
        input_ids=input_ids,
        positions=positions,
        requests=requests
    )
