
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


from collections import deque
from typing import Optional




from engine.types import RequestState



class WaitingQueue:
    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self._queue = deque()

    def enqueue(self, request: RequestState) -> None:
        """
        Add a request to the waiting queue
        """
        if len(self._queue) >= self.max_waiting:
            raise RuntimeError("Waiting queue is full")

        self._queue.append(request)

    def  pop(self) -> RequestState:
        if not self._queue:
            raise IndexError("Waiting queue is empty")

        return self._queue.popleft()
    def peek(self) -> RequestState:
        if not self._queue:
            raise IndexError("Waiting queue is empty")

        return self._queue[0]


    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)



class ActiveSet:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self._requests: dict[str, RequestState] = {}

    def admit(self, request: RequestState) -> bool:
        if len(self._requests) >= self.max_batch_size:
            return False

        if request.request_id in self._requests:
            raise ValueError(
                f"Request already active: {request.request_id}"
            )

        self._requests[request.request_id] = request

        return True

    def evict(self, request_id: str) ->  RequestState:
        try:
            return self._requests.pop(request_id)
        except KeyError:
            raise KeyError(f"Request is not active: {request_id}")

    def evict_finished(self) -> list[RequestState]:
        """Remove and return all finished requests."""

        removed = []
        for request_id, request in list(self._requests.items()):
            if request.finished:
                removed.append(request)
                del self._requests[request_id]

        return removed

    def get(self, request_id: str) -> Optional[RequestState]:
        return self._requests.get(request_id)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self):
        return iter(self._requests.values())



