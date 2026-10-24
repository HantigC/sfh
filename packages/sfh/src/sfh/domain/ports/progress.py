from typing import Protocol


class ProgressReporter(Protocol):
    """Lets long-running steps report progress without knowing who is listening."""

    def start(self, total: int) -> None: ...
    def advance(self, n: int = 1) -> None: ...
    def finish(self) -> None: ...


class NullProgress:
    def start(self, total: int) -> None:
        pass

    def advance(self, n: int = 1) -> None:
        pass

    def finish(self) -> None:
        pass
