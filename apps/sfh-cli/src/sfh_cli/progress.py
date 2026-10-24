from types import TracebackType
from typing import Self

from rich.progress import Progress, TaskID


class RichProgress:
    """ProgressReporter that draws a progress bar in the terminal."""

    def __init__(self, description: str) -> None:
        self._description = description
        self._progress = Progress(transient=True)
        self._task: TaskID | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.finish()

    def start(self, total: int) -> None:
        self._progress.start()
        self._task = self._progress.add_task(self._description, total=total)

    def advance(self, n: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, n)

    def finish(self) -> None:
        self._progress.stop()
