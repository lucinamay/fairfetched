from typing import Callable, Iterable

try:
    from tqdm import (  # ty: ignore[unresolved-import,unused-ignore-comment] # ty: ignore[unused-ignore-comment]
        tqdm as tqdm_track,
    )
except ImportError:

    def tqdm_track(iterable, *args, **kwargs):
        return iter(iterable)


try:
    from rich.progress import (
        BarColumn,
        Console,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        ProgressType,
        StyleType,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    def rich_track(
        sequence: Iterable[ProgressType],
        description: str = "Working...",
        total: float | None = None,
        completed: int = 0,
        auto_refresh: bool = True,
        console: Console | None = None,
        transient: bool = False,
        get_time: Callable[[], float] | None = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        update_period: float = 0.1,
        disable: bool = False,
        show_speed: bool = True,
    ) -> Iterable[ProgressType]:
        columns: list["ProgressColumn"] = (
            [TextColumn("[progress.description]{task.description}")]
            if description
            else []
        )
        columns.extend(
            (
                BarColumn(
                    style=style,
                    complete_style=complete_style,
                    finished_style=finished_style,
                    pulse_style=pulse_style,
                ),
                TaskProgressColumn(show_speed=show_speed),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            )
        )
        progress = Progress(
            *columns,
            auto_refresh=auto_refresh,
            console=console,
            transient=transient,
            get_time=get_time,
            refresh_per_second=refresh_per_second or 10,
            disable=disable,
        )

        with progress:
            yield from progress.track(
                sequence,
                total=total,
                completed=completed,
                description=description,
                update_period=update_period,
            )

except ImportError:

    def rich_track(iterable, *args, **kwargs):
        return iter(iterable)


try:
    from marimo import (  # ty: ignore[unresolved-import,unused-ignore-comment] # ty: ignore[unused-ignore-comment]
        running_in_notebook as in_marimo,
    )

except ImportError:

    def in_marimo():
        return False
