from typing import Iterable

from _optional import in_marimo, rich_track, tqdm_track


def track(iterable: Iterable, desc: str = "", total: int | None = None):
    """Progress bar that adapts to the environment (marimo or terminal)."""
    if not total:
        try:
            total = len(iterable)  # ty: ignore[invalid-argument-type]
        except TypeError:
            pass
    if in_marimo():
        # Use tqdm with thin bar characters for a slimmer look
        return tqdm_track(
            iterable,
            desc=desc,
            total=total,
            bar_format="{desc} {bar:20} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii="─━",
            leave=True,
        )
    return rich_track(
        iterable,
        description=desc,
        total=total,
        transient=False,
        style="green",
        complete_style="bold green",
        console=None,
    )
