import sys
from collections.abc import Iterable

from ._optional import HAS_RICH, HAS_TQDM, in_marimo, rich_track, tqdm_track


def _simple_track(
    iterable: Iterable,
    desc: str | None = "",
    total: int | None = None,
):
    label = f"{desc} " if desc else ""
    update_every = max(1, total // 100) if total is not None else 100

    def render(count: int, done: bool = False):
        if total is not None:
            percent = min(100, int((count / total) * 100)) if total else 100
            message = f"\r{label}{percent:3d}% {count}/{total}"
        else:
            suffix = " done" if done else ""
            message = f"\r{label}{count} items{suffix}"
        print(message, end="", file=sys.stderr, flush=True)

    def iterator():
        count = 0
        for item in iterable:
            count += 1
            if count % update_every == 0 or total is not None and count >= total:
                render(count)
            yield item

        render(count, done=True)
        print(file=sys.stderr, flush=True)

    return iterator()


def track(
    iterable: Iterable,
    desc: str | None = "",
    total: int | None = None,
    disable: bool = False,
):
    """Progress bar that adapts to the environment (marimo or terminal)."""
    if disable:
        return iter(iterable)

    if total is None:
        try:
            total = len(iterable)  # ty: ignore[invalid-argument-type]
        except TypeError:
            pass
    if (in_marimo() or not HAS_RICH) and HAS_TQDM:
        # Use tqdm with thin bar characters for a slimmer look
        return tqdm_track(
            iterable,
            desc=desc if desc else "",
            total=total,
            bar_format="{desc} {bar:20} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ascii="─━",
            leave=True,
        )
    if HAS_RICH:
        return rich_track(
            iterable,
            description=desc if desc else "",
            total=total,
            transient=False,
            style="green",
            complete_style="bold green",
            console=None,
        )
    return _simple_track(iterable, desc=desc, total=total)
