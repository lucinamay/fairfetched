try:
    from tqdm import (  # ty: ignore[unresolved-import,unused-ignore-comment] # ty: ignore[unused-ignore-comment]
        tqdm as tqdm_track,
    )
except ImportError:

    def tqdm_track(iterable, *args, **kwargs):
        return iter(iterable)


try:
    from rich.progress import (  # ty: ignore[unresolved-import,unused-ignore-comment] # ty: ignore[unused-ignore-comment]
        track as rich_track,
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
