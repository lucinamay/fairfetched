# hand-kept: Papyrus ships two source files (see papyrus.SOURCE_URLS).
"""Raw Papyrus source tables as attributes on :class:`fairfetched.get.Papyrus`."""

from polars import LazyFrame

from fairfetched.get._tables import table


class PapyrusTables:
    protein: LazyFrame = table("protein")
    bioactivity: LazyFrame = table("bioactivity")
