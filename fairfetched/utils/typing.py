from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from polars import LazyFrame


class ComposedLFDict(TypedDict):
    bioactivity: LazyFrame
    compounds: LazyFrame
    full: NotRequired[LazyFrame]
    proteins: NotRequired[LazyFrame]
    components: NotRequired[LazyFrame]


# ComposedDict = TypedDict(

#     "ComposedDict",
#     {
#         "bioactivity": pl.LazyFrame,
#         "compounds": pl.LazyFrame,
#         "proteins": NotRequired[pl.LazyFrame],
#     },
#     total=False,
# )
