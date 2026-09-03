# run with `uv run --extra adrecs python -m dev.inspect_adrecs [--raw-dir DIR]`
"""Reconnaissance on the *raw* ADReCS / ADReCS-Target downloads, before any
parsing.

get.adrecs / get.adrecs_target make claims about these files (which separator,
which sheet, which null tokens, which BOM'd header). This script is how those
claims were checked and how to re-check them when a new version lands: it dumps
every sheet of every raw file with original headers, dtypes, a sample, and the
frequency of each suspected null token per string column.

By default it downloads via ``ensure_raw_files`` into the normal cache; pass
``--raw-dir`` to point at files you already have.
"""

import argparse
import gzip
import io
from collections import Counter
from pathlib import Path

import polars as pl

from fairfetched.get import adrecs, adrecs_target

NULL_TOKENS = ("Not Available", "---", "null", "-", "", "NA", "N/A")


def _frames(path: Path) -> dict[str, pl.DataFrame]:
    """Every sheet (xlsx) or the single table (txt/txt.gz), unparsed except for
    separator detection. Header row is kept as data so we see it verbatim."""
    name = path.name.lower()
    if name.endswith(".xlsx"):
        return pl.read_excel(path, sheet_id=0, has_header=False)
    raw = gzip.decompress(path.read_bytes()) if name.endswith(".gz") else path.read_bytes()
    text = raw.decode("utf-8", "replace")
    sep = "#" if "#" in text.splitlines()[0] else "\t"
    df = pl.read_csv(io.StringIO(text), separator=sep, has_header=False, infer_schema_length=0)
    return {f"(sep={sep!r})": df}


def _dump(source: str, raw_paths: dict[str, Path]) -> None:
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    for key, path in raw_paths.items():
        print(f"\n### {key}  <-  {path.name}  ({path.stat().st_size:,} B)")
        for sheet, df in _frames(path).items():
            header = [str(h) for h in df.row(0)]
            body = df.slice(1)
            print(f"  sheet {sheet}  {df.shape}")
            print(f"  header row: {header}")
            with pl.Config(fmt_str_lengths=25, tbl_cols=-1, tbl_rows=3):
                print(body.head(3))
            for name, col in zip(header, df.columns):
                hits = Counter(v for v in body[col].to_list() if v in NULL_TOKENS)
                if hits:
                    print(f"    {name!r}: null-ish {dict(hits)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, help="reuse existing raw files instead of downloading")
    ap.add_argument("--adrecs-version", default=adrecs.latest())
    args = ap.parse_args()

    if args.raw_dir:
        by_name = {p.name: p for p in args.raw_dir.iterdir() if p.is_file()}
        a_raw = {k: by_name[Path(u).name] for k, u in adrecs.source_urls(args.adrecs_version).items() if Path(u).name in by_name}
        t_raw = {k: by_name[Path(u).name] for k, u in adrecs_target.source_urls("1.0").items() if Path(u).name in by_name}
    else:
        a_raw = adrecs.ensure_raw_files(args.adrecs_version)
        t_raw = adrecs_target.ensure_raw_files()

    _dump("ADReCS", a_raw)
    _dump("ADReCS-Target", t_raw)


if __name__ == "__main__":
    main()
