"""Benchmark: does downcasting parquet column dtypes before the ChEMBL view
joins pay for itself, and how does polars compare to doing the anchor joins in
duckdb?

Run one mode per process so peak RSS (getrusage, monotonic) is per-scenario:

    uv run python dev/bench_views.py baseline
    uv run python dev/bench_views.py shrink_pass
    uv run python dev/bench_views.py shrunk
    uv run python dev/bench_views.py duckdb

`dev/bench_views.py all` drives the four in subprocesses and prints a table.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

from fairfetched.get import chembl
from fairfetched.utils import BASE_DIR

SRC = Path(f"{BASE_DIR}/chembl/37/parquet")
SCRATCH = Path("/tmp/claude-bench-views")
SHRUNK = SCRATCH / "shrunk"
OUT = SCRATCH / "out"

# tables build_views actually touches
TABLES = list(chembl._REQUIRED)


def _peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # darwin: bytes, linux: kB


def _sink(lf: pl.LazyFrame, name: str) -> tuple[float, int, int]:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.parquet"
    t = time.perf_counter()
    lf.sink_parquet(p)
    dt = time.perf_counter() - t
    meta = pl.read_parquet_schema(p)
    n = pl.scan_parquet(p).select(pl.len()).collect().item()
    return dt, n, p.stat().st_size, len(meta)


def _build_and_sink(src: Path) -> dict:
    paths = {t: src / f"{t}.parquet" for t in TABLES}
    views = chembl.build_views(paths)
    res = {}
    t0 = time.perf_counter()
    for name, lf in views.items():
        dt, n, sz, ncol = _sink(lf, name)
        res[name] = {"s": round(dt, 2), "rows": n, "out_mb": round(sz / 1e6, 1)}
    res["total_s"] = round(time.perf_counter() - t0, 2)
    res["peak_mb"] = round(_peak_mb(), 0)
    return res


# ---- shrink pass ---------------------------------------------------------------

def _shrink_expr(name: str, dtype: pl.DataType, lo, hi) -> pl.Expr | None:
    col = pl.col(name)
    if dtype == pl.Int64:
        if lo is None:
            return None
        for cand in (pl.Int8, pl.Int16, pl.Int32):
            if lo >= _imin(cand) and hi <= _imax(cand):
                return col.cast(cand)
        return None
    if dtype == pl.Float64:
        return col.cast(pl.Float32)
    return None


def _imin(t):
    return {pl.Int8: -128, pl.Int16: -32768, pl.Int32: -(2**31)}[t]


def _imax(t):
    return {pl.Int8: 127, pl.Int16: 32767, pl.Int32: 2**31 - 1}[t]


def shrink_pass() -> dict:
    SHRUNK.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    src_bytes = tgt_bytes = 0
    rewritten = 0
    for t in TABLES:
        sp = SRC / f"{t}.parquet"
        lf = pl.scan_parquet(sp)
        sch = lf.collect_schema()
        num = [c for c, d in sch.items() if d in (pl.Int64, pl.Float64)]
        casts = []
        if num:
            stats = lf.select(
                [pl.col(c).min().alias(f"{c}__lo") for c in num]
                + [pl.col(c).max().alias(f"{c}__hi") for c in num]
            ).collect().row(0, named=True)
            for c in num:
                e = _shrink_expr(c, sch[c], stats[f"{c}__lo"], stats[f"{c}__hi"])
                if e is not None:
                    casts.append(e)
        src_bytes += sp.stat().st_size
        if not casts:
            # symlink through; no rewrite cost
            dp = SHRUNK / f"{t}.parquet"
            if dp.exists() or dp.is_symlink():
                dp.unlink()
            dp.symlink_to(sp)
            tgt_bytes += sp.stat().st_size
            continue
        rewritten += 1
        dp = SHRUNK / f"{t}.parquet"
        lf.with_columns(casts).sink_parquet(dp)
        tgt_bytes += dp.stat().st_size
    return {
        "total_s": round(time.perf_counter() - t0, 2),
        "peak_mb": round(_peak_mb(), 0),
        "tables_rewritten": rewritten,
        "src_mb": round(src_bytes / 1e6, 0),
        "shrunk_mb": round(tgt_bytes / 1e6, 0),
    }


# ---- duckdb anchor joins -----------------------------------------------------

def duckdb_bench() -> dict:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    def rel(t):
        return f"read_parquet('{SRC / (t + '.parquet')}')"

    v = {t: rel(t) for t in TABLES}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.perf_counter()

    bio_sql = f"""
    SELECT * FROM {v['activities']} a
    LEFT JOIN (SELECT record_id, compound_key, compound_name FROM {v['compound_records']}) r USING (record_id)
    LEFT JOIN {v['action_type']} act USING (action_type)
    LEFT JOIN (SELECT activity_id, bei, sei, le, lle FROM {v['ligand_eff']}) le USING (activity_id)
    LEFT JOIN (SELECT * EXCLUDE (doc_id, src_id), doc_id AS doc_id_assay, src_id AS src_id_assay FROM {v['assays']}) asy USING (assay_id)
    LEFT JOIN {v['assay_type']} aty USING (assay_type)
    LEFT JOIN {v['confidence_score_lookup']} csl USING (confidence_score)
    LEFT JOIN {v['relationship_type']} rt USING (relationship_type)
    LEFT JOIN (SELECT variant_id, mutation, accession FROM {v['variant_sequences']}) vs USING (variant_id)
    LEFT JOIN (SELECT tid, pref_name, target_type, organism, tax_id, chembl_id, species_group_flag FROM {v['target_dictionary']}) td USING (tid)
    LEFT JOIN (SELECT doc_id, journal, year, pubmed_id, doi, chembl_id AS chembl_id_doc, title, doc_type FROM {v['docs']}) d USING (doc_id)
    LEFT JOIN (SELECT src_id, src_description, src_short_name FROM {v['source']}) s USING (src_id)
    """
    con.execute(
        f"COPY ({bio_sql}) TO '{OUT / 'bioactivity_duck.parquet'}' (FORMAT parquet)"
    )
    res["bioactivity"] = {"s": round(time.perf_counter() - t0, 2)}

    t1 = time.perf_counter()
    comp_sql = f"""
    SELECT * FROM {v['molecule_dictionary']} m
    LEFT JOIN {v['compound_structures']} cs USING (molregno)
    LEFT JOIN {v['compound_properties']} cp USING (molregno)
    LEFT JOIN {v['molecule_hierarchy']} mh USING (molregno)
    LEFT JOIN (SELECT molregno, helm_notation FROM {v['biotherapeutics']}) b USING (molregno)
    LEFT JOIN (
        SELECT molregno, list(DISTINCT alert_name) AS structural_alerts
        FROM {v['compound_structural_alerts']} JOIN {v['structural_alerts']} USING (alert_id)
        GROUP BY molregno
    ) sa USING (molregno)
    """
    con.execute(
        f"COPY ({comp_sql}) TO '{OUT / 'compounds_duck.parquet'}' (FORMAT parquet)"
    )
    res["compounds"] = {"s": round(time.perf_counter() - t1, 2)}

    res["total_s"] = round(time.perf_counter() - t0, 2)
    res["peak_mb"] = round(_peak_mb(), 0)
    return res


# ---- driver ----------------------------------------------------------------

MODES = {
    "baseline": lambda: _build_and_sink(SRC),
    "shrink_pass": shrink_pass,
    "shrunk": lambda: _build_and_sink(SHRUNK),
    "duckdb": duckdb_bench,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in MODES:
        print(json.dumps({mode: MODES[mode]()}))
        return
    results = {}
    for m in ("baseline", "shrink_pass", "shrunk", "duckdb"):
        print(f"--- {m} ---", file=sys.stderr)
        out = subprocess.run(
            ["uv", "run", "python", __file__, m], capture_output=True, text=True
        )
        sys.stderr.write(out.stderr)
        try:
            results.update(json.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:
            print("FAILED", m, out.stdout, file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
