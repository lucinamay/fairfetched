# run with `uv run python -m dev.check_missing_tables --csv`
"""Check missing ChEMBL tables by parsing schema_documentation.txt from each version.

Run to print table availability matrix to stdout, or with --csv to save to chembl_table_status.csv.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.request import urlopen

from fairfetched.get import chembl

# only v15+ has schema
VERSIONS = [v for v in chembl.available_versions() if float(v.replace("_", ".")) >= 15]


def fetch_schema_doc(version: str) -> str | None:
    """Fetch schema_documentation.txt for a version using the correct naming pattern."""
    # Use _format_version to parse and validate
    try:
        formatted = chembl._format_version(version)
        v_num = float(formatted.replace("_", "."))
    except (ValueError, TypeError):
        print(f"  ✗ v{version}: invalid version format")
        return None

    if v_num < 27.0 and v_num != 23.0:
        url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{version}/chembl_{version}_schema_documentation.txt"
    else:
        url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{version}/schema_documentation.txt"

    try:
        with urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ✗ v{version} {url}: {e}")
        return None


def extract_tables(schema_text: str) -> set[str]:
    """Extract table names from schema documentation."""
    tables = set()

    for line in schema_text.split("\n"):
        # uppercase name followed by colon, then description (^TABLE_NAME:)
        match = re.match(r"^([A-Z_][A-Z_0-9]*):$", line.strip())
        if match:
            tables.add(match.group(1))

    return tables


def save_csv(schemas: dict[str, set[str]], output_file: Path) -> None:
    """Save version-by-version status to CSV."""
    latest = sorted(schemas.keys())[-1]
    latest_tables = schemas[latest]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Version", "Total Tables", "Missing Tables"])

        for version in sorted(schemas.keys()):
            tables = schemas[version]
            missing = latest_tables - tables
            missing_str = ", ".join(sorted(missing)) if missing else "(none)"
            writer.writerow([f"v{version}", len(tables), missing_str])

    print(f"✓ Saved to {output_file}", file=sys.stderr)


def main():
    """Fetch and compare schemas across versions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="Save results to CSV file")
    args = parser.parse_args()

    schemas = {}

    print("Fetching schema documentation from ChEMBL versions...", file=sys.stderr)
    for version in VERSIONS:
        print(f"  v{version}...", end=" ", flush=True, file=sys.stderr)
        doc = fetch_schema_doc(version)
        if doc:
            tables = extract_tables(doc)
            schemas[version] = tables
            print(f"✓ {len(tables)} tables", file=sys.stderr)
        else:
            print(f"✗ failed", file=sys.stderr)

    if not schemas:
        print("Failed to fetch any schemas", file=sys.stderr)
        return

    if args.csv:
        output_file = Path(__file__).parent / "output" / "chembl_table_status.csv"
        save_csv(schemas, output_file)
        return

    # Use latest as reference
    latest = sorted(schemas.keys())[-1]
    latest_tables = schemas[latest]

    print(f"\n{'Version':<10} {'Total':<6} {'Missing tables (vs v{latest})'}")
    print("=" * 80)

    for version in sorted(schemas.keys()):
        tables = schemas[version]
        missing = latest_tables - tables
        count = len(tables)

        if missing:
            missing_str = ", ".join(sorted(missing)[:5])
            if len(missing) > 5:
                missing_str += f" ... +{len(missing) - 5} more"
        else:
            missing_str = "(all present)"

        print(f"{version:<10} {count:<6} {missing_str}")

    # Show when each table was introduced
    print(f"\n{'Table':<40} {'Introduced in v':<16}")
    print("=" * 80)
    all_tables = set()
    for tables in schemas.values():
        all_tables.update(tables)

    table_intro = {}
    for table in sorted(all_tables):
        for version in sorted(schemas.keys()):
            if table in schemas[version]:
                table_intro[table] = version
                break

    for table in sorted(all_tables):
        intro_version = table_intro.get(table, "unknown")
        print(f"{table:<40} {intro_version:<16}")


if __name__ == "__main__":
    main()
