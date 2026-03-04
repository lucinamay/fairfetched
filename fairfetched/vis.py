from sqlalchemy import create_engine, inspect


def print_sqlite_schema(db_path: str) -> None:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """)
        tables = [r[0] for r in cur.fetchall()]

        for table in tables:
            print(f"\n{table}")
            cur.execute(f"PRAGMA table_info('{table}')")
            for cid, name, coltype, notnull, default, pk in cur.fetchall():
                nn = " NOT NULL" if notnull else ""
                pkf = " PK" if pk else ""
                dflt = f" DEFAULT {default}" if default is not None else ""
                print(f"  {name} ({coltype}){nn}{pkf}{dflt}")


def load_schema(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    schema = {}

    for table in inspector.get_table_names():
        cols = inspector.get_columns(table)
        pk_info = inspector.get_pk_constraint(table)
        pks = set(pk_info.get("constrained_columns") or [])
        fks = inspector.get_foreign_keys(table)

        fk_cols = {
            fk["constrained_columns"][0] for fk in fks if fk.get("constrained_columns")
        }

        schema[table] = {
            "columns": [
                {
                    "name": c["name"],
                    "type": str(c["type"]).split("(")[0],
                    "pk": c["name"] in pks,
                    "fk": c["name"] in fk_cols,
                }
                for c in cols
            ],
            "foreign_keys": [
                {
                    "column": fk["constrained_columns"][0],
                    "target_table": fk["referred_table"],
                }
                for fk in fks
                if fk.get("constrained_columns") and fk.get("referred_table")
            ],
        }

    return schema


def schema_to_mermaid(schema):
    lines = ["erDiagram"]

    # Tables
    for table, data in schema.items():
        lines.append(f"    {table} {{")
        for col in data["columns"]:
            if col["pk"]:
                tag = " PK"
            elif col["fk"]:
                tag = " FK"
            else:
                tag = ""
            lines.append(f"        {col['type']} {col['name']}{tag}")
        lines.append("    }")

    # Relationships
    for table, data in schema.items():
        for fk in data["foreign_keys"]:
            lines.append(f"    {table} }}|--|| {fk['target_table']} : {fk['column']}")

    return "\n".join(lines)


def db_to_mermaid(db_path: str):
    return schema_to_mermaid(load_schema(db_path))
