"""Query bundled CSV with DuckDB; optionally round-trip Parquet in a temporary folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


DATA = Path(__file__).resolve().parent / "sales.csv"
SUMMARY_SQL = """
    SELECT category,
           SUM(quantity) AS units,
           SUM(quantity * unit_price_cents) AS revenue_cents
    FROM sales
    WHERE quantity >= ?
    GROUP BY category
    ORDER BY revenue_cents DESC, category
"""


def load_sales(connection: Any) -> None:
    """Use a trusted bundled path and explicit types rather than CSV type guessing."""
    connection.read_csv(
        str(DATA),
        header=True,
        columns={
            "order_id": "INTEGER",
            "category": "VARCHAR",
            "quantity": "INTEGER",
            "unit_price_cents": "INTEGER",
        },
    ).create_view("sales")


def summary(connection: Any, min_quantity: int = 1) -> list[tuple[str, int, int]]:
    if type(min_quantity) is not int or min_quantity < 1:
        raise ValueError("min_quantity must be a positive integer")
    # Parameterize values. Do not interpolate user input into SQL text.
    return connection.execute(SUMMARY_SQL, [min_quantity]).fetchall()


def parquet_roundtrip(connection: Any) -> tuple[int, list[tuple[str, int, int]]]:
    """Write only inside a newly created temporary test directory, then remove it."""
    with TemporaryDirectory(prefix="learn-duckdb-") as directory:
        target = Path(directory) / "sales.parquet"
        connection.table("sales").write_parquet(str(target))
        connection.read_parquet(str(target)).create_view("sales_copy", replace=True)
        try:
            restored = connection.execute(SUMMARY_SQL.replace("FROM sales", "FROM sales_copy"),
                                          [1]).fetchall()
            return target.stat().st_size, restored
        finally:
            # Release the view before its backing temporary file is removed.
            connection.execute("DROP VIEW sales_copy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="store_true", help="write a temporary Parquet file")
    args = parser.parse_args()
    try:
        import duckdb
    except ModuleNotFoundError:
        parser.exit(2, "DuckDB is not installed. Read this directory's README first.\n")

    with duckdb.connect(":memory:") as connection:
        load_sales(connection)
        original = summary(connection)
        print("category, units, revenue_cents")
        for row in original:
            print(row)
        if args.parquet:
            size, restored = parquet_roundtrip(connection)
            if original != restored:
                raise RuntimeError("Parquet round-trip changed the summary")
            print(f"Parquet round-trip matches; temporary file size: {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
