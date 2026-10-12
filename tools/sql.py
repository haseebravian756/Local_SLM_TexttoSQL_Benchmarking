"""Run SQL on one of the BIRD Mini-Dev SQLite databases and print the result as a table.

Usage (run from the Prototyping folder):
    py tools/sql.py <database> "<SQL or command>"

Examples:
    py tools/sql.py superhero ".tables"                     list all tables
    py tools/sql.py superhero ".schema hero_power"          show how a table was created
    py tools/sql.py superhero "SELECT * FROM superpower LIMIT 5"

The database is opened read-only, so nothing you type can change it.
"""
import sqlite3
import sys
import time
from pathlib import Path

DB_ROOT = Path(__file__).resolve().parent.parent / "data" / "minidev" / "dev_databases"
MAX_ROWS = 50   # print at most this many rows
MAX_WIDTH = 40  # cut long cell values to this many characters


def print_table(columns, rows):
    cells = [[str(v)[:MAX_WIDTH] for v in row] for row in rows[:MAX_ROWS]]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(line)
    print("| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |")
    print(line)
    for r in cells:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    print(line)
    if len(rows) > MAX_ROWS:
        print(f"... showing {MAX_ROWS} of {len(rows)} rows")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # some data has non-English characters
    if len(sys.argv) != 3:
        print(__doc__)
        print("Databases:", ", ".join(sorted(p.name for p in DB_ROOT.iterdir())))
        return
    db, sql = sys.argv[1], sys.argv[2].strip()
    path = DB_ROOT / db / f"{db}.sqlite"
    if not path.exists():
        print(f"No database '{db}'. Available:", ", ".join(sorted(p.name for p in DB_ROOT.iterdir())))
        return
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)

    if sql == ".tables":
        sql = "SELECT name AS table_name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    elif sql.startswith(".schema"):
        table = sql.split(maxsplit=1)[1]
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        print(row[0] if row else f"No table '{table}'")
        return

    try:
        t0 = time.perf_counter()
        cur = conn.execute(sql)
        rows = cur.fetchall()
        ms = (time.perf_counter() - t0) * 1000
    except sqlite3.Error as e:
        print("SQL error:", e)
        return
    print_table([d[0] for d in cur.description], rows)
    print(f"{len(rows)} row(s) in {ms:.1f} ms")


if __name__ == "__main__":
    main()
