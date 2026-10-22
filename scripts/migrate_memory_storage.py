"""离线迁移/只读检查记忆结构；不调用LLM、不发送QQ消息。"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import Database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="执行迁移（先停止bot）；默认只读检查")
    args = parser.parse_args()
    path = args.database.resolve(strict=True)
    backup = None
    if args.apply:
        db = Database(path)
        backup = str(db.memory_migration_backup) if db.memory_migration_backup else None
        db.close()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = "memory_records" if "memory_records" in tables else "memories" if "memories" in tables else None
        result = {"storage": table, "legacy_table_present": "memories" in tables, "backup": backup}
        if table:
            result["total"] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if {"layer", "status"} <= columns:
                result["counts"] = [{"layer": layer, "status": status, "count": count}
                                    for layer, status, count in conn.execute(f"SELECT layer,status,count(*) FROM {table} GROUP BY layer,status")]
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
