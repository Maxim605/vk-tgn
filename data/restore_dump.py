"""
Восстановление дампа dump_2026-03-26T20-29-16 в ArangoDB.

Формат дампа: jsonl.gz (каждая строка — JSON-документ)
Запуск: python learner/data/restore_dump.py

Коллекции из meta.json:
  document: users, comment, groups, migrations, posts
  edge:     subscriptions, interactions, friendships
"""

import gzip
import json
import time
from pathlib import Path
from arango import ArangoClient

DUMP_DIR    = Path(__file__).parent.parent.parent / "dump_2026-03-26T20-29-16"
ARANGO_URL  = "http://localhost:8529"
ARANGO_DB   = "_system"
ARANGO_USER = "root"
ARANGO_PASS = "test"
BATCH_SIZE  = 5000

client = ArangoClient(hosts=ARANGO_URL)
db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)

meta = json.loads((DUMP_DIR / "meta.json").read_text())
collections_meta = {c["name"]: c for c in meta["collections"]}

def ensure_collection(name: str, col_type: str):
    """Создаёт коллекцию если не существует."""
    if db.has_collection(name):
        print(f"  [{name}] уже существует, пропускаем создание")
        return db.collection(name)
    if col_type == "edge":
        col = db.create_collection(name, edge=True)
    else:
        col = db.create_collection(name)
    print(f"  [{name}] создана ({col_type})")
    return col


def import_collection(name: str, col_type: str):
    col = ensure_collection(name, col_type)
    gz_file = DUMP_DIR / f"{name}.jsonl.gz"
    if not gz_file.exists():
        print(f"  [{name}] файл не найден: {gz_file}")
        return

    existing = col.count()
    expected = collections_meta[name]["count"]
    if existing >= expected:
        print(f"  [{name}] уже загружено {existing:,} / {expected:,} — пропускаем")
        return

    print(f"  [{name}] загрузка {expected:,} документов...")
    t0 = time.time()
    batch = []
    total = 0
    errors = 0

    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            batch.append(doc)
            if len(batch) >= BATCH_SIZE:
                try:
                    col.import_bulk(batch, on_duplicate="replace", halt_on_error=False)
                except Exception as e:
                    print(f"    batch error: {e}")
                total += len(batch)
                batch = []
                print(f"    {total:,} / {expected:,}  ({total/expected*100:.1f}%)", end="\r")

    if batch:
        try:
            col.import_bulk(batch, on_duplicate="replace", halt_on_error=False)
        except Exception as e:
            print(f"    batch error: {e}")
        total += len(batch)

    elapsed = time.time() - t0
    print(f"    {total:,} документов за {elapsed:.1f}s  (ошибок парсинга: {errors})")


def main():
    print(f"=== Восстановление дампа ===")
    print(f"Источник: {DUMP_DIR}")
    print(f"База:     {ARANGO_DB} @ {ARANGO_URL}\n")

    # Порядок важен: сначала документы, потом рёбра
    order = ["users", "groups", "posts", "comment", "migrations",
             "subscriptions", "interactions", "friendships"]

    for name in order:
        if name not in collections_meta:
            continue
        col_type = collections_meta[name]["type"]
        print(f"\n[{name}] type={col_type}, ожидается {collections_meta[name]['count']:,}")
        import_collection(name, col_type)

    print("\n=== Итог ===")
    for name in order:
        if not db.has_collection(name):
            continue
        count = db.collection(name).count()
        expected = collections_meta.get(name, {}).get("count", "?")
        status = "✅" if count >= expected else "⚠️"
        print(f"  {status} {name}: {count:,} / {expected:,}")


if __name__ == "__main__":
    main()
