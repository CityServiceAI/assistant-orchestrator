import csv
from pathlib import Path
from typing import List

from app.deps.litellm_client import client
from .categories import CategoryRecord

EMBEDDING_MODEL = "codemie-text-embedding-ada-002"
CATEGORIES: List[CategoryRecord] = []


def load_categories(path: str = "app/data/categories.csv") -> List[CategoryRecord]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"CSV file not found: {fp}")

    records: List[CategoryRecord] = []

    with fp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        if reader.fieldnames is None:
            raise ValueError("Categories CSV has no header row")

        cleaned_fieldnames = [
            name.lstrip("\ufeff").strip() for name in reader.fieldnames
        ]
        reader.fieldnames = cleaned_fieldnames

        required = [
            "L1 Код",
            "L1 Назва",
            "L2 Код",
            "L2 Назва",
            "L3 Код",
            "L3 Назва (проблема)",
            "Коли обирати цю категорію",
            "Чим відрізняється від схожих категорій",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Categories CSV missing columns: {missing}. Found: {reader.fieldnames}"
            )

        for row in reader:
            records.append(
                CategoryRecord(
                    l1_code=row["L1 Код"].strip(),
                    l1_name=row["L1 Назва"].strip(),
                    l2_code=row["L2 Код"].strip(),
                    l2_name=row["L2 Назва"].strip(),
                    l3_code=row["L3 Код"].strip(),
                    l3_name=row["L3 Назва (проблема)"].strip(),
                    when_to_use=row["Коли обирати цю категорію"].strip(),
                    diff_from_similar=row[
                        "Чим відрізняється від схожих категорій"
                    ].strip(),
                )
            )

    return records


def init_categories(path: str = "app/data/categories.csv") -> None:
    """
    Викликаємо один раз при старті (в main.py).
    Вантажимо всі категорії й рахуємо для них embeddings.
    """
    global CATEGORIES
    CATEGORIES = load_categories(path)

    if not CATEGORIES:
        raise RuntimeError("No categories loaded from CSV")

    texts = [
        f"{c.l3_name}. Коли обирати: {c.when_to_use}. Відмінність: {c.diff_from_similar}"
        for c in CATEGORIES
    ]

    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in resp.data]

    if len(vectors) != len(CATEGORIES):
        raise RuntimeError(
            f"Embeddings count ({len(vectors)}) != categories count ({len(CATEGORIES)})"
        )

    for c, emb in zip(CATEGORIES, vectors):
        c.embedding = emb
