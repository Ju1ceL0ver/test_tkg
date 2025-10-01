#!/usr/bin/env python3
"""Utility to create or update meta.json for Alpha Temporal (or похожий) датасет."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create meta.json with period and num_relations")
    parser.add_argument("--dataset", default="alpha_temporal", help="Имя директории датасета внутри data/")
    parser.add_argument("--dataset_root", default="./data", help="Корень, где лежат датасеты")
    parser.add_argument("--period", type=int, required=True, help="Шаг временной шкалы")
    parser.add_argument("--num_relations", type=int, required=True, help="Количество отношений")
    parser.add_argument("--force", action="store_true", help="Перезаписать существующий meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.period <= 0:
        raise ValueError("period должен быть положительным")
    if args.num_relations <= 0:
        raise ValueError("num_relations должен быть положительным")

    dataset_dir = Path(args.dataset_root).expanduser() / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "meta.json"
    if meta_path.exists() and not args.force:
        raise FileExistsError(
            f"meta.json уже существует: {meta_path}. Запустите с --force для перезаписи"
        )

    payload = {
        "period": int(args.period),
        "num_relations": int(args.num_relations),
    }

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"meta.json сохранён в {meta_path}")


if __name__ == "__main__":
    main()
