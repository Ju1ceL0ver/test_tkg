#!/usr/bin/env python3
"""Build alpha_temporal-style dataset from synthetic parquet sources."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert synthetic nodes/edges parquet into GenTKG dataset")
    parser.add_argument("--nodes", default="data/synthetic_nodes.parquet", help="Путь к parquet с вершинами")
    parser.add_argument("--edges", default="data/synthetic_edges.parquet", help="Путь к parquet с рёбрами")
    parser.add_argument("--dataset", default="alpha_temporal", help="Имя датасета для папки data/<dataset>")
    parser.add_argument("--data_root", default="./data", help="Корневой каталог для датасетов")
    parser.add_argument("--train_splits", type=int, default=-2,
                        help="Сколько временных точек использовать для train (при значении -2 берутся все кроме двух последних)")
    parser.add_argument("--valid_splits", type=int, default=1,
                        help="Сколько последних временных точек валидации")
    parser.add_argument("--test_splits", type=int, default=1,
                        help="Сколько последних временных точек теста")
    parser.add_argument("--write_meta", action="store_true", help="Создать meta.json с period и num_relations")
    return parser.parse_args()


def normalise_entities(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def build_mappings(edges: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    entities = sorted(set(edges["source"]).union(edges["target"]))
    entity2id = {entity: idx for idx, entity in enumerate(entities)}

    relations = sorted(edges["relation"].unique())
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    timestamps = sorted(edges["timestamp"].unique())
    ts2id = {timestamp: idx for idx, timestamp in enumerate(timestamps)}

    return {
        "entity2id": entity2id,
        "relation2id": relation2id,
        "ts2id": ts2id,
    }


def write_json(path: Path, payload: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_txt(path: Path, rows: Iterable[Sequence[int]], desc: Optional[str] = None) -> None:
    iterator: Iterable[Sequence[int]]
    total = len(rows) if hasattr(rows, "__len__") else None
    iterator = tqdm(rows, total=total, desc=desc) if desc else rows
    with path.open("w", encoding="utf-8") as fh:
        for row in iterator:
            fh.write("\t".join(str(item) for item in row))
            fh.write("\n")


def assign_split(labels: List[str], train_splits: int, valid_splits: int, test_splits: int) -> Dict[str, str]:
    if not labels:
        return {}
    labels_sorted = sorted(labels)

    if train_splits < 0:
        train_cutoff = max(len(labels_sorted) + train_splits - (valid_splits + test_splits), 0)
    else:
        train_cutoff = min(train_splits, len(labels_sorted))

    valid_start = max(train_cutoff, len(labels_sorted) - (valid_splits + test_splits))
    test_start = max(len(labels_sorted) - test_splits, valid_start)

    allocation: Dict[str, str] = {}
    for idx, ts in enumerate(labels_sorted):
        if idx < valid_start:
            allocation[ts] = "train"
        elif idx < test_start:
            allocation[ts] = "valid"
        else:
            allocation[ts] = "test"
    return allocation


def main() -> None:
    args = parse_args()

    nodes_path = Path(args.nodes).expanduser()
    edges_path = Path(args.edges).expanduser()
    if not edges_path.exists():
        raise FileNotFoundError(f"Не найден файл рёбер: {edges_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Не найден файл вершин: {nodes_path}")

    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)

    edges_df = edges_df.copy()
    edges_df["source"] = normalise_entities(edges_df["source"])
    edges_df["target"] = normalise_entities(edges_df["target"])
    edges_df["timestamp"] = pd.to_datetime(edges_df["score_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    edges_df = edges_df.dropna(subset=["timestamp", "source", "target"])
    edges_df["relation"] = edges_df.apply(
        lambda row: f"rel_{int(row['rel_type'])}_is_gendir_{int(row['is_gendir'])}", axis=1
    )

    mappings = build_mappings(edges_df)

    dataset_dir = Path(args.data_root).expanduser() / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    write_json(dataset_dir / "entity2id.json", mappings["entity2id"])
    write_json(dataset_dir / "relation2id.json", mappings["relation2id"])
    write_json(dataset_dir / "ts2id.json", mappings["ts2id"])

    ts_allocation = assign_split(list(mappings["ts2id"].keys()), args.train_splits, args.valid_splits, args.test_splits)

    encoded_rows: List[List[int]] = []
    split_rows = {"train": [], "valid": [], "test": []}

    for row in tqdm(edges_df.itertuples(index=False), total=len(edges_df), desc="Encoding rows"):
        encoded = [
            mappings["entity2id"][row.source],
            mappings["relation2id"][row.relation],
            mappings["entity2id"][row.target],
            mappings["ts2id"][row.timestamp],
        ]
        encoded_rows.append(encoded)
        split = ts_allocation.get(row.timestamp, "train")
        split_rows[split].append(encoded)

    write_txt(dataset_dir / "all_facts.txt", encoded_rows, desc="Writing all_facts.txt")

    for split, rows in split_rows.items():
        write_txt(dataset_dir / f"{split}.txt", rows, desc=f"Writing {split}.txt")

    if args.write_meta:
        meta = {
            "period": 1,
            "num_relations": len(mappings["relation2id"]),
        }
        write_json(dataset_dir / "meta.json", meta)

    print(f"Датасет сохранён в {dataset_dir}")


if __name__ == "__main__":
    main()
