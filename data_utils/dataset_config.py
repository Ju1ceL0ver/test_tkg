from pathlib import Path
from typing import Optional, Tuple

from basic import read_json


DEFAULT_PERIODS = {
    "icews14": 24,
    "icews18": 24,
}


def _safe_int(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer-like value, got {value!r}") from exc


def _load_meta(dataset_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return None, None

    meta = read_json(meta_path)
    period = _safe_int(meta.get("period"))
    num_relations = _safe_int(meta.get("num_relations"))
    return period, num_relations


def resolve_dataset_params(
    dataset_name: str,
    dataset_root: Path,
    period: Optional[int] = None,
    num_relations: Optional[int] = None,
) -> Tuple[int, int]:
    """Infer dataset-specific parameters used across preprocessing steps.

    Preference order:
        1. Explicit CLI overrides (``period``/``num_relations`` arguments).
        2. ``meta.json`` file located in the dataset directory.
        3. Hard-coded defaults for well-known datasets.
        4. Automatically counted relations from ``relation2id.json``.

    Args:
        dataset_name: Name of the dataset (e.g. ``icews14``).
        dataset_root: Base directory where dataset folders are stored.
        period: Optional manual override for the temporal period.
        num_relations: Optional manual override for the number of relations.

    Returns:
        Tuple of ``(period, num_relations)``.
    """

    dataset_dir = dataset_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    meta_period, meta_num_relations = _load_meta(dataset_dir)

    period = _safe_int(period) if period is not None else meta_period
    num_relations = _safe_int(num_relations) if num_relations is not None else meta_num_relations

    if num_relations is None:
        relation_path = dataset_dir / "relation2id.json"
        if not relation_path.exists():
            raise FileNotFoundError(
                "Unable to infer num_relations. Provide --num_relations or ensure "
                f"{relation_path} exists."
            )
        relations = read_json(relation_path)
        num_relations = len(relations)

    if period is None:
        period = DEFAULT_PERIODS.get(dataset_name.lower(), 1)

    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if num_relations <= 0:
        raise ValueError(f"num_relations must be positive, got {num_relations}")

    return int(period), int(num_relations)
