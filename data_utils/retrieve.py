import argparse
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from TLR import Retriever
from basic import read_txt_as_list, read_json, write_txt
from dataset_config import resolve_dataset_params
from id_words import convert_dataset


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--retrieve_type", "-t", default="TLogic-3", type=str)
    parser.add_argument("--name_of_rules_file", "-r", default="", type=str)
    parser.add_argument("--period", type=int, default=None,
                        help="Temporal period override. Defaults to meta.json or built-in presets.")
    parser.add_argument("--num_relations", type=int, default=None,
                        help="Number of relations override. If omitted, relation2id.json is used.")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--output_root", type=str, default="./output",
                        help="Root directory where rule banks are stored.")
    parser.add_argument("--processed_root", type=str, default="./data/processed_new",
                        help="Destination root for processed files.")
    parser.add_argument("--splits", type=str, default="train,valid,test",
                        help="Comma-separated list of dataset splits to process.")
    parsed = vars(parser.parse_args())
    return parsed


def _prepare_rule_path(path_out_tl: Path, name_rules: str) -> Path:
    if name_rules:
        rule_path = path_out_tl / name_rules
        if not rule_path.exists():
            raise FileNotFoundError(f"Specified rule file not found: {rule_path}")
        return rule_path

    rule_candidates = sorted(path_out_tl.glob("*rules.json"))
    if not rule_candidates:
        raise FileNotFoundError(f"No rule files found in {path_out_tl}")
    return rule_candidates[0]


def _iter_splits(splits: str) -> List[str]:
    return [split.strip() for split in splits.split(',') if split.strip()]


if __name__ == "__main__":
    parsed = parser()
    retrieve_type = parsed["retrieve_type"]
    type_dataset = parsed["dataset"]
    name_rules = parsed["name_of_rules_file"]

    data_root = Path(parsed["data_root"]).expanduser()
    dataset_dir = data_root / type_dataset
    output_root = Path(parsed["output_root"]).expanduser()
    path_out_tl = output_root / type_dataset
    processed_root = Path(parsed["processed_root"]).expanduser()
    path_save_root = processed_root / type_dataset

    path_save_root.mkdir(parents=True, exist_ok=True)

    period, num_relations = resolve_dataset_params(
        type_dataset,
        data_root,
        period=parsed["period"],
        num_relations=parsed["num_relations"],
    )

    chains = {}
    rule_path = None
    if retrieve_type.lower() == "bs":
        print("Using bs retrieval without rule bank")
    else:
        rule_path = _prepare_rule_path(path_out_tl, name_rules)
        print(f"Using rules from: {rule_path}")
        chains = read_json(rule_path)

    relations = read_json(dataset_dir / "relation2id.json")
    entities = read_json(dataset_dir / "entity2id.json")
    times_id = read_json(dataset_dir / "ts2id.json")
    rel_keys = list(relations.keys())

    all_facts_path = dataset_dir / "all_facts.txt"
    all_facts_raw = read_txt_as_list(all_facts_path)
    all_facts = convert_dataset(all_facts_raw, dataset_dir, period=period)

    splits = _iter_splits(parsed["splits"])

    for split in tqdm(splits, desc="Splits"):
        print(f"Processing split: {split}")
        split_file = dataset_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Skipping {split}: missing file {split_file}")
            continue
        test_ans_raw = read_txt_as_list(split_file)
        test_ans = convert_dataset(test_ans_raw, dataset_dir, period=period)

        rtr = Retriever(
            test_ans,
            all_facts,
            entities,
            relations,
            times_id,
            num_relations,
            chains,
            rel_keys,
            dataset=type_dataset,
            retrieve_type=retrieve_type,
            period=period,
        )
        test_idx, test_text = rtr.get_output()

        history_dir = path_save_root / split / "history_facts"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_stem = history_dir / f"history_facts_{type_dataset}"
        path_file_word = history_stem.with_suffix(".txt")
        path_file_id = history_dir / f"{history_stem.name}_idx_fine_tune_all.txt"

        write_txt(path_file_id, test_text, desc=f"Writing {split} history (idx)")
        with open(path_file_word, "w", encoding="utf-8") as f:
            for record in tqdm(test_text, desc=f"Writing {split} history (words)"):
                f.write((record[0] if record else "") + "\n")
        print(f"Saved history facts to {path_file_word} and {path_file_id}")

        answer_dir = path_save_root / split / "test_answers"
        answer_dir.mkdir(parents=True, exist_ok=True)
        path_answer = answer_dir / f"test_answers_{type_dataset}.txt"
        write_txt(path_answer, test_ans, head="", desc=f"Writing {split} answers")
        print(f"Saved answers to {path_answer}")
