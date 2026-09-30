import csv
import os
from typing import Dict, Set

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


def read_sources(folder: str, prefix: str) -> Dict[str, pd.DataFrame]:
    return {f"s{k}": read_tsv(os.path.join(folder, f"{prefix}_source{k}.tsv")) for k in (1, 2, 3)}


def read_truth(path: str) -> Dict[str, Set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return {a: {x for x in b.split(",") if x} for a, b in zip(gt.iloc[:, 0], gt.iloc[:, 1])}


def write_id_lists(path: str, header2: str, s1_ids, mapping: Dict[str, Set[str]]):
    """One row per Source 1 id; ids sorted, comma-joined, empty when none."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for e in s1_ids:
            f.write(f"{e}\t{','.join(sorted(mapping.get(e, ())))}\n")


def self_validate(match_path: str, cand_path: str, s1_ids, s2_ids, s3_ids) -> list:
    """Mirror of the official rules (the official utils/validate_submission.py is authoritative)."""
    issues = []
    valid = set(s2_ids) | set(s3_ids)
    s1_set = set(s1_ids)
    tables = {}
    for name, path, col in [("matching", match_path, "matched_entity_ids"),
                            ("candidate", cand_path, "candidate_entity_ids")]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{name}: bad header {list(df.columns)}")
        if df["source1_entity_id"].duplicated().any():
            issues.append(f"{name}: duplicate source1 rows")
        if set(df["source1_entity_id"]) != s1_set:
            issues.append(f"{name}: source1 id coverage mismatch")
        m = {}
        for a, b in zip(df["source1_entity_id"], df[col]):
            ids = [x for x in b.split(",") if x]
            if len(ids) != len(set(ids)):
                issues.append(f"{name}: duplicate ids in list for {a}")
            bad = [x for x in ids if x not in valid]
            if bad:
                issues.append(f"{name}: unknown/invalid ids for {a}: {bad[:3]}")
            m[a] = set(ids)
        tables[name] = m
    for a, ids in tables.get("matching", {}).items():
        if not ids <= tables["candidate"].get(a, set()):
            issues.append(f"matched id not in candidates for {a}")
            break
    return issues
