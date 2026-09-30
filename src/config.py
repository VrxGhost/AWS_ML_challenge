from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    # ---- data -----------------------------------------------------------
    data_dir: str = "dataset"
    work_dir: str = "work"
    out_dir: str = "output"
    # Train on a subsample of Source 1 entities (None = all). Sampling keeps a
    # consistent sub-universe: sampled S1 + their true matches + the same
    # fraction of unmatched S2/S3 records.
    train_sample_s1: Optional[int] = 400_000
    seed: int = 42
    n_jobs: int = 4

    # ---- normalisation (normalization.py) --------------------------------
    # A fitted state from `python src/run_normalization.py fit` (work/norm_state.json).
    # If it exists, train() reuses it instead of fitting again (saves ~minutes).
    norm_state: str = "work/norm_state.json"
    mine_abbrev_min_count: int = 25
    use_cache: bool = True             # cache normalised frames in work/cache/ (re-runs skip normalising)
    cache_mode: str = "full"           # "low" = small word caches per worker (less RAM, a bit slower)

    # ---- blocking -------------------------------------------------------
    use_country_in_keys: bool = True
    max_block_target: int = 300        # drop keys shared by more target records than this
    max_block_pairs: int = 20_000      # drop keys whose S1 x target block exceeds this
    block_chunk: int = 200_000         # S1 rows per join chunk (memory knob)
    topk_joint: int = 20              # keep top-K by cheap joint score per (entity, source)
    topk_name: int = 5                 # ... plus top-K by name cosine
    topk_addr: int = 5                 # ... plus top-K by address cosine
    min_cheap: float = 0.10
    ref_records: int = 0               # records in the TRAINING universe (set by training)

    # ---- model ----------------------------------------------------------
    holdout_frac: float = 0.2          # entity-level validation split
    n_folds: int = 3                   # out-of-fold stage-1 scores for context features
    neg_keep: float = 1.0              # negative down-sampling for model fitting
    use_context: bool = True           # two-stage model with rank / reverse-rank features

    def to_dict(self):
        return asdict(self)
