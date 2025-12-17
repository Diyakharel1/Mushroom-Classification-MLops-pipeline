"""
ETL • Transform stage for the Mushroom Classification project.
- Cleans, imputes, encodes and expands features for modelling.
- Keeps behaviour equivalent to the previous notebook-driven pipeline,
  but with clearer structure, type hints, and configurable knobs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import zscore
from .redis_conn import redis_connection
import pickle

# Setting up redis store
redis_store = redis_connection()

# ---------------------- paths & logging ------------------------------------- #

# Project root = two levels above this file (…/src/transform.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _build_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in interactive runs
    if not logger.handlers:
        fh = logging.FileHandler(LOGS_DIR / "transform.log")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


logger = _build_logger(__name__)


# ---------------------- core transformer ------------------------------------ #

@dataclass
class MushroomTransformer:
    """
    Data transformer with parameters exposed for easy experimentation.

    Parameters
    ----------
    rare_threshold : int
        Minimum count required for a category to be kept; rarer values become "Other".
    z_thresh : float
        Z-score threshold for outlier filtering on numeric features.
    seed : int
        Random seed for reproducible stochastic imputations.
    drop_cols : Optional[Sequence[str]]
        Columns to drop up-front due to excessive missingness or irrelevance.
        If None, sensible defaults are used.
    """
    rare_threshold: int = 1000
    z_thresh: float = 2.5
    seed: int = 42
    drop_cols: Optional[Sequence[str]] = None

    # internal state (not passed by caller)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.drop_cols is None:
            # defaults mirror the original notebook’s choices
            self.drop_cols = (
                "gill_spacing",
                "stem_surface",
                "stem_root",
                "spore_print_color",
                "veil_type",
                "veil_color",
            )

    # ------------------ public API ------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full transformation pipeline and return an ML-ready dataframe.
        """
        logger.info("Starting mushroom data transformation")
        work = df.copy()

        # 1) Column drops (high missingness)
        work = self._drop_columns(work, self.drop_cols)

        # 2) Target & binary encodings kept as dedicated numeric fields
        work = self._encode_target_and_binaries(work)

        # 3) Light imputations for a few specific categoricals (sampling from observed values)
        for col in ("cap_surface", "gill_attachment", "ring_type"):
            if col in work.columns:
                work[col] = self._sample_impute_categorical(work[col], col_name=col)

        # 4) Consolidate rare categories to "Other" on selected features
        for col in (
            "habitat",
            "stem_color",
            "gill_color",
            "cap_color",
            "cap_shape",
            "cap_surface",
            "ring_type",
        ):
            if col in work.columns:
                work[col] = self._consolidate_rare(work[col], min_count=self.rare_threshold, col_name=col)

        # 5) Drop original target/binary text columns after numeric encodings exist
        for col in ("class", "does_bruise_or_bleed", "has_ring"):
            if col in work.columns:
                work = work.drop(columns=col)

        # 6) Outlier filtering on numeric features via z-score
        work = self._filter_outliers(work, numeric_cols=("cap_diameter", "stem_height", "stem_width"))

        # 7) One-hot encode remaining categoricals
        work = self._one_hot_encode(work)

        # 8) Final tidy-up
        work = work.reset_index(drop=True)
        logger.info("Transformation complete | final shape=%s", work.shape)
        return work

    # ------------------ steps (private) ------------------ #
    @staticmethod
    def _drop_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        existing = [c for c in cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            logger.info("Dropped columns (high missingness): %s", existing)
        return df

    @staticmethod
    def _encode_target_and_binaries(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numeric encodings without mutating original labels first.
        - class_encoded: edible=0, poisonous=1 (gracefully handles e/p or words)
        - does_bruise_or_bleed_encoded: no=0, yes=1
        - has_ring_encoded: no=0, yes=1
        """
        def _bin_map(series: pd.Series, true_vals: set[str], false_vals: set[str], name: str) -> pd.Series:
            s = series.astype(str).str.lower()
            out = pd.Series(np.nan, index=s.index)
            out = out.mask(s.isin(false_vals), 0)
            out = out.mask(s.isin(true_vals), 1)
            if out.isna().any():
                # if ambiguous/missing, impute with mode (bias toward majority)
                mode_val = out.dropna().mode()
                fill_val = int(mode_val.iloc[0]) if not mode_val.empty else 0
                out = out.fillna(fill_val)
                logger.info("Imputed ambiguous values in %s with %s", name, fill_val)
            return out.astype(int)

        if "class" in df.columns:
            # Support e/p or words
            s = df["class"].astype(str).str.lower()
            class_enc = pd.Series(np.where(s.isin({"p", "poison", "poisonous"}), 1, 0), index=df.index)
            df = df.assign(class_encoded=class_enc)

        if "does_bruise_or_bleed" in df.columns:
            df = df.assign(
                does_bruise_or_bleed_encoded=_bin_map(
                    df["does_bruise_or_bleed"],
                    true_vals={"t", "true", "yes", "y", "1"},
                    false_vals={"f", "false", "no", "n", "0"},
                    name="does_bruise_or_bleed_encoded",
                )
            )

        if "has_ring" in df.columns:
            df = df.assign(
                has_ring_encoded=_bin_map(
                    df["has_ring"],
                    true_vals={"t", "true", "yes", "y", "1"},
                    false_vals={"f", "false", "no", "n", "0"},
                    name="has_ring_encoded",
                )
            )
            logger.info("Encoded target and binary flags")
        return df

    def _sample_impute_categorical(self, s: pd.Series, *, col_name: str) -> pd.Series:
        """
        Replace missing/ambiguous entries by sampling from observed values.
        No label encoding round-trip; stays in original string space.
        """
        ser = s.astype("string")
        # Treat literal 'nan', '?', '' as missing
        mask_missing = ser.isna() | (ser.str.strip().isin({"", "nan", "?"}))
        if mask_missing.any():
            candidates = ser[~mask_missing].dropna().to_numpy()
            if candidates.size > 0:
                ser.loc[mask_missing] = self._rng.choice(candidates, size=int(mask_missing.sum()))
                logger.info("Imputed %d missing values in %s by sampling", mask_missing.sum(), col_name)
            else:
                ser.loc[mask_missing] = "Unknown"
                logger.info("Filled %d missing values in %s with 'Unknown'", mask_missing.sum(), col_name)
        return ser

    @staticmethod
    def _consolidate_rare(s: pd.Series, *, min_count: int, col_name: str) -> pd.Series:
        ser = s.astype("string")
        counts = ser.value_counts(dropna=False)
        rare_vals = set(counts[counts < min_count].index.tolist())
        if rare_vals:
            ser = ser.where(~ser.isin(rare_vals), other="Other")
            logger.info("Consolidated rare categories in %s (threshold=%d)", col_name, min_count)
        return ser

    def _filter_outliers(self, df: pd.DataFrame, *, numeric_cols: Sequence[str]) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            if col in df.columns:
                col_z = pd.Series(zscore(df[col].astype(float), nan_policy="omit"), index=df.index)
                keep = col_z.abs() < self.z_thresh
                mask &= keep.fillna(True)  # if zscore is nan, keep the row
                logger.info("Outlier filter applied on %s (|z| < %.2f)", col, self.z_thresh)
        filtered = df.loc[mask]
        logger.info("Outlier removal: kept %d / %d rows", filtered.shape[0], df.shape[0])
        return filtered

    @staticmethod
    def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        # Identify categoricals by dtype=object/string
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            logger.info("One-hot encoded categorical columns: %s", cat_cols)
        return df


# ---------------------- convenience function -------------------------------- #

def transform_data(
                   rare_threshold: int = 1000,
                   z_thresh: float = 2.5,
                   seed: int = 42,
                   drop_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Backwards-compatible convenience wrapper (original public entry point).
    """
    unserialized_df = redis_store.get('df')
    df = pickle.loads(unserialized_df)
    print(df.head())
    tx = MushroomTransformer(
        rare_threshold=rare_threshold,
        z_thresh=z_thresh,
        seed=seed,
        drop_cols=drop_cols,
    )
    transformed_data = tx.transform(df)
    redis_store.set('transformed_df', pickle.dumps(transformed_data))
    return transformed_data.shape