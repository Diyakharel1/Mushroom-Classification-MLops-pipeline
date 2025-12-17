#!/usr/bin/env python3
"""
Batch prediction script (chunked, memory-safe).

Usage:
  python3 predict.py \
    --in data/raw/secondary_data.csv \
    --out artifacts/predictions.csv \
    --model artifacts/mushroom_pipeline.joblib \
    [--features artifacts/feature_names.json] \
    [--already-transformed] \
    [--chunksize 5000]
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

from src.transform import transform_data

DEF_IN  = "data/raw/secondary_data.csv"
DEF_OUT = "artifacts/predictions.csv"
DEF_MODEL = "artifacts/mushroom_pipeline.joblib"
DEF_FEATS = "artifacts/feature_names.json"
DEF_CHUNK = 5000

def _read_csv_auto(path: str, chunksize: int = None):
    # Secondary Mushroom dataset uses ';' delimiter
    if chunksize:
        return pd.read_csv(path, sep=";", engine="python", chunksize=chunksize)
    return pd.read_csv(path, sep=";", engine="python")

def _align_columns(X: pd.DataFrame, feature_file: Path) -> pd.DataFrame:
    """Align X to the training feature order; create the feature file if missing."""
    if feature_file.exists():
        feat = json.loads(feature_file.read_text())
        missing = [c for c in feat if c not in X.columns]
        for c in missing:
            X[c] = 0
        X = X.reindex(columns=feat, fill_value=0)
    else:
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        feature_file.write_text(json.dumps(list(X.columns)))
        print(f"ℹ️ Saved feature list -> {feature_file}")
    return X

def _predict_chunk(pipe, X: pd.DataFrame) -> pd.DataFrame:
    preds = pipe.predict(X)
    try:
        proba = pipe.predict_proba(X)
        out = pd.DataFrame({"pred": preds})
        if proba is not None and proba.shape[1] == 2:
            out["proba_0"] = proba[:, 0]
            out["proba_1"] = proba[:, 1]
    except Exception:
        out = pd.DataFrame({"pred": preds})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEF_IN, help="Input file: raw CSV or processed parquet/CSV")
    ap.add_argument("--out", dest="out_path", default=DEF_OUT, help="Output CSV for predictions")
    ap.add_argument("--model", dest="model_path", default=DEF_MODEL, help="Joblib model path")
    ap.add_argument("--features", dest="feat_path", default=DEF_FEATS, help="Feature list json path")
    ap.add_argument("--already-transformed", action="store_true",
                    help="Set if input is already transformed with the exact training columns")
    ap.add_argument("--chunksize", type=int, default=DEF_CHUNK, help="CSV rows per chunk (for raw CSV)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    model_path = Path(args.model_path)
    feat_path = Path(args.feat_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Did you save artifacts/mushroom_pipeline.joblib?")

    pipe = joblib.load(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Case A: already-transformed input (prefer smaller files; parquet read is not chunked here)
    if args.already_transformed:
        if in_path.suffix.lower() == ".parquet":
            X = pd.read_parquet(in_path)
        else:
            X = pd.read_csv(in_path)
        X = _align_columns(X, feat_path)
        out = _predict_chunk(pipe, X)
        out.to_csv(out_path, index=False)
        print(f"✅ Wrote predictions -> {out_path}  (rows={len(out)})")
        return

    # Case B: raw CSV → stream in chunks → transform → align → predict → append
    if in_path.suffix.lower() == ".csv":
        reader = _read_csv_auto(str(in_path), chunksize=args.chunksize)
        header_written = False
        total_rows = 0

        for i, chunk in enumerate(reader, 1):
            print(f"Processing chunk {i} (rows={len(chunk)})...")
            df_t = transform_data(chunk)

            # Drop target columns if present
            for tgt in ("class_encoded", "class"):
                if tgt in df_t.columns:
                    df_t = df_t.drop(columns=[tgt])

            df_t = _align_columns(df_t, feat_path)

            out_chunk = _predict_chunk(pipe, df_t)
            out_chunk.to_csv(out_path, index=False, mode="a", header=not header_written)
            header_written = True
            total_rows += len(out_chunk)

        print(f"✅ Wrote predictions -> {out_path}  (rows={total_rows})")
        return

    # Case C: raw parquet (loads fully; if huge, convert to CSV first)
    if in_path.suffix.lower() == ".parquet":
        df_raw = pd.read_parquet(in_path)
        df_t = transform_data(df_raw)
        for tgt in ("class_encoded", "class"):
            if tgt in df_t.columns:
                df_t = df_t.drop(columns=[tgt])
        df_t = _align_columns(df_t, feat_path)
        out = _predict_chunk(pipe, df_t)
        out.to_csv(out_path, index=False)
        print(f"✅ Wrote predictions -> {out_path}  (rows={len(out)})")
        return

    raise ValueError("Input must be .csv or .parquet")

if __name__ == "__main__":
    main()
