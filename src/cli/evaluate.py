# src/cli/evaluate.py

import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config_loader import load_config
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)


def _load_eval_data(
    cfg,
    eval_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load and process the evaluation CSV into features."""
    paths = cfg["paths_resolved"]
    if eval_csv_path is None:
        eval_csv_path = paths["eval_csv"]

    eval_csv_path = os.path.abspath(eval_csv_path)
    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"Eval CSV not found: {eval_csv_path}")

    logger.info("Loading eval data from %s", eval_csv_path)
    df = pd.read_csv(eval_csv_path)

    if "File Path" not in df.columns or "Document types" not in df.columns:
        raise ValueError("Eval CSV must contain 'File Path' and 'Document types' columns.")

    df["File Path"] = df["File Path"].apply(os.path.normpath)

    # Filter missing files
    exists_mask = df["File Path"].apply(os.path.exists)
    missing = df[~exists_mask]
    if not missing.empty:
        logger.warning("Dropping %d eval rows because file does not exist.", len(missing))
    df = df[exists_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid eval rows after filtering missing files.")

    # Init processors (reuse same settings as training)
    paths = cfg["paths_resolved"]
    PDFProcessor = init_pdf_processor(
        cache_dir=paths["cache_dir"],
        tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
        threshold_ocr=cfg["ocr"]["threshold_ocr"],
    )

    text_processor = TextProcessor(cache_dir=paths["cache_dir"])
    text_processor.load_cache()

    logger.info("Processing eval documents ( OCR + text + visual features )...")
    df["Raw_Text"] = df["File Path"].apply(
        lambda x: PDFProcessor.extract_text(x, cfg["processing"]["version"])
    )
    df["Processed_Text"] = df["Raw_Text"].apply(text_processor.preprocess)
    df["Visual_Features"] = df["File Path"].apply(
        lambda x: PDFProcessor.extract_visual_features(x, cfg["processing"]["version"])
    )

    text_processor.save_cache()

    return df


def _evaluate_single_version(
    cfg,
    version: int,
    df_eval: pd.DataFrame,
    save_metrics: bool = True,
) -> dict:
    """Evaluate a given model version on df_eval and optionally persist metrics."""
    paths = cfg["paths_resolved"]
    base = paths["model_base_name"]
    models_dir = paths["models_dir"]

    model_filename = f"{base}_v{version}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Evaluating model version %d from %s", version, model_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_dict = bundle["label_dict"]
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Ensure eval labels exist in model
    valid_mask = df_eval["Document types"].isin(label_dict.keys())
    dropped = df_eval[~valid_mask]
    if not dropped.empty:
        logger.warning(
            "Dropping %d eval rows with unseen classes: %s",
            len(dropped),
            dropped["Document types"].unique().tolist(),
        )

    df_eval_used = df_eval[valid_mask].reset_index(drop=True)
    if df_eval_used.empty:
        raise ValueError("No eval rows left after filtering for known classes.")

    # Map to numeric labels using model's label_dict
    df_eval_used["Label"] = df_eval_used["Document types"].map(label_dict)

    X = df_eval_used[["Processed_Text", "Visual_Features"]]
    y_true = df_eval_used["Label"].values

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=sorted(inv_label_dict.keys()),
        target_names=[inv_label_dict[i] for i in sorted(inv_label_dict.keys())],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=sorted(inv_label_dict.keys()))

    logger.info("Eval for version %d: Accuracy = %.4f, Macro F1 = %.4f", version, acc, macro_f1)

    # Pretty print report to console
    print(f"\n=== Evaluation for model v{version} ===")
    print(f"Samples used: {len(df_eval_used)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    # Re-print the human-readable classification report
    from sklearn.metrics import classification_report as cr_print

    print(cr_print(
        y_true,
        y_pred,
        labels=sorted(inv_label_dict.keys()),
        target_names=[inv_label_dict[i] for i in sorted(inv_label_dict.keys())],
        zero_division=0,
    ))

    # Metrics dict for saving
    metrics = {
        "version": version,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "n_eval": int(len(df_eval_used)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1": float(v["f1-score"]),
                "support": int(v["support"]),
            }
            for name, v in cls_report.items()
            if name not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {
            "labels": [inv_label_dict[i] for i in sorted(inv_label_dict.keys())],
            "matrix": cm.tolist(),
        },
    }

    if save_metrics:
        metrics_filename = f"{base}_v{version}.metrics.json"
        metrics_path = os.path.join(models_dir, metrics_filename)

        # Merge with existing metrics if present
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        else:
            existing = {}

        existing["version"] = version
        existing["eval"] = metrics

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        logger.info("Saved eval metrics to %s", metrics_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more model versions on a fixed test set."
    )
    parser.add_argument(
        "--version",
        type=int,
        nargs="+",
        help="Model version(s) to evaluate. If omitted, evaluates only the latest version.",
    )
    parser.add_argument(
        "--eval-csv",
        default=None,
        help="Path to eval CSV (default: paths.eval_csv from config).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write metrics JSON files, only print them.",
    )

    args = parser.parse_args()
    cfg = load_config()
    paths = cfg["paths_resolved"]

    vm = VersionManager(paths["version_file"])
    latest = vm.get_last_version()
    if latest <= 0:
        raise RuntimeError("No trained model versions found. Train a model first.")

    if args.version is None:
        versions: List[int] = [latest]
    else:
        versions = args.version

    df_eval = _load_eval_data(cfg, eval_csv_path=args.eval_csv)

    all_metrics = []
    for v in versions:
        metrics = _evaluate_single_version(
            cfg,
            version=v,
            df_eval=df_eval,
            save_metrics=not args.no_save,
        )
        all_metrics.append(metrics)

    # If multiple versions, print a quick comparison table
    if len(all_metrics) > 1:
        print("\n=== Version comparison (eval) ===")
        print("Version | N_eval | Accuracy | Macro-F1")
        print("--------------------------------------")
        for m in all_metrics:
            print(
                f"{m['version']:7d} | {m['n_eval']:6d} | {m['accuracy']:.4f} | {m['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
