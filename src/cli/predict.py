import argparse
import os
import logging
from pathlib import Path

import pandas as pd
import joblib
from datetime import datetime

from src.config_loader import load_config
from src.logging_config import setup_logging
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)
REJECT_THRESHOLD = 0.75

def prepare_features_for_pdfs(pdf_paths, cfg):
    """
    Given a list of PDF paths, run them through PDFProcessor + TextProcessor
    and return a DataFrame with Processed_Text and Visual_Features.
    """
    paths_resolved = cfg["paths_resolved"]
    cache_dir = paths_resolved["cache_dir"]

    PDFProcessor = init_pdf_processor(
        cache_dir=cache_dir,
        tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
        threshold_ocr=cfg["ocr"]["threshold_ocr"],
    )

    text_processor = TextProcessor(cache_dir=cache_dir)
    text_processor.load_cache()

    records = []
    for p in pdf_paths:
        p_norm = os.path.normpath(p)
        if not os.path.exists(p_norm):
            logger.warning("File does not exist, skipping: %s", p_norm)
            continue

        raw_text = PDFProcessor.extract_text(p_norm, cfg["processing"]["version"])
        processed_text = text_processor.preprocess(raw_text)
        visual_features = PDFProcessor.extract_visual_features(p_norm, cfg["processing"]["version"])

        records.append({
            "File Path": p_norm,
            "Processed_Text": processed_text,
            "Visual_Features": visual_features,
        })

    # optional: save updated text cache
    text_processor.save_cache()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)

def collect_pdfs(input_path: str, recursive: bool = False):
    """
    If input_path is a file -> return [that file].
    If it's a directory -> scan for .pdf files.
    """
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return [str(f) for f in p.glob(pattern)]
    raise FileNotFoundError(f"Input path not found: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict document types for PDFs using latest model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If input is a directory, search PDFs recursively.",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
        help="Optional: path to save a CSV report of predictions. "
             "If not provided, a default file under data/predictions/ will be used.",
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg["paths_resolved"]
    setup_logging(log_dir=paths["logs_dir"])

    # Determine latest model version
    vm = VersionManager(paths["version_file"])
    last_version = vm.get_last_version()
    if last_version <= 0:
        raise RuntimeError("No trained model version found. Run training first.")

    model_filename = f"{paths['model_base_name']}_v{last_version}.pkl"
    model_path = os.path.join(paths["models_dir"], model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model bundle from %s (version %d)", model_path, last_version)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_dict = bundle["label_dict"]
    inv_label_dict = {v: k for k, v in label_dict.items()}

    # Collect PDFs
    pdf_paths = collect_pdfs(args.input, recursive=args.recursive)
    if not pdf_paths:
        logger.warning("No PDFs found at input path: %s", args.input)
        return

    logger.info("Found %d PDFs. Preparing features...", len(pdf_paths))
    df_feats = prepare_features_for_pdfs(pdf_paths, cfg)
    if df_feats.empty:
        logger.warning("No valid PDFs to process.")
        return

    X = df_feats[["Processed_Text", "Visual_Features"]]
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        max_proba = proba.max(axis=1)
    else:
        max_proba = [None] * len(preds)

    print("\nPredictions:")

    rows_for_csv = []

    for i, row in df_feats.iterrows():
        file_path = row["File Path"]

        label_idx = preds[i]
        best_label = inv_label_dict.get(label_idx, f"CLASS_{label_idx}")

        conf = max_proba[i] if max_proba is not None else None

        # Decide status + display prediction
        if conf is not None and conf < REJECT_THRESHOLD:
            status = "UNKNOWN"
            display_prediction = "UNKNOWN"
            print(f"- {file_path} -> UNKNOWN (best: {best_label}, conf: {conf:.2f})")
        else:
            status = "OK"
            display_prediction = best_label
            if conf is not None:
                print(f"- {file_path} -> {best_label} (conf: {conf:.2f})")
            else:
                print(f"- {file_path} -> {best_label}")

        rows_for_csv.append({
            "file_path": file_path,
            "status": status,              # "OK" or "UNKNOWN"
            "prediction": display_prediction,
            "best_label": best_label,      # model's top class even if UNKNOWN
            "confidence": conf,
        })

    # ----- Save CSV report -----
    if rows_for_csv:
        df_report = pd.DataFrame(rows_for_csv)

        # Determine output path
        if args.output_csv:
            csv_path = args.output_csv
        else:
            # Default: data/predictions/predictions_YYYYmmdd_HHMMSS.csv
            paths = cfg["paths_resolved"]
            data_dir = paths["data_dir"]
            reports_dir = os.path.join(data_dir, "predictions")
            os.makedirs(reports_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(reports_dir, f"predictions_{timestamp}.csv")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_report.to_csv(csv_path, index=False)
        logger.info("Saved prediction report to %s", csv_path)
        print(f"\nPrediction report saved to: {csv_path}")

if __name__ == "__main__":
    main()
