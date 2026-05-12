"""
Update Document Classification Model

IMPORTANT:
If the new CSV contains a new document type, add it first to the config file:

training:
  trusted_classes:
    - Existing Type
    - New Document Type

This script updates/retrains the model using newly labeled data.
"""

import argparse
import gc
import logging
import os
import pandas as pd

from src.config_loader import load_config
from src.logging_config import setup_logging
from src.model.updater import ModelUpdater

logger = logging.getLogger(__name__)


def validate_csv(csv_path, required_columns):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV file is missing required columns: {missing_columns}"
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Update document classifier with newly labeled data."
    )

    parser.add_argument(
        "--new-data",
        required=True,
        help="Path to CSV with new labeled data. Required columns: 'File Path', 'Document types'.",
    )

    parser.add_argument(
        "--unlabeled-extra",
        required=False,
        help="Optional CSV with extra unlabeled PDFs. Required column: 'File Path'.",
    )

    args = parser.parse_args()

    cfg = load_config()
    setup_logging(log_dir=cfg["paths_resolved"]["logs_dir"])

    validate_csv(
        args.new_data,
        required_columns=["File Path", "Document types"]
    )

    if args.unlabeled_extra:
        validate_csv(
            args.unlabeled_extra,
            required_columns=["File Path"]
        )

    logger.info("Starting model update...")
    logger.info("New labeled data: %s", args.new_data)

    if args.unlabeled_extra:
        logger.info("Extra unlabeled data: %s", args.unlabeled_extra)

    updater = ModelUpdater(cfg=cfg)
    updater.update(
        new_data_csv=args.new_data,
        unlabeled_extra_csv=args.unlabeled_extra
    )

    logger.info("Model update completed successfully.")


def cleanup():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
    except Exception:
        pass

    gc.collect()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()