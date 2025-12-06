"""if new csv has new document type do not forget to add new document to the config file 
under training->trusted_classes
THIS IS THE SCRIPT TO UPDATE THE MODEL WITH NEW LABELED DATA
"""

import argparse
import gc
import logging

from src.config_loader import load_config
from src.logging_config import setup_logging
from src.model.updater import ModelUpdater

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Update document classifier (semi-supervised).")
    parser.add_argument(
        "--new-data",
        required=True,
        help="Path to CSV with new labeled data (must have 'File Path' and 'Document types').",
    )
    parser.add_argument(
        "--unlabeled-extra",
        required=False,
        help="Optional path to CSV with extra unlabeled PDFs (must have 'File Path').",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(log_dir=cfg["paths_resolved"]["logs_dir"])

    updater = ModelUpdater(cfg=cfg)
    updater.update(new_data_csv=args.new_data, unlabeled_extra_csv=args.unlabeled_extra)

def cleanup():
    import tkinter as tk
    root = tk.Tk()
    root.destroy()
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
