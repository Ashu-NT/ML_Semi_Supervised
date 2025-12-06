import os
import gc
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config_loader import load_config
from src.logging_config import setup_logging
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.data.cache_manager import DataManager
from src.model.pipeline import build_model_pipeline
from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)

def main():
    cfg = load_config()
    paths = cfg["paths_resolved"]

    paths = cfg["paths_resolved"]
    cache_dir = paths["cache_dir"]
    training_csv = paths["training_csv"]
    models_dir = paths["models_dir"]
    logs_dir = paths["logs_dir"]
    missing_log_path = paths["missing_files_log"]
    model_base_name = paths["model_base_name"]
    version_file = paths["version_file"]

    setup_logging(log_dir=logs_dir)

    os.makedirs(models_dir, exist_ok=True)

    # Init OCR / PDFProcessor with config
    PDFProcessor = init_pdf_processor(
        cache_dir=cache_dir,
        tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
        threshold_ocr=cfg["ocr"]["threshold_ocr"],
    )

    # Components
    text_processor = TextProcessor(cache_dir=cache_dir)
    text_processor.load_cache()
    data_manager = DataManager(
        cache_dir=cache_dir,
        processing_version=cfg["processing"]["version"],
        PDFProcessor=PDFProcessor,
    )

    # Load data
    logger.info(f"Loading training data from {training_csv}")
    df = pd.read_csv(training_csv)
    df["File Path"] = df["File Path"].apply(os.path.normpath)

    # Filter missing files
    exists_mask = df["File Path"].apply(os.path.exists)
    missing = df[~exists_mask]
    if not missing.empty:
        logger.warning(f"Dropping {len(missing)} rows because file does not exist.")
        os.makedirs(os.path.dirname(missing_log_path), exist_ok=True)
        missing.to_csv(missing_log_path, index=False)

    df = df[exists_mask].reset_index(drop=True)

    # Process / cache
    df = data_manager.load_or_process(df, text_processor)
    text_processor.save_cache()
    
    # Drop rows where PDF processing clearly failed
    failed_mask = (df["Raw_Text"] == "") | df["Raw_Text"].isna()
    failed = df[failed_mask]

    if not failed.empty:
        print(f"Dropping {len(failed)} rows where PDF could not be opened or text is empty.")
        failed.to_csv("data/failed_pdfs_log.csv", index=False)

    df = df[~failed_mask].reset_index(drop=True)


    # Labels
    trusted_classes = cfg["training"].get("trusted_classes") or []

    if trusted_classes:
        # Use only trusted classes, in the given order
        df = df[df["Document types"].isin(trusted_classes)].reset_index(drop=True)
        labels = trusted_classes
    else:
        # Fallback: discover from data
        labels = sorted(df["Document types"].unique())

    label_dict = {label: idx for idx, label in enumerate(labels)}
    df["Label"] = df["Document types"].map(label_dict)
    class_names = list(label_dict.keys())
    logger.info(f"Label mapping: {label_dict}")

    # Split
    X = df[["Processed_Text", "Visual_Features"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        stratify=y,
        random_state=cfg["training"]["random_state"],
    )

    # Train
    model = build_model_pipeline()
    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.show()

    # CV
    cv = StratifiedKFold(
        n_splits=cfg["training"]["n_splits_cv"],
        shuffle=True,
        random_state=cfg["training"]["random_state"],
    )
    cv_scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    logger.info(
        f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})"
    )

    # Save model
    vm = VersionManager(version_file)
    last_version = vm.get_last_version()
    new_version = last_version + 1

    model_filename = f"{model_base_name}_v{new_version}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    bundle = {
        "model": model,
        "label_dict": label_dict,
    }
    joblib.dump(bundle, model_path, protocol=4)
    logger.info("Model saved to %s (version %d)", model_path, new_version)
    print(f"\nModel saved to {model_path}")

    vm.set_version(new_version)


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
