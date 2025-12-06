import os
import logging
from typing import Optional, List

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from src.config_loader import load_config
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.data.cache_manager import DataManager
from src.model.pipeline import build_model_pipeline

from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)


class ModelUpdater:
    """
    Semi-supervised model updater.

    - Loads existing cached processed data (from DataManager cache).
    - Processes new labeled data and merges it.
    - Uses trusted classes for supervised training.
    - Uses 'Delivery Documentation' + optional extra unlabeled PDFs as unlabeled pool.
    - Pseudo-labels unlabeled pool with high confidence.
    - Retrains model on labeled + pseudo-labeled data.
    - Overwrites the current model file.
    """

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.paths = cfg["paths_resolved"]
        
        # Config bits
        self.models_dir = self.paths["models_dir"]
        self.model_base_name = self.paths["model_base_name"]
        self.version_file = self.paths["version_file"]
        self.vm = VersionManager(self.version_file)
        self.last_version = self.vm.get_last_version()
        self.new_version = self.last_version + 1

        self.cache_dir = self.paths["cache_dir"]
        self.processing_version = cfg["processing"]["version"]
        self.trusted_classes: List[str] = cfg["training"]["trusted_classes"] or []
        self.test_size = cfg["training"]["test_size"]
        self.random_state = cfg["training"]["random_state"]
        self.n_splits_cv = cfg["training"]["n_splits_cv"]

        self.conf_threshold = cfg["semi_supervised"]["confidence_threshold"]
        self.max_per_class = cfg["semi_supervised"]["max_per_class"]

        # Instantiate processors
        PDFProcessor = init_pdf_processor(
            cache_dir=self.cache_dir,
            tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
            threshold_ocr=cfg["ocr"]["threshold_ocr"],
        )

        self.PDFProcessor = PDFProcessor
        self.text_processor = TextProcessor(cache_dir=self.cache_dir)
        self.text_processor.load_cache()
        self.data_manager = DataManager(
            cache_dir=self.cache_dir,
            processing_version=self.processing_version,
            PDFProcessor=self.PDFProcessor,
        )

        # Try to load existing model, if any
        self.model = self._load_existing_model()

        # Load previous cached processed data (if exists)
        self.previous_cache = self._load_previous_cache()

        # Will be set when we build labels
        self.label_dict = None

    # ------------------------------------------------------------------ utils

    def _load_existing_model(self):
        if self.last_version <= 0:
            logger.warning("No previous model version found (last_version=%d).", self.last_version)
            return None

        model_filename = f"{self.model_base_name}_v{self.last_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        if os.path.exists(model_path):
            logger.info("Loading existing model from %s (version %d)", model_path, self.last_version)
            bundle = joblib.load(model_path)
            # Also set label_dict if saved
            self.label_dict = bundle.get("label_dict")
            return bundle["model"]

        logger.warning("Expected previous model at %s but file does not exist.", model_path)
        return None


    def _load_previous_cache(self) -> pd.DataFrame:
        """
        Use DataManager's cache file if it exists.
        """
        cache_file = os.path.join(self.cache_dir, "processed_data.joblib")
        if os.path.exists(cache_file):
            logger.info(f"Loading previous processed data cache from {cache_file}")
            cached = joblib.load(cache_file)
            return cached["data_frame"]
        logger.info("No previous processed data cache found, starting from empty.")
        return pd.DataFrame()

    @staticmethod
    def _filter_existing_files(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
        df = df.copy()
        df["File Path"] = df["File Path"].apply(os.path.normpath)
        exists_mask = df["File Path"].apply(os.path.exists)
        missing = df[~exists_mask]
        if not missing.empty:
            logger.warning(f"[{label_name}] Dropping {len(missing)} rows because file does not exist.")
        return df[exists_mask].reset_index(drop=True)

    # ----------------------------------------------------------------- process

    def process_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge new labeled data with previous cache.
        Only processes new rows whose File Path wasn't seen before.
        """
        if "File Path" not in new_data.columns:
            raise ValueError("New data must contain a 'File Path' column.")
        if "Document types" not in new_data.columns:
            raise ValueError("New data must contain a 'Document types' column.")

        new_data = self._filter_existing_files(new_data, label_name="New-data")

        # Deduplicate by File Path vs previous cache
        if not self.previous_cache.empty and "File Path" in self.previous_cache.columns:
            new_data = new_data[
                ~new_data["File Path"].isin(self.previous_cache["File Path"])
            ]

        if new_data.empty:
            logger.info("No new data to process; using previous cache only.")
            return self.previous_cache

        logger.info("Processing new documents...")
        logger.info("New data class distribution:\n%s", new_data["Document types"].value_counts())

        new_data["Raw_Text"] = new_data["File Path"].apply(
            lambda x: self.PDFProcessor.extract_text(x, self.processing_version)
        )
        new_data["Processed_Text"] = new_data["Raw_Text"].apply(
            self.text_processor.preprocess
        )
        new_data["Visual_Features"] = new_data["File Path"].apply(
            lambda x: self.PDFProcessor.extract_visual_features(x, self.processing_version)
        )
        
        # drop rows where processing failed
        new_data = self._drop_failed_rows(new_data, label_name="New-data")

        if new_data.empty:
            logger.info("All new rows failed processing; nothing to add.")
            return self.previous_cache

        # Merge with previous cache
        updated_data = pd.concat([self.previous_cache, new_data], ignore_index=True)

        # Update DataManager's cache file
        cache_data = {
            "data_frame": updated_data,
            "file_signatures": updated_data["File Path"].apply(
                self.PDFProcessor.file_signature
            ).values,
            "version": self.processing_version,
        }
        cache_file = os.path.join(self.cache_dir, "processed_data.joblib")
        joblib.dump(cache_data, cache_file)
        logger.info(f"Updated processed data cache saved to {cache_file}")

        # Persist text preprocessor cache also
        self.text_processor.save_cache()

        return updated_data

    # ------------------------------------------------------- pseudo-labelling

    def _pseudo_label_unlabeled(self, unlabeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Use current self.model to pseudo-label unlabeled data.
        Returns a DataFrame with Processed_Text, Visual_Features, Label, Document types.
        """
        if unlabeled_df.empty:
            logger.info("[Pseudo] No unlabeled data to pseudo-label.")
            return pd.DataFrame()

        if self.model is None:
            logger.warning("[Pseudo] No model available; cannot pseudo-label.")
            return pd.DataFrame()

        X_unlabeled = unlabeled_df[["Processed_Text", "Visual_Features"]]
        proba = self.model.predict_proba(X_unlabeled)
        preds = self.model.predict(X_unlabeled)
        max_conf = proba.max(axis=1)

        high_conf_mask = max_conf >= self.conf_threshold
        high_conf_df = unlabeled_df[high_conf_mask].copy()

        if high_conf_df.empty:
            logger.info("[Pseudo] No samples above confidence %.2f.", self.conf_threshold)
            return pd.DataFrame()

        high_conf_df["Label"] = preds[high_conf_mask]

        # reverse label map
        inv_label_dict = {v: k for k, v in self.label_dict.items()}
        high_conf_df["Document types"] = high_conf_df["Label"].map(inv_label_dict)

        # cap per class
        balanced_dfs = []
        for label_id in np.unique(high_conf_df["Label"]):
            class_rows = high_conf_df[high_conf_df["Label"] == label_id]
            if len(class_rows) > self.max_per_class:
                class_rows = class_rows.sample(self.max_per_class, random_state=self.random_state)
            balanced_dfs.append(class_rows)

        pseudo_df = pd.concat(balanced_dfs).reset_index(drop=True)
        logger.info("[Pseudo] Selected %d pseudo-labeled samples.", len(pseudo_df))
        logger.info("[Pseudo] Class distribution:\n%s", pseudo_df["Document types"].value_counts())
        return pseudo_df

    def _process_extra_unlabeled(self, unlabeled_csv_path: Optional[str]) -> pd.DataFrame:
        """Optional external unlabeled CSV with at least 'File Path' column."""
        if unlabeled_csv_path is None:
            return pd.DataFrame()

        if not os.path.exists(unlabeled_csv_path):
            logger.warning("[Unlabeled-extra] File not found: %s", unlabeled_csv_path)
            return pd.DataFrame()

        df_u = pd.read_csv(unlabeled_csv_path)
        if "File Path" not in df_u.columns:
            logger.warning("[Unlabeled-extra] CSV missing 'File Path' column.")
            return pd.DataFrame()

        df_u = self._filter_existing_files(df_u, label_name="Unlabeled-extra")
        if df_u.empty:
            logger.info("[Unlabeled-extra] No existing files to process.")
            return df_u

        logger.info("[Unlabeled-extra] Processing %d documents...", len(df_u))
        df_u["Raw_Text"] = df_u["File Path"].apply(
            lambda x: self.PDFProcessor.extract_text(x, self.processing_version)
        )
        df_u["Processed_Text"] = df_u["Raw_Text"].apply(self.text_processor.preprocess)
        df_u["Visual_Features"] = df_u["File Path"].apply(
            lambda x: self.PDFProcessor.extract_visual_features(x, self.processing_version)
        )
        return df_u
    
    @staticmethod
    def _drop_failed_rows(df, label_name="data"):
        """
        Drop rows where PDF processing clearly failed:
        - Raw_Text is empty or NaN
        - OR Visual_Features look like all zeros
        """
        df = df.copy()

        # Raw text missing/empty
        mask_bad_text = (df["Raw_Text"] == "") | df["Raw_Text"].isna()

        # Visual features all zeros (list, tuple, or np.array of zeros)
        def is_all_zero(v):
            if v is None:
                return True
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.array(v).astype(float)
                return np.all(arr == 0)
            return False

        mask_bad_vis = df["Visual_Features"].apply(is_all_zero)

        bad_mask = mask_bad_text | mask_bad_vis
        bad_rows = df[bad_mask]

        if not bad_rows.empty:
            logger.warning(
                "[%s] Dropping %d rows where PDF processing failed (empty text / zero features).",
                label_name, len(bad_rows)
            )
            # Optional: save for later inspection
            out_path = f"data/failed_pdfs_{label_name}.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            bad_rows.to_csv(out_path, index=False)
            logger.warning("[%s] Saved failed PDF rows to %s", label_name, out_path)

        return df[~bad_mask].reset_index(drop=True)

    # ---------------------------------------------------------- retraining

    def retrain_model(self, df: pd.DataFrame, unlabeled_extra_path: Optional[str] = None):
        """
        Retrain the model with updated data, using semi-supervision:
        - Supervised on trusted classes.
        - Pseudo-labeled on 'Delivery Documentation' + optional extra unlabeled PDFs.
        - Overwrites model at self.model_path.
        """
        df = df.copy()
        df = self._filter_existing_files(df, label_name="All-data")

        # Split into trusted vs default/fallback
        if self.trusted_classes:
            df_trusted = df[df["Document types"].isin(self.trusted_classes)].reset_index(drop=True)
        else:
            # If no trusted classes configured, treat all as trusted
            df_trusted = df.reset_index(drop=True)

        df_default = df[df["Document types"] == "Delivery Documentation"].reset_index(drop=True)

        logger.info("[Train] Trusted class distribution:\n%s",
                    df_trusted["Document types"].value_counts())
        logger.info("[Train] Default/fallback 'Delivery Documentation' rows: %d", len(df_default))

        # Label mapping for trusted classes only
        if self.trusted_classes:
            self.label_dict = {label: idx for idx, label in enumerate(self.trusted_classes)}
        else:
            class_order = sorted(df_trusted["Document types"].unique().tolist())
            self.label_dict = {label: idx for idx, label in enumerate(class_order)}

        df_trusted["Label"] = df_trusted["Document types"].map(self.label_dict)
        class_names = list(self.label_dict.keys())

        # Supervised training on trusted data only
        X = df_trusted[["Processed_Text", "Visual_Features"]]
        y = df_trusted["Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        self.model = build_model_pipeline()
        logger.info("Retraining model (supervised) on trusted classes...")
        logger.info("Training label distribution:\n%s", y_train.value_counts())
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Supervised Model Accuracy: %.2f%%", acc * 100)
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
        plt.title("Supervised Confusion Matrix (trusted classes)")
        plt.show()

        cv = StratifiedKFold(
            n_splits=self.n_splits_cv,
            shuffle=True,
            random_state=self.random_state,
        )
        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        logger.info(
            "Supervised CV Accuracy: %.2f%% (±%.2f%%)",
            cv_scores.mean() * 100,
            cv_scores.std() * 100,
        )

        # Semi-supervised augmentation
        unlabeled_parts = []

        if not df_default.empty:
            logger.info("[Semi] Using 'Delivery Documentation' rows as unlabeled pool.")
            unlabeled_parts.append(df_default)

        extra_unlabeled = self._process_extra_unlabeled(unlabeled_extra_path)
        if not extra_unlabeled.empty:
            unlabeled_parts.append(extra_unlabeled)

        if not unlabeled_parts:
            logger.info("[Semi] No unlabeled data available; keeping supervised model.")
            joblib.dump(self.model, self.model_path, protocol=4)
            logger.info("Saved supervised model to %s", self.model_path)
            return

        unlabeled_df = pd.concat(unlabeled_parts, ignore_index=True)
        pseudo_df = self._pseudo_label_unlabeled(unlabeled_df)

        if pseudo_df.empty:
            logger.info("[Semi] No pseudo-labeled samples; keeping supervised model.")
            joblib.dump(self.model, self.model_path, protocol=4)
            logger.info("Saved supervised model to %s", self.model_path)
            return

        # Retrain on labeled + pseudo-labeled
        X_train_aug = pd.concat(
            [X_train, pseudo_df[["Processed_Text", "Visual_Features"]]]
        )
        y_train_aug = pd.concat([y_train, pseudo_df["Label"]])

        semi_model = build_model_pipeline()
        logger.info("[Semi] Retraining model on labeled + pseudo-labeled data...")
        semi_model.fit(X_train_aug, y_train_aug)

        # Evaluate on held-out labeled test set
        y_pred_semi = semi_model.predict(X_test)
        acc_semi = accuracy_score(y_test, y_pred_semi)
        logger.info("Semi-supervised Model Accuracy: %.2f%%", acc_semi * 100)
        print(classification_report(y_test, y_pred_semi, target_names=class_names))

        cm_semi = confusion_matrix(y_test, y_pred_semi)
        sns.heatmap(
            cm_semi,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Semi-supervised Confusion Matrix (trusted classes)")
        plt.show()

        cv_scores_semi = cross_val_score(
            semi_model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        logger.info(
            "Semi-supervised CV Accuracy: %.2f%% (±%.2f%%)",
            cv_scores_semi.mean() * 100,
            cv_scores_semi.std() * 100,
        )

        # Save semi-supervised model
        self.model = semi_model
        model_filename = f"{self.model_base_name}_v{self.new_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        bundle = {
            "model": self.model,         # or semi_model
            "label_dict": self.label_dict,
        }
        joblib.dump(bundle, model_path, protocol=4)
        logger.info("Saved model to %s (version %d)", model_path, self.new_version)
        self.vm.set_version(self.new_version)

    # --------------------------------------------------------------- public API

    def update(self, new_data_csv: str, unlabeled_extra_csv: Optional[str] = None):
        """
        Main function:
        - Read new labeled CSV.
        - Process & merge with previous cache.
        - Retrain model (semi-supervised).
        """
        if not os.path.exists(new_data_csv):
            raise FileNotFoundError(f"New data CSV not found: {new_data_csv}")

        logger.info("Loading new labeled data from %s", new_data_csv)
        new_data = pd.read_csv(new_data_csv)
        updated_data = self.process_new_data(new_data)

        if updated_data.empty:
            logger.info("Updated data is empty; no model update performed.")
            return

        self.retrain_model(updated_data, unlabeled_extra_path=unlabeled_extra_csv)
