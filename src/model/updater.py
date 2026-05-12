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
    """

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        self.paths = cfg["paths_resolved"]

        self.models_dir = self.paths["models_dir"]
        self.model_base_name = self.paths["model_base_name"]
        self.version_file = self.paths["version_file"]

        self.vm = VersionManager(self.version_file)
        self.last_version = self.vm.get_last_version()
        self.new_version = self.last_version + 1

        self.cache_dir = self.paths["cache_dir"]
        self.processing_version = cfg["processing"]["version"]

        self.trusted_classes: List[str] = cfg["training"].get("trusted_classes") or []
        self.test_size = cfg["training"]["test_size"]
        self.random_state = cfg["training"]["random_state"]
        self.n_splits_cv = cfg["training"]["n_splits_cv"]

        self.conf_threshold = cfg["semi_supervised"]["confidence_threshold"]
        self.max_per_class = cfg["semi_supervised"]["max_per_class"]

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

        self.label_dict = None
        self.model = self._load_existing_model()
        self.previous_cache = self._load_previous_cache()

    # Loading / Saving

    def _load_existing_model(self):
        if self.last_version <= 0:
            logger.warning(
                "No previous model version found. last_version=%d",
                self.last_version,
            )
            return None

        model_filename = f"{self.model_base_name}_v{self.last_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        if not os.path.exists(model_path):
            logger.warning("Previous model file not found: %s", model_path)
            return None

        logger.info(
            "Loading existing model from %s, version %d",
            model_path,
            self.last_version,
        )

        bundle = joblib.load(model_path)

        self.label_dict = bundle.get("label_dict")

        return bundle["model"]

    def _save_model(self, model, accuracy=None, cv_mean=None, cv_std=None):
        """
        Save model as a versioned bundle.
        This keeps prediction.py compatible.
        """
        os.makedirs(self.models_dir, exist_ok=True)

        model_filename = f"{self.model_base_name}_v{self.new_version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        bundle = {
            "model": model,
            "label_dict": self.label_dict,
            "model_version": self.new_version,
            "previous_version": self.last_version,
            "trusted_classes": self.trusted_classes,
            "pipeline_type": "TFIDF_LogisticRegression",
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

        joblib.dump(bundle, model_path, protocol=4)

        self.vm.set_version(self.new_version)

        logger.info(
            "Saved model to %s, version %d",
            model_path,
            self.new_version,
        )

        print(f"\nModel saved to: {model_path}")

    def _load_previous_cache(self) -> pd.DataFrame:
        cache_file = os.path.join(self.cache_dir, "processed_data.joblib")

        if os.path.exists(cache_file):
            logger.info("Loading previous processed data cache from %s", cache_file)
            cached = joblib.load(cache_file)
            return cached["data_frame"]

        logger.info("No previous processed data cache found.")
        return pd.DataFrame()

    # Helpers

    @staticmethod
    def _filter_existing_files(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
        df = df.copy()

        if "File Path" not in df.columns:
            raise ValueError(f"[{label_name}] Missing required column: File Path")

        df["File Path"] = df["File Path"].apply(os.path.normpath)

        exists_mask = df["File Path"].apply(os.path.exists)
        missing = df[~exists_mask]

        if not missing.empty:
            logger.warning(
                "[%s] Dropping %d rows because files do not exist.",
                label_name,
                len(missing),
            )

        return df[exists_mask].reset_index(drop=True)

    @staticmethod
    def _drop_failed_rows(df: pd.DataFrame, label_name="data") -> pd.DataFrame:
        df = df.copy()

        mask_bad_text = df["Raw_Text"].isna() | (df["Raw_Text"].astype(str).str.strip() == "")

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
                "[%s] Dropping %d rows because PDF processing failed.",
                label_name,
                len(bad_rows),
            )

            out_path = f"data/failed_pdfs_{label_name}.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            bad_rows.to_csv(out_path, index=False)

            logger.warning("[%s] Failed rows saved to %s", label_name, out_path)

        return df[~bad_mask].reset_index(drop=True)

    # Processing

    def process_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        if "File Path" not in new_data.columns:
            raise ValueError("New data must contain 'File Path' column.")

        if "Document types" not in new_data.columns:
            raise ValueError("New data must contain 'Document types' column.")

        initial_count = len(new_data)
        new_data = self._filter_existing_files(new_data, label_name="New-data")

        if not self.previous_cache.empty and "File Path" in self.previous_cache.columns:
            new_data["File Path"] = new_data["File Path"].apply(os.path.normpath)
            self.previous_cache["File Path"] = self.previous_cache["File Path"].apply(os.path.normpath)

            in_cache_mask = new_data["File Path"].isin(self.previous_cache["File Path"])

            logger.info(
                "New-data rows: %d | Already in cache: %d | To process: %d",
                initial_count,
                int(in_cache_mask.sum()),
                int((~in_cache_mask).sum()),
            )

            new_data = new_data[~in_cache_mask].reset_index(drop=True)

        if new_data.empty:
            logger.info("No new data to process. Using previous cache only.")
            return self.previous_cache

        logger.info("Processing new labeled documents...")
        logger.info(
            "New data class distribution:\n%s",
            new_data["Document types"].value_counts(),
        )

        new_data["Raw_Text"] = new_data["File Path"].apply(
            lambda x: self.PDFProcessor.extract_text(x, self.processing_version)
        )

        new_data["Processed_Text"] = new_data["Raw_Text"].apply(
            self.text_processor.preprocess
        )

        new_data["Visual_Features"] = new_data["File Path"].apply(
            lambda x: self.PDFProcessor.extract_visual_features(
                x,
                self.processing_version,
            )
        )

        new_data = self._drop_failed_rows(new_data, label_name="New-data")

        if new_data.empty:
            logger.info("All new rows failed processing. Nothing to add.")
            return self.previous_cache

        updated_data = pd.concat(
            [self.previous_cache, new_data],
            ignore_index=True,
        )

        cache_data = {
            "data_frame": updated_data,
            "file_signatures": updated_data["File Path"].apply(
                self.PDFProcessor.file_signature
            ).values,
            "version": self.processing_version,
        }

        cache_file = os.path.join(self.cache_dir, "processed_data.joblib")
        joblib.dump(cache_data, cache_file)

        logger.info("Updated processed data cache saved to %s", cache_file)

        self.text_processor.save_cache()

        return updated_data

    def _process_extra_unlabeled(self, unlabeled_csv_path: Optional[str]) -> pd.DataFrame:
        if unlabeled_csv_path is None:
            return pd.DataFrame()

        if not os.path.exists(unlabeled_csv_path):
            logger.warning("[Unlabeled-extra] File not found: %s", unlabeled_csv_path)
            return pd.DataFrame()

        df_u = pd.read_csv(unlabeled_csv_path)

        if "File Path" not in df_u.columns:
            logger.warning("[Unlabeled-extra] Missing 'File Path' column.")
            return pd.DataFrame()

        df_u = self._filter_existing_files(df_u, label_name="Unlabeled-extra")

        if df_u.empty:
            logger.info("[Unlabeled-extra] No existing files to process.")
            return df_u

        logger.info("[Unlabeled-extra] Processing %d documents...", len(df_u))

        df_u["Raw_Text"] = df_u["File Path"].apply(
            lambda x: self.PDFProcessor.extract_text(x, self.processing_version)
        )

        df_u["Processed_Text"] = df_u["Raw_Text"].apply(
            self.text_processor.preprocess
        )

        df_u["Visual_Features"] = df_u["File Path"].apply(
            lambda x: self.PDFProcessor.extract_visual_features(
                x,
                self.processing_version,
            )
        )

        df_u = self._drop_failed_rows(df_u, label_name="Unlabeled-extra")

        return df_u

    # Pseudo-labeling

    def _pseudo_label_unlabeled(self, unlabeled_df: pd.DataFrame) -> pd.DataFrame:
        if unlabeled_df.empty:
            logger.info("[Pseudo] No unlabeled data.")
            return pd.DataFrame()

        if self.model is None:
            logger.warning("[Pseudo] No existing model available.")
            return pd.DataFrame()

        if self.label_dict is None:
            logger.warning("[Pseudo] No label dictionary available.")
            return pd.DataFrame()

        if not hasattr(self.model, "predict_proba"):
            logger.warning(
                "[Pseudo] Current model does not support predict_proba. "
                "Skipping pseudo-labeling."
            )
            return pd.DataFrame()

        X_unlabeled = unlabeled_df[["Processed_Text", "Visual_Features"]]

        proba = self.model.predict_proba(X_unlabeled)
        preds = self.model.predict(X_unlabeled)
        max_conf = proba.max(axis=1)

        high_conf_mask = max_conf >= self.conf_threshold
        high_conf_df = unlabeled_df[high_conf_mask].copy()

        if high_conf_df.empty:
            logger.info(
                "[Pseudo] No samples above confidence threshold %.2f.",
                self.conf_threshold,
            )
            return pd.DataFrame()

        high_conf_df["Label"] = preds[high_conf_mask]

        inv_label_dict = {v: k for k, v in self.label_dict.items()}
        high_conf_df["Document types"] = high_conf_df["Label"].map(inv_label_dict)
        high_conf_df["Pseudo_Confidence"] = max_conf[high_conf_mask]

        balanced_dfs = []

        for label_id in np.unique(high_conf_df["Label"]):
            class_rows = high_conf_df[high_conf_df["Label"] == label_id]

            if len(class_rows) > self.max_per_class:
                class_rows = class_rows.sample(
                    self.max_per_class,
                    random_state=self.random_state,
                )

            balanced_dfs.append(class_rows)

        pseudo_df = pd.concat(balanced_dfs, ignore_index=True)

        logger.info("[Pseudo] Selected %d pseudo-labeled samples.", len(pseudo_df))
        logger.info(
            "[Pseudo] Class distribution:\n%s",
            pseudo_df["Document types"].value_counts(),
        )

        return pseudo_df

    # Training

    def retrain_model(
        self,
        df: pd.DataFrame,
        unlabeled_extra_path: Optional[str] = None,
    ):
        df = df.copy()
        df = self._filter_existing_files(df, label_name="All-data")

        if self.trusted_classes:
            df_trusted = df[df["Document types"].isin(self.trusted_classes)].reset_index(drop=True)
        else:
            df_trusted = df.reset_index(drop=True)

        if df_trusted.empty:
            logger.warning("[Train] No trusted data available. Training aborted.")
            return

        df_default = df[df["Document types"] == "Delivery Documentation"].reset_index(drop=True)

        logger.info(
            "[Train] Trusted class distribution:\n%s",
            df_trusted["Document types"].value_counts(),
        )

        logger.info(
            "[Train] Default/fallback 'Delivery Documentation' rows: %d",
            len(df_default),
        )

        if self.trusted_classes:
            self.label_dict = {
                label: idx for idx, label in enumerate(self.trusted_classes)
            }
        else:
            class_order = sorted(df_trusted["Document types"].unique().tolist())
            self.label_dict = {
                label: idx for idx, label in enumerate(class_order)
            }

        df_trusted["Label"] = df_trusted["Document types"].map(self.label_dict)

        df_trusted = df_trusted.dropna(subset=["Label"]).reset_index(drop=True)
        df_trusted["Label"] = df_trusted["Label"].astype(int)

        class_names = list(self.label_dict.keys())

        X = df_trusted[["Processed_Text", "Visual_Features"]]
        y = df_trusted["Label"]

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            logger.warning("[Train] Need at least 2 classes to train. Training aborted.")
            return

        min_class_count = class_counts.min()

        if min_class_count < 2:
            logger.warning(
                "[Train] At least one class has fewer than 2 samples. "
                "Stratified split may fail. Training aborted."
            )
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        self.model = build_model_pipeline()

        logger.info("[Train] Retraining supervised model...")
        logger.info("[Train] Training label distribution:\n%s", y_train.value_counts())

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
        plt.title("Supervised Confusion Matrix")
        plt.show()

        cv_mean = None
        cv_std = None

        if min_class_count >= self.n_splits_cv:
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

            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            logger.info(
                "Supervised CV Accuracy: %.2f%% (±%.2f%%)",
                cv_mean * 100,
                cv_std * 100,
            )
        else:
            logger.warning(
                "[CV] Skipping CV because smallest class has only %d samples.",
                min_class_count,
            )

        # ---------------- Semi-supervised part ----------------

        unlabeled_parts = []

        if not df_default.empty:
            logger.info("[Semi] Using 'Delivery Documentation' as unlabeled pool.")
            unlabeled_parts.append(df_default)

        extra_unlabeled = self._process_extra_unlabeled(unlabeled_extra_path)

        if not extra_unlabeled.empty:
            unlabeled_parts.append(extra_unlabeled)

        if not unlabeled_parts:
            logger.info("[Semi] No unlabeled data. Saving supervised model.")
            self._save_model(
                self.model,
                accuracy=acc,
                cv_mean=cv_mean,
                cv_std=cv_std,
            )
            return

        unlabeled_df = pd.concat(unlabeled_parts, ignore_index=True)

        pseudo_df = self._pseudo_label_unlabeled(unlabeled_df)

        if pseudo_df.empty:
            logger.info("[Semi] No pseudo-labeled samples. Saving supervised model.")
            self._save_model(
                self.model,
                accuracy=acc,
                cv_mean=cv_mean,
                cv_std=cv_std,
            )
            return

        X_train_aug = pd.concat(
            [
                X_train,
                pseudo_df[["Processed_Text", "Visual_Features"]],
            ],
            ignore_index=True,
        )

        y_train_aug = pd.concat(
            [
                y_train,
                pseudo_df["Label"],
            ],
            ignore_index=True,
        )

        semi_model = build_model_pipeline()

        logger.info("[Semi] Retraining on labeled + pseudo-labeled data...")
        semi_model.fit(X_train_aug, y_train_aug)

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
        plt.title("Semi-supervised Confusion Matrix")
        plt.show()

        self.model = semi_model

        self._save_model(
            self.model,
            accuracy=acc_semi,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )

    # Public API

    def update(
        self,
        new_data_csv: str,
        unlabeled_extra_csv: Optional[str] = None,
    ):
        if not os.path.exists(new_data_csv):
            raise FileNotFoundError(f"New data CSV not found: {new_data_csv}")

        logger.info("Loading new labeled data from %s", new_data_csv)

        new_data = pd.read_csv(new_data_csv)

        updated_data = self.process_new_data(new_data)

        if updated_data.empty:
            logger.info("Updated data is empty. No model update performed.")
            return

        self.retrain_model(
            updated_data,
            unlabeled_extra_path=unlabeled_extra_csv,
        )