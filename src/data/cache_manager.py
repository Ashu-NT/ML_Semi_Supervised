import os
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, cache_dir: str, processing_version: str, PDFProcessor):
        self.processing_version = processing_version
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "processed_data.joblib")
        self.PDFProcessor = PDFProcessor

    def needs_processing(self, df):
        """Check if data needs reprocessing"""
        if not os.path.exists(self.cache_file):
            logger.info("No existing cache found; processing needed.")
            return True

        cached = joblib.load(self.cache_file)
        if cached["version"] != self.processing_version:
            logger.info("Processing version changed; reprocessing needed.")
            return True

        current_sigs = df["File Path"].apply(self.PDFProcessor.file_signature)
        if not np.array_equal(current_sigs.values, cached["file_signatures"]):
            logger.info("File signatures changed; reprocessing needed.")
            return True

        return False

    def load_or_process(self, df, text_processor):
        """Main data processing workflow"""
        if not self.needs_processing(df):
            logger.info("Loading cached processed data...")
            return joblib.load(self.cache_file)["data_frame"]

        logger.info("Processing data (this might take a while)...")
        df["Raw_Text"] = df["File Path"].apply(
            lambda x: self.PDFProcessor.extract_text(x, self.processing_version)
        )
        df["Processed_Text"] = df["Raw_Text"].apply(text_processor.preprocess)
        df["Visual_Features"] = df["File Path"].apply(
            lambda x: self.PDFProcessor.extract_visual_features(x, self.processing_version)
        )

        cache_data = {
            "data_frame": df,
            "file_signatures": df["File Path"].apply(self.PDFProcessor.file_signature).values,
            "version": self.processing_version,
        }
        joblib.dump(cache_data, self.cache_file)
        logger.info(f"Saved processed data cache to {self.cache_file}")
        return df
