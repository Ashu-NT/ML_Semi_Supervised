from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import os
import joblib
import logging
from typing import List
import io
import pandas as pd

from src.config_loader import load_config
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)
REJECT_THRESHOLD = 0.70  # same as CLI 

app = FastAPI(title="Document Classifier API")

# --------- Load config + model at startup ---------

cfg = load_config()
paths = cfg["paths_resolved"]

vm = VersionManager(paths["version_file"])
last_version = vm.get_last_version()

if last_version <= 0:
    raise RuntimeError("No trained model version found. Run training first.")

model_filename = f"{paths['model_base_name']}_v{last_version}.pkl"
model_path = os.path.join(paths["models_dir"], model_filename)

bundle = joblib.load(model_path)
model = bundle["model"]
label_dict = bundle["label_dict"]
inv_label_dict = {v: k for k, v in label_dict.items()}

# Processors (reuse same as predict.py)
PDFProcessor = init_pdf_processor(
    cache_dir=paths["cache_dir"],
    tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
    threshold_ocr=cfg["ocr"]["threshold_ocr"],
)
text_processor = TextProcessor(cache_dir=paths["cache_dir"])
text_processor.load_cache()

# --------- Helpers ---------


def prepare_features_for_temp_pdf(tmp_path: str):
    raw_text = PDFProcessor.extract_text(tmp_path, cfg["processing"]["version"])
    processed_text = text_processor.preprocess(raw_text)
    visual_features = PDFProcessor.extract_visual_features(tmp_path, cfg["processing"]["version"])

    df = pd.DataFrame(
        [{
            "File Path": tmp_path,
            "Processed_Text": processed_text,
            "Visual_Features": visual_features,
        }]
    )

    # optional: save cache
    text_processor.save_cache()
    return df


def predict_single_pdf(tmp_path: str):
    df_feats = prepare_features_for_temp_pdf(tmp_path)
    X = df_feats[["Processed_Text", "Visual_Features"]]

    proba = model.predict_proba(X)
    preds = model.predict(X)

    label_idx = int(preds[0])
    best_label = inv_label_dict.get(label_idx, f"CLASS_{label_idx}")
    conf = float(proba.max(axis=1)[0])

    if conf < REJECT_THRESHOLD:
        status = "UNKNOWN"
        prediction = "UNKNOWN"
    else:
        status = "OK"
        prediction = best_label

    return {
        "status": status,
        "prediction": prediction,
        "best_label": best_label,
        "confidence": conf,
    }


# --------- API endpoints ---------


@app.post("/predict")
async def predict_pdf(file: UploadFile = File(...)):
    """
    Predict document type for a single uploaded PDF file.
    """
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported."},
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        result = predict_single_pdf(tmp_path)
        result["file_name"] = file.filename
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict document types for multiple uploaded PDF files in one request.
    Returns a list of results, one per file.
    """
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "No files uploaded."},
        )

    results = []

    for file in files:
        # Validate extension
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "file_name": file.filename,
                "status": "ERROR",
                "error": "Only PDF files are supported.",
            })
            # Skip this file, continue with others
            continue

        # Save to temp and run prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            pred = predict_single_pdf(tmp_path)
            pred["file_name"] = file.filename
            results.append(pred)
        except Exception as e:
            logger.exception("Error predicting file %s: %s", file.filename, e)
            results.append({
                "file_name": file.filename,
                "status": "ERROR",
                "error": str(e),
            })
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return results

@app.post("/predict-batch-csv")
async def predict_batch_csv(files: List[UploadFile] = File(...)):
    """
    Predict document types for multiple uploaded PDF files and return
    a CSV report as a downloadable response.
    """
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "No files uploaded."},
        )

    rows_for_csv = []

    for file in files:
        file_name = file.filename

        if not file_name.lower().endswith(".pdf"):
            rows_for_csv.append({
                "file_name": file_name,
                "status": "ERROR",
                "prediction": "",
                "best_label": "",
                "confidence": "",
                "error": "Only PDF files are supported.",
            })
            # Skip processing this file further
            continue

        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            pred = predict_single_pdf(tmp_path)
            rows_for_csv.append({
                "file_name": file_name,
                "status": pred["status"],
                "prediction": pred["prediction"],
                "best_label": pred["best_label"],
                "confidence": pred["confidence"],
                "error": "",
            })
        except Exception as e:
            logger.exception("Error predicting file %s: %s", file_name, e)
            rows_for_csv.append({
                "file_name": file_name,
                "status": "ERROR",
                "prediction": "",
                "best_label": "",
                "confidence": "",
                "error": str(e),
            })
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Build CSV in memory
    df_report = pd.DataFrame(rows_for_csv, columns=[
        "file_name",
        "status",
        "prediction",
        "best_label",
        "confidence",
        "error",
    ])

    csv_buffer = io.StringIO()
    df_report.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Create streaming response
    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predictions_report.csv"'
        },
    )
