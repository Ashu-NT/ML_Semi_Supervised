from typing import List, Optional
import os
import tempfile
import io
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config_loader import load_config
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.model.versioning import VersionManager
from src.cli.predict import REJECT_THRESHOLD

logger = logging.getLogger(__name__)

# FastAPI app with OpenAPI / Swagger metadata
app = FastAPI(
    title="Semi-Supervised Document Classifier API",
    description=(
        "Multimodal OCR + PDF classification pipeline.\n\n"
        "This API accepts PDF documents, runs OCR + text processing + visual-feature "
        "extraction, and predicts a document type such as 'Manual', 'Drawing', "
        "'Datasheet', or 'Certificate'.\n\n"
        "Low-confidence predictions are flagged as `UNKNOWN`."
    ),
    version="1.0.0",
    contact={
        "name": "Document ML Pipeline",
        "url": "https://github.com/Ashu-NT/ML_Semi_Supervised",  
    },
    license_info={
        "name": "Proprietary / Internal",  
    },
)

# Pydantic models for OpenAPI

class PredictionResult(BaseModel):
    file_name: str = Field(..., example="seatel_manual.pdf")
    status: str = Field(
        ...,
        description="OK = confident prediction, UNKNOWN = too low confidence, ERROR = processing failed.",
        example="OK",
    )
    prediction: Optional[str] = Field(
        None,
        description="Final prediction used by the system. May be 'UNKNOWN' if below threshold.",
        example="Manual",
    )
    best_label: Optional[str] = Field(
        None,
        description="Model's highest-probability label, even when status is UNKNOWN.",
        example="Manual",
    )
    confidence: Optional[float] = Field(
        None,
        description="Model probability of best_label, between 0 and 1.",
        example=0.83,
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status == 'ERROR'.",
        example="Only PDF files are supported.",
    )


# Load config, model, and processors at startup

cfg = load_config()
paths = cfg["paths_resolved"]

# Model versioning
vm = VersionManager(paths["version_file"])
last_version = vm.get_last_version()
if last_version <= 0:
    raise RuntimeError("No trained model version found. Run training first.")

model_filename = f"{paths['model_base_name']}_v{last_version}.pkl"
model_path = os.path.join(paths["models_dir"], model_filename)

logger.info("Loading model bundle from %s (version %d)", model_path, last_version)
bundle = joblib.load(model_path)
model = bundle["model"]
label_dict = bundle["label_dict"]
inv_label_dict = {v: k for k, v in label_dict.items()}

# Processors
PDFProcessor = init_pdf_processor(
    cache_dir=paths["cache_dir"],
    tesseract_cmd=cfg["ocr"]["tesseract_cmd"],
    threshold_ocr=cfg["ocr"]["threshold_ocr"],
)
text_processor = TextProcessor(cache_dir=paths["cache_dir"])
text_processor.load_cache()

# Helper functions

def prepare_features_for_temp_pdf(tmp_path: str) -> pd.DataFrame:
    """
    Run OCR + preprocessing + visual-feature extraction on a single temp PDF path
    and return a one-row DataFrame with the feature columns expected by the model.
    """
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

    # Persist updated text cache
    text_processor.save_cache()
    return df


def predict_single_pdf(tmp_path: str) -> PredictionResult:
    """
    Core prediction logic: given a temp PDF path, return a structured prediction.
    """
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

    return PredictionResult(
        file_name=os.path.basename(tmp_path),  # will be overridden for API
        status=status,
        prediction=prediction,
        best_label=best_label,
        confidence=conf,
        error=None,
    )

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Document Classifier API is running.",
        "docs": "/docs",
        "endpoints": ["/predict", "/predict-batch", "/predict-batch-csv"],
    }
    
@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict a document type for a single PDF",
    tags=["Prediction"],
)
async def predict_pdf(file: UploadFile = File(..., description="A single PDF document to classify.")):
    """
    Upload a **single PDF file** and receive a JSON prediction.

    - **status**:
        - `OK` → confident prediction
        - `UNKNOWN` → model not confident enough (below threshold)
        - `ERROR` → processing failed
    - **prediction**:
        - Final class used by the system (may be `"UNKNOWN"`).
    - **best_label**:
        - Model's top class even if status is UNKNOWN.
    - **confidence**:
        - Probability of `best_label` in [0, 1].
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        pred = predict_single_pdf(tmp_path)
        # Use original filename in API response
        pred.file_name = file.filename
        return pred
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post(
    "/predict-batch",
    response_model=List[PredictionResult],
    summary="Predict document types for multiple PDFs",
    tags=["Prediction"],
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="One or more PDF files to classify.")
):
    """
    Upload **multiple PDFs** in one request and receive a JSON list of predictions.

    Each element in the returned list corresponds to one input file and follows
    the same schema as `/predict`.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results: List[PredictionResult] = []

    for file in files:
        file_name = file.filename

        if not file_name.lower().endswith(".pdf"):
            results.append(
                PredictionResult(
                    file_name=file_name,
                    status="ERROR",
                    prediction=None,
                    best_label=None,
                    confidence=None,
                    error="Only PDF files are supported.",
                )
            )
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            pred = predict_single_pdf(tmp_path)
            pred.file_name = file_name
            results.append(pred)
        except Exception as e:
            logger.exception("Error predicting file %s: %s", file_name, e)
            results.append(
                PredictionResult(
                    file_name=file_name,
                    status="ERROR",
                    prediction=None,
                    best_label=None,
                    confidence=None,
                    error=str(e),
                )
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return results


@app.post(
    "/predict-batch-csv",
    summary="Predict types for multiple PDFs and download a CSV report",
    tags=["Prediction"],
    responses={
        200: {
            "content": {
                "text/csv": {
                    "example": (
                        "file_name,status,prediction,best_label,confidence,error\n"
                        "manual1.pdf,OK,Manual,Manual,0.83,\n"
                        "contract.pdf,UNKNOWN,UNKNOWN,Drawing,0.27,\n"
                        "image.png,ERROR,,,,Only PDF files are supported.\n"
                    )
                }
            },
            "description": "CSV file containing one row per input document.",
        }
    },
)
async def predict_batch_csv(
    files: List[UploadFile] = File(..., description="One or more PDF files to classify.")
):
    """
    Upload **multiple PDFs** and receive a **CSV report** as a downloadable response.

    The CSV contains these columns:

    - `file_name`
    - `status` (OK / UNKNOWN / ERROR)
    - `prediction`
    - `best_label`
    - `confidence`
    - `error`
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

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
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            pred = predict_single_pdf(tmp_path)
            rows_for_csv.append({
                "file_name": file_name,
                "status": pred.status,
                "prediction": pred.prediction,
                "best_label": pred.best_label,
                "confidence": pred.confidence,
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

    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predictions_report.csv"'
        },
    )
