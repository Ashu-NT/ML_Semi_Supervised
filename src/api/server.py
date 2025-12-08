from typing import List, Optional
import os
import tempfile
import io
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional

import glob
import json

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datetime import datetime

from src.config_loader import load_config
from src.data.ocr import init_pdf_processor
from src.data.text_processing import TextProcessor
from src.model.versioning import VersionManager
from src.cli.predict import REJECT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class ModelSummary(BaseModel):
    version: int = Field(..., example=2)
    is_current: bool = Field(
        ...,
        description="True if this version is the one pointed to by version.txt.",
        example=True,
    )
    model_file: str = Field(
        ...,
        description="Filename of the model bundle.",
        example="cached_multimodal_doc_classifier_v2.pkl",
    )
    has_metrics: bool = Field(
        ...,
        description="Whether metrics JSON exists for this version.",
        example=True,
    )
    accuracy: Optional[float] = Field(
        None,
        description="Last recorded evaluation accuracy on the fixed test set.",
        example=0.91,
    )
    macro_f1: Optional[float] = Field(
        None,
        description="Last recorded macro-averaged F1 on the fixed test set.",
        example=0.89,
    )
    n_eval: Optional[int] = Field(
        None,
        description="Number of eval samples used in last evaluation.",
        example=98,
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

def _load_eval_data(cfg, eval_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and process the evaluation CSV into features.
    This mirrors the CLI evaluation behavior.
    """
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

    exists_mask = df["File Path"].apply(os.path.exists)
    missing = df[~exists_mask]
    if not missing.empty:
        logger.warning("Dropping %d eval rows because file does not exist.", len(missing))
    df = df[exists_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid eval rows after filtering missing files.")

    # Reuse processors
    PDFProc = PDFProcessor  # already initialized globally
    tp = text_processor

    logger.info("Processing eval documents ( OCR + text + visual features )...")
    df["Raw_Text"] = df["File Path"].apply(
        lambda x: PDFProc.extract_text(x, cfg["processing"]["version"])
    )
    df["Processed_Text"] = df["Raw_Text"].apply(tp.preprocess)
    df["Visual_Features"] = df["File Path"].apply(
        lambda x: PDFProc.extract_visual_features(x, cfg["processing"]["version"])
    )

    tp.save_cache()
    return df

def _evaluate_model_version(
    cfg,
    version: int,
    df_eval: pd.DataFrame,
    save_metrics: bool = True,
) -> dict:
    """
    Evaluate a given model version on df_eval and optionally persist metrics JSON.
    Returns a metrics dict.
    """
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

    # Filter to only classes known by this model
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

        payload = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        payload["version"] = version
        payload["eval"] = metrics

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved eval metrics to %s", metrics_path)

    return metrics

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Document Classifier API is running.",
        "docs": "/docs",
        "endpoints": ["/predict", "/predict-batch", "/predict-batch-csv"],
    }

@app.get(
    "/models",
    response_model=List[ModelSummary],
    summary="List available model versions and their metrics",
    tags=["Models"],
)
async def list_models():
    """
    List all available model versions, show whether they're current,
    and summarize evaluation metrics if available.
    """
    paths = cfg["paths_resolved"]
    models_dir = paths["models_dir"]
    base = paths["model_base_name"]

    vm = VersionManager(paths["version_file"])
    current_version = vm.get_last_version()

    pattern = os.path.join(models_dir, f"{base}_v*.pkl")
    model_files = sorted(glob.glob(pattern))

    summaries: List[ModelSummary] = []

    for path in model_files:
        fname = os.path.basename(path)
        try:
            ver_str = fname.split("_v")[-1].split(".pkl")[0]
            ver = int(ver_str)
        except Exception:
            logger.warning("Could not parse version from %s", fname)
            continue

        metrics_path = os.path.join(models_dir, f"{base}_v{ver}.metrics.json")
        has_metrics = os.path.exists(metrics_path)

        acc = None
        macro_f1 = None
        n_eval = None

        if has_metrics:
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                eval_m = data.get("eval", {})
                acc = eval_m.get("accuracy")
                macro_f1 = eval_m.get("macro_f1")
                n_eval = eval_m.get("n_eval")
            except Exception as e:
                logger.warning("Failed to load metrics from %s: %s", metrics_path, e)

        summaries.append(
            ModelSummary(
                version=ver,
                is_current=(ver == current_version),
                model_file=fname,
                has_metrics=has_metrics,
                accuracy=acc,
                macro_f1=macro_f1,
                n_eval=n_eval,
            )
        )

    return summaries

@app.get(
    "/models/{version}/metrics",
    summary="Get stored evaluation metrics for a specific model version",
    tags=["Models"],
)
async def get_model_metrics(version: int):
    """
    Return the saved evaluation metrics JSON for a given model version, if present.
    Does NOT trigger a new evaluation; just reads the metrics file.
    """
    paths = cfg["paths_resolved"]
    base = paths["model_base_name"]
    models_dir = paths["models_dir"]

    metrics_path = os.path.join(models_dir, f"{base}_v{version}.metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail=f"No metrics file found for version {version}. "
                   f"Run evaluation first (CLI or /models/{version}/evaluate).",
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.exception("Error reading metrics for version %d: %s", version, e)
        raise HTTPException(status_code=500, detail="Failed to read metrics file.")

@app.post(
    "/models/{version}/evaluate",
    summary="Run evaluation on a specific model version using the fixed eval set",
    tags=["Models"],
)
async def evaluate_model_version(version: int, save_metrics: bool = True):
    """
    Trigger evaluation of a given model version on the configured eval CSV.
    This may take time (OCR + model inference).
    """
    try:
        df_eval = _load_eval_data(cfg)
        metrics = _evaluate_model_version(cfg, version=version, df_eval=df_eval, save_metrics=save_metrics)
        return metrics
    except FileNotFoundError as e:
        logger.error("Eval error: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error("Eval error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during evaluation: %s", e)
        raise HTTPException(status_code=500, detail="Evaluation failed due to an internal error.")
 
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
