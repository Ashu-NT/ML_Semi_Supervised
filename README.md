cd src
python -m src.cli.train

python -m src.cli.predict --input path/to/file.pdf
python -m src.cli.predict --input path/to/folder
python -m src.cli.predict --input path/to/folder --recursive

if new csv has new document type do not forget to add new document to the config

# Document Classifier – OCR + NLP + Semi-Supervised Learning

A production-grade document classification system for PDFs using:

- **OCR (PyMuPDF + Tesseract)**
- **Text preprocessing + lemmatization**
- **Visual feature extraction**
- **Supervised learning (initial training)**
- **Semi-supervised incremental updates (self-learning)**
- **Versioned models (`v1`, `v2`, …)**  
- **Caching of all expensive OCR & preprocessing steps**

This system classifies engineering PDFs into categories such as:

- Manual  
- Drawing  
- Datasheet  
- Certificate  
- Delivery Documentation (treated as *unlabeled* during updates)

---

## Project Structure

```
ML_Semi_Supervised/
├─ src/
│  └── 
│     ├─ config/
│     │  └─ base.yaml
│     ├─ config_loader.py
│     ├─ logging_config.py
│     ├─ data/
│     │  ├─ ocr.py
│     │  ├─ text_processing.py
│     │  └─ cache_manager.py
│     ├─ model/
│     │  ├─ pipeline.py
│     │  ├─ updater.py
│     │  └─ versioning.py
│     └─ cli/
│        ├─ train.py
│        ├─ update.py
│        └─ predict.py
├─ data/
│  ├─ training/
│  └─ cache/
├─ models/
├─ logs/
├─ version.txt
├─ requirements.txt
└─ README.md
```

---

## Installation

### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR  
Download from: https://github.com/UB-Mannheim/tesseract/wiki

Update YAML config:

```yaml
ocr:
  tesseract_cmd: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

---

## Configuration

All settings are located in:
```
src/doc_classifier/config/base.yaml
```

Example:
```yaml
paths:
  data_dir: "data"
  cache_subdir: "cache"
  training_subpath: "training/training_data.csv"
  models_dir: "models"
  logs_dir: "logs"
  model_base_name: "cached_multimodal_doc_classifier"
  version_file: "version.txt"
```

---

##  Training and Updating Workflow

### 1️ Initial Training (Supervised only)
Run once:
```bash
python -m src.cli.train
```

This will:
- Train a supervised model on trusted labels
- Save `models/modelname_v1.pkl`
- Write `1` to `version.txt`

### Incremental Update (Semi-supervised)
```bash
python -m src.cli.update --new-data data/training/new_batch.csv
```

Optional extra unlabeled PDFs:
```bash
python -m src.cli.update   --new-data data/training/new_batch.csv   --unlabeled-extra data/training/unlabeled_docs.csv
```

The updater:
- Uses `"Delivery Documentation"` as unlabeled  
- Pseudo-labels unlabeled samples  
- Retrains model  
- Saves new version `_v(N+1).pkl`

---

## 🔎 Predict PDFs

Single file:
```bash
python -m src.cli.predict --input path/to/file.pdf
```

Folder:
```bash
python -m src.cli.predict --input path/to/folder
```

Recursive:
```bash
python -m src.cli.predict --input path/to/folder --recursive
```

---

##  First-Time Setup Checklist

1. Install Python + Tesseract  
2. Create `.venv` and install requirements  
3. Create folders:
   ```
   data/training/
   data/cache/
   models/
   logs/
   ```
4. Place your labeled CSV in:
   ```
   data/training/training_data.csv
   ```
5. Set `version.txt` to `0`  
6. Run:
   ```bash
   python -m src.cli.train
   ```
 