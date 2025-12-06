import yaml
from pathlib import Path

# This file is at: <project_root>/src/config_loader.py
BASE_DIR = Path(__file__).resolve().parent      # .../src
PROJECT_ROOT = BASE_DIR.parent                  # .../ML_Semi_Supervised

def load_config(env: str = "base"):
    """
    Load YAML config and resolve paths relative to the project root.
    Assumes:
      - config files in: src/config/
      - data/, models/, logs/ in project root.
    """
    cfg_path = BASE_DIR / "config" / f"{env}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})

    data_dir = PROJECT_ROOT / paths.get("data_dir", "data")
    cache_dir = data_dir / paths.get("cache_subdir", "cache")
    training_csv = data_dir / paths.get("training_subpath", "training/training_data.csv")
    models_dir = PROJECT_ROOT / paths.get("models_dir", "models")
    logs_dir = PROJECT_ROOT / paths.get("logs_dir", "logs")
    model_base_name = paths.get("model_base_name", "cached_multimodal_doc_classifier")
    version_file = PROJECT_ROOT / paths.get("version_file", "version.txt")
    missing_files_log = PROJECT_ROOT / paths.get("missing_files_log", "data/missing_files_log.csv")

    cfg["paths_resolved"] = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir),
        "training_csv": str(training_csv),
        "models_dir": str(models_dir),
        "logs_dir": str(logs_dir),
        "model_base_name": model_base_name,
        "version_file": str(version_file),
        "missing_files_log": str(missing_files_log),
    }

    return cfg
