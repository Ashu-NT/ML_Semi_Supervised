# src/cli/info.py

import argparse
import glob
import json
import logging
import os

from src.config_loader import load_config
from src.model.versioning import VersionManager

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Show available model versions and their metrics (if present)."
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg["paths_resolved"]
    models_dir = paths["models_dir"]
    base = paths["model_base_name"]

    vm = VersionManager(paths["version_file"])
    latest = vm.get_last_version()

    pattern = os.path.join(models_dir, f"{base}_v*.pkl")
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        print("No model files found.")
        return

    print(f"Models directory: {models_dir}")
    print(f"Latest version (from {paths['version_file']}): {latest}\n")

    print("Available models:")
    print("Version | File name                                          | Eval accuracy | N_eval")
    print("--------+----------------------------------------------------+--------------+-------")

    for path in model_files:
        fname = os.path.basename(path)
        # parse version number from "..._vX.pkl"
        try:
            ver_str = fname.split("_v")[-1].split(".pkl")[0]
            ver = int(ver_str)
        except Exception:
            logger.warning("Could not parse version from filename %s", fname)
            continue

        metrics_path = os.path.join(models_dir, f"{base}_v{ver}.metrics.json")
        acc_str = "-"
        n_eval_str = "-"

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                eval_m = m.get("eval", {})
                acc_str = f"{eval_m.get('accuracy', 0.0):.4f}"
                n_eval_str = str(eval_m.get("n_eval", "-"))
            except Exception as e:
                logger.warning("Failed to load metrics from %s: %s", metrics_path, e)

        latest_marker = "*" if ver == latest else " "
        print(
            f"{ver:7d}{latest_marker} | {fname:50s} | {acc_str:12s} | {n_eval_str:5s}"
        )

    print("\n(*) marks the version referenced by version.txt as 'current'.")
    print("Run `python -m src.cli.evaluate --version N` to (re)evaluate a model on the test set.")


if __name__ == "__main__":
    main()
