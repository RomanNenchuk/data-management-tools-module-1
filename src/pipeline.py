import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Config
from .logger import setup_logger
from .io import ensure_dirs, load_csv, save_csv
from .preprocess import build_preprocessor
from .train import build_model
from .evaluate import evaluate

def run_pipeline(config_path: str):
    cfg = Config.load(config_path)
    seed = int(cfg.get("project", "seed", default=42))
    np.random.seed(seed)
    
    # Створення Run ID та директорій
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = cfg.get("output", "artifacts_dir", default="artifacts")
    run_dir = Path(artifacts_dir) / f"run_{run_id}"
    logs_dir = cfg.get("output", "logs_dir", default="logs")
    ensure_dirs(str(run_dir / "models"), str(run_dir / "predictions"),
                str(run_dir / "metrics"), str(run_dir / "reports"), logs_dir)

    logger, _ = setup_logger(logs_dir, run_id)
    logger.info(f"--- Start Run {run_id} (Variant 8: Regression + Top-10 Errors) ---")

    # Завантаження та підготовка даних
    input_csv = cfg.get("data", "input_csv")
    target = cfg.get("data", "target")
    num_cols = cfg.get("data", "num_cols", default=[])
    cat_cols = cfg.get("data", "cat_cols", default=[])
    
    df = load_csv(input_csv)
    logger.info(f"Data loaded: {df.shape}")
    
    X = df[num_cols + cat_cols].copy()
    y = df[target].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("data", "test_size", default=0.2), random_state=seed
    )

    # Побудова та тренування Pipeline
    pre = build_preprocessor(num_cols, cat_cols)
    model = build_model(cfg.get("model", "task"), cfg.get("model", "name"), cfg.get("model", "params"))
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    
    pipe.fit(X_train, y_train)
    logger.info("Model trained successfully.")

    # Прогнозування та оцінка
    y_pred = pipe.predict(X_test)
    metrics = evaluate("regression", y_test, y_pred)
    logger.info(f"Metrics: {metrics}")

    # --- ФІШКА ВАРІАНТА 8: ТОП-10 ПОМИЛОК  ---
    error_df = X_test.copy()
    error_df["y_true"] = y_test.values
    error_df["y_pred"] = y_pred
    error_df["abs_error"] = np.abs(error_df["y_true"] - error_df["y_pred"])
    
    top_10_errors = error_df.sort_values(by="abs_error", ascending=False).head(10)
    top_10_path = run_dir / "reports" / "top_10_errors.csv"
    save_csv(top_10_errors, str(top_10_path))
    logger.info(f"Top 10 errors saved to {top_10_path}")
    # -------------------------------------------------

    # Збереження артефактів
    joblib.dump(pipe, run_dir / "models" / "model.joblib")
    (run_dir / "metrics" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_csv(error_df.drop(columns=["abs_error"]), str(run_dir / "predictions" / "predictions.csv"))

    logger.info(f"Run finished. All artifacts are in {run_dir}")