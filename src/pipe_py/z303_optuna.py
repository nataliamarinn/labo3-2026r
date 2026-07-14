"""
z303_optuna.py  –  Grupo 3: Banegas - Marín - Mengoni - Rey
Búsqueda de hiperparámetros con Optuna + MLflow.
Lee dataset_fe.parquet. Guarda best_params JSON + SQLite a local y GCS.
Uso:  python z303_optuna.py [--config config.yaml] [--modo B_zero_full]
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import yaml
from google.cloud import storage as gcs
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
def cargar_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return 0.0
    return np.abs(y_true - y_pred).sum() / denom


# ---------------------------------------------------------------------------
def cargar_datos(cfg: dict):
    ruta = Path(cfg["paths"]["fe_out"]) / "dataset_fe.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}. Correr z302_fe.py primero.")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE ds AS SELECT * FROM read_parquet('{ruta}')")

    todas_cols = [r[0] for r in con.execute("DESCRIBE ds").fetchall()]
    excluir = set(cfg["fe"]["cols_excluir_leakage"] + ["target"])
    feature_cols = [c for c in todas_cols if c not in excluir]

    cols_sql = ", ".join(feature_cols + ["target"])
    datos = con.execute(f"SELECT {cols_sql} FROM ds").fetchnumpy()

    X = np.column_stack([datos[c] for c in feature_cols])
    y = datos["target"]
    return X, y, feature_cols


# ---------------------------------------------------------------------------
def make_objective(X, y, cfg, tipo_target, B0=None, B1=None):
    opt_cfg = cfg["optuna"]
    space = cfg["lgbm_space"]

    def objective(trial):
        params = {
            "objective":        opt_cfg["objective_lgbm"],
            "num_leaves":       trial.suggest_int("num_leaves", *space["num_leaves"]),
            "learning_rate":    trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "n_estimators":     trial.suggest_int("n_estimators", *space["n_estimators"]),
            "min_child_samples":trial.suggest_int("min_child_samples", *space["min_child_samples"]),
            "subsample":        trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "reg_alpha":        trial.suggest_float("reg_alpha", *space["reg_alpha"]),
            "reg_lambda":       trial.suggest_float("reg_lambda", *space["reg_lambda"]),
            "verbosity":        -1,
            "n_jobs":           -1,
            "random_state":     opt_cfg["semilla"],
        }

        kf = KFold(n_splits=opt_cfg["n_folds"], shuffle=True, random_state=opt_cfg["semilla"])
        wapes = []

        for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
            Xtr, Xva = X[idx_tr], X[idx_va]
            ytr, yva = y[idx_tr], y[idx_va]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            pred = model.predict(Xva)

            # si target=delta reconstruir nivel para medir wape real
            if tipo_target == "delta" and B0 is not None and B1 is not None:
                # nivel_pred = (delta_pred + tn{salto}_norm) * B1 + B0
                # aca no tenemos tn_salto en este scope; usamos proxy directo
                pred_nivel = pred * B1[idx_va] + B0[idx_va]
                real_nivel = yva * B1[idx_va] + B0[idx_va]
                w = wape(real_nivel, pred_nivel)
            else:
                w = wape(yva, pred)

            wapes.append(w)
            trial.report(w, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(wapes))

    return objective


# ---------------------------------------------------------------------------
def subir_a_gcs(ruta_local: Path, bucket_name: str, blob_path: str):
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(ruta_local))
    uri = f"gs://{bucket_name}/{blob_path}"
    print(f"  subido → {uri}")
    return uri


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--modo", default=None, help="Ej: B_zero_full")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    opt_cfg = cfg["optuna"]
    p = cfg["paths"]
    g = cfg["gcs"]

    pr = cfg["preproc"]
    modo = args.modo or f"{pr['group_mode']}_{pr['missing_strategy']}_{pr['densify_strategy']}"

    print(f"[z303] Optuna  modo={modo}  trials={opt_cfg['n_trials']}")

    # MLflow
    mlflow.set_tracking_uri(p["mlflow_uri"])
    mlflow.set_experiment(opt_cfg["experiment_name"])

    X, y, feature_cols = cargar_datos(cfg)
    print(f"  X shape: {X.shape}")

    tipo_target = cfg["fe"]["tipo_target"]
    sqlite_path = opt_cfg["sqlite_path"].replace("{MODO}", modo)

    sampler = optuna.samplers.TPESampler(seed=opt_cfg["semilla"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    storage = f"sqlite:///{sqlite_path}"

    study = optuna.create_study(
        study_name=f"labo3_{modo}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(X, y, cfg, tipo_target)

    with mlflow.start_run(run_name=f"optuna_{modo}") as run:
        mlflow.log_params({
            "modo":         modo,
            "n_trials":     opt_cfg["n_trials"],
            "n_folds":      opt_cfg["n_folds"],
            "tipo_target":  tipo_target,
            "metrica":      opt_cfg["metrica"],
        })

        t0 = time.time()
        study.optimize(objective, n_trials=opt_cfg["n_trials"], show_progress_bar=True)
        elapsed = time.time() - t0

        best = study.best_trial
        print(f"\n  Mejor trial #{best.number}  WAPE={best.value:.4f}  ({elapsed:.0f}s)")

        mlflow.log_metric("best_wape", best.value)
        mlflow.log_metric("n_trials_completados", len(study.trials))
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_param("run_id", run.info.run_id)

    # guardar resultado
    resultado = {
        "modo":         modo,
        "tipo_target":  tipo_target,
        "metrica":      opt_cfg["metrica"],
        "best_wape":    best.value,
        "best_params":  best.params,
        "features":     feature_cols,
        "n_trials":     len(study.trials),
        "mlflow_run_id": run.info.run_id,
    }

    ruta_json = Path(p["optuna_out"]) / f"z303_hiper_{modo}.json"
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w") as f:
        json.dump(resultado, f, indent=2)
    print(f"  hiperparámetros → {ruta_json}")

    # subir json y sqlite a GCS
    subir_a_gcs(ruta_json, g["bucket"], f"{g['prefix_optuna']}/z303_hiper_{modo}.json")
    subir_a_gcs(Path(sqlite_path), g["bucket"], f"{g['prefix_optuna']}/z303_optuna_{modo}.db")

    print("[z303] OK")
    print(f"\nVer MLflow UI:  mlflow ui --backend-store-uri {p['mlflow_uri']} --port 5000")
    print(f"Ver Optuna:     optuna-dashboard sqlite:///{sqlite_path}")


if __name__ == "__main__":
    main()
