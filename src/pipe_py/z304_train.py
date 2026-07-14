"""
z304_train.py  –  Grupo 3: Banegas - Marín - Mengoni - Rey
Entrenamiento final con ensemble de semillas + submit a Kaggle.
Lee dataset_fe.parquet + z303_hiper_{modo}.json.
Uso:  python z304_train.py [--config config.yaml] [--modo B_zero_full] [--no-submit]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import yaml
from google.cloud import storage as gcs


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
def cargar_hiperparametros(cfg: dict, modo: str) -> dict:
    ruta = Path(cfg["paths"]["optuna_out"]) / f"z303_hiper_{modo}.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}. Correr z303_optuna.py primero.")
    with open(ruta) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def cargar_datos(cfg: dict, split: str = "train"):
    """
    split='train' → todos los datos hasta periodo_max
    split='infer' → período inferencia (202002)
    """
    ruta = Path(cfg["paths"]["fe_out"]) / "dataset_fe.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}.")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE ds AS SELECT * FROM read_parquet('{ruta}')")

    todas_cols = [r[0] for r in con.execute("DESCRIBE ds").fetchall()]
    excluir = set(cfg["fe"]["cols_excluir_leakage"] + ["target"])
    feature_cols = [c for c in todas_cols if c not in excluir]

    if split == "infer":
        periodo_inf = cfg["train"]["periodo_inferencia"]
        where = f"WHERE periodo = {periodo_inf}" if "periodo" in todas_cols else ""
    else:
        where = ""

    cols_sql = ", ".join(feature_cols + ["target"])
    datos = con.execute(f"SELECT {cols_sql} FROM ds {where}").fetchnumpy()

    X = np.column_stack([datos[c] for c in feature_cols])
    y = datos["target"]

    # B0, B1 para desnormalizar si tipo_target=delta
    B0 = datos.get("B0")
    B1 = datos.get("B1")

    return X, y, feature_cols, B0, B1


# ---------------------------------------------------------------------------
def entrenar_ensemble(X, y, best_params, semillas, objetivo="regression"):
    modelos = []
    for semilla in semillas:
        params = {**best_params, "random_state": semilla, "n_jobs": -1, "verbosity": -1,
                  "objective": objetivo}
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
        modelos.append(model)
        print(f"  semilla {semilla} → OK")
    return modelos


# ---------------------------------------------------------------------------
def predecir_ensemble(modelos, X):
    preds = np.stack([m.predict(X) for m in modelos], axis=0)
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
def desnormalizar(pred_norm, B0, B1):
    return pred_norm * B1 + B0


# ---------------------------------------------------------------------------
def guardar_submit(pred_nivel, cfg: dict, modo: str):
    import pandas as pd  # solo para CSV de submit
    ruta = Path(cfg["paths"]["submit_out"]) / f"submit_{modo}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"product_id": range(len(pred_nivel)), "tn": pred_nivel})
    df.to_csv(ruta, index=False)
    print(f"  submit guardado → {ruta}")
    return ruta


# ---------------------------------------------------------------------------
def submit_kaggle(ruta_csv: Path, competencia: str, mensaje: str):
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competencia,
        "-f", str(ruta_csv),
        "-m", mensaje,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  [WARNING] kaggle submit error: {result.stderr}")


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
    parser.add_argument("--modo", default=None)
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    p = cfg["paths"]
    g = cfg["gcs"]
    tr = cfg["train"]

    pr = cfg["preproc"]
    modo = args.modo or f"{pr['group_mode']}_{pr['missing_strategy']}_{pr['densify_strategy']}"

    print(f"[z304] Entrenamiento final  modo={modo}")

    hiper = cargar_hiperparametros(cfg, modo)
    best_params  = hiper["best_params"]
    tipo_target  = hiper["tipo_target"]
    feature_cols = hiper["features"]

    # datos de entrenamiento
    X_tr, y_tr, _, B0_tr, B1_tr = cargar_datos(cfg, split="train")
    print(f"  X_train: {X_tr.shape}")

    # MLflow
    mlflow.set_tracking_uri(p["mlflow_uri"])
    mlflow.set_experiment(cfg["optuna"]["experiment_name"])

    with mlflow.start_run(run_name=f"train_{modo}") as run:
        mlflow.log_params({"modo": modo, "tipo_target": tipo_target,
                           "n_semillas": len(tr["semillas_ensemble"])})
        mlflow.log_params({f"hiper_{k}": v for k, v in best_params.items()})

        t0 = time.time()
        print(f"  entrenando ensemble ({len(tr['semillas_ensemble'])} semillas)...")
        modelos = entrenar_ensemble(
            X_tr, y_tr, best_params, tr["semillas_ensemble"],
            objetivo=cfg["optuna"]["objective_lgbm"],
        )

        # diagnóstico in-sample
        pred_tr = predecir_ensemble(modelos, X_tr)
        if tipo_target == "delta" and B0_tr is not None:
            pred_nivel_tr = desnormalizar(pred_tr, B0_tr, B1_tr)
            real_nivel_tr = desnormalizar(y_tr, B0_tr, B1_tr)
        else:
            pred_nivel_tr = pred_tr
            real_nivel_tr = y_tr

        wape_insample = wape(real_nivel_tr, pred_nivel_tr)
        print(f"  WAPE in-sample (nivel): {wape_insample:.4f}")
        mlflow.log_metric("wape_insample", wape_insample)

        # inferencia
        X_inf, _, _, B0_inf, B1_inf = cargar_datos(cfg, split="infer")
        print(f"  X_infer: {X_inf.shape}")
        pred_inf = predecir_ensemble(modelos, X_inf)

        if tipo_target == "delta" and B0_inf is not None:
            pred_nivel_inf = desnormalizar(pred_inf, B0_inf, B1_inf)
        else:
            pred_nivel_inf = pred_inf

        # forzar no negativos
        pred_nivel_inf = np.clip(pred_nivel_inf, 0, None)

        elapsed = time.time() - t0
        mlflow.log_metric("elapsed_seg", elapsed)

        ruta_csv = guardar_submit(pred_nivel_inf, cfg, modo)
        subir_a_gcs(ruta_csv, g["bucket"], f"{g['prefix_submit']}/submit_{modo}.csv")

        if not args.no_submit and tr.get("submit", False):
            print(f"  enviando a Kaggle → {tr['kaggle_competition']}")
            submit_kaggle(ruta_csv, tr["kaggle_competition"],
                          f"labo3 grupo3 modo={modo} wape_is={wape_insample:.4f}")
        else:
            print("  submit omitido (--no-submit o submit=false en config)")

        mlflow.log_artifact(str(ruta_csv))

    print(f"[z304] OK  ({elapsed:.0f}s)")
    print(f"\nVer MLflow UI:  mlflow ui --backend-store-uri {p['mlflow_uri']} --port 5000")


if __name__ == "__main__":
    main()
