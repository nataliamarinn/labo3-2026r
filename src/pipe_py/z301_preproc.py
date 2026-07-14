"""
z301_preproc.py  –  Grupo 3: Banegas - Marín - Mengoni - Rey
Preprocesamiento de sell-in: agrupacion, nulos, densificacion.
Corre en GCP VM, sin Colab. Lee raw, escribe parquet a local + GCS.
Uso:  python z301_preproc.py [--config config.yaml]
"""

import argparse
import sys
import time
import tempfile
from pathlib import Path

import polars as pl
import yaml
from google.cloud import storage as gcs


# ---------------------------------------------------------------------------
def cargar_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
def leer_sellin(ruta: str) -> pl.LazyFrame:
    return pl.scan_csv(
        ruta,
        separator="\t",
        infer_schema_length=50_000,
    )


# ---------------------------------------------------------------------------
def agrupar(lf: pl.LazyFrame, group_mode: str) -> pl.LazyFrame:
    """
    A = cliente-producto-mes  →  suma tn por (customer_id, product_id, periodo)
    B = producto-mes          →  suma tn por (product_id, periodo)
    """
    if group_mode == "A":
        keys = ["customer_id", "product_id", "periodo"]
    elif group_mode == "B":
        keys = ["product_id", "periodo"]
    else:
        raise ValueError(f"group_mode desconocido: {group_mode}")

    return lf.group_by(keys).agg(pl.col("tn").sum())


# ---------------------------------------------------------------------------
def aplicar_missing(lf: pl.LazyFrame, strategy: str) -> pl.LazyFrame:
    if strategy == "zero":
        return lf.with_columns(pl.col("tn").fill_null(0.0))
    elif strategy == "null":
        return lf  # deja los nulls tal cual
    else:
        raise ValueError(f"missing_strategy desconocido: {strategy}")


# ---------------------------------------------------------------------------
def densificar(
    lf: pl.LazyFrame,
    strategy: str,
    periodo_min: int,
    periodo_max: int,
    group_mode: str,
) -> pl.LazyFrame:
    """
    full      → grilla completa de periodos para cada entidad
    lifecycle → solo periodos donde la entidad tiene al menos un registro
    """
    periodos = list(range(periodo_min, periodo_max + 1))
    # filtramos solo meses válidos (YYYYMM donde MM in 01..12)
    periodos = [p for p in periodos if 1 <= (p % 100) <= 12]

    if strategy == "lifecycle":
        return lf  # sin densificación extra

    # full: cross join entidades x periodos, left join datos
    if group_mode == "A":
        id_cols = ["customer_id", "product_id"]
    else:
        id_cols = ["product_id"]

    entidades = lf.select(id_cols).unique()
    grilla_periodos = pl.LazyFrame({"periodo": periodos})
    grilla = entidades.join(grilla_periodos, how="cross")

    lf_denso = grilla.join(lf, on=id_cols + ["periodo"], how="left")
    return lf_denso.with_columns(pl.col("tn").fill_null(0.0))


# ---------------------------------------------------------------------------
def agregar_agrupacion_id(lf: pl.LazyFrame, group_mode: str) -> pl.LazyFrame:
    """Genera Agrupacion_ID para compatibilidad con FE anterior."""
    if group_mode == "A":
        return lf.with_columns(
            (pl.col("customer_id") * 100_000 + pl.col("product_id")).alias(
                "Agrupacion_ID"
            )
        )
    else:
        return lf.with_columns(pl.col("product_id").alias("Agrupacion_ID"))


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
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    p = cfg["paths"]
    g = cfg["gcs"]
    pr = cfg["preproc"]

    grupo = f"{pr['group_mode']}_{pr['missing_strategy']}_{pr['densify_strategy']}"
    nombre_archivo = f"preprocesado_{grupo}.parquet"
    ruta_out = Path(p["preproc_out"]) / nombre_archivo

    print(f"[z301] Preprocesamiento  grupo={grupo}")
    print(f"  raw   → {p['raw_sellin']}")
    print(f"  salida → {ruta_out}")

    t0 = time.time()

    lf = leer_sellin(p["raw_sellin"])
    lf = agrupar(lf, pr["group_mode"])
    lf = aplicar_missing(lf, pr["missing_strategy"])
    lf = densificar(
        lf,
        pr["densify_strategy"],
        pr["periodo_min"],
        pr["periodo_max"],
        pr["group_mode"],
    )
    lf = agregar_agrupacion_id(lf, pr["group_mode"])

    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(str(ruta_out), compression="snappy")

    elapsed = time.time() - t0
    n_rows = pl.read_parquet(str(ruta_out)).shape[0]
    print(f"  filas escritas: {n_rows:,}  ({elapsed:.1f}s)")

    blob_path = f"{g['prefix_preproc']}/{nombre_archivo}"
    subir_a_gcs(ruta_out, g["bucket"], blob_path)

    print("[z301] OK")


if __name__ == "__main__":
    main()
