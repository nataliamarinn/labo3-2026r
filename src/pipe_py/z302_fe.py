"""
z302_fe.py  –  Grupo 3: Banegas - Marín - Mengoni - Rey
Feature Engineering: lags, normalización, deltas, target explícito.
Lee z301 output (parquet). Escribe dataset_fe.parquet a local + GCS.
Uso:  python z302_fe.py [--config config.yaml]
"""

import argparse
import time
import tempfile
from pathlib import Path

import duckdb
import polars as pl
import yaml
from google.cloud import storage as gcs


# ---------------------------------------------------------------------------
def cargar_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
def encontrar_preproc(cfg: dict) -> Path:
    pr = cfg["preproc"]
    grupo = f"{pr['group_mode']}_{pr['missing_strategy']}_{pr['densify_strategy']}"
    ruta = Path(cfg["paths"]["preproc_out"]) / f"preprocesado_{grupo}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Correr z301_preproc.py primero."
        )
    return ruta


# ---------------------------------------------------------------------------
def construir_lags(con: duckdb.DuckDBPyConnection, t0: int, max_lags: int, group_mode: str) -> str:
    """
    Genera columnas tn0, tn1, ..., tn{max_lags} por producto (o agrupacion).
    tn0 = t0, tn1 = t0-1 mes, etc.
    Retorna nombre de la tabla creada en duckdb.
    """
    if group_mode == "A":
        id_col = "Agrupacion_ID"
    else:
        id_col = "product_id"

    # lista de periodos reales (retrocediendo desde t0)
    def periodo_menos_n(t0: int, n: int) -> int:
        anio = t0 // 100
        mes = t0 % 100
        mes -= n
        while mes <= 0:
            mes += 12
            anio -= 1
        return anio * 100 + mes

    periodos = [periodo_menos_n(t0, k) for k in range(max_lags + 1)]

    # pivoteamos: una fila por entidad, columnas tn0..tn{max_lags}
    pivot_cols = ", ".join(
        f"MAX(CASE WHEN periodo = {p} THEN tn ELSE NULL END) AS tn{k}"
        for k, p in enumerate(periodos)
    )

    sql = f"""
    CREATE OR REPLACE TABLE lags AS
    SELECT
        {id_col},
        {pivot_cols}
    FROM preproc
    GROUP BY {id_col}
    """
    con.execute(sql)
    return "lags"


# ---------------------------------------------------------------------------
def normalizar_recta(con: duckdb.DuckDBPyConnection, max_lags: int) -> str:
    """
    Normalización lineal (recta) por fila: tn_norm = (tn - B0) / B1
    B1 = max - min, B0 = min. Si B1=0 → tn_norm=0.
    Genera tn0_norm..tn{max_lags}_norm + B0, B1 para desnormalizar.
    """
    tn_cols = [f"tn{k}" for k in range(max_lags + 1)]
    min_expr = " + ".join([f"COALESCE(tn{k}, 0)" for k in range(max_lags + 1)])

    # B0=min, B1=max-min por fila
    minmax_cols = (
        f"LEAST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))}) AS B0, "
        f"GREATEST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))}) - "
        f"LEAST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))}) AS B1"
    )

    norm_cols = ", ".join(
        f"CASE WHEN (GREATEST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))}) - "
        f"LEAST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))})) = 0 THEN 0.0 "
        f"ELSE (COALESCE(tn{k}, 0) - LEAST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))})) / "
        f"(GREATEST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))}) - "
        f"LEAST({', '.join(f'COALESCE(tn{k},0)' for k in range(max_lags+1))})) END AS tn{k}_norm"
        for k in range(max_lags + 1)
    )

    sql = f"""
    CREATE OR REPLACE TABLE normalizado AS
    SELECT
        product_id,
        {', '.join(tn_cols)},
        {minmax_cols},
        {norm_cols}
    FROM lags
    """
    con.execute(sql)
    return "normalizado"


# ---------------------------------------------------------------------------
def agregar_deltas(con: duckdb.DuckDBPyConnection, max_lags: int, salto: int = 2) -> str:
    delta_cols = ", ".join(
        f"(tn{k}_norm - tn{k+salto}_norm) AS tn{k}_delta"
        for k in range(max_lags + 1 - salto)
    )
    sql = f"""
    CREATE OR REPLACE TABLE con_deltas AS
    SELECT *, {delta_cols}
    FROM normalizado
    """
    con.execute(sql)
    return "con_deltas"


# ---------------------------------------------------------------------------
def agregar_target(con: duckdb.DuckDBPyConnection, tipo_target: str) -> str:
    """
    Agrega una sola columna 'target' con el valor correcto.
    clase_tn       = tn0_norm  (nivel normalizado del período objetivo)
    clase_tn_delta = tn0_norm - tn{salto}_norm
    """
    if tipo_target == "nivel":
        target_expr = "tn0_norm"
    elif tipo_target == "delta":
        target_expr = "tn0_delta"
    else:
        raise ValueError(f"tipo_target desconocido: {tipo_target}")

    sql = f"""
    CREATE OR REPLACE TABLE dataset_fe AS
    SELECT
        *,
        {target_expr} AS target
    FROM con_deltas
    """
    con.execute(sql)
    return "dataset_fe"


# ---------------------------------------------------------------------------
def detectar_leakage(
    con: duckdb.DuckDBPyConnection,
    umbral_corr: float,
    umbral_multi: float,
    cols_excluir: list,
) -> list:
    """
    Retorna lista de columnas a eliminar por leakage o multicolinealidad.
    """
    # columnas disponibles (excluimos ids y target conocidos)
    todas = [
        r[0]
        for r in con.execute("DESCRIBE dataset_fe").fetchall()
        if r[0] not in cols_excluir + ["target"]
    ]

    eliminar = set()

    # correlación con target
    for col in todas:
        try:
            corr = con.execute(
                f"SELECT ABS(CORR({col}, target)) FROM dataset_fe"
            ).fetchone()[0]
            if corr is not None and corr >= umbral_corr:
                print(f"  [leakage] {col}  corr={corr:.4f}")
                eliminar.add(col)
        except Exception:
            pass

    # multicolinealidad entre features (solo entre las que quedan)
    restantes = [c for c in todas if c not in eliminar]
    for i, c1 in enumerate(restantes):
        for c2 in restantes[i + 1:]:
            try:
                corr = con.execute(
                    f"SELECT ABS(CORR({c1}, {c2})) FROM dataset_fe"
                ).fetchone()[0]
                if corr is not None and corr >= umbral_multi:
                    print(f"  [multicol] {c1} <-> {c2}  corr={corr:.4f}")
                    eliminar.add(c2)
            except Exception:
                pass

    return list(eliminar)


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
    fe = cfg["fe"]
    lk = cfg["leakage"]

    ruta_preproc = encontrar_preproc(cfg)
    ruta_out = Path(p["fe_out"]) / "dataset_fe.parquet"

    print(f"[z302] Feature Engineering")
    print(f"  preproc → {ruta_preproc}")
    print(f"  t0={fe['t0']}  max_lags={fe['max_lags']}  tipo_target={fe['tipo_target']}")
    print(f"  salida  → {ruta_out}")

    t0 = time.time()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs")

    con.execute(f"CREATE TABLE preproc AS SELECT * FROM read_parquet('{ruta_preproc}')")
    print(f"  filas leídas: {con.execute('SELECT COUNT(*) FROM preproc').fetchone()[0]:,}")

    construir_lags(con, fe["t0"], fe["max_lags"], cfg["preproc"]["group_mode"])
    normalizar_recta(con, fe["max_lags"])

    if fe["agregar_deltas"]:
        agregar_deltas(con, fe["max_lags"], fe["salto_delta"])
    else:
        con.execute("CREATE OR REPLACE TABLE con_deltas AS SELECT * FROM normalizado")

    agregar_target(con, fe["tipo_target"])

    print("  detectando leakage...")
    cols_a_eliminar = detectar_leakage(
        con,
        lk["umbral_correlacion"],
        lk["umbral_multicolineal"],
        fe["cols_excluir_leakage"],
    )
    if cols_a_eliminar:
        print(f"  eliminando {len(cols_a_eliminar)} columnas con leakage/multicol")
        todas_cols = [
            r[0]
            for r in con.execute("DESCRIBE dataset_fe").fetchall()
            if r[0] not in cols_a_eliminar
        ]
        cols_sql = ", ".join(todas_cols)
        con.execute(f"CREATE OR REPLACE TABLE dataset_fe AS SELECT {cols_sql} FROM dataset_fe")

    n_rows = con.execute("SELECT COUNT(*) FROM dataset_fe").fetchone()[0]
    n_cols = len(con.execute("DESCRIBE dataset_fe").fetchall())
    print(f"  dataset_fe: {n_rows:,} filas x {n_cols} columnas")

    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY dataset_fe TO '{ruta_out}' (FORMAT PARQUET, COMPRESSION SNAPPY)")

    elapsed = time.time() - t0
    print(f"  escrito en {elapsed:.1f}s")

    blob_path = f"{g['prefix_fe']}/dataset_fe.parquet"
    subir_a_gcs(ruta_out, g["bucket"], blob_path)

    print("[z302] OK")


if __name__ == "__main__":
    main()
