"""
Ejecutar UNA SOLA VEZ para convertir archivos pesados a Parquet.
Después la app carga 10-20x más rápido.

    python convertir_datos.py
"""
import os
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(fname):
    return os.path.join(BASE_DIR, fname)

def mb(fname):
    return os.path.getsize(p(fname)) / 1e6

# ── 1. tabla_clusters: Excel 60 MB → Parquet con geo pre-fusionado ────────────
print("=" * 60)
print("[1/3] tabla_clusters.xlsx → Parquet")
print("      Leyendo Excel (puede tardar 30-60 seg)…")
t0 = time.time()
df = pd.read_excel(p("tabla_clusters.xlsx"))
print(f"      Leído en {time.time()-t0:.0f}s · {len(df):,} filas · {len(df.columns)} cols")

geo_cols = ["REGION", "DEPARTAMENTO", "CIUDAD_STD", "Ciudad", "Estrato"]
missing  = [c for c in geo_cols if c not in df.columns]
if missing:
    geo_path = p("data_limpia.csv")
    if os.path.exists(geo_path):
        key = next((c for c in ["Contrato", "NIT", "Nit"] if c in df.columns), None)
        if key:
            print(f"      Fusionando geografía {missing} desde data_limpia.csv por '{key}'…")
            geo_src = pd.read_csv(
                geo_path,
                usecols=lambda c: c in [key] + geo_cols,
                low_memory=False,
            )
            geo_src = geo_src.drop_duplicates(subset=key)
            df = df.merge(
                geo_src[[key] + [c for c in geo_cols if c in geo_src.columns]],
                on=key, how="left", suffixes=("", "_geo"),
            )
            print(f"      Columnas geo añadidas: {[c for c in geo_cols if c in df.columns]}")

df.to_parquet(p("tabla_clusters.parquet"), index=False)
print(f"  ✓  tabla_clusters.parquet · {mb('tabla_clusters.parquet'):.1f} MB")

# ── 2. clientes_segmentados: CSV 200 MB → Parquet ─────────────────────────────
print()
print("[2/3] clientes_segmentados.csv → Parquet")
csv_cs = p("clientes_segmentados.csv")
if os.path.exists(csv_cs):
    print("      Leyendo CSV 200 MB…")
    t0 = time.time()
    df2 = pd.read_csv(csv_cs, low_memory=False)
    print(f"      Leído en {time.time()-t0:.0f}s · {len(df2):,} filas")
    df2.to_parquet(p("clientes_segmentados.parquet"), index=False)
    print(f"  ✓  clientes_segmentados.parquet · {mb('clientes_segmentados.parquet'):.1f} MB")
else:
    print("  ⚠  clientes_segmentados.csv no encontrado — omitido.")

# ── 3. df_res: Excel pequeño → Parquet ───────────────────────────────────────
print()
print("[3/3] df_res.xlsx → Parquet")
df3 = pd.read_excel(p("df_res.xlsx"))
df3.to_parquet(p("df_res.parquet"), index=False)
print(f"  ✓  df_res.parquet · {mb('df_res.parquet'):.1f} MB")

print()
print("=" * 60)
print("¡Conversión completa! Reinicia la app con: streamlit run app.py")
print("Los archivos .xlsx y .csv originales se conservan como respaldo.")
