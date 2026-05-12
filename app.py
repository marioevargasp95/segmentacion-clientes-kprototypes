"""
app.py — Segmentación de Clientes Corporativo
Streamlit dashboard con 5 páginas:
  Dashboard          → KPIs, donut, radar, tabla resumen
  PCA Interactivo    → scatter 2D interactivo con Plotly
  Explorar registros → tabla filtrable + exportar CSV/Excel
  Clasificar cliente → formulario de inferencia en tiempo real
  Estadísticas       → análisis empresarial y familias
"""

import os
import io
import json
import uuid
import hashlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOMBRES = {
    0: "Titulares Maduros",
    1: "Familias Premium",
    2: "Jóvenes en Crecimiento",
}
ACCIONES = {
    0: "Campaña de renovación anticipada + beneficio exclusivo por antigüedad.",
    1: "Oferta plan familiar extendido (agregar dependientes) + descuento por volumen.",
    2: "Descuento primer año + pack bienvenida digital + recordatorio de chequeo.",
}

EXPORT_ENABLED = False

# ── Sesiones persistidas en disco ─────────────────────────────────────────────
INACTIVITY_TIMEOUT = 30   # minutos sin actividad para cerrar sesión
SESSIONS_FILE      = os.path.join(BASE_DIR, "sessions.json")

def _load_sessions() -> dict:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE, encoding="utf-8") as f:
        return json.load(f)

def _save_sessions(sessions: dict) -> None:
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f)

# ── Paleta corporativa ────────────────────────────────────────────────────────
C_VERDE    = '#24743c'
C_V_PAL    = '#79a65c'
C_AMARILLO = '#f39c14'
C_A_PAL    = '#f3eb8b'
PALETTE    = [C_VERDE, C_V_PAL, C_AMARILLO, C_A_PAL]
COL_STATS  = PALETTE   # alias — todas las gráficas usan esta paleta
PAL        = {0: C_VERDE, 1: C_V_PAL, 2: C_AMARILLO}  # colores corporativos por segmento

# ── Template Plotly corporativo ───────────────────────────────────────────────
pio.templates['corp'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='white',
        plot_bgcolor='#f5fbf7',
        colorway=PALETTE,
        font=dict(family='sans-serif', size=11, color='#333333'),
        title=dict(font=dict(size=13, color=C_VERDE)),
        xaxis=dict(
            linecolor='#b0c8b8', linewidth=1, showgrid=False,
            tickfont=dict(color='#444444'), ticks='outside', tickcolor='#b0c8b8',
        ),
        yaxis=dict(
            linecolor='#b0c8b8', linewidth=1,
            gridcolor='#c0d8c8', gridwidth=1, showgrid=True,
            tickfont=dict(color='#444444'), ticks='outside', tickcolor='#b0c8b8',
        ),
        margin=dict(t=40, b=40, l=10, r=10),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#b0c8b8', borderwidth=1),
    )
)
pio.templates.default = 'plotly+corp'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
_logo_img = Image.open(os.path.join(BASE_DIR, "imagenes", "Logo.png")).convert("RGBA")
_lw, _lh  = _logo_img.size
# Cuadrado con lado = alto de la imagen, recortado desde el ícono circular
_ix      = int(_lw * 0.09)
_icon    = _logo_img.crop((_ix, 0, _ix + _lh, _lh))
# Fondo transparente
_favicon = Image.new("RGBA", (_lh, _lh), (0, 0, 0, 0))
_favicon.paste(_icon, (0, 0), _icon)

st.set_page_config(
    page_title="Segmentación Corporativo",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.kpi-card {
    background:#f5fbf7; border-radius:12px; padding:18px 14px;
    box-shadow:0 2px 8px rgba(36,116,60,.10);
    border-left:4px solid #24743c;
    text-align:center; margin-bottom:8px;
}
.kpi-title { font-size:12px; color:#79a65c; margin-bottom:4px;
             text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
.kpi-value { font-size:24px; font-weight:700; color:#24743c; }
.seg-chip  { display:inline-block; border-radius:20px; padding:4px 16px;
             font-size:13px; font-weight:600; color:#fff; margin:4px 2px; }
.progress-wrap { background:#e4f0e8; border-radius:8px; height:13px; margin:3px 0; }
.progress-bar  { height:13px; border-radius:8px; background:#24743c;
                 transition:width .5s ease; }
/* sidebar corporativa */
section[data-testid="stSidebar"] { background:#f0f8f2 !important; }
/* ocultar botón de descarga nativo del st.dataframe */
[data-testid="stDataFrameResizable"] toolbar,
[data-testid="stElementToolbar"],
[data-testid="stDataFrame"] [data-testid="stElementToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTENTICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _load_users() -> dict:
    path = os.path.join(BASE_DIR, "users.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _login_page() -> None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("## 🔐 Acceso al sistema")
        st.caption("Segmentación Corporativo")
        st.divider()
        with st.form("login_form"):
            usuario = st.text_input("Usuario", placeholder="usuario")
            clave   = st.text_input("Contraseña", type="password", placeholder="••••••••")
            btn     = st.form_submit_button("Ingresar", use_container_width=True, type="primary")
        if btn:
            _USERS = _load_users()
            if usuario in _USERS and _hash_pw(clave) == _USERS[usuario]:
                token    = str(uuid.uuid4())
                sessions = _load_sessions()
                sessions[token] = {
                    "usuario":       usuario,
                    "last_activity": datetime.datetime.now().isoformat(),
                }
                _save_sessions(sessions)
                st.query_params["session"] = token
                st.session_state["auth"]    = True
                st.session_state["usuario"] = usuario
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

def _check_auth() -> None:
    if st.session_state.get("auth"):
        return

    token = st.query_params.get("session")

    if not token:
        _login_page()
        st.stop()

    sessions = _load_sessions()
    if token not in sessions:
        st.query_params.clear()
        _login_page()
        st.stop()

    sesion      = sessions[token]
    last_act    = datetime.datetime.fromisoformat(sesion["last_activity"])
    inactividad = (datetime.datetime.now() - last_act).total_seconds() / 60

    if inactividad > INACTIVITY_TIMEOUT:
        del sessions[token]
        _save_sessions(sessions)
        st.query_params.clear()
        st.session_state.clear()
        st.warning(f"Sesión cerrada por inactividad ({INACTIVITY_TIMEOUT} min).")
        _login_page()
        st.stop()

    sessions[token]["last_activity"] = datetime.datetime.now().isoformat()
    _save_sessions(sessions)
    st.session_state["auth"]    = True
    st.session_state["usuario"] = sesion["usuario"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title: str, value: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def barra(valor: float, color: str = "#047c5c") -> str:
    v = max(0.0, min(float(valor), 100.0))
    return (
        f'<div class="progress-wrap">'
        f'<div class="progress-bar" style="width:{v}%;background:{color};"></div>'
        f'</div>'
    )


def chip_segmento(cluster: int) -> str:
    color  = PAL.get(cluster, "#888")
    nombre = NOMBRES.get(cluster, f"Cluster {cluster}")
    return f'<span class="seg-chip" style="background:{color};">{nombre}</span>'


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte #RRGGBB o #RGB a rgba(r,g,b,a), formato aceptado por Plotly."""
    hc = str(hex_color).strip().lstrip("#")
    if len(hc) == 3:
        hc = "".join(ch * 2 for ch in hc)
    if len(hc) != 6:
        return f"rgba(136, 136, 136, {alpha})"
    r = int(hc[0:2], 16)
    g = int(hc[2:4], 16)
    b = int(hc[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _limpiar_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina caracteres de control que openpyxl rechaza."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: "".join(c for c in str(x) if ord(c) >= 32) if pd.notna(x) else x
        )
    return df


def top_pct(series: pd.Series) -> tuple:
    s = series.dropna()
    if s.empty:
        return "N/A", 0.0
    vc = s.value_counts(normalize=True)
    return str(vc.index[0]), float(vc.iloc[0] * 100)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (cacheados) — lee Parquet si existe, xlsx como respaldo
# ─────────────────────────────────────────────────────────────────────────────
def _leer_tabla_clusters() -> pd.DataFrame:
    pq = os.path.join(BASE_DIR, "tabla_clusters.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    # Respaldo: Excel + merge geo
    df = pd.read_excel(os.path.join(BASE_DIR, "tabla_clusters.xlsx"))
    geo_cols = ["REGION", "DEPARTAMENTO", "CIUDAD_STD", "Ciudad", "Estrato"]
    missing  = [c for c in geo_cols if c not in df.columns]
    if missing:
        geo_path = os.path.join(BASE_DIR, "data_limpia.csv")
        if os.path.exists(geo_path):
            key = next((c for c in ["Contrato", "NIT", "Nit"] if c in df.columns), None)
            if key:
                geo_src = pd.read_csv(geo_path, usecols=lambda c: c in [key] + geo_cols,
                                      low_memory=False)
                geo_src = geo_src.drop_duplicates(subset=key)
                df = df.merge(
                    geo_src[[key] + [c for c in geo_cols if c in geo_src.columns]],
                    on=key, how="left", suffixes=("", "_geo"),
                )
    return df


@st.cache_data(show_spinner="Cargando datos…")
def cargar_datos() -> tuple:
    pq_res = os.path.join(BASE_DIR, "df_res.parquet")
    df_res  = pd.read_parquet(pq_res) if os.path.exists(pq_res) \
              else pd.read_excel(os.path.join(BASE_DIR, "df_res.xlsx"))
    df_full = _leer_tabla_clusters()
    return df_res, df_full


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo() -> tuple:
    """Carga pipeline RF + artefactos de preprocesamiento."""
    try:
        import joblib
    except ImportError:
        st.warning("joblib no instalado. Inferencia deshabilitada.")
        return None, None, {}, {}

    def _load(fname):
        fp = os.path.join(BASE_DIR, fname)
        return joblib.load(fp) if os.path.exists(fp) else None

    modelo    = _load("modelo_clusters_rf.pkl")
    pt        = _load("power_transformers.pkl")
    imp_stats = _load("imputation_stats.pkl") or {}
    meta      = {}
    mp = os.path.join(BASE_DIR, "modelo_metadata.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    return modelo, pt, imp_stats, meta


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DATOS COMPLETOS — solo al entrar a Estadísticas (carga diferida)
# Lee Parquet si existe, CSV como respaldo
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset completo…")
def cargar_datos_full() -> pd.DataFrame:
    pq = os.path.join(BASE_DIR, "clientes_segmentados.parquet")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    else:
        _cols = [
            "Contrato", "Nit", "Entidad", "Ciudad", "CIUDAD_STD", "DEPARTAMENTO", "REGION",
            "Edad", "Sexo", "Estadocivil", "Estrato", "Fechaingreso",
            "Tienepadres", "Tieneesposa", "Tienehijos", "Tieneperro", "Tienegato",
            "Cantidad_mascotas", "PromEdadMascotas",
            "Producto", "Canal", "Valormensual", "Valortotalplan", "Cuotas", "Act.valor",
            "Total_afiliados", "Rango_afiliados", "Antigedad", "Rango_antigedad",
            "Rentabilidad", "Rango_produccion", "Rango_siniestralidad", "Rango_tarifa",
            "SECTOR_EMPLEADOR", "ACTIVIDAD_ECONOMICA", "Tiposseguros_ajuste",
            "TienePadres_V", "TieneEsposa_V", "TieneHijos_V",
            "Salud", "Bicicleta", "Repatriacion", "Expatriacion",
            "Cluster",
        ]
        df = pd.read_csv(
            os.path.join(BASE_DIR, "clientes_segmentados.csv"),
            low_memory=False, usecols=lambda c: c in _cols,
        )
    rename = {"Cluster": "cluster", "Estadocivil": "EstadoCivil", "Nit": "NIT"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "CIUDAD_STD" not in df.columns and "Ciudad" in df.columns:
        df["CIUDAD_STD"] = df["Ciudad"]
    for bin_col, src in [("TienePadres_V", "Tienepadres"), ("TieneEsposa_V", "Tieneesposa"),
                          ("TieneHijos_V",  "Tienehijos")]:
        if bin_col not in df.columns and src in df.columns:
            df[bin_col] = (df[src] == "Y").astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENRIQUECIMIENTO DEL RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def enriquecer(df_res: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de producto, canal, región, estrato y departamento
    por cluster. Añade nombre_segmento y distribución de sexo.
    """
    def metricas(g: pd.DataFrame) -> pd.Series:
        d = {}
        for col, key in [
            ("Producto",            "producto"),
            ("Canal",               "canal"),
            ("REGION",              "region"),
            ("Estrato",             "estrato"),
            ("DEPARTAMENTO",        "departamento"),
            ("Ciudad",              "ciudad"),
            ("SECTOR_EMPLEADOR",    "sector_empleador"),
            ("ACTIVIDAD_ECONOMICA", "actividad_economica"),
        ]:
            t, p = top_pct(g[col]) if col in g.columns else ("N/A", 0.0)
            d[f"{key}_top"] = t
            d[f"{key}_pct"] = p
        return pd.Series(d)

    extra = df_full.groupby("cluster", group_keys=False).apply(metricas).reset_index()
    # Drop columns already in df_res to prevent _x/_y suffix conflicts on merge
    extra = extra.drop(columns=[c for c in extra.columns if c != "cluster" and c in df_res.columns])

    # Género
    sx = df_full["Sexo"].astype(str).str.strip().str.upper().replace(
        {"MASCULINO": "M", "FEMENINO": "F"}
    )
    sp = (
        df_full.assign(Sexo_norm=sx)
        .groupby(["cluster", "Sexo_norm"])
        .size()
        .reset_index(name="n")
    )
    sp["pct"] = sp.groupby("cluster")["n"].transform(lambda x: x / x.sum() * 100)
    piv = sp.pivot(index="cluster", columns="Sexo_norm", values="pct").fillna(0)
    for c in ["M", "F"]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.rename(columns={"M": "sexo_M_pct", "F": "sexo_F_pct"})[
        ["sexo_M_pct", "sexo_F_pct"]
    ].reset_index()

    return (
        df_res
        .merge(extra, on="cluster", how="left")
        .merge(piv,   on="cluster", how="left")
        .assign(nombre_segmento=lambda d: d["cluster"].map(NOMBRES))
        .sort_values("%_cluster", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETACIÓN DEL SEGMENTO
# ─────────────────────────────────────────────────────────────────────────────
def texto_interpretacion(row: pd.Series) -> str:
    costo    = str(row.get("valor_promedio_plan", "")).lower()
    nivel    = "alto" if "alto" in costo else "medio" if "medio" in costo else "bajo"
    edad_rng = f"{int(row.get('edad_q1',0))}–{int(row.get('edad_q3',0))} años"
    tam_rng  = f"{int(row.get('tamano_grupo_q1',0))}–{int(row.get('tamano_grupo_q3',0))} contratos"
    prod  = row.get("producto_top", "N/A");  prod_p  = float(row.get("producto_pct", 0))
    canal = row.get("canal_top",    "N/A");  canal_p = float(row.get("canal_pct",    0))
    reg   = row.get("region_top",   "N/A");  reg_p   = float(row.get("region_pct",   0))
    est   = row.get("estrato_top",  "N/A");  est_p   = float(row.get("estrato_pct",  0))
    pareja = float(row.get("% Con pareja", 0))
    hijos  = float(row.get("% Con hijos",  0))
    padres = float(row.get("% Con padres", 0))
    sexo_m = float(row.get("sexo_M_pct",   0))
    sexo_f = float(row.get("sexo_F_pct",   0))
    sexo_txt = (
        "equilibrada" if abs(sexo_m - sexo_f) < 5
        else f"predominio M ({sexo_m:.1f}%)" if sexo_m > sexo_f
        else f"predominio F ({sexo_f:.1f}%)"
    )
    cl     = int(row.get("cluster", 0))
    nom    = row.get("nombre_segmento", f"Cluster {cl}")
    accion = ACCIONES.get(cl, "—")

    return f"""
**{nom}** · {row['%_cluster']:.1f}% del total de clientes

| Dimensión          | Detalle |
|--------------------|---------|
| Plan               | Nivel **{nivel}** · entidad **{row.get('categoria_entidad','N/A')}** |
| Edad               | {edad_rng} |
| Tamaño grupo       | {tam_rng} |
| Producto dominante | **{prod}** ({prod_p:.1f}%) |
| Canal principal    | **{canal}** ({canal_p:.1f}%) |
| Región             | **{reg}** ({reg_p:.1f}%) |
| Estrato            | **{est}** ({est_p:.1f}%) |
| Familia            | Pareja {pareja:.1f}% · Hijos {hijos:.1f}% · Padres {padres:.1f}% |
| Género             | {sexo_txt} |

**Acción sugerida:** {accion}
"""


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────
def predecir_cluster(datos: dict, modelo, pt, imp_stats: dict) -> tuple:
    """
    Replica el pipeline de NB02:
      1. Imputa NaN con estadísticos de entrenamiento
      2. Fusiona TieneEsposa_V + TieneHijos_V → tiene_nucleo_familiar (si aplica)
      3. PowerTransformer (Yeo-Johnson) sobre columnas numéricas
      4. RF pipeline → predict + predict_proba
    """
    df_inf = pd.DataFrame([datos])

    # 1. Imputación
    for col, val in imp_stats.items():
        if col in df_inf.columns and df_inf[col].isna().any():
            df_inf[col] = df_inf[col].fillna(val)

    # 2. Fusión nucleo familiar (por si vienen variables originales en vez de fusionada)
    if "TieneEsposa_V" in df_inf.columns or "TieneHijos_V" in df_inf.columns:
        a = df_inf.get("TieneEsposa_V", pd.Series(["0"], index=df_inf.index))
        b = df_inf.get("TieneHijos_V",  pd.Series(["0"], index=df_inf.index))
        df_inf["tiene_nucleo_familiar"] = (
            (a.astype(str) == "1") | (b.astype(str) == "1")
        ).map({True: "SI", False: "NO"})
        df_inf.drop(
            columns=[c for c in ["TieneEsposa_V", "TieneHijos_V"] if c in df_inf.columns],
            inplace=True,
        )

    # 3. PowerTransformer
    if pt is not None and hasattr(pt, "feature_names_in_"):
        num_cols = [c for c in pt.feature_names_in_ if c in df_inf.columns]
        if num_cols:
            df_inf[num_cols] = pt.transform(df_inf[num_cols])

    # 4. Predicción
    cluster = int(modelo.predict(df_inf)[0])
    proba   = (
        modelo.predict_proba(df_inf)[0]
        if hasattr(modelo, "predict_proba")
        else None
    )
    return cluster, proba


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def pagina_dashboard(df_summary: pd.DataFrame, df_full: pd.DataFrame) -> None:
    _c1, _c2 = st.columns([0.06, 0.94])
    with _c1:
        st.image(os.path.join(BASE_DIR, "imagenes", "dashboard-icon-23660.png"), width=42)
    with _c2:
        st.header("Dashboard — Resumen ejecutivo")

    # KPIs globales
    n_seg  = df_summary["cluster"].nunique()
    total  = len(df_full)
    kpis   = st.columns(3 + n_seg)
    with kpis[0]: kpi_card("Total clientes", f"{total:,}")
    with kpis[1]: kpi_card("Segmentos", str(n_seg))
    with kpis[2]: kpi_card("Algoritmo", "K-Prototypes")
    for j, (_, row) in enumerate(df_summary.iterrows()):
        with kpis[3 + j]:
            kpi_card(
                row.get("nombre_segmento", f"Cluster {int(row['cluster'])}"),
                f"{row['%_cluster']:.1f}%",
            )

    st.divider()

    # Donut + Barras de valor
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución de segmentos")
        fig_d = px.pie(
            df_summary,
            names="nombre_segmento",
            values="%_cluster",
            hole=0.48,
            color="cluster",
            color_discrete_map={int(r["cluster"]): PAL.get(int(r["cluster"]), "#888")
                                 for _, r in df_summary.iterrows()},
        )
        fig_d.update_traces(textinfo="percent+label", textfont_size=13)
        fig_d.update_layout(margin=dict(t=10, b=10), height=320, showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True, key="pc_001")

    with c2:
        st.subheader("Valor del plan por segmento (Q1 – Q3)")
        fig_b = go.Figure()
        for _, row in df_summary.iterrows():
            cl  = int(row["cluster"])
            q1  = float(row.get("valor_total_pesos_q1", 0))
            q3  = float(row.get("valor_total_pesos_q3", 0))
            nom = row.get("nombre_segmento", f"Cluster {cl}")
            fig_b.add_trace(go.Bar(
                name=nom, x=[nom], y=[q3 - q1], base=[q1],
                marker_color=PAL.get(cl, "#888"),
                text=[f"${q1:,.0f}–${q3:,.0f}"],
                textposition="outside",
            ))
        fig_b.update_layout(
            height=320, barmode="group", showlegend=False,
            yaxis_title="Valor plan ($COP)",
            yaxis=dict(tickformat="$,.0f"),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_b, use_container_width=True, key="pc_002")

    st.divider()

    # Distribución geográfica
    st.subheader("Distribución geográfica por segmento")
    geo_cols_avail = [c for c in ["REGION", "DEPARTAMENTO", "Ciudad"] if c in df_full.columns]
    if geo_cols_avail:
        gcol_ctrl, gcol_chart = st.columns([1, 3])
        with gcol_ctrl:
            geo_var = st.selectbox("Variable geográfica", geo_cols_avail, key="geo_var_dash")
            top_n   = st.slider("Top N categorías", 3, 10, 5, key="geo_topn_dash")

        grp = (
            df_full.groupby(["cluster", geo_var])
            .size()
            .reset_index(name="n")
        )
        grp["pct"] = (
            grp.groupby("cluster")["n"]
            .transform(lambda x: (x / x.sum() * 100).round(1))
        )
        grp["Segmento"] = grp["cluster"].map(NOMBRES)
        top_cats = (
            grp.groupby(geo_var)["n"].sum()
            .nlargest(top_n).index.tolist()
        )
        grp_f = grp[grp[geo_var].isin(top_cats)].copy()

        chart_h = max(320, top_n * 60)
        fig_geo = px.bar(
            grp_f,
            y=geo_var, x="pct",
            color="Segmento",
            barmode="group",
            orientation="h",
            color_discrete_map={v: PAL.get(k, "#888") for k, v in NOMBRES.items()},
            text=grp_f["pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"pct": "% del segmento", geo_var: ""},
            height=chart_h,
        )
        fig_geo.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22),
            margin=dict(t=10, b=10, l=10, r=80),
            yaxis=dict(autorange="reversed"),
        )
        fig_geo.update_traces(textposition="outside",
                              textfont=dict(size=12, color="#24743c", family="sans-serif"))
        with gcol_chart:
            st.plotly_chart(fig_geo, use_container_width=True, key="pc_003")
    else:
        st.info("No se encontraron columnas geográficas (REGION, DEPARTAMENTO, Ciudad) en los datos.")

    st.divider()

    # Actividad económica y sector empleador
    act_cols_avail = [c for c in ["SECTOR_EMPLEADOR", "ACTIVIDAD_ECONOMICA"] if c in df_full.columns]
    if act_cols_avail:
        st.subheader("Actividad económica y sector empleador")
        ac1, ac2 = st.columns([1, 3])
        with ac1:
            act_var = st.selectbox("Variable", act_cols_avail, key="act_var_dash")
            act_n   = st.slider("Top N", 3, 15, 8, key="act_topn_dash")

        df_act = df_full[["cluster", act_var]].copy()
        df_act[act_var] = df_act[act_var].fillna("Sin categoría identificada").astype(str).str.strip()
        df_act[act_var] = df_act[act_var].replace({"": "Sin categoría identificada", "nan": "Sin categoría identificada"})

        agrp = (
            df_act.groupby(["cluster", act_var])
            .size()
            .reset_index(name="n")
        )
        agrp["pct"] = (
            agrp.groupby("cluster")["n"]
            .transform(lambda x: (x / x.sum() * 100).round(1))
        )
        agrp["Segmento"] = agrp["cluster"].map(NOMBRES)

        # Top N excluyendo "Sin categoría" para el ranking, pero siempre incluyéndola
        ranked = (
            agrp[agrp[act_var] != "Sin categoría identificada"]
            .groupby(act_var)["n"].sum()
            .nlargest(act_n).index.tolist()
        )
        top_act = ranked + ["Sin categoría identificada"]
        agrp_f = agrp[agrp[act_var].isin(top_act)].copy()

        fig_act = px.bar(
            agrp_f,
            y=act_var, x="pct",
            color="Segmento",
            barmode="group",
            orientation="h",
            color_discrete_map={v: PAL.get(k, "#888") for k, v in NOMBRES.items()},
            text=agrp_f["pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"pct": "% del segmento", act_var: ""},
            height=max(320, act_n * 60),
        )
        fig_act.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22),
            margin=dict(t=10, b=10, l=10, r=80),
            yaxis=dict(autorange="reversed"),
        )
        fig_act.update_traces(textposition="outside",
                              textfont=dict(size=12, color="#24743c", family="sans-serif"))
        with ac2:
            st.plotly_chart(fig_act, use_container_width=True, key="pc_004")

    st.divider()

    # Radar comparativo
    st.subheader("Radar comparativo — 6 dimensiones normalizadas")
    with st.expander("¿Cómo leer este radar?", expanded=False):
        st.markdown("""
**El radar compara los 3 segmentos en 6 dimensiones clave del perfil del cliente.**
Cada eje muestra la posición relativa del segmento — no valores absolutos, sino quién es
mayor o menor que los otros en esa dimensión.

| Eje | Qué mide | Valor hacia afuera significa |
|-----|----------|------------------------------|
| **Edad mediana** | Edad central del segmento | Clientes más maduros |
| **Estrato** | Nivel socioeconómico dominante | Mayor capacidad adquisitiva |
| **Tamaño grupo** | Tamaño mediano del grupo asegurado | Grupos más grandes |
| **Valor plan** | Precio mediano del plan contratado | Planes de mayor valor |
| **% Con pareja** | % con cónyuge o unión libre | Mayor vínculo de pareja |
| **% Con hijos** | % con hijos en el plan | Mayor carga familiar |

**Escala — Normalización Min-Max con piso del 15 %:**

Cada eje aplica la fórmula:
```
posición en el radar = 0.15 + 0.85 × (valor − mínimo) / (máximo − mínimo)
```
El segmento con el valor **más bajo** en un eje aparece al 15 % del radio.
El segmento con el valor **más alto** aparece al 100 %.
El piso del 15 % evita que un segmento desaparezca en el centro cuando domina en otras dimensiones.

⚠️ **Lo que importa es quién está más afuera que quién en cada eje, no la distancia al centro.**

> Ejemplo: un polígono angosto en "Con hijos" pero amplio en "Valor plan"
> describe grupos sin dependientes pero de alto ticket.
        """)

    dims_cands = [
        ("edad_q50",              "Edad mediana"),
        ("estrato_q50",           "Estrato"),
        ("tamano_grupo_q50",      "Tamaño grupo"),
        ("valor_total_pesos_q50", "Valor plan"),
        ("% Con pareja",          "Con pareja"),
        ("% Con hijos",           "Con hijos"),
    ]
    dims   = [d for d, _ in dims_cands if d in df_summary.columns]
    labels = [l for d, l in dims_cands if d in df_summary.columns]

    if len(dims) >= 3:
        vals   = df_summary[dims].values.astype(float)
        mn     = np.nanmin(vals, axis=0)
        mx     = np.nanmax(vals, axis=0)
        mx     = np.where(mx == mn, mn + 1, mx)
        # Normalizar a [0.15, 1.0] para que el segmento mínimo tenga presencia visual
        raw_n  = np.where(np.isnan(vals), 0.0, (vals - mn) / (mx - mn))
        vals_n = 0.15 + 0.85 * raw_n

        # Diagnóstico: muestra valores reales y normalizados
        with st.expander("Diagnóstico radar — valores por segmento", expanded=False):
            segs = df_summary["nombre_segmento"].tolist()
            df_diag_raw  = pd.DataFrame(vals,   columns=dims, index=segs).round(1)
            df_diag_norm = pd.DataFrame(vals_n, columns=dims, index=segs).round(3)
            st.caption("**Valores reales** (de df_res.parquet)")
            st.dataframe(df_diag_raw,  use_container_width=True)
            st.caption("**Valores normalizados** (0–1, min-max entre segmentos)")
            st.dataframe(df_diag_norm, use_container_width=True)
            nans_por_col = pd.isna(vals).sum(axis=0)
            if nans_por_col.any():
                st.warning(f"Columnas con NaN: { {d: int(n) for d, n in zip(dims, nans_por_col) if n > 0} }")

        fig_r = go.Figure()
        for idx, (_, row) in enumerate(df_summary.iterrows()):
            cl    = int(row["cluster"])
            nom   = row.get("nombre_segmento", f"Cluster {cl}")
            v     = vals_n[idx].tolist() + [vals_n[idx][0]]
            hex_c = PAL.get(cl, "#888")
            fig_r.add_trace(go.Scatterpolar(
                r=v,
                theta=labels + [labels[0]],
                name=nom,
                fill="toself",
                mode="lines+markers",
                line=dict(color=hex_c, width=2.5),
                fillcolor=hex_to_rgba(hex_c, 0.25),
                marker=dict(size=5, color=hex_c),
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(categoryarray=labels),
            ),
            height=440, margin=dict(t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="pc_005")
    else:
        st.info("Radar no disponible: faltan columnas `_q50` en df_res.xlsx. Ejecuta NB03 completo primero.")

    # Tabla resumen
    st.subheader("Tabla resumen de segmentos")
    cols_show = [c for c in [
        "cluster", "nombre_segmento", "%_cluster",
        "edad_q1", "edad_q50", "edad_q3",
        "valor_total_pesos_q1", "valor_total_pesos_q50", "valor_total_pesos_q3",
        "% Con pareja", "% Con hijos", "% Con padres",
        "producto_top", "canal_top",
        "region_top", "departamento_top", "ciudad_top",
        "generacion_moda", "categoria_entidad",
    ] if c in df_summary.columns]
    st.dataframe(df_summary[cols_show], use_container_width=True)

    # Detalle por segmento (tabs)
    st.divider()
    st.subheader("Detalle por segmento")
    tab_labels = [
        f"Segmento {int(r['cluster'])} · {r['%_cluster']:.1f}%"
        for _, r in df_summary.iterrows()
    ]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            row = df_summary.iloc[i]
            cl  = int(row["cluster"])
            st.markdown(chip_segmento(cl), unsafe_allow_html=True)
            col1, col2 = st.columns([1.1, 2])

            with col1:
                st.metric("% del total",        f"{row['%_cluster']:.1f}%")
                st.metric("Nivel plan",          row.get("valor_promedio_plan", "N/A"))
                st.metric("Categoría entidad",   row.get("categoria_entidad",   "N/A"))
                st.markdown("**Demografía**")
                st.write(f"Edad: **{int(row.get('edad_q1',0))} – {int(row.get('edad_q3',0))} años**")
                st.write(f"Generación: **{row.get('generacion_moda','N/A')}**")
                st.write(f"Contratos/entidad: **{int(row.get('tamano_grupo_q1',0))} – {int(row.get('tamano_grupo_q3',0))}**")
                st.markdown("**Producto · Canal**")
                st.write(f"**{row.get('producto_top','N/A')}** ({float(row.get('producto_pct',0)):.1f}%)")
                st.write(f"**{row.get('canal_top','N/A')}** ({float(row.get('canal_pct',0)):.1f}%)")

            with col2:
                st.markdown("**Rango económico (Q1–Q3)**")
                q1 = float(row.get("valor_total_pesos_q1", 0))
                q3 = float(row.get("valor_total_pesos_q3", 0))
                rq1, rq2 = st.columns(2)
                with rq1: st.metric("Q1", f"${q1:,.0f}")
                with rq2: st.metric("Q3", f"${q3:,.0f}")

                st.markdown("**Composición familiar**")
                for lbl, key, col in [
                    ("Con pareja", "% Con pareja", C_VERDE),
                    ("Con hijos",  "% Con hijos",  C_V_PAL),
                    ("Con padres", "% Con padres", C_AMARILLO),
                ]:
                    v = float(row.get(key, 0))
                    st.markdown(barra(v, col), unsafe_allow_html=True)
                    st.caption(f"{lbl}: {v:.1f}%")

                st.markdown("**Distribución por género**")
                m  = float(row.get("sexo_M_pct", 0))
                f_ = float(row.get("sexo_F_pct", 0))
                st.markdown(barra(m, C_VERDE), unsafe_allow_html=True)
                st.caption(f"Masculino: {m:.1f}%")
                st.markdown(barra(f_, C_V_PAL), unsafe_allow_html=True)
                st.caption(f"Femenino: {f_:.1f}%")

                st.markdown("**Geografía**")
                gc1, gc2, gc3, gc4 = st.columns(4)
                with gc1: st.metric("Región",   f"{row.get('region_top','N/A')} ({float(row.get('region_pct',0)):.1f}%)")
                with gc2: st.metric("Depto",    f"{row.get('departamento_top','N/A')} ({float(row.get('departamento_pct',0)):.1f}%)")
                with gc3: st.metric("Ciudad",   f"{row.get('ciudad_top','N/A')} ({float(row.get('ciudad_pct',0)):.1f}%)")
                with gc4: st.metric("Estrato",  f"{row.get('estrato_top','N/A')} ({float(row.get('estrato_pct',0)):.1f}%)")

            st.divider()
            st.info(texto_interpretacion(row))


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — PCA INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
def pagina_pca(df_full: pd.DataFrame) -> None:
    _c1, _c2 = st.columns([0.06, 0.94])
    with _c1:
        st.image(os.path.join(BASE_DIR, "imagenes", "PCA.png"), width=42)
    with _c2:
        st.header("PCA 2D Interactivo")
    st.caption(
        "PCA calculado en tiempo real sobre variables numéricas del modelo. "
        "Cada punto = un cliente. Pasa el cursor para ver su perfil."
    )

    NUM_COLS = [
        "Total_afiliados", "Cuotas", "Cantidad_mascotas",
        "Edad", "ValorTotal_scaled", "Estrato", "ValorTotalPlan", "tamano_grupo",
    ]
    available = [c for c in NUM_COLS if c in df_full.columns]

    if len(available) < 2:
        st.error("Columnas numéricas insuficientes en `tabla_clusters.parquet`.")
        return

    with st.sidebar:
        st.divider()
        n_max   = min(50_000, len(df_full))
        muestra = st.slider("Muestra (registros)", 1_000, n_max, min(15_000, n_max), 1_000)
        color_by = st.selectbox(
            "Colorear por",
            ["Segmento"] + [c for c in ["Producto", "Canal", "REGION"] if c in df_full.columns],
        )
        opacidad = st.slider("Opacidad", 0.1, 1.0, 0.60, 0.05)

    df_s = (
        df_full
        .sample(n=min(muestra, len(df_full)), random_state=42)
        .dropna(subset=available)
        .copy()
    )

    X_sc   = StandardScaler().fit_transform(df_s[available].values)
    pca_   = PCA(n_components=2, random_state=42).fit(X_sc)
    coords = pca_.transform(X_sc)
    exp_v  = pca_.explained_variance_ratio_ * 100

    df_s["PC1"] = coords[:, 0]
    df_s["PC2"] = coords[:, 1]
    df_s["nombre_segmento"] = df_s["cluster"].map(NOMBRES)

    hover_cols = [c for c in ["nombre_segmento", "Edad", "ValorTotalPlan",
                               "Estrato", "Producto", "Canal"] if c in df_s.columns]

    if color_by == "Segmento":
        color_col = "nombre_segmento"
        cmap      = {v: PAL[k] for k, v in NOMBRES.items() if k in df_s["cluster"].values}
    else:
        color_col = color_by
        cmap      = None

    fig_pca = px.scatter(
        df_s, x="PC1", y="PC2",
        color=color_col,
        color_discrete_map=cmap,
        hover_data=hover_cols,
        opacity=opacidad,
        labels={
            "PC1": f"PC1 ({exp_v[0]:.1f}% var.)",
            "PC2": f"PC2 ({exp_v[1]:.1f}% var.)",
        },
        title=f"PCA 2D — {len(df_s):,} registros",
        height=580,
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(legend_title=color_by, margin=dict(t=40, b=10))
    st.plotly_chart(fig_pca, use_container_width=True, key="pc_006")

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Varianza PC1",           f"{exp_v[0]:.1f}%")
    with m2: st.metric("Varianza PC2",           f"{exp_v[1]:.1f}%")
    with m3: st.metric("Varianza total PC1+PC2", f"{sum(exp_v):.1f}%")

    with st.expander("Variables usadas para PCA"):
        st.write(available)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — EXPLORAR REGISTROS
# ─────────────────────────────────────────────────────────────────────────────
def pagina_explorar(df_full: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    _c1, _c2 = st.columns([0.06, 0.94])
    with _c1:
        st.image(os.path.join(BASE_DIR, "imagenes", "exportar.png"), width=42)
    with _c2:
        st.header("Explorar registros por segmento")

    try:
        df_view = cargar_datos_full()
    except Exception:
        df_view = df_full.copy()
        st.warning("No se pudo cargar clientes_segmentados.parquet. Se usa tabla_clusters como respaldo.")

    if "nombre_segmento" not in df_view.columns and "cluster" in df_view.columns:
        df_view["nombre_segmento"] = df_view["cluster"].map(NOMBRES)

    cf1, cf2, cf3 = st.columns([1.5, 1.5, 1])

    with cf1:
        opciones = ["Todos"] + [
            f"{int(r['cluster'])} · {r.get('nombre_segmento','')}"
            for _, r in df_summary.iterrows()
        ]
        seg_sel = st.selectbox("Segmento", opciones)

    if seg_sel != "Todos" and "cluster" in df_view.columns:
        cl_num  = int(seg_sel.split("·")[0].strip())
        df_view = df_view[df_view["cluster"] == cl_num]

    with cf2:
        cat_cols = [c for c in [
            "Producto", "Canal", "REGION", "DEPARTAMENTO", "Ciudad", "Sexo",
            "SECTOR_EMPLEADOR", "ACTIVIDAD_ECONOMICA",
        ] if c in df_view.columns]
        col_f = st.selectbox("Filtro adicional", ["—"] + cat_cols)
        if col_f != "—":
            vals_f = ["Todos"] + sorted(df_view[col_f].dropna().astype(str).unique().tolist())
            val_f  = st.selectbox(f"Valor de {col_f}", vals_f)
            if val_f != "Todos":
                df_view = df_view[df_view[col_f].astype(str) == val_f]

    with cf3:
        st.metric("Registros filtrados", f"{len(df_view):,}")

    col_prio = [
        "cluster", "nombre_segmento", "Contrato", "Entidad",
        "Edad", "Sexo", "Estrato", "Producto", "Canal",
        "ValorTotalPlan", "tamano_grupo", "REGION", "DEPARTAMENTO", "Ciudad",
        "generacion_moda", "categoria_entidad",
        "SECTOR_EMPLEADOR", "ACTIVIDAD_ECONOMICA",
    ]
    _excluir_default = {"ValorTotal_scaled"}
    col_prio  = [c for c in col_prio if c in df_view.columns]
    resto_cols = [c for c in df_view.columns if c not in col_prio and c not in _excluir_default]
    default_cols = col_prio + resto_cols

    cols_sel = st.multiselect(
        "Columnas a mostrar",
        options=df_view.columns.tolist(),
        default=default_cols,
    )
    if not cols_sel:
        cols_sel = default_cols

    MAX_FILAS = 5_000
    df_display = df_view[cols_sel].reset_index(drop=True)
    if len(df_display) > MAX_FILAS:
        st.caption(f"Mostrando {MAX_FILAS:,} de {len(df_display):,} registros. Usa los filtros o exporta para ver todos.")
        df_display = df_display.head(MAX_FILAS)
    st.dataframe(df_display, use_container_width=True, height=520)

    # ── Análisis por sector y actividad económica ──────────────────────────
    if "SECTOR_EMPLEADOR" in df_view.columns or "ACTIVIDAD_ECONOMICA" in df_view.columns:
        st.divider()
        st.subheader("🏭 Análisis por entidad y sector")

        SIN_INFO       = "Sin información"
        COLOR_SIN_INFO = "#AAAAAA"

        def _cmap(extra_seg=None):
            cm = {v: PAL[k] for k, v in NOMBRES.items()}
            cm[SIN_INFO] = COLOR_SIN_INFO
            if extra_seg:
                cm.update(extra_seg)
            return cm

        if "SECTOR_EMPLEADOR" in df_view.columns:
            _sec = df_view["SECTOR_EMPLEADOR"].fillna(SIN_INFO)
            _seg = df_view["nombre_segmento"].fillna(SIN_INFO)
            sec_tot = _sec.value_counts().reset_index()
            sec_tot.columns = ["Sector", "n"]
            n_sectores  = len(sec_tot)
            chart_h     = max(340, n_sectores * 42)
            n_sin_sec   = int((_sec == SIN_INFO).sum())

            sd1, sd2 = st.columns(2)

            with sd1:
                st.markdown("#### Sector empleador — contratos")
                if n_sin_sec:
                    st.caption(f"⚠️ {n_sin_sec:,} registros sin sector (excluidos del ranking).")
                sec_plot = sec_tot[sec_tot["Sector"] != SIN_INFO]
                fig_se = px.bar(
                    sec_plot.sort_values("n"),
                    x="n", y="Sector", orientation="h",
                    text="n",
                    color="Sector",
                    color_discrete_sequence=PALETTE * ((n_sectores // len(PALETTE)) + 1),
                    height=chart_h,
                    labels={"n": "Contratos", "Sector": ""},
                )
                fig_se.update_traces(texttemplate="%{text:,}", textposition="outside",
                                     textfont=dict(size=12, color="#24743c", family="sans-serif"),
                                     marker_line_color="white", marker_line_width=0.6)
                fig_se.update_layout(showlegend=False,
                                     margin=dict(t=10, b=10, r=80),
                                     xaxis=dict(tickformat=",.0f"),
                                     paper_bgcolor="white", plot_bgcolor="#f5fbf7")
                st.plotly_chart(fig_se, use_container_width=True, key="pc_007")

            with sd2:
                st.markdown("#### Composición por segmento")
                sec_seg = (
                    pd.DataFrame({"SECTOR_EMPLEADOR": _sec, "nombre_segmento": _seg})
                    .groupby(["SECTOR_EMPLEADOR", "nombre_segmento"])
                    .size().reset_index(name="n")
                )
                sec_seg["pct"] = sec_seg.groupby("SECTOR_EMPLEADOR")["n"].transform(
                    lambda x: (x / x.sum() * 100).round(1)
                )
                fig_ss = px.bar(
                    sec_seg, x="pct", y="SECTOR_EMPLEADOR", color="nombre_segmento",
                    color_discrete_map=_cmap(),
                    barmode="stack", orientation="h", height=chart_h,
                    labels={"pct": "% clientes", "SECTOR_EMPLEADOR": ""},
                    text="pct",
                )
                fig_ss.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                                     textfont_size=11)
                fig_ss.update_layout(
                    margin=dict(t=10, b=40, l=10, r=10),
                    legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
                    xaxis=dict(ticksuffix="%", range=[0, 100]),
                    yaxis=dict(tickfont=dict(size=11)),
                    paper_bgcolor="white", plot_bgcolor="#f5fbf7",
                )
                st.plotly_chart(fig_ss, use_container_width=True, key="pc_008")

        if "ACTIVIDAD_ECONOMICA" in df_view.columns:
            st.markdown("#### Actividad económica — sector CIIU")
            _act = df_view["ACTIVIDAD_ECONOMICA"].fillna(SIN_INFO)
            _seg = df_view["nombre_segmento"].fillna(SIN_INFO)
            top_n = st.slider("Top actividades a mostrar", 5, 25, 15, key="top_ciiu")

            top_act = (
                pd.DataFrame({"ACTIVIDAD_ECONOMICA": _act, "nombre_segmento": _seg})
                .groupby(["ACTIVIDAD_ECONOMICA", "nombre_segmento"])
                .size().reset_index(name="n")
            )
            totales = top_act.groupby("ACTIVIDAD_ECONOMICA")["n"].sum()
            sin_info_existe = SIN_INFO in totales.index
            top_sin_sin = totales.drop(index=SIN_INFO, errors="ignore").nlargest(top_n)
            orden = top_sin_sin.index.tolist()
            if sin_info_existe:
                orden = [SIN_INFO] + orden
            orden_cat = orden[::-1]

            top_act = top_act[top_act["ACTIVIDAD_ECONOMICA"].isin(orden)]
            top_act["ACTIVIDAD_ECONOMICA"] = pd.Categorical(
                top_act["ACTIVIDAD_ECONOMICA"], categories=orden_cat, ordered=True
            )
            fig_ae = px.bar(
                top_act.sort_values("ACTIVIDAD_ECONOMICA"),
                x="n", y="ACTIVIDAD_ECONOMICA", color="nombre_segmento",
                color_discrete_map=_cmap(),
                barmode="stack", orientation="h",
                height=max(420, (top_n + (1 if sin_info_existe else 0)) * 34),
                labels={"n": "Clientes", "ACTIVIDAD_ECONOMICA": ""},
                text="n",
            )
            fig_ae.update_traces(texttemplate="%{text:,}", textposition="inside",
                                 textfont_size=11)
            fig_ae.update_layout(
                margin=dict(t=10, b=10, l=10),
                legend=dict(orientation="h", y=-0.10, font=dict(size=13)),
                xaxis=dict(tickformat=",.0f", tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12)),
            )
            if sin_info_existe:
                n_sin_act = int(totales[SIN_INFO])
                st.caption(f"⚠️ {n_sin_act:,} registros sin actividad económica (barra gris).")
            st.plotly_chart(fig_ae, use_container_width=True, key="pc_009")

        if "ACTIVIDAD_ECONOMICA" in df_view.columns and "nombre_segmento" in df_view.columns:
            with st.expander("Tabla cruzada: Segmento x Actividad económica"):
                _act = df_view["ACTIVIDAD_ECONOMICA"].fillna(SIN_INFO)
                _seg = df_view["nombre_segmento"].fillna(SIN_INFO)
                cross = pd.crosstab(
                    _act, _seg,
                    margins=True, margins_name="Total"
                ).sort_values("Total", ascending=False)
                total_cross = cross.loc["Total", "Total"]
                total_view  = len(df_view)
                if total_cross != total_view:
                    st.warning(f"La tabla incluye {total_cross:,} de {total_view:,} registros ({total_cross/total_view*100:.1f}%).")
                else:
                    st.caption(f"Total: {total_cross:,} registros — todos incluidos.")
                st.dataframe(cross, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 — CLASIFICAR NUEVO CLIENTE
# ─────────────────────────────────────────────────────────────────────────────
def pagina_clasificar(
    modelo,
    pt,
    imp_stats: dict,
    df_summary: pd.DataFrame,
) -> None:
    _c1, _c2 = st.columns([0.06, 0.94])
    with _c1:
        st.image(os.path.join(BASE_DIR, "imagenes", "Cliente.png"), width=42)
    with _c2:
        st.header("Clasificar nuevo cliente")

    if modelo is None:
        st.error(
            "**Modelo no encontrado.** Asegúrate de que `modelo_clusters_rf.pkl` "
            "esté en el mismo directorio que `app.py`."
        )
        return

    _, df_full = cargar_datos()
    canal_opts    = sorted(df_full["Canal"].dropna().astype(str).unique().tolist())    if "Canal"   in df_full.columns else []
    producto_opts = sorted(df_full["Producto"].dropna().astype(str).unique().tolist()) if "Producto" in df_full.columns else []
    region_opts   = sorted(df_full["REGION"].dropna().astype(str).unique().tolist())   if "REGION"  in df_full.columns else []
    sexo_opts     = sorted(df_full["Sexo"].dropna().astype(str).unique().tolist())     if "Sexo"    in df_full.columns else ["F", "M"]
    estado_col    = "EstadoCivil" if "EstadoCivil" in df_full.columns else "Estadocivil"
    estado_opts   = sorted(df_full[estado_col].dropna().astype(str).unique().tolist()) if estado_col in df_full.columns else ["CAS", "DIV", "OTR", "SEP", "SOL", "UNI", "VIU"]
    estado_labels = {"SOL": "Soltero", "CAS": "Casado", "UNI": "Union libre", "SEP": "Separado", "DIV": "Divorciado", "VIU": "Viudo", "OTR": "Otro"}


    st.caption("La app usa las mismas categorias y transformaciones del modelo entrenado.")

    with st.form("form_nuevo_cliente", clear_on_submit=False):
        st.subheader("Variables categoricas")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            canal    = st.selectbox("Canal *",    canal_opts,    help="Canal comercial real observado en entrenamiento")
            producto = st.selectbox("Producto *", producto_opts)

        with fc2:
            region = st.selectbox("REGION *", region_opts)
            sexo   = st.selectbox("Género *",   sexo_opts)

        with fc3:
            tiene_padres = st.selectbox("TienePadres_V *", ["0", "1"], format_func=lambda x: "Si" if x == "1" else "No")
            estadocivil  = st.selectbox("Estado civil *", estado_opts, format_func=lambda x: estado_labels.get(x, x))
            tiene_nucleo = st.selectbox("tiene_nucleo_familiar *", ["SI", "NO"])

        st.subheader("Variables numericas")
        fn1, fn2, fn3 = st.columns(3)

        with fn1:
            total_afiliados = st.number_input("Total_afiliados *", min_value=1, max_value=10000, value=5)
            cuotas          = st.number_input("Cuotas *", min_value=1, max_value=36, value=12)

        with fn2:
            cantidad_mascotas = st.number_input("Cantidad_mascotas", min_value=0, max_value=20, value=1)
            edad              = st.number_input("Edad *", min_value=18, max_value=100, value=40)

        with fn3:
            valor_plan = st.number_input("ValorTotalPlan * ($COP)", min_value=0.0, max_value=50000000.0, value=150000.0, step=1000.0)
            estrato    = st.number_input("Estrato *", min_value=1, max_value=6, value=3)

        st.caption("`ValorTotal_scaled` se calcula con la media y desviacion observadas en `tabla_clusters.parquet`.")
        submitted = st.form_submit_button("Clasificar cliente", type="primary", use_container_width=True)

    if not submitted:
        return

    errores = []
    if not (18 <= edad <= 100):
        errores.append("Edad debe estar entre 18 y 100.")
    if not (1 <= estrato <= 6):
        errores.append("Estrato debe estar entre 1 y 6.")
    if not (1 <= cuotas <= 36):
        errores.append("Cuotas debe estar entre 1 y 36.")
    if total_afiliados < 1:
        errores.append("Total_afiliados debe ser >= 1.")
    if valor_plan < 0:
        errores.append("ValorTotalPlan debe ser >= 0.")
    if errores:
        for e in errores:
            st.error(e)
        return

    val_col = "ValorTotalPlan" if "ValorTotalPlan" in df_full.columns else "Valortotalplan"
    if val_col in df_full.columns:
        vt_series = pd.to_numeric(df_full[val_col], errors="coerce")
        vt_mean   = float(vt_series.mean())
        vt_std    = float(vt_series.std())
    else:
        vt_mean, vt_std = 150000.0, 80000.0
    vt_std       = vt_std if vt_std > 0 else 1.0
    valor_scaled = (valor_plan - vt_mean) / vt_std

    datos = {
        "Canal":                 canal,
        "Producto":              producto,
        "REGION":                region,
        "Sexo":                  sexo,
        "TienePadres_V":         tiene_padres,
        "Estadocivil":           estadocivil,
        "Estado":                "ACT",
        "tiene_nucleo_familiar": tiene_nucleo,
        "Total_afiliados":       float(total_afiliados),
        "Cuotas":                float(cuotas),
        "Cantidad_mascotas":     float(cantidad_mascotas),
        "Edad":                  float(edad),
        "ValorTotal_scaled":     float(valor_scaled),
        "Estrato":               float(estrato),
    }

    with st.spinner("Ejecutando pipeline de inferencia..."):
        try:
            cluster_pred, proba = predecir_cluster(datos, modelo, pt, imp_stats)
        except Exception as ex:
            st.error(f"Error en la prediccion: {ex}")
            st.code(str(ex))
            return

    nombre_seg = NOMBRES.get(cluster_pred, f"Cluster {cluster_pred}")
    st.success(f"### Segmento asignado: **{nombre_seg}** (Cluster {cluster_pred})")
    st.markdown(chip_segmento(cluster_pred), unsafe_allow_html=True)
    st.write("")

    rc1, rc2 = st.columns(2)

    with rc1:
        st.subheader("Perfil del segmento asignado")
        seg_row = df_summary[df_summary["cluster"] == cluster_pred]
        if not seg_row.empty:
            st.info(texto_interpretacion(seg_row.iloc[0]))

    with rc2:
        st.subheader("Probabilidades por segmento")
        if proba is not None:
            n_classes = len(proba)
            prob_df = pd.DataFrame({
                "Cluster":      list(range(n_classes)),
                "Segmento":     [NOMBRES.get(c, f"Cluster {c}") for c in range(n_classes)],
                "Probabilidad": proba,
            })
            fig_p = px.bar(
                prob_df,
                x="Segmento",
                y="Probabilidad",
                color="Segmento",
                color_discrete_map={v: PAL.get(k, "#888") for k, v in NOMBRES.items()},
                text=[f"{p:.1%}" for p in proba],
                range_y=[0, 1],
                height=320,
            )
            fig_p.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(t=10))
            st.plotly_chart(fig_p, use_container_width=True, key="pc_010")
        else:
            st.info("El modelo no retorna probabilidades.")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 5 — ESTADÍSTICAS EMPRESARIALES Y FAMILIAS
# ─────────────────────────────────────────────────────────────────────────────
def _fig_vacio(titulo: str = "", height: int = 320) -> go.Figure:
    """Figura vacía con mensaje cuando no hay datos para el filtro actual."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=35, b=10),
        title=dict(text=titulo, font=dict(size=13, color=C_VERDE)) if titulo else {},
        annotations=[dict(
            text="Sin datos para el filtro seleccionado",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=13, color="#aaaaaa"),
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _bar_h(df_grp: pd.DataFrame, x_col: str, y_col: str,
           n_top: int = 10, height: int = 340, color: str = C_VERDE) -> go.Figure:
    if df_grp.empty or x_col not in df_grp.columns or y_col not in df_grp.columns:
        return _fig_vacio(height=height)
    totales = df_grp.groupby(y_col)[x_col].sum().nlargest(n_top).index[::-1]
    if totales.empty:
        return _fig_vacio(height=height)
    df_f = df_grp[df_grp[y_col].isin(totales)].copy()
    df_f[y_col] = pd.Categorical(df_f[y_col], categories=totales, ordered=True)
    fig = px.bar(df_f.sort_values(y_col), x=x_col, y=y_col, orientation="h",
                 height=max(height, n_top * 34), text=x_col,
                 labels={x_col: "", y_col: ""})
    fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                      textfont=dict(size=12, color="#24743c", family="sans-serif"),
                      marker_color=color,
                      marker_line_color='white', marker_line_width=0.8)
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, r=70),
                      xaxis=dict(tickformat=",.0f"),
                      paper_bgcolor='white', plot_bgcolor='#f5fbf7')
    return fig


def _donut(series: pd.Series, title: str = "", color_map: dict = None) -> go.Figure:
    clean = series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})
    clean = clean.dropna()
    if clean.empty:
        return _fig_vacio(titulo=title, height=320)
    cnt = clean.value_counts().reset_index()
    cnt.columns = ["cat", "n"]
    pie_kwargs = dict(hole=0.48, title=title, height=320)
    if color_map:
        pie_kwargs["color"] = "cat"
        pie_kwargs["color_discrete_map"] = color_map
    else:
        pie_kwargs["color_discrete_sequence"] = PALETTE
    fig = px.pie(cnt, names="cat", values="n", **pie_kwargs)
    fig.update_traces(textinfo="percent+label", textfont_size=12,
                      marker=dict(line=dict(color='white', width=2)))
    fig.update_layout(margin=dict(t=35, b=10), showlegend=False,
                      paper_bgcolor='white')
    return fig


def _filtros_panel(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 Filtros", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            reg_opts = sorted(df["REGION"].dropna().unique()) if "REGION" in df.columns else []
            reg_sel = st.multiselect("Región", reg_opts, key="st_reg")
        with c2:
            df_r = df[df["REGION"].isin(reg_sel)] if reg_sel else df
            dept_opts = sorted(df_r["DEPARTAMENTO"].dropna().unique()) if "DEPARTAMENTO" in df_r.columns else []
            dept_sel = st.multiselect("Departamento", dept_opts, key="st_dept")
        with c3:
            df_d = df_r[df_r["DEPARTAMENTO"].isin(dept_sel)] if dept_sel else df_r
            ciu_col = next((c for c in ["CIUDAD_STD", "Ciudad"] if c in df.columns), None)
            ciu_opts = sorted(df_d[ciu_col].dropna().unique()) if ciu_col else []
            ciu_sel = st.multiselect("Ciudad", ciu_opts, key="st_ciu")
        with c4:
            canal_opts = sorted(df["Canal"].dropna().unique()) if "Canal" in df.columns else []
            canal_sel = st.multiselect("Canal", canal_opts, key="st_canal")
        with c5:
            prod_opts = sorted(df["Producto"].dropna().unique()) if "Producto" in df.columns else []
            prod_sel = st.multiselect("Producto", prod_opts, key="st_prod")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            ent_opts = sorted(df["Entidad"].dropna().unique()) if "Entidad" in df.columns else []
            ent_sel = st.multiselect("Entidad", ent_opts, key="st_ent")
        with e2:
            nit_txt = st.text_input("NIT", placeholder="800123456…", key="st_nit")
        with e3:
            ra_opts = sorted(df["Rango_afiliados"].dropna().unique()) if "Rango_afiliados" in df.columns else []
            ra_sel = st.multiselect("Rango afiliados", ra_opts, key="st_ra")
        with e4:
            ant_opts = sorted(df["Rango_antigedad"].dropna().unique()) if "Rango_antigedad" in df.columns else []
            ant_sel = st.multiselect("Antigüedad", ant_opts, key="st_ant")

        f1, f2, f3 = st.columns(3)
        with f1:
            est_raw  = sorted(pd.to_numeric(df["Estrato"], errors="coerce").dropna().astype(int).unique()) if "Estrato" in df.columns else []
            est_sel  = st.multiselect("Estrato", est_raw, key="st_estrato")
        with f2:
            ec_col_f = next((c for c in ["EstadoCivil", "Estadocivil"] if c in df.columns), None)
            ec_opts  = sorted(df[ec_col_f].dropna().unique()) if ec_col_f else []
            ec_sel   = st.multiselect("Estado civil", ec_opts, key="st_estadocivil")
        with f3:
            sec_opts = sorted(df["SECTOR_EMPLEADOR"].dropna().unique()) if "SECTOR_EMPLEADOR" in df.columns else []
            sec_sel  = st.multiselect("Sector económico", sec_opts, key="st_sector")

    df_f = df.copy()
    if reg_sel:   df_f = df_f[df_f["REGION"].isin(reg_sel)]
    if dept_sel:  df_f = df_f[df_f["DEPARTAMENTO"].isin(dept_sel)]
    if ciu_col and ciu_sel: df_f = df_f[df_f[ciu_col].isin(ciu_sel)]
    if canal_sel: df_f = df_f[df_f["Canal"].isin(canal_sel)]
    if prod_sel:  df_f = df_f[df_f["Producto"].isin(prod_sel)]
    if ent_sel:   df_f = df_f[df_f["Entidad"].isin(ent_sel)]
    if nit_txt and "NIT" in df_f.columns:
        df_f = df_f[df_f["NIT"].astype(str).str.contains(nit_txt.strip(), na=False)]
    if ra_sel:    df_f = df_f[df_f["Rango_afiliados"].isin(ra_sel)]
    if ant_sel:   df_f = df_f[df_f["Rango_antigedad"].isin(ant_sel)]
    if est_sel and "Estrato" in df_f.columns:
        est_num = pd.to_numeric(df_f["Estrato"], errors="coerce").astype("Int64")
        df_f = df_f[est_num.isin(est_sel)]
    if ec_sel and ec_col_f and ec_col_f in df_f.columns:
        df_f = df_f[df_f[ec_col_f].isin(ec_sel)]
    if sec_sel and "SECTOR_EMPLEADOR" in df_f.columns:
        df_f = df_f[df_f["SECTOR_EMPLEADOR"].isin(sec_sel)]
    return df_f


def _tab_empresariales(df: pd.DataFrame) -> None:
    val_col = next((c for c in ["Valortotalplan", "ValorTotalPlan"] if c in df.columns), None)
    vt      = pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.Series(dtype=float)
    n_nits  = df["NIT"].nunique() if "NIT" in df.columns else (
              df["Entidad"].nunique() if "Entidad" in df.columns else 0)

    # ── KPIs ──────────────────────────────────────────────────────────────
    kc = st.columns(4)
    with kc[0]: kpi_card("Empresas únicas (NIT)", f"{n_nits:,}")
    with kc[1]: kpi_card("Total contratos",        f"{len(df):,}")
    with kc[2]:
        if "Total_afiliados" in df.columns and "Entidad" in df.columns:
            kpi_card("Promedio afiliados/empresa",
                     f"{df.groupby('Entidad')['Total_afiliados'].first().mean():,.0f}")
        else:
            kpi_card("Promedio afiliados/empresa", "N/A")
    with kc[3]: kpi_card("Valor promedio plan", f"${vt.mean():,.0f}" if not vt.empty else "N/A")

    st.divider()

    # ── 10.1 NITs por tamaño de empresa ───────────────────────────────────
    if "Rango_afiliados" in df.columns and "NIT" in df.columns:
        st.markdown("#### 10.1 · Empresas por tamaño (NITs únicos por rango de afiliados)")
        ORDEN_RANGO = ["0-100","100-300","300-500","500-1000",
                       "1000-2000","2000-5000","5000-10000","Mas 10.000"]
        nits_tam = (
            df.groupby("Rango_afiliados")["NIT"].nunique()
            .reindex([r for r in ORDEN_RANGO if r in df["Rango_afiliados"].values])
            .dropna().reset_index()
        )
        nits_tam.columns = ["Rango", "NITs"]
        fig_ra = px.bar(nits_tam, x="Rango", y="NITs", text="NITs",
                        color_discrete_sequence=COL_STATS, height=340,
                        labels={"Rango": "Rango de afiliados", "NITs": "Empresas (NITs únicos)"})
        fig_ra.update_traces(texttemplate="%{y:,}", textposition="outside",
                             textfont=dict(size=13, color="#24743c", family="sans-serif"),
                             marker_color=COL_STATS[0])
        fig_ra.update_layout(showlegend=False, yaxis=dict(showticklabels=False, showgrid=False),
                             margin=dict(t=30, b=10))
        st.plotly_chart(fig_ra, use_container_width=True, key="pc_011")
        st.divider()

    # ── NITs por canal ─────────────────────────────────────────────────────
    if "Canal" in df.columns and "NIT" in df.columns:
        st.markdown("#### Empresas únicas por canal")
        nits_canal = df.groupby("Canal")["NIT"].nunique().sort_values(ascending=False).reset_index()
        nits_canal.columns = ["Canal", "NITs"]
        cn1, cn2 = st.columns([1, 3])
        with cn1:
            top_n_canal = st.slider("Top N canales", 5, 20, 10, key="nit_canal_n")
        with cn2:
            st.plotly_chart(_bar_h(nits_canal, "NITs", "Canal",
                                   n_top=top_n_canal, color=COL_STATS[1]),
                            use_container_width=True, key="pc_012")
        st.divider()

    # ── NITs por región ────────────────────────────────────────────────────
    if "REGION" in df.columns and "Contrato" in df.columns:
        st.markdown("#### Contratos únicos por región y departamento")
        rg1, rg2 = st.columns(2)
        with rg1:
            reg_ct = df.groupby("REGION")["Contrato"].nunique().sort_values(ascending=False).reset_index()
            reg_ct.columns = ["Región", "Contratos"]
            fig_reg = px.bar(reg_ct, x="Contratos", y="Región", orientation="h",
                             text="Contratos", height=300,
                             color="Región", color_discrete_sequence=COL_STATS,
                             labels={"Contratos": "Contratos únicos", "Región": ""})
            fig_reg.update_traces(texttemplate="%{text:,}", textposition="outside",
                                  textfont=dict(size=13, color="#24743c", family="sans-serif"))
            fig_reg.update_layout(showlegend=False, margin=dict(t=10, b=10, r=80))
            st.plotly_chart(fig_reg, use_container_width=True, key="pc_013")
        with rg2:
            if "DEPARTAMENTO" in df.columns:
                dep_ctrl, dep_chart = st.columns([1, 2])
                with dep_ctrl:
                    top_dep = st.slider("Top departamentos", 5, 20, 10, key="nit_dep_n")
                dep_ct = (df.groupby("DEPARTAMENTO")["Contrato"].nunique()
                          .sort_values(ascending=False).head(top_dep).reset_index())
                dep_ct.columns = ["Departamento", "Contratos"]
                with dep_chart:
                    st.plotly_chart(_bar_h(dep_ct, "Contratos", "Departamento",
                                           n_top=top_dep, color=COL_STATS[0]),
                                    use_container_width=True, key="pc_014")
        st.divider()

    # ── Contratos por ciudad (top N) ──────────────────────────────────────
    ciu_col = next((c for c in ["CIUDAD_STD", "Ciudad"] if c in df.columns), None)
    if ciu_col and "Contrato" in df.columns:
        st.markdown("#### Top ciudades por número de contratos únicos")
        cc1, cc2 = st.columns([1, 3])
        with cc1:
            top_ciu = st.slider("Top ciudades", 5, 20, 10, key="nit_ciu_n")
        ciu_ct = (df.groupby(ciu_col)["Contrato"].nunique()
                  .sort_values(ascending=False).head(top_ciu).reset_index())
        ciu_ct.columns = ["Ciudad", "Contratos"]
        with cc2:
            st.plotly_chart(_bar_h(ciu_ct, "Contratos", "Ciudad",
                                   n_top=top_ciu, color=COL_STATS[2]),
                            use_container_width=True, key="pc_015")
        st.divider()

    # ── Principales entidades ─────────────────────────────────────────────
    if "Entidad" in df.columns:
        st.markdown("#### Principales entidades por contratos Únicos")
        ctrl1, _ = st.columns([1, 3])
        with ctrl1:
            top_n = st.slider("Top N", 5, 30, 15, key="ent_topn")

        if "Contrato" in df.columns:
            ent_cnt = (df.groupby("Entidad")["Contrato"].nunique()
                       .sort_values(ascending=False).reset_index())
            ent_cnt.columns = ["Entidad", "Contratos únicos"]
            met_col = "Contratos únicos"
        st.plotly_chart(_bar_h(ent_cnt, met_col, "Entidad", n_top=top_n, color=C_VERDE),
                        use_container_width=True, key="pc_016")
        st.divider()

    # ── Sector empleador (NITs únicos) ────────────────────────────────────
    if "SECTOR_EMPLEADOR" in df.columns and "NIT" in df.columns:
        st.markdown("#### Sector empleador (NITs únicos)")
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            sec_n = st.slider("Top sectores", 5, 20, 10, key="sec_topn")
        sec_cnt = (df.groupby(df["SECTOR_EMPLEADOR"].fillna("Sin información"))["NIT"]
                   .nunique().sort_values(ascending=False).reset_index())
        sec_cnt.columns = ["Sector", "NITs"]
        with sc2:
            st.plotly_chart(_bar_h(sec_cnt, "NITs", "Sector", n_top=sec_n, color=COL_STATS[1]),
                            use_container_width=True, key="pc_018")
        st.divider()

    # ── Clasificación por sector económico y segmento ─────────────────────
    if "SECTOR_EMPLEADOR" in df.columns and "cluster" in df.columns and "NIT" in df.columns:
        st.markdown("#### Clasificación del portafolio por sector económico y segmento")
        scs1, scs2 = st.columns([1, 3])
        with scs1:
            sec_seg_n = st.slider("Top sectores", 5, 20, 10, key="sec_seg_topn")

        sec_grp = (
            df.assign(_sector=df["SECTOR_EMPLEADOR"].fillna("Sin información"))
            .groupby(["_sector", "cluster"])["NIT"]
            .nunique()
            .reset_index()
        )
        sec_grp.columns = ["Sector", "cluster", "NITs"]
        sec_grp["Segmento"] = sec_grp["cluster"].map(NOMBRES)

        top_secs = (
            sec_grp.groupby("Sector")["NITs"].sum()
            .nlargest(sec_seg_n).index.tolist()
        )
        sec_grp_f = sec_grp[sec_grp["Sector"].isin(top_secs)].copy()

        fig_sec_seg = px.bar(
            sec_grp_f,
            y="Sector", x="NITs",
            color="Segmento",
            barmode="stack",
            orientation="h",
            color_discrete_map={v: PAL.get(k, "#888") for k, v in NOMBRES.items()},
            height=max(340, sec_seg_n * 46),
            labels={"NITs": "Empresas (NITs únicos)", "Sector": ""},
            text="NITs",
        )
        fig_sec_seg.update_traces(
            texttemplate="%{text:,}", textposition="inside",
            textfont=dict(size=11, color="white"),
        )
        fig_sec_seg.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.18),
            margin=dict(t=10, b=10, r=10),
            yaxis=dict(autorange="reversed"),
        )
        with scs2:
            st.plotly_chart(fig_sec_seg, use_container_width=True, key="pc_sec_seg")
        st.divider()

    # ── Actividad económica CIIU 869 — NITs únicos ────────────────────────
    if "ACTIVIDAD_ECONOMICA" in df.columns and "NIT" in df.columns:
        df_869 = df[df["ACTIVIDAD_ECONOMICA"].astype(str).str.startswith("869")]
        if not df_869.empty:
            st.markdown("#### Actividad económica CIIU 869 — NITs únicos")
            ae_cnt = (df_869.groupby(df_869["ACTIVIDAD_ECONOMICA"].astype(str))["NIT"]
                      .nunique().sort_values(ascending=False).reset_index())
            ae_cnt.columns = ["CIIU", "NITs"]
            fig_ae = px.bar(ae_cnt, x="CIIU", y="NITs", text="NITs",
                            color_discrete_sequence=COL_STATS, height=340,
                            labels={"CIIU": "Código CIIU 869", "NITs": "Empresas (NITs únicos)"})
            fig_ae.update_traces(texttemplate="%{y:,}", textposition="outside",
                                 textfont=dict(size=13, color="#24743c", family="sans-serif"),
                                 marker_color=COL_STATS[0])
            fig_ae.update_layout(showlegend=False, yaxis=dict(showticklabels=False, showgrid=False),
                                 margin=dict(t=30, b=10))
            st.plotly_chart(fig_ae, use_container_width=True, key="pc_019")
            st.divider()

    # ── Seguro / Póliza — Con/Sin + breakdown AP/PFI/SOLICANASTA ──────────
    if "Tiposseguros_ajuste" in df.columns:
        df_ct_pol = df.drop_duplicates(subset="Contrato").copy() if "Contrato" in df.columns else df.copy()
        TIPOS_VALIDOS_POL = ["AP", "PFI", "SOLICANASTA"]
        df_ct_pol["_tiene_poliza"] = df_ct_pol["Tiposseguros_ajuste"].isin(TIPOS_VALIDOS_POL)
        total_pol = len(df_ct_pol)
        con_pol   = int(df_ct_pol["_tiene_poliza"].sum())
        sin_pol   = total_pol - con_pol

        st.markdown("#### Seguro")
        pol1, pol2 = st.columns(2)
        with pol1:
            pol_cs = pd.DataFrame({
                "Estado":    ["Con Seguro", "Sin Seguro"],
                "Contratos": [con_pol, sin_pol],
                "Pct":       [con_pol / max(total_pol, 1) * 100, sin_pol / max(total_pol, 1) * 100],
            })
            pol_cs["texto"] = pol_cs.apply(
                lambda r: f"{int(r['Contratos']):,}<br>({r['Pct']:.1f}%)", axis=1)
            fig_pol = px.bar(pol_cs, x="Estado", y="Contratos", text="texto",
                             color="Estado",
                             color_discrete_map={"Con Seguro": C_VERDE, "Sin Seguro": C_A_PAL},
                             height=300, labels={"Estado": "", "Contratos": "Contratos únicos"})
            fig_pol.update_traces(textposition="outside",
                                  textfont=dict(size=11, color="#24743c", family="sans-serif"))
            fig_pol.update_layout(showlegend=False,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  margin=dict(t=30, b=10))
            st.plotly_chart(fig_pol, use_container_width=True, key="pc_pol_cs")
        with pol2:
            if con_pol > 0:
                tipo_df = (df_ct_pol[df_ct_pol["_tiene_poliza"]]["Tiposseguros_ajuste"]
                           .value_counts().reset_index())
                tipo_df.columns = ["Tipo", "Contratos"]
                tipo_df["pct"]   = tipo_df["Contratos"] / con_pol * 100
                tipo_df["texto"] = tipo_df.apply(
                    lambda r: f"{int(r['Contratos']):,}<br>({r['pct']:.1f}%)", axis=1)
                fig_tipo = px.bar(tipo_df, x="Tipo", y="Contratos", text="texto",
                                  color="Tipo", color_discrete_sequence=COL_STATS,
                                  height=300,
                                  labels={"Tipo": "Tipo de seguro", "Contratos": "Contratos únicos"})
                fig_tipo.update_traces(textposition="outside",
                                       textfont=dict(size=9, color="#24743c", family="sans-serif"))
                fig_tipo.update_layout(showlegend=False,
                                       yaxis=dict(showticklabels=False, showgrid=False),
                                       margin=dict(t=30, b=10))
                st.plotly_chart(fig_tipo, use_container_width=True, key="pc_pol_tipo")
        st.divider()

    # ── Coberturas adicionales — Salud / Bicicleta / Repatriacion / Expatriacion
    _COBERTURAS_EXTRA = ["Salud", "Bicicleta", "Repatriacion", "Expatriacion"]
    _cob_presentes = [c for c in _COBERTURAS_EXTRA if c in df.columns]
    if _cob_presentes:
        df_ct_cob  = df.drop_duplicates(subset="Contrato") if "Contrato" in df.columns else df
        total_base = len(df_ct_cob)
        st.markdown("#### Coberturas adicionales")
        cob_rows = []
        for col in _cob_presentes:
            con = int(pd.to_numeric(df_ct_cob[col], errors="coerce").fillna(0).gt(0).sum())
            sin = total_base - con
            cob_rows.append({"Cobertura": col, "Estado": "Con", "Contratos": con,
                             "pct": con / max(total_base, 1) * 100})
            cob_rows.append({"Cobertura": col, "Estado": "Sin", "Contratos": sin,
                             "pct": sin / max(total_base, 1) * 100})
        cob_long = pd.DataFrame(cob_rows)
        cob_long["texto"] = cob_long.apply(
            lambda r: f"{int(r['Contratos']):,}<br>({r['pct']:.1f}%)", axis=1)
        fig_cob = px.bar(
            cob_long, x="Estado", y="Contratos", color="Estado",
            facet_col="Cobertura", text="texto",
            color_discrete_map={"Con": C_VERDE, "Sin": C_A_PAL},
            height=360,
            labels={"Estado": "", "Contratos": "Contratos únicos"},
        )
        fig_cob.update_traces(textposition="outside",
                              textfont=dict(size=9, color="#24743c", family="sans-serif"))
        fig_cob.update_layout(showlegend=True, margin=dict(t=30, b=10),
                              legend=dict(orientation="h", y=-0.18))
        fig_cob.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_cob, use_container_width=True, key="pc_cob")
        st.divider()

    # ── Participación de mercado — combinaciones exactas + totales por dimensión
    _SEIS_PRODS = ["Poliza", "Salud", "Bicicleta", "Repatriacion", "Expatriacion", "Mascotas"]
    df_pen = df.drop_duplicates(subset="Contrato").copy() if "Contrato" in df.columns else df.copy()
    if "Poliza" not in df_pen.columns and "Tiposseguros_ajuste" in df_pen.columns:
        df_pen["Poliza"] = df_pen["Tiposseguros_ajuste"].isin(["AP", "PFI", "SOLICANASTA"]).astype(int)
    if "Mascotas" not in df_pen.columns:
        _perro = df_pen["Tieneperro"].isin({"Y", "S"}) if "Tieneperro" in df_pen.columns else pd.Series(False, index=df_pen.index)
        _gato  = df_pen["Tienegato"].isin({"Y", "S"})  if "Tienegato"  in df_pen.columns else pd.Series(False, index=df_pen.index)
        df_pen["Mascotas"] = (_perro | _gato).astype(int)
    _seis_disp = [c for c in _SEIS_PRODS if c in df_pen.columns]

    if len(_seis_disp) >= 2:
        st.markdown(f"#### Participación de mercado  |  {len(df_pen):,} contratos únicos")
        total_penet = len(df_pen)

        # Normalizar a 0/1
        for _c in _seis_disp:
            df_pen[_c] = pd.to_numeric(df_pen[_c], errors="coerce").fillna(0).gt(0).astype(int)

        # Combinaciones exactas
        df_pen["_n_prod"]   = df_pen[_seis_disp].sum(axis=1)
        df_pen["_etiqueta"] = df_pen[_seis_disp].apply(
            lambda r: " + ".join(c for c in _seis_disp if r[c] == 1) or "Sin productos", axis=1
        )
        combos = (
            df_pen.groupby(["_n_prod", "_etiqueta"])
            .size()
            .reset_index(name="Contratos")
            .sort_values(["_n_prod", "Contratos"], ascending=[False, False])
            .reset_index(drop=True)
        )
        combos["Pct"] = (combos["Contratos"] / total_penet * 100).round(2)
        combos["texto"] = combos.apply(
            lambda r: f"{int(r['Contratos']):,}  ({r['Pct']:.1f}%)", axis=1)

        # Totales independientes por dimensión
        totales_dim = {c: int(df_pen[c].sum()) for c in _seis_disp}
        dim_df = pd.DataFrame([
            {"Producto": k, "Contratos": v, "Pct": v / max(total_penet, 1) * 100}
            for k, v in totales_dim.items()
        ])
        dim_df["texto"] = dim_df.apply(
            lambda r: f"{int(r['Contratos']):,}<br>({r['Pct']:.1f}%)", axis=1)

        # Colores por n_productos para combinaciones
        _pal_n = {6: COL_STATS[0], 5: COL_STATS[1], 4: COL_STATS[2],
                  3: COL_STATS[3], 2: COL_STATS[0], 1: COL_STATS[1], 0: COL_STATS[2]}

        top_combos = combos[combos["Contratos"] > 0].sort_values("Contratos", ascending=True)

        pen1, pen2 = st.columns(2)

        # Panel izquierdo — combinaciones exactas (horizontal)
        with pen1:
            _n_rows = max(len(top_combos), 1)
            fig_combos = go.Figure()
            fig_combos.add_trace(go.Bar(
                x=top_combos["Contratos"],
                y=top_combos["_etiqueta"],
                orientation="h",
                text=top_combos["texto"],
                textposition="outside",
                textfont=dict(size=8, color="#24743c", family="sans-serif"),
                marker_color=[_pal_n.get(int(n), COL_STATS[0]) for n in top_combos["_n_prod"]],
                marker_line_color="white",
                marker_line_width=0.6,
            ))
            fig_combos.update_layout(
                height=max(400, _n_rows * 28),
                title=dict(text="Combinaciones exactas<br>(de mayor a menor)",
                           font=dict(size=12, color=C_VERDE)),
                xaxis=dict(title="Contratos únicos",
                           range=[0, top_combos["Contratos"].max() * 1.38]),
                yaxis=dict(autorange=True),
                margin=dict(t=50, b=10, r=10, l=10),
                paper_bgcolor="white", plot_bgcolor="#f5fbf7",
                showlegend=False,
            )
            st.plotly_chart(fig_combos, use_container_width=True, key="pc_combos")

        # Panel derecho — total por dimensión (vertical)
        with pen2:
            fig_dim = px.bar(
                dim_df.sort_values("Contratos", ascending=False),
                x="Producto", y="Contratos", text="texto",
                color="Producto", color_discrete_sequence=COL_STATS * 2,
                height=max(400, _n_rows * 28),
                labels={"Producto": "", "Contratos": "Contratos"},
                title="Total contratos por dimensión<br>(un contrato puede tener varias)",
            )
            fig_dim.update_traces(
                textposition="outside",
                textfont=dict(size=9, color="#24743c", family="sans-serif"),
            )
            fig_dim.update_layout(
                showlegend=False,
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(t=50, b=10),
                title=dict(font=dict(size=12, color=C_VERDE)),
            )
            st.plotly_chart(fig_dim, use_container_width=True, key="pc_penet")

        st.caption("Nota: sumar los totales por dimensión NO da el total — un contrato puede tener varios productos.")
        st.divider()



def _tab_familias(df: pd.DataFrame) -> None:
    # ── Base deduplicada por Contrato (un registro por titular) ──────────
    df_ct = df.drop_duplicates(subset="Contrato") if "Contrato" in df.columns else df.copy()

    # ── Act.valor numérico para secciones de familias ─────────────────────
    av_num = pd.to_numeric(df["Act.valor"], errors="coerce") if "Act.valor" in df.columns else None
    df_av  = df.assign(**{"Act.valor": av_num}) if av_num is not None else df

    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 👤 Por titular (Contrato único)")
    st.caption(f"Base: {len(df_ct):,} contratos únicos")
    st.divider()

    # ── KPIs composición familiar ─────────────────────────────────────────
    pct_esp  = df_ct["TieneEsposa_V"].mean() * 100  if "TieneEsposa_V" in df_ct.columns else 0
    pct_hij  = df_ct["TieneHijos_V"].mean()  * 100  if "TieneHijos_V"  in df_ct.columns else 0
    pct_pad  = df_ct["TienePadres_V"].mean() * 100  if "TienePadres_V" in df_ct.columns else 0
    _VALID_PET = {"Y", "S"}
    pct_perr = df_ct["Tieneperro"].isin(_VALID_PET).mean() * 100 if "Tieneperro" in df_ct.columns else 0
    pct_gato = df_ct["Tienegato"].isin(_VALID_PET).mean()  * 100 if "Tienegato"  in df_ct.columns else 0

    kf = st.columns(5)
    with kf[0]: kpi_card("Con pareja (%)", f"{pct_esp:.1f}%")
    with kf[1]: kpi_card("Con hijos (%)",  f"{pct_hij:.1f}%")
    with kf[2]: kpi_card("Con padres (%)", f"{pct_pad:.1f}%")
    with kf[3]: kpi_card("Con perro (%)",  f"{pct_perr:.1f}%")
    with kf[4]: kpi_card("Con gato (%)",   f"{pct_gato:.1f}%")

    st.divider()

    # ── Composición familiar ──────────────────────────────────────────────
    st.markdown("#### Composición familiar")
    n_total   = len(df_ct)
    n_esp     = int(df_ct["TieneEsposa_V"].sum())  if "TieneEsposa_V" in df_ct.columns else 0
    n_hij     = int(df_ct["TieneHijos_V"].sum())   if "TieneHijos_V"  in df_ct.columns else 0
    n_pad     = int(df_ct["TienePadres_V"].sum())  if "TienePadres_V" in df_ct.columns else 0
    fam_df = pd.DataFrame({
        "Tipo":       ["Con pareja", "Con hijos", "Con padres"],
        "Porcentaje": [pct_esp,      pct_hij,     pct_pad],
        "Contratos":  [n_esp,        n_hij,       n_pad],
    })
    fam_df["Etiqueta"] = fam_df.apply(
        lambda r: f"{r['Contratos']:,} contratos<br>({r['Porcentaje']:.1f}%)", axis=1
    )
    fig_fam = px.bar(fam_df, x="Tipo", y="Porcentaje", color="Tipo",
                     text="Etiqueta",
                     color_discrete_sequence=COL_STATS, height=340,
                     labels={"Tipo": "", "Porcentaje": "%"})
    fig_fam.update_traces(textposition="outside",
                          textfont=dict(size=13, color="#24743c", family="sans-serif"))
    fig_fam.update_layout(showlegend=False, yaxis_range=[0, 125],
                          yaxis=dict(showticklabels=False, showgrid=False),
                          margin=dict(t=30, b=10))
    st.plotly_chart(fig_fam, use_container_width=True, key="pc_020")
    st.divider()

    # ── Distribución por género ───────────────────────────────────────────
    if "Sexo" in df_ct.columns:
        st.markdown("#### Distribución por género")
        sexo_clean = (df_ct["Sexo"].astype(str).str.strip().str.upper()
                      .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA, "N/A": pd.NA})
                      .dropna())
        sexo_clean = sexo_clean[sexo_clean.isin(["M", "F", "MASCULINO", "FEMENINO",
                                                  "HOMBRE", "MUJER"])]
        sx_cnt = sexo_clean.value_counts().reset_index()
        sx_cnt.columns = ["Género", "Contratos"]
        sx_cnt["Etiqueta"] = sx_cnt.apply(
            lambda r: f"{r['Contratos']:,}<br>({r['Contratos']/sx_cnt['Contratos'].sum()*100:.1f}%)", axis=1
        )
        sg1, sg2 = st.columns(2)
        with sg1:
            if sx_cnt.empty:
                st.plotly_chart(_fig_vacio("Género", height=320), use_container_width=True, key="pc_021")
            else:
                fig_sx = px.bar(sx_cnt, x="Género", y="Contratos", color="Género",
                                text="Etiqueta", height=320,
                                color_discrete_sequence=[COL_STATS[0], COL_STATS[2]],
                                labels={"Género": "", "Contratos": "Contratos"})
                fig_sx.update_traces(textposition="outside",
                                     textfont=dict(size=13, color="#24743c", family="sans-serif"))
                fig_sx.update_layout(showlegend=False,
                                     yaxis=dict(showticklabels=False, showgrid=False),
                                     margin=dict(t=30, b=10))
                st.plotly_chart(fig_sx, use_container_width=True, key="pc_021")
        with sg2:
            if "Producto" in df_ct.columns:
                sp = (df_ct.assign(_sx=sexo_clean)
                      .dropna(subset=["_sx"])
                      .groupby(["Producto", "_sx"]).size().reset_index(name="n"))
                sp.rename(columns={"_sx": "Sexo"}, inplace=True)
                sp["pct"] = sp.groupby("Producto")["n"].transform(lambda x: x / x.sum() * 100)
                sp["texto"] = sp.apply(lambda r: f"{int(r['n']):,}\n({r['pct']:.1f}%)" if r["pct"] >= 8 else "", axis=1)
                fig_sp = px.bar(sp, x="pct", y="Producto", color="Sexo",
                                barmode="stack", orientation="h", height=320,
                                color_discrete_sequence=[COL_STATS[0], COL_STATS[2]],
                                labels={"pct": "% género", "Producto": "", "Sexo": "Género"},
                                text="texto")
                fig_sp.update_traces(textposition="inside",
                                     textfont=dict(size=11, color="white", family="sans-serif"))
                fig_sp.update_layout(xaxis=dict(ticksuffix="%", range=[0, 100]),
                                     legend=dict(orientation="h", y=-0.22),
                                     margin=dict(t=10, b=10))
                st.plotly_chart(fig_sp, use_container_width=True, key="pc_022")
        st.divider()

    # ── Estado civil ──────────────────────────────────────────────────────
    ec_col = next((c for c in ["EstadoCivil", "Estadocivil"] if c in df_ct.columns), None)
    if ec_col:
        st.markdown("#### Estado civil")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.plotly_chart(_donut(df_ct[ec_col], "Estado civil"), use_container_width=True, key="pc_023")
        with ec2:
            ec_cnt = df_ct[ec_col].value_counts().reset_index()
            ec_cnt.columns = ["Estado", "n"]
            st.plotly_chart(_bar_h(ec_cnt, "n", "Estado", n_top=10, color=COL_STATS[1]),
                            use_container_width=True, key="pc_024")
        st.divider()

    # ── Distribución por estrato socioeconómico ───────────────────────────
    if "Estrato" in df_ct.columns:
        st.markdown("#### Distribución por estrato socioeconómico")
        est_num = pd.to_numeric(df_ct["Estrato"], errors="coerce").dropna().astype(int)
        est_cnt = est_num.value_counts().sort_index().reset_index()
        est_cnt.columns = ["Estrato", "n"]
        if est_cnt.empty:
            st.info("Sin datos de estrato para el filtro seleccionado.")
        else:
            fig_est = px.bar(est_cnt, x="Estrato", y="n", text="n",
                             color="Estrato", color_continuous_scale=["#f3eb8b", "#24743c"],
                             height=320, labels={"Estrato": "Estrato", "n": "Contratos"})
            fig_est.update_traces(texttemplate="%{y:,}", textposition="outside",
                                  textfont=dict(size=13, color="#24743c", family="sans-serif"))
            fig_est.update_layout(showlegend=False, coloraxis_showscale=False,
                                  yaxis=dict(showticklabels=False, showgrid=False),
                                  margin=dict(t=30, b=10))
            st.plotly_chart(fig_est, use_container_width=True, key="pc_025")
        st.divider()

    # ── 10.3 · Rango de edad (bandas actuariales FASECOLDA) ──────────────
    _ORDEN_EDAD  = ["Menor", "Joven", "Adulto Joven", "Adulto",
                    "Adulto Mayor", "Pre-Jubilado", "Jubilado", "Tercera Edad"]
    _BINS_EDAD   = [0, 17, 25, 35, 45, 55, 65, 75, 130]

    _df_e = df_ct.copy()
    if "Rango_edad" in _df_e.columns:
        re_col = "Rango_edad"
    elif "Edad" in _df_e.columns:
        _df_e["Rango_edad"] = pd.cut(
            pd.to_numeric(_df_e["Edad"], errors="coerce"),
            bins=_BINS_EDAD, labels=_ORDEN_EDAD,
            right=True, include_lowest=True
        ).astype(str)
        re_col = "Rango_edad"
    else:
        re_col = None

    if re_col:
        st.markdown("#### 10.3 · Distribución por rango de edad (bandas actuariales)")
        orden_presente = [c for c in _ORDEN_EDAD if c in _df_e[re_col].unique()]
        edad_dist = (_df_e[re_col].value_counts().reindex(orden_presente, fill_value=0))

        rng_df = edad_dist.reset_index()
        rng_df.columns = ["Rango_edad", "n"]
        fig_rng = px.bar(
            rng_df, x="Rango_edad", y="n", text="n",
            color="Rango_edad", color_discrete_sequence=COL_STATS * 2,
            category_orders={"Rango_edad": orden_presente},
            height=340, labels={"Rango_edad": "Rango de edad", "n": "Contratos"},
        )
        fig_rng.update_traces(texttemplate="%{text:,}", textposition="outside",
                              textfont=dict(size=13, color="#24743c", family="sans-serif"))
        fig_rng.update_layout(showlegend=False, yaxis=dict(showticklabels=False, showgrid=False),
                              margin=dict(t=30, b=60),
                              xaxis=dict(tickangle=-35))
        st.plotly_chart(fig_rng, use_container_width=True, key="pc_033")

        with st.expander("📋 Bandas actuariales de referencia (FASECOLDA / Swiss Re)"):
            st.markdown("""
| # | Rango | Etiqueta | Perfil de riesgo |
|:-:|-------|----------|-----------------|
| 1 | 0 – 17 | **Menor** | Dependiente, siniestralidad baja |
| 2 | 18 – 25 | **Joven** | Inicio vida laboral, riesgo moderado |
| 3 | 26 – 35 | **Adulto Joven** | Formación familia, riesgo bajo-medio |
| 4 | 36 – 45 | **Adulto** | Productividad máxima, riesgo medio |
| 5 | 46 – 55 | **Adulto Mayor** | Inicio enfermedades crónicas, riesgo medio-alto |
| 6 | 56 – 65 | **Pre-Jubilado** | Alta siniestralidad, mayor uso de beneficios |
| 7 | 66 – 75 | **Jubilado** | Riesgo alto, enfermedades degenerativas |
| 8 | 76 – 130 | **Tercera Edad** | Muy alta siniestralidad, dependencia |
""")

        if "Sexo" in _df_e.columns:
            st.markdown("##### Distribución por rango de edad y género")
            _sx = (_df_e["Sexo"].astype(str).str.strip().str.upper()
                   .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA}))
            _df_e2 = _df_e.assign(Sexo_n=_sx).dropna(subset=["Sexo_n"])
            cruce = (pd.crosstab(_df_e2[re_col], _df_e2["Sexo_n"])
                     .reindex(orden_presente, fill_value=0))
            cruce_long = cruce.reset_index().melt(id_vars=re_col, var_name="Sexo", value_name="n")
            totales_b  = cruce.sum(axis=1).to_dict()
            cruce_long["pct"]   = cruce_long.apply(
                lambda r: r["n"] / totales_b.get(r[re_col], 1) * 100, axis=1)
            cruce_long["texto"] = cruce_long.apply(
                lambda r: f"{int(r['n']):,}<br>({r['pct']:.0f}%)" if r["pct"] >= 5 else "", axis=1)
            fig_ctab = px.bar(
                cruce_long, x=re_col, y="n", color="Sexo", text="texto",
                barmode="stack",
                color_discrete_sequence=[COL_STATS[0], COL_STATS[2]],
                category_orders={re_col: orden_presente},
                height=380,
                labels={re_col: "Rango de edad", "n": "Contratos", "Sexo": "Género"},
            )
            fig_ctab.update_traces(textposition="inside", insidetextanchor="middle")
            fig_ctab.update_layout(legend=dict(orientation="h", y=-0.22),
                                   margin=dict(t=10, b=60), xaxis=dict(tickangle=-35))
            st.plotly_chart(fig_ctab, use_container_width=True, key="pc_034")
        st.divider()

    # ── Distribución Valor Total Plan ─────────────────────────────────────
    _val_col = next((c for c in ["Valortotalplan", "ValorTotalPlan"] if c in df_ct.columns), None)
    if _val_col:
        st.markdown("#### Distribución Valor Total Plan")
        _serie_vtp = pd.to_numeric(df_ct[_val_col], errors="coerce")
        _serie_vtp = _serie_vtp[_serie_vtp > 0].dropna()
        if not _serie_vtp.empty:
            _q1, _med, _q3 = _serie_vtp.quantile([0.25, 0.50, 0.75])
            _iqr = _q3 - _q1
            st.caption(
                f"Q1: ${int(_q1):,} · Mediana: ${int(_med):,} · Q3: ${int(_q3):,} · "
                f"IQR: ${int(_iqr):,} · Máx: ${int(_serie_vtp.max()):,}"
            )
            _vb1, _vb2 = st.columns(2)
            with _vb1:
                _fig_vtp1 = go.Figure()
                _fig_vtp1.add_trace(go.Box(
                    y=_serie_vtp, name="Original",
                    boxmean=True, marker_color=C_VERDE,
                    line_color=C_VERDE,
                ))
                _fig_vtp1.update_layout(
                    height=340, title="Escala original",
                    yaxis_title="Valor Total Plan ($)",
                    margin=dict(t=40, b=10), showlegend=False,
                    paper_bgcolor="white", plot_bgcolor="#f5fbf7",
                )
                st.plotly_chart(_fig_vtp1, use_container_width=True, key="pc_vtp1")
            with _vb2:
                _upper    = _q3 + 1.5 * _iqr
                _sin_out  = _serie_vtp[_serie_vtp <= _upper]
                _fig_vtp2 = go.Figure()
                _fig_vtp2.add_trace(go.Box(
                    y=_sin_out, name="Sin outliers",
                    boxmean=True, marker_color=C_V_PAL,
                    line_color=C_V_PAL,
                ))
                _fig_vtp2.update_layout(
                    height=340, title="Sin outliers (1.5 × IQR)",
                    yaxis_title="Valor Total Plan ($)",
                    margin=dict(t=40, b=10), showlegend=False,
                    paper_bgcolor="white", plot_bgcolor="#f5fbf7",
                )
                st.plotly_chart(_fig_vtp2, use_container_width=True, key="pc_vtp2")
        st.divider()

    # ── Mascotas ──────────────────────────────────────────────────────────
    _vp_m = df_ct["Tieneperro"].isin({"Y", "S"}) if "Tieneperro" in df_ct.columns else pd.Series(False, index=df_ct.index)
    _vg_m = df_ct["Tienegato"].isin({"Y", "S"})  if "Tienegato"  in df_ct.columns else pd.Series(False, index=df_ct.index)
    df_masc = df_ct[_vp_m | _vg_m].copy()

    if len(df_masc) > 0:
        st.markdown("#### Mascotas")
        _cant_m = pd.to_numeric(df_masc["Cantidad_mascotas"], errors="coerce").fillna(0) if "Cantidad_mascotas" in df_masc.columns else pd.Series(0, index=df_masc.index)
        total_contratos_masc = len(df_masc)
        total_sum_mascotas   = int(_cant_m.sum())

        # Gráfico 1 — Contratos con mascotas vs Suma total mascotas
        _mc_simple = pd.DataFrame({
            "Etiqueta": ["Contratos\ncon mascotas", "Suma total\nmascotas"],
            "Valor":    [total_contratos_masc, total_sum_mascotas],
            "Color":    [COL_STATS[0], COL_STATS[2]],
        })
        fig_mcs = go.Figure(go.Bar(
            x=_mc_simple["Etiqueta"], y=_mc_simple["Valor"],
            marker_color=_mc_simple["Color"], marker_line_color="white", marker_line_width=0.8,
            text=[f"{v:,}" for v in _mc_simple["Valor"]], textposition="outside",
            textfont=dict(size=14, color="#24743c", family="sans-serif"),
            width=[0.4, 0.4],
        ))
        fig_mcs.update_layout(
            height=360, title="Mascotas — Contratos y cantidad total",
            yaxis=dict(showticklabels=False, showgrid=False),
            margin=dict(t=40, b=10), showlegend=False,
            paper_bgcolor="white", plot_bgcolor="#f5fbf7",
        )
        st.plotly_chart(fig_mcs, use_container_width=True, key="pc_masc_simple")

        # Tipo de mascota
        def _tipo_masc(row):
            p = row.get("Tieneperro", "N"); g = row.get("Tienegato", "N")
            if p in ("Y","S") and g in ("Y","S"): return "Perro + Gato"
            if p in ("Y","S"): return "Solo Perro"
            if g in ("Y","S"): return "Solo Gato"
            return "Sin clasificar"
        df_masc["Tipo_mascota"] = df_masc.apply(_tipo_masc, axis=1)

        # Suma por tipo
        tipo_sum = (df_masc.groupby("Tipo_mascota")["Cantidad_mascotas"]
                    .sum().sort_values(ascending=False)
                    if "Cantidad_mascotas" in df_masc.columns
                    else df_masc["Tipo_mascota"].value_counts())
        total_sum_tipo = tipo_sum.sum()

        # Gráfico 4 paneles — 2x2
        st.markdown(f"##### Análisis detallado  |  {total_contratos_masc:,} contratos  |  {int(total_sum_tipo):,} mascotas")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        # Panel 1: suma por tipo
        with r1c1:
            _ts_df = tipo_sum.reset_index()
            _ts_df.columns = ["Tipo", "N"]
            _ts_df["pct"] = _ts_df["N"] / max(total_sum_tipo, 1) * 100
            _ts_df["texto"] = _ts_df.apply(lambda r: f"{int(r['N']):,}<br>({r['pct']:.1f}%)", axis=1)
            fig_ts = px.bar(_ts_df, x="Tipo", y="N", text="texto",
                            color="Tipo", color_discrete_sequence=COL_STATS,
                            height=320,
                            title=f"Suma de mascotas por tipo  |  {int(total_sum_tipo):,} total",
                            labels={"Tipo": "", "N": "N° mascotas"})
            fig_ts.update_traces(textposition="outside",
                                 textfont=dict(size=10, color="#24743c", family="sans-serif"))
            fig_ts.update_layout(showlegend=False,
                                 yaxis=dict(showticklabels=False, showgrid=False),
                                 margin=dict(t=45, b=10))
            st.plotly_chart(fig_ts, use_container_width=True, key="pc_masc_tipo")

        # Panel 2: distribución cantidad por contrato
        with r1c2:
            if "Cantidad_mascotas" in df_masc.columns:
                _cant_clip = _cant_m.astype(int).clip(upper=5)
                _cant_clip = _cant_clip.replace(5, "5+").astype(str)
                cant_cnt = (_cant_clip.value_counts()
                            .reindex([str(i) for i in range(1, 5)] + ["5+"], fill_value=0)
                            .reset_index())
                cant_cnt.columns = ["N_masc", "n"]
                cant_cnt["texto"] = cant_cnt.apply(
                    lambda r: f"{int(r['n']):,}<br>({r['n']/max(total_contratos_masc,1)*100:.1f}%)" if r["n"] > 0 else "", axis=1)
                fig_cant = px.bar(cant_cnt, x="N_masc", y="n", text="texto",
                                  color="N_masc", color_discrete_sequence=COL_STATS,
                                  height=320,
                                  title="Cantidad de mascotas por contrato",
                                  labels={"N_masc": "N° mascotas", "n": "Contratos"})
                fig_cant.update_traces(textposition="outside",
                                       textfont=dict(size=10, color="#24743c", family="sans-serif"))
                fig_cant.update_layout(showlegend=False,
                                       yaxis=dict(showticklabels=False, showgrid=False),
                                       margin=dict(t=45, b=10))
                st.plotly_chart(fig_cant, use_container_width=True, key="pc_masc_cant")

        # Panel 3: histograma edades individuales (parsear Edadesmascotas, ≤30 años)
        with r2c1:
            EDAD_MAX_MASCOTA = 30
            _edad_col = next((c for c in ["Edadesmascotas", "Edades_raw"] if c in df_masc.columns), None)
            if _edad_col:
                def _parsear_edades_m(val):
                    if pd.isna(val): return []
                    s = str(val).strip().replace("--", "-").strip("-")
                    if not s or s.lower() in ("nan", "none", ""): return []
                    edades = []
                    for parte in s.split("-"):
                        try:
                            e = float(parte.strip())
                            if 0 <= e <= EDAD_MAX_MASCOTA:
                                edades.append(e)
                        except (ValueError, AttributeError):
                            pass
                    return edades
                _edades = (
                    df_masc[_edad_col].apply(_parsear_edades_m).explode().dropna().astype(float)
                )
                if len(_edades) > 0:
                    fig_edad = go.Figure()
                    fig_edad.add_trace(go.Histogram(
                        x=_edades, nbinsx=25,
                        marker_color=COL_STATS[0], marker_line_color="white", marker_line_width=0.6,
                        name="Edades",
                    ))
                    fig_edad.add_vline(x=_edades.median(), line_dash="dash", line_color=C_AMARILLO,
                                       annotation_text=f"Mediana: {_edades.median():.1f} años",
                                       annotation_position="top right")
                    fig_edad.add_vline(x=_edades.mean(), line_dash="dash", line_color=C_V_PAL,
                                       annotation_text=f"Media: {_edades.mean():.1f} años",
                                       annotation_position="top left")
                    fig_edad.update_layout(
                        height=320,
                        title=f"Edad de mascotas — {len(_edades):,} válidas (≤{EDAD_MAX_MASCOTA} años)",
                        xaxis_title="Edad (años)", yaxis_title="Mascotas",
                        margin=dict(t=45, b=10), showlegend=False,
                        paper_bgcolor="white", plot_bgcolor="#f5fbf7",
                    )
                    st.plotly_chart(fig_edad, use_container_width=True, key="pc_masc_edad")

        # Panel 4: top razas
        with r2c2:
            _col_razas = next((c for c in ["Razasmascotas", "Razas"] if c in df_masc.columns), None)
            if _col_razas:
                _razas = (df_masc[_col_razas].dropna().astype(str).str.strip()
                          .replace({"nan": "", "None": "", "none": ""}))
                _razas = _razas[_razas != ""].value_counts().head(10)
                if len(_razas) > 0:
                    _rz_df = _razas.reset_index()
                    _rz_df.columns = ["Raza", "n"]
                    fig_rz = px.bar(_rz_df.sort_values("n"), x="n", y="Raza",
                                    orientation="h", text="n",
                                    color_discrete_sequence=COL_STATS,
                                    height=320, title="Top 10 razas",
                                    labels={"n": "Contratos", "Raza": ""})
                    fig_rz.update_traces(
                        texttemplate="%{x:,}", textposition="outside",
                        textfont=dict(size=9, color="#24743c", family="sans-serif"),
                        marker_color=COL_STATS[1],
                    )
                    fig_rz.update_layout(showlegend=False,
                                         xaxis=dict(range=[0, _razas.max() * 1.18]),
                                         margin=dict(t=45, b=10))
                    st.plotly_chart(fig_rz, use_container_width=True, key="pc_masc_razas")
        st.divider()

    # ── Canal comercial vs Tipo de asistencia — 6 productos ──────────────
    _SEIS_CAN = ["Poliza", "Salud", "Bicicleta", "Repatriacion", "Expatriacion", "Mascotas"]
    if "Canal" in df_ct.columns:
        st.markdown("#### Canal comercial vs Tipo de asistencia — 6 productos")
        df_can = df_ct.copy()
        # Construir columnas de productos
        if "Poliza" not in df_can.columns and "Tiposseguros_ajuste" in df_can.columns:
            df_can["Poliza"] = df_can["Tiposseguros_ajuste"].isin(["AP", "PFI", "SOLICANASTA"]).astype(int)
        if "Mascotas" not in df_can.columns:
            _vp_c = df_can["Tieneperro"].isin({"Y","S"}) if "Tieneperro" in df_can.columns else pd.Series(False, index=df_can.index)
            _vg_c = df_can["Tienegato"].isin({"Y","S"})  if "Tienegato"  in df_can.columns else pd.Series(False, index=df_can.index)
            df_can["Mascotas"] = (_vp_c | _vg_c).astype(int)
        _seis_can = [c for c in _SEIS_CAN if c in df_can.columns]
        for _c in _seis_can:
            df_can[_c] = pd.to_numeric(df_can[_c], errors="coerce").fillna(0).gt(0).astype(int)

        # Etiqueta de combinación exacta
        df_can["Tipo_asistencia"] = df_can[_seis_can].apply(
            lambda r: " + ".join(c for c in _seis_can if r[c] == 1) or "Sin productos", axis=1
        )

        total_can = len(df_can)
        _umbral   = total_can * 0.005
        _counts_c = df_can["Tipo_asistencia"].value_counts()
        _top_c    = _counts_c[_counts_c >= _umbral].index.tolist()
        _otros_c  = _counts_c[_counts_c < _umbral].index.tolist()
        if _otros_c:
            df_can.loc[df_can["Tipo_asistencia"].isin(_otros_c), "Tipo_asistencia"] = "Otras combinaciones"

        # Orden: de mayor a menor, Sin productos y Otras al final
        _orden_can = (df_can["Tipo_asistencia"].value_counts()
                      .sort_values(ascending=False).index.tolist())
        for _last in ["Sin productos", "Otras combinaciones"]:
            if _last in _orden_can:
                _orden_can.remove(_last)
                _orden_can.append(_last)

        # Agrupar por Canal x Tipo_asistencia
        _cnt_col = "Contrato" if "Contrato" in df_can.columns else None
        if _cnt_col:
            cruce_can = (df_can.groupby(["Canal", "Tipo_asistencia"])[_cnt_col]
                         .nunique().reset_index())
            cruce_can.columns = ["Canal", "Tipo_asistencia", "Contratos"]
        else:
            cruce_can = (df_can.groupby(["Canal", "Tipo_asistencia"])
                         .size().reset_index(name="Contratos"))

        # Totales por canal para etiqueta superior
        totales_can = cruce_can.groupby("Canal")["Contratos"].sum().to_dict()
        cruce_can["total_canal"] = cruce_can["Canal"].map(totales_can)
        cruce_can = cruce_can.sort_values("total_canal", ascending=False)

        # Solo mostrar etiqueta si el segmento es >=4% del total del canal
        cruce_can["_texto"] = cruce_can.apply(
            lambda r: f"{int(r['Contratos']):,}" if r["Contratos"] / max(r["total_canal"], 1) >= 0.04 else "",
            axis=1
        )
        fig_can = px.bar(
            cruce_can,
            x="Canal", y="Contratos", color="Tipo_asistencia",
            barmode="stack", height=440,
            color_discrete_sequence=COL_STATS * 4,
            category_orders={"Tipo_asistencia": _orden_can},
            labels={"Canal": "Canal comercial", "Contratos": "Contratos únicos",
                    "Tipo_asistencia": "Combinación"},
            text="_texto",
        )
        fig_can.update_traces(
            texttemplate="%{text}", textposition="inside",
            textfont=dict(size=8, color="white"),
        )
        # Totales sobre cada barra
        for _canal, _tot in totales_can.items():
            fig_can.add_annotation(
                x=_canal, y=_tot, text=f"<b>{int(_tot):,}</b>",
                showarrow=False, yanchor="bottom",
                font=dict(size=9, color=C_VERDE),
                yshift=4,
            )
        fig_can.update_layout(
            legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
            margin=dict(t=20, b=10),
            xaxis=dict(tickangle=-35),
        )
        st.plotly_chart(fig_can, use_container_width=True, key="pc_canal_6prod")
        st.caption(f"Base: {total_can:,} contratos únicos · combinaciones con <0.5% agrupadas en 'Otras combinaciones'")
        st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 👨‍👩‍👧 Por familias (beneficiarios activos)")
    # Act.valor = 1 por cada beneficiario activo → contar filas = contar familias
    df_fam = df[pd.to_numeric(df["Act.valor"], errors="coerce").fillna(0).gt(0)] \
             if "Act.valor" in df.columns else df.copy()
    n_fam = len(df_fam)
    st.caption(f"Base: **{n_fam:,}** beneficiarios activos (Act.valor = 1 por persona activa)")
    st.divider()

    # ── 10.5 · Beneficiarios activos por canal ────────────────────────────
    if "Canal" in df_fam.columns:
        st.markdown("#### 10.5 · Beneficiarios activos por canal comercial")
        seg_canal = (df_fam.groupby("Canal").size()
                     .sort_values(ascending=False).reset_index())
        seg_canal.columns = ["Canal", "Beneficiarios"]
        fig_seg = px.bar(seg_canal, x="Canal", y="Beneficiarios", text="Beneficiarios",
                         color_discrete_sequence=COL_STATS, height=320,
                         labels={"Canal": "", "Beneficiarios": "Beneficiarios activos"})
        fig_seg.update_traces(texttemplate="%{text:,}", textposition="outside",
                              textfont=dict(size=13, color="#24743c", family="sans-serif"),
                              marker_color=COL_STATS[0])
        fig_seg.update_layout(showlegend=False, yaxis=dict(showticklabels=False, showgrid=False),
                              margin=dict(t=30, b=10))
        st.plotly_chart(fig_seg, use_container_width=True, key="pc_035")
        st.divider()

    # ── 10.2 · Beneficiarios activos por geografía ────────────────────────
    st.markdown("#### 10.2 · Beneficiarios activos por geografía")
    fa1, fa2 = st.columns(2)
    if "REGION" in df_fam.columns:
        with fa1:
            reg_fam = (df_fam.groupby("REGION").size()
                       .sort_values(ascending=False).reset_index())
            reg_fam.columns = ["Región", "Beneficiarios"]
            fig_rf = px.bar(reg_fam, x="Región", y="Beneficiarios", text="Beneficiarios",
                            color="Región", color_discrete_sequence=COL_STATS, height=320,
                            labels={"Región": "", "Beneficiarios": "Beneficiarios activos"})
            fig_rf.update_traces(texttemplate="%{text:,}", textposition="outside",
                                 textfont=dict(size=13, color="#24743c", family="sans-serif"))
            fig_rf.update_layout(showlegend=False, yaxis=dict(showticklabels=False, showgrid=False),
                                 margin=dict(t=30, b=10))
            st.plotly_chart(fig_rf, use_container_width=True, key="pc_036")

    if "DEPARTAMENTO" in df_fam.columns:
        with fa2:
            fd_ctrl, fd_chart = st.columns([1, 2])
            with fd_ctrl:
                top_dep_f = st.slider("Top departamentos", 5, 20, 10, key="fam_dep_n")
            dep_fam = (df_fam.groupby("DEPARTAMENTO").size()
                       .sort_values(ascending=False).head(top_dep_f).reset_index())
            dep_fam.columns = ["Departamento", "Beneficiarios"]
            with fd_chart:
                st.plotly_chart(_bar_h(dep_fam, "Beneficiarios", "Departamento",
                                       n_top=top_dep_f, color=COL_STATS[1]),
                                use_container_width=True, key="pc_037")

    ciu_col_fam = next((c for c in ["CIUDAD_STD", "Ciudad"] if c in df_fam.columns), None)
    if ciu_col_fam:
        fc_ctrl, fc_chart = st.columns([1, 3])
        with fc_ctrl:
            top_ciu_f = st.slider("Top ciudades", 5, 20, 10, key="fam_ciu_n")
        ciu_fam = (df_fam.groupby(ciu_col_fam).size()
                   .sort_values(ascending=False).head(top_ciu_f).reset_index())
        ciu_fam.columns = ["Ciudad", "Beneficiarios"]
        with fc_chart:
            st.plotly_chart(_bar_h(ciu_fam, "Beneficiarios", "Ciudad",
                                   n_top=top_ciu_f, color=COL_STATS[2]),
                            use_container_width=True, key="pc_038")
    st.divider()


def pagina_estadisticas(df_full: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    _c1, _c2 = st.columns([0.06, 0.94])
    with _c1:
        st.image(os.path.join(BASE_DIR, "imagenes", "estadisticas.png"), width=42)
    with _c2:
        st.header("Estadísticas Empresariales y Familias")

    try:
        df_ext = cargar_datos_full()
    except Exception as e:
        st.warning(f"No se pudo cargar clientes_segmentados.parquet ({e}). Usando tabla_clusters.")
        df_ext = df_full.copy()

    total_orig = len(df_ext)
    df_filt    = _filtros_panel(df_ext)
    total_filt = len(df_filt)
    pct_filt   = total_filt / max(total_orig, 1) * 100
    st.caption(f"Registros: **{total_filt:,}** de {total_orig:,} ({pct_filt:.1f}%)")

    if total_filt == 0:
        st.warning("Sin registros para los filtros seleccionados. Ajusta los criterios.")
        return

    tab_emp, tab_fam = st.tabs(["🏢 Empresariales", "👨‍👩‍👧 Titulares y Familias"])
    with tab_emp:
        _tab_empresariales(df_filt)
    with tab_fam:
        _tab_familias(df_filt)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _check_auth()

    # ── Navegación con imágenes ───────────────────────────────────────────────
    _PAGINAS = [
        ("Dashboard",          "dashboard-icon-23660.png"),
        ("PCA Interactivo",    "PCA.png"),
        ("Explorar registros", "exportar.png"),
        ("Clasificar cliente", "Cliente.png"),
        ("Estadísticas",       "estadisticas.png"),
    ]
    if "pagina" not in st.session_state:
        st.session_state["pagina"] = "Dashboard"

    with st.sidebar:
        _logo = os.path.join(BASE_DIR, "imagenes", "Logo.png")
        if os.path.exists(_logo):
            st.image(_logo, use_container_width=True)
        st.title("Segmentación\nCorporativo")
        st.caption("K-Prototypes")
        st.divider()

        for _label, _img_file in _PAGINAS:
            _img_path = os.path.join(BASE_DIR, "imagenes", _img_file)
            _col_i, _col_b = st.columns([1, 4])
            with _col_i:
                if os.path.exists(_img_path):
                    st.image(_img_path, width=28)
            with _col_b:
                _active = st.session_state["pagina"] == _label
                if st.button(
                    _label,
                    key=f"nav_{_label}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    st.session_state["pagina"] = _label
                    st.rerun()

        st.divider()
        st.caption("Pipeline: NB01 → NB02 → NB03 → App")
        st.divider()
        usuario_actual = st.session_state.get("usuario", "")
        st.caption(f"👤 {usuario_actual}")
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state["auth"]    = False
            st.session_state["usuario"] = ""
            st.rerun()

    pagina = st.session_state["pagina"]

    try:
        df_res, df_full = cargar_datos()
    except FileNotFoundError as e:
        st.error(
            f"**Archivo no encontrado:** `{e}`\n\n"
            "Asegúrate de que `df_res.xlsx` y `tabla_clusters.xlsx` estén "
            "en el mismo directorio que `app.py`. Ejecuta NB02 y NB03 primero."
        )
        st.stop()

    df_summary = enriquecer(df_res, df_full)

    if   pagina == "Dashboard":
        pagina_dashboard(df_summary, df_full)
    elif pagina == "PCA Interactivo":
        pagina_pca(df_full)
    elif pagina == "Explorar registros":
        pagina_explorar(df_full, df_summary)
    elif pagina == "Clasificar cliente":
        # Modelo pesado (82 MB) — solo se carga al entrar a esta página
        modelo, pt, imp_stats, meta = cargar_modelo()
        pagina_clasificar(modelo, pt, imp_stats, df_summary)
    elif pagina == "Estadísticas":
        pagina_estadisticas(df_full, df_summary)


main()
