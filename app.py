"""
app.py — Dashboard E-Visor: Predicción de Consumo Energético por Clúster
Streamlit Cloud deployment
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from predictor import (
    cargar_modelos,
    predecir_bloque,
    predecir_cluster,
    predecir_todos,
    get_historia_bloque,
    CLUSTER_MAP,
    CLUSTER_NAMES,
    CLUSTER_COLORS,
    PRED_COLOR,
)

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Visor · Predicción Energética",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.info(
    "**Nota sobre las predicciones:** Los modelos fueron entrenados con 47 días de mediciones "
    "(Feb 10 – Mar 28, 2026). A medida que se incorporen más datos históricos, "
    "la precisión de las predicciones seguirá mejorando."
)

# ── CSS personalizado ──────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f1117; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card {
        background: #1a1d27;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border-left: 3px solid var(--accent);
    }
    .cluster-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    h1 { font-size: 1.6rem !important; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Rutas de archivos ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "data" / "datos_limpios_09042026.csv"

CUTOFF = pd.Timestamp("2026-03-29")   # inicio apagón

# ── Carga de datos y modelos (cacheado) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["time_index_colombia"] = pd.to_datetime(df["time_index_colombia"])
    df = df[df["time_index_colombia"] < CUTOFF].copy()
    df = df.sort_values(["entity_id", "time_index_colombia"])
    return df

@st.cache_resource(show_spinner=False)
def cargar_modelos_cached():
    return cargar_modelos(str(MODELS_DIR))


# ── Helpers de visualización ───────────────────────────────────────────────
def nombre_corto(entity_id: str) -> str:
    return entity_id.replace("SmartMeter_SM_", "")


def fig_bloque(entity_id: str, historia: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    """
    Gráfico principal: historia (7 días) + predicción (24h).
    Para C2 incluye banda p10–p90.
    """
    cluster_id  = CLUSTER_MAP[entity_id]
    hist_color  = CLUSTER_COLORS[cluster_id]
    nombre      = nombre_corto(entity_id)

    fig = go.Figure()

    # Historia real
    fig.add_trace(go.Scatter(
        x=historia["time_index_colombia"],
        y=historia["activepower"],
        name="Medición real",
        line=dict(color=hist_color, width=1.8),
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:,.0f} W</b><extra></extra>",
    ))

    # Línea de corte — convertir a ms para compatibilidad con Plotly
    last_real = historia["time_index_colombia"].max()
    fig.add_vline(
        x=last_real.timestamp() * 1000,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text=" Inicio predicción",
        annotation_font_size=11,
        annotation_font_color="rgba(255,255,255,0.5)",
    )

    # Banda p10–p90 solo para C2
    if cluster_id == 2 and "y_pred_p10" in pred.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([pred["timestamp"], pred["timestamp"].iloc[::-1]]),
            y=pd.concat([pred["y_pred_p90"], pred["y_pred_p10"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(255,140,0,0.15)",
            line=dict(color="rgba(255,140,0,0)"),
            name="Banda p10–p90",
            hoverinfo="skip",
        ))

    # Predicción p50
    fig.add_trace(go.Scatter(
        x=pred["timestamp"],
        y=pred["y_pred"],
        name="Predicción 24h",
        line=dict(color=PRED_COLOR, width=2, dash="dot"),
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:,.0f} W</b><extra>predicción</extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{nombre}</b>   ·   {CLUSTER_NAMES[cluster_id]}",
            font=dict(size=15),
            x=0,
        ),
        height=360,
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=True,
            range=[
                historia["time_index_colombia"].min(),
                pred["timestamp"].max(),
            ],
            gridcolor="rgba(255,255,255,0.06)",
            tickformat="%d %b\n%H:%M",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            ticksuffix=" W",
            tickfont=dict(size=11),
            tickformat=",.0f",
        ),
        hovermode="x unified",
    )
    return fig


def fig_cluster_grid(cluster_id: int, historia_df: pd.DataFrame, pred_df: pd.DataFrame) -> list:
    """Devuelve lista de figuras, una por bloque del clúster."""
    bloques = [b for b, c in CLUSTER_MAP.items() if c == cluster_id]
    figs    = []
    for bloque in sorted(bloques):
        hist = get_historia_bloque(bloque, historia_df, dias=7)
        pred = pred_df[pred_df["entity_id"] == bloque].copy()
        if hist.empty or pred.empty:
            continue
        figs.append((bloque, fig_bloque(bloque, hist, pred)))
    return figs


def metricas_card(pred: pd.DataFrame, cluster_id: int):
    """Muestra tarjetas de estadísticas de la predicción."""
    cols = st.columns(4)
    stats = [
        ("Promedio predicho",  f"{pred['y_pred'].mean():,.0f} W"),
        ("Máximo predicho",    f"{pred['y_pred'].max():,.0f} W"),
        ("Mínimo predicho",    f"{pred['y_pred'].min():,.0f} W"),
        ("Hora pico predicha",    pred["timestamp"].iloc[pred["y_pred"].argmax()].strftime("%H:%M")),
    ]
    for col, (label, value) in zip(cols, stats):
        col.metric(label, value)

    if cluster_id == 2 and "y_pred_p10" in pred.columns:
        st.caption(
            f"Banda de incertidumbre p10–p90: "
            f"{pred['y_pred_p10'].mean():,.0f} W — {pred['y_pred_p90'].mean():,.0f} W (promedio)"
        )


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## E-Visor")
    st.markdown("**Predicción de consumo energético**  \nUniversidad Pontificia Bolivariana")
    st.divider()

    modo = st.radio(
        "Vista",
        ["Por clúster", "Por bloque individual"],
        index=0,
    )

    st.divider()

    if modo == "Por clúster":
        cluster_sel = st.selectbox(
            "Clúster",
            options=sorted(set(CLUSTER_MAP.values())),
            format_func=lambda c: f"C{c} — {CLUSTER_NAMES[c]}",
        )
        bloque_sel = None
    else:
        bloque_sel = st.selectbox(
            "Bloque",
            options=sorted(CLUSTER_MAP.keys()),
            format_func=nombre_corto,
        )
        cluster_sel = CLUSTER_MAP[bloque_sel]

    st.divider()
    st.caption("Datos: Feb 10 – Mar 28, 2026  \nModelo entrenado con XGBoost / LightGBM  \nHorizonte 24h")


# ── Carga inicial ──────────────────────────────────────────────────────────
with st.spinner("Cargando datos y modelos..."):
    try:
        df_full = cargar_datos()
        models  = cargar_modelos_cached()
        carga_ok = True
    except FileNotFoundError as e:
        st.error(f"**Error al cargar archivos**\n\n{e}")
        st.info(
            "Asegúrate de que la carpeta `models/` contiene los 5 archivos `.joblib` "
            "y que `data/datos_limpios_09042026.csv` existe."
        )
        carga_ok = False

if not carga_ok:
    st.stop()


# ── Vista: Por clúster ─────────────────────────────────────────────────────
if modo == "Por clúster":
    cluster_id   = cluster_sel
    cluster_name = CLUSTER_NAMES[cluster_id]
    color_hex    = CLUSTER_COLORS[cluster_id]

    st.markdown(f"## C{cluster_id} — {cluster_name}")
    st.caption(
        f"Predicción de las próximas 24 horas para los "
        f"**{sum(1 for c in CLUSTER_MAP.values() if c == cluster_id)} bloques** del clúster"
    )

    with st.spinner(f"Generando predicciones para C{cluster_id}..."):
        pred_cluster = predecir_cluster(cluster_id, df_full, models, horizon_h=24)

    if pred_cluster.empty:
        st.warning("No se pudieron generar predicciones para este clúster.")
        st.stop()

    # Métricas agregadas
    metricas_card(pred_cluster, cluster_id)
    st.divider()

    # Gráficos por bloque (2 columnas)
    figs = fig_cluster_grid(cluster_id, df_full, pred_cluster)
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figs):
                bloque, fig = figs[i + j]
                col.plotly_chart(fig, use_container_width=True)

    # Tabla de predicciones
    with st.expander("Ver tabla de predicciones hora a hora"):
        tabla = pred_cluster.copy()
        tabla["bloque"] = tabla["entity_id"].apply(nombre_corto)
        tabla["hora"]   = tabla["timestamp"].dt.strftime("%d %b %H:%M")
        cols_show = ["hora", "bloque", "y_pred"]
        if cluster_id == 2:
            cols_show += ["y_pred_p10", "y_pred_p90"]

        rename = {"hora": "Hora", "bloque": "Bloque", "y_pred": "Predicción p50 (W)",
                  "y_pred_p10": "Piso p10 (W)", "y_pred_p90": "Techo p90 (W)"}
        st.dataframe(
            tabla[cols_show].rename(columns=rename).set_index("Hora"),
            use_container_width=True,
        )


# ── Vista: Por bloque individual ───────────────────────────────────────────
else:
    entity_id    = bloque_sel
    cluster_id   = CLUSTER_MAP[entity_id]
    cluster_name = CLUSTER_NAMES[cluster_id]
    nombre       = nombre_corto(entity_id)

    st.markdown(f"## {nombre}")
    st.caption(f"Clúster: **C{cluster_id} — {cluster_name}**")

    with st.spinner(f"Generando predicción para {nombre}..."):
        pred = predecir_bloque(entity_id, df_full, models, horizon_h=24)

    hist = get_historia_bloque(entity_id, df_full, dias=7)

    # Métricas del bloque
    metricas_card(pred, cluster_id)
    st.divider()

    # Gráfico principal
    fig = fig_bloque(entity_id, hist, pred)
    st.plotly_chart(fig, use_container_width=True)

    # Detalle de la semana anterior
    with st.expander("Ver historia de la última semana"):
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist["time_index_colombia"],
            y=hist["activepower"],
            line=dict(color=CLUSTER_COLORS[cluster_id], width=1.5),
            hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:,.0f} W</b><extra></extra>",
        ))
        fig_hist.update_layout(
            height=250, margin=dict(t=20, b=30, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat="%d %b\n%H:%M"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", ticksuffix=" W", tickformat=",.0f"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tabla de predicción
    with st.expander("Ver predicción hora a hora"):
        tabla = pred[["timestamp", "y_pred"]].copy()
        tabla["timestamp"] = tabla["timestamp"].dt.strftime("%d %b %H:%M")
        rename = {"timestamp": "Hora", "y_pred": "Predicción p50 (W)"}
        if cluster_id == 2:
            tabla["y_pred_p10"] = pred["y_pred_p10"].values
            tabla["y_pred_p90"] = pred["y_pred_p90"].values
            rename.update({"y_pred_p10": "Piso p10 (W)", "y_pred_p90": "Techo p90 (W)"})
        st.dataframe(
            tabla.rename(columns=rename).set_index("Hora"),
            use_container_width=True,
        )
