"""
predictor.py — Motor de predicción E-Visor
Carga modelos .joblib y genera predicciones por bloque y por clúster.
"""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

# ── Configuración de clústeres ─────────────────────────────────────────────
CLUSTER_MAP = {
    'SmartMeter_SM_B10_ARQ': 0, 'SmartMeter_SM_B4_PRIM': 0,
    'SmartMeter_SM_B5_BACH': 0, 'SmartMeter_SM_B7_TAC':  0,
    'SmartMeter_SM_B8_LABS': 0, 'SmartMeter_SM_B9_SFA2': 0,
    'SmartMeter_SM_B12_DERE': 2, 'SmartMeter_SM_B17_POLI': 2,
    'SmartMeter_SM_B3_RECT':  2, 'SmartMeter_SM_B8_AA':   2,
    'SmartMeter_SM_B8_CPA':   2, 'SmartMeter_SM_ECOVILLA': 2,
    'SmartMeter_SM_B18_PARQ': 1, 'SmartMeter_SM_B9_SFA1': 1,
    'SmartMeter_SM_B7_CTIC':  3,
}

CLUSTER_NAMES = {
    0: 'Académico estándar',
    1: 'Alto consumo eficiente',
    2: 'Aulas ineficientes',
    3: 'Operación constante',
}

CLUSTER_COLORS = {
    0: "#A9B04C",   # azul académico
    1: '#55A868',   # verde alto consumo
    2: "#B04EC4",   # rojo aulas ineficientes
    3: "#78CEE3",   # púrpura operación constante
}

PRED_COLOR    = '#FF8C00'   # naranja para predicciones (todos los clústeres)
BAND_COLOR    = 'rgba(255,140,0,0.15)'  # banda p10–p90 C2

FESTIVOS = pd.to_datetime(['2026-03-23'])

# ── Carga de modelos ────────────────────────────────────────────────────────

_MODELS_CACHE: dict = {}

def cargar_modelos(models_dir: str) -> dict:
    """
    Carga todos los modelos y metadata desde models_dir.
    Usa caché en memoria para no re-leer en cada interacción de Streamlit.
    """
    global _MODELS_CACHE
    if _MODELS_CACHE:
        return _MODELS_CACHE

    p = Path(models_dir)
    required = [
        'modelo_c0_academico.joblib',
        'modelo_c1_alto_consumo.joblib',
        'modelos_c2_aulas_cuantil.joblib',
        'modelo_c3_constante.joblib',
        'meta_cluster.joblib',
    ]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos de modelo en '{models_dir}':\n" + "\n".join(missing)
        )

    _MODELS_CACHE = {
        'c0': joblib.load(p / 'modelo_c0_academico.joblib'),
        'c1': joblib.load(p / 'modelo_c1_alto_consumo.joblib'),
        'c2': joblib.load(p / 'modelos_c2_aulas_cuantil.joblib'),   # dict {0.1, 0.5, 0.9}
        'c3': joblib.load(p / 'modelo_c3_constante.joblib'),
        'meta': joblib.load(p / 'meta_cluster.joblib'),
    }
    return _MODELS_CACHE


# ── Feature engineering (idéntico al notebook) ────────────────────────────

def build_features(df_bloque: pd.DataFrame, festivos=FESTIVOS) -> pd.DataFrame:
    df = df_bloque.copy()

    df['hora_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dia_sin']  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dia_cos']  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['mes_sin']  = np.sin(2 * np.pi * df.index.month / 12)
    df['mes_cos']  = np.cos(2 * np.pi * df.index.month / 12)

    df['es_fin_de_semana'] = (df.index.dayofweek >= 5).astype(int)
    df['es_festivo']       = df.index.normalize().isin(festivos).astype(int)
    df['dia_no_laboral']   = df[['es_fin_de_semana', 'es_festivo']].max(axis=1)
    df['es_hora_pico'] = (
        (df.index.hour >= 7) & (df.index.hour <= 20) &
        (df.index.dayofweek < 5) & (df['dia_no_laboral'] == 0)
    ).astype(int)

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'lag_{lag}h'] = df['y'].shift(lag)

    y_shifted = df['y'].shift(1)
    df['media_24h']  = y_shifted.rolling(window=24,  min_periods=12).mean()
    df['std_24h']    = y_shifted.rolling(window=24,  min_periods=12).std()
    df['min_24h']    = y_shifted.rolling(window=24,  min_periods=12).min()
    df['max_24h']    = y_shifted.rolling(window=24,  min_periods=12).max()
    df['media_168h'] = y_shifted.rolling(window=168, min_periods=84).mean()
    df['std_168h']   = y_shifted.rolling(window=168, min_periods=84).std()

    return df


# ── Predicción por bloque ──────────────────────────────────────────────────

def predecir_bloque(
    entity_id: str,
    df_historico: pd.DataFrame,
    models: dict,
    horizon_h: int = 24,
) -> pd.DataFrame:
    """
    Genera predicción iterativa para un bloque dado.

    Args:
        entity_id:    Nombre completo del bloque.
        df_historico: DataFrame con columnas ['time_index_colombia', 'entity_id', 'activepower'].
                      Debe incluir al menos 192h (8 días) de historia previa para este bloque.
        models:       Dict devuelto por cargar_modelos().
        horizon_h:    Horas a predecir (default 24).

    Returns:
        DataFrame con columnas: timestamp, y_pred
        Para C2 también: y_pred_p10, y_pred_p90
    """
    meta       = models['meta']
    cluster_id = CLUSTER_MAP[entity_id]
    feature_cols = meta['feature_cols']

    # Seleccionar modelo(s) y encoding
    enc_key = f'bloque_enc_c{cluster_id}'
    bloque_enc_map = meta.get(enc_key, {})

    if cluster_id == 0:
        modelos_pred = {0.5: models['c0']}
    elif cluster_id == 1:
        modelos_pred = {0.5: models['c1']}
    elif cluster_id == 2:
        modelos_pred = models['c2']           # {0.1, 0.5, 0.9}
    else:
        modelos_pred = {0.5: models['c3']}

    # Preparar serie histórica
    sub = (
        df_historico[df_historico['entity_id'] == entity_id]
        .set_index('time_index_colombia')
        .sort_index()[['activepower']]
        .rename(columns={'activepower': 'y'})
        .asfreq('h')
    )
    if sub['y'].isna().sum() > 0:
        sub['y'] = sub['y'].interpolate(method='time', limit=3)
        sub = sub.dropna()

    last_ts   = sub.index[-1]
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_h, freq='h')

    bloque_enc_val = float(bloque_enc_map.get(entity_id, 0))

    # Predicción iterativa por cuantil
    results_by_q = {}
    for q, modelo in modelos_pred.items():
        # Mantener historia como DataFrame (no Series) para que concat sea consistente
        hist_df = sub[['y']].copy()
        preds   = []

        for ts in future_ts:
            tmp      = pd.DataFrame({'y': [np.nan]}, index=[ts])
            full_tmp = pd.concat([hist_df, tmp])   # DataFrame + DataFrame → limpio
            feat_row = build_features(full_tmp).iloc[[-1]].copy()
            feat_row['bloque_enc'] = bloque_enc_val
            feat_row = feat_row[feature_cols]

            pred = float(modelo.predict(feat_row)[0])
            if pred < 0:
                print(f"[WARN] prediccion negativa ({pred:.1f} W) en {ts} — modelo fuera de distribucion?")
            pred = max(0.0, pred)
            preds.append(pred)

            # Añadir predicción al historial para el siguiente paso
            hist_df = pd.concat([hist_df, pd.DataFrame({'y': [pred]}, index=[ts])])

        results_by_q[q] = preds

    result = pd.DataFrame({'timestamp': future_ts, 'y_pred': results_by_q[0.5]})
    if cluster_id == 2:
        result['y_pred_p10'] = results_by_q[0.1]
        result['y_pred_p90'] = results_by_q[0.9]

    return result


# ── Predicción por clúster ─────────────────────────────────────────────────

def predecir_cluster(
    cluster_id: int,
    df_historico: pd.DataFrame,
    models: dict,
    horizon_h: int = 24,
) -> pd.DataFrame:
    """
    Genera predicciones para todos los bloques de un clúster.

    Returns:
        DataFrame con columnas: timestamp, entity_id, cluster_id, y_pred
        (más y_pred_p10, y_pred_p90 para C2)
    """
    bloques = [b for b, c in CLUSTER_MAP.items() if c == cluster_id]
    frames  = []

    for bloque in bloques:
        try:
            pred = predecir_bloque(bloque, df_historico, models, horizon_h)
            pred['entity_id']  = bloque
            pred['cluster_id'] = cluster_id
            frames.append(pred)
        except Exception as e:
            # Bloque sin datos suficientes → se omite con advertencia
            print(f"[WARN] {bloque}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def predecir_todos(
    df_historico: pd.DataFrame,
    models: dict,
    horizon_h: int = 24,
) -> pd.DataFrame:
    """Genera predicciones para todos los clústeres y bloques."""
    frames = []
    for cluster_id in sorted(set(CLUSTER_MAP.values())):
        df_c = predecir_cluster(cluster_id, df_historico, models, horizon_h)
        if not df_c.empty:
            frames.append(df_c)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Preparar ventana de historia para visualización ───────────────────────

def get_historia_bloque(
    entity_id: str,
    df_full: pd.DataFrame,
    dias: int = 7,
) -> pd.DataFrame:
    """
    Devuelve los últimos `dias` días de historia real para un bloque.
    df_full debe tener columnas: time_index_colombia, entity_id, activepower.
    """
    sub = (
        df_full[df_full['entity_id'] == entity_id]
        .sort_values('time_index_colombia')
        .copy()
    )
    cutoff = sub['time_index_colombia'].max() - pd.Timedelta(days=dias)
    return sub[sub['time_index_colombia'] > cutoff][['time_index_colombia', 'activepower']]
