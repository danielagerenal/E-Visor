# ⚡ E-Visor — Dashboard de Predicción Energética

Dashboard en Streamlit para visualizar el consumo energético histórico y predecir las próximas 24 horas por clúster y por bloque, usando modelos XGBoost / LightGBM entrenados con datos de smart meters de la UPB.

---

## Estructura del proyecto

```
E-Visor/
├── app.py                  ← App principal de Streamlit
├── predictor.py            ← Motor de predicción (carga modelos, genera forecasts)
├── requirements.txt        ← Dependencias Python
├── .gitignore
├── data/                   ← NO se sube a GitHub (ver instrucciones)
│   └── datos_limpios_09042026.csv
└── models/                 ← NO se sube a GitHub (ver instrucciones)
    ├── modelo_c0_academico.joblib
    ├── modelo_c1_alto_consumo.joblib
    ├── modelos_c2_aulas_cuantil.joblib
    ├── modelo_c3_constante.joblib
    └── meta_cluster.joblib
```

---

## Despliegue en Streamlit Cloud (paso a paso)

### 1. Preparar el repositorio local

```bash
# Clonar tu repo existente
git clone https://github.com/danielagerenal/E-Visor.git
cd E-Visor

# Copiar los archivos nuevos al repo
cp /ruta/donde/descargaste/app.py .
cp /ruta/donde/descargaste/predictor.py .
cp /ruta/donde/descargaste/requirements.txt .
cp /ruta/donde/descargaste/.gitignore .
```

### 2. Añadir datos y modelos (localmente, NO se suben a GitHub)

```bash
mkdir -p data models

# Copiar el CSV de datos
cp /ruta/a/datos_limpios_09042026.csv data/

# Descargar los .joblib desde Google Drive y copiarlos
cp /ruta/a/modelo_c0_academico.joblib        models/
cp /ruta/a/modelo_c1_alto_consumo.joblib     models/
cp /ruta/a/modelos_c2_aulas_cuantil.joblib   models/
cp /ruta/a/modelo_c3_constante.joblib        models/
cp /ruta/a/meta_cluster.joblib               models/
```

### 3. Probar localmente antes de subir

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre http://localhost:8501 y verifica que las predicciones se generan correctamente.

### 4. Subir solo el código a GitHub

Los archivos `data/` y `models/` están en `.gitignore` — no se subirán.

```bash
git add app.py predictor.py requirements.txt .gitignore README.md
git commit -m "feat: dashboard E-Visor con predicción por clúster"
git push origin main
```

### 5. Configurar Streamlit Cloud

Los modelos y datos son demasiado grandes para el repositorio de GitHub. La solución es subirlos como **Streamlit Secrets** (para archivos pequeños) o, como en este caso, usar **Streamlit Cloud Files** mediante la siguiente configuración:

#### Opción A — Subir modelos como archivos al repo (si pesan < 100 MB en total)

Si los 5 `.joblib` + el CSV pesan menos de 100 MB en conjunto, puedes quitarlos del `.gitignore` y subirlos directamente:

```bash
# Edita .gitignore y comenta las líneas de data/ y models/
git add data/ models/
git commit -m "feat: añadir datos y modelos"
git push origin main
```

Luego en Streamlit Cloud:
1. Ve a https://share.streamlit.io
2. Click en "New app"
3. Repositorio: `danielagerenal/E-Visor`
4. Branch: `main`
5. Main file: `app.py`
6. Click "Deploy"

#### Opción B — Modelos grandes (> 100 MB): usar Google Drive + gdown

Si los archivos son grandes, añade `gdown` a `requirements.txt` y agrega esta función al inicio de `app.py` para descargarlos automáticamente al arrancar:

```python
import gdown, os

def descargar_modelos_si_faltan():
    GDRIVE_IDS = {
        "models/modelo_c0_academico.joblib":       "TU_ID_DE_GDRIVE_C0",
        "models/modelo_c1_alto_consumo.joblib":    "TU_ID_DE_GDRIVE_C1",
        "models/modelos_c2_aulas_cuantil.joblib":  "TU_ID_DE_GDRIVE_C2",
        "models/modelo_c3_constante.joblib":        "TU_ID_DE_GDRIVE_C3",
        "models/meta_cluster.joblib":               "TU_ID_DE_GDRIVE_META",
        "data/datos_limpios_09042026.csv":          "TU_ID_DE_GDRIVE_CSV",
    }
    os.makedirs("models", exist_ok=True)
    os.makedirs("data",   exist_ok=True)
    for path, gdrive_id in GDRIVE_IDS.items():
        if not os.path.exists(path):
            gdown.download(id=gdrive_id, output=path, quiet=False)
```

Para obtener el ID de Google Drive de cada archivo:
- Clic derecho en el archivo en Drive → "Compartir" → "Cualquier persona con el enlace puede ver"
- Copia el enlace: `https://drive.google.com/file/d/ESTE_ES_EL_ID/view`

---

## Clústeres y modelos

| Clúster | Nombre | Bloques | Modelo | MAPE |
|---|---|---|---|---|
| C0 | Académico estándar | 6 | XGBoost pooled + walk-forward | 13.8% |
| C1 | Alto consumo eficiente | 2 | XGBoost pooled + walk-forward | 12.2% |
| C2 | Aulas ineficientes | 6 | LightGBM cuantil p10/p50/p90 | 23.1% |
| C3 | Operación constante | 1 | XGBoost simple | 5.3% |

---

## Uso del dashboard

- **Vista por clúster**: muestra todos los bloques del clúster seleccionado en una cuadrícula 2×N, con métricas agregadas y tabla descargable.
- **Vista por bloque individual**: gráfico detallado con historia de 7 días (línea sólida) + predicción 24h (línea punteada naranja). Para C2 incluye la banda de incertidumbre p10–p90.
