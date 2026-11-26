# Código

| Archivo / carpeta | Contenido |
|-------------------|-----------|
| `TP_CC219_Steam_EDA.ipynb` | Trabajo parcial: carga desde Kaggle, limpieza, normalización, análisis exploratorio y modelo base. |
| `TF_CC219_Steam_Modelos.ipynb` | Trabajo final: modelos de P1 (Naive Bayes, regresión logística, SVM lineal y DistilBERT), P2 (regresión logística y LightGBM), P3 (NMF y LDA), E1 (recomendador), E2 (VADER vs. modelo) y E3. Guarda figuras, tablas y modelos. |
| `figuras/` | Gráficos generados por el notebook del TF. |
| `resultados/` | Tablas de métricas en CSV. |
| `app/` | Aplicación web en Streamlit (`app.py`), módulo de limpieza compartido (`preprocesamiento.py`), dependencias y modelos entrenados. |

Ejecutar la aplicación:

```bash
cd code/app
pip install -r requirements.txt
streamlit run app.py
```
