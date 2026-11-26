# CC219-TP-TF-2026-2-CC92

## Minería de textos sobre reseñas de videojuegos en Steam

Trabajo Parcial y Trabajo Final del curso **1ACC0219 – Aplicaciones de Data Science** (sección 4896), Universidad Peruana de Ciencias Aplicadas (UPC), ciclo 2026-2.

---

## Objetivo del trabajo

Aplicar técnicas de **minería de textos y procesamiento de lenguaje natural (NLP)** sobre más de 700 mil reseñas de jugadores de Steam para construir modelos que, a partir del texto de una reseña, permitan responder las siguientes preguntas, y poner esos modelos al alcance de estudios de videojuegos y jugadores mediante una aplicación web:

| # | Pregunta | Tipo de tarea |
|---|----------|---------------|
| P1 | ¿La reseña recomienda el juego o no? | Clasificación binaria |
| P2 | ¿La reseña será votada como útil (2 votos o más)? | Clasificación binaria (texto y contexto) |
| P3 | ¿De qué tema trata la queja o el elogio? | Modelado de tópicos y clasificación multiclase |
| E1 *(extra)* | ¿Qué juegos recomendar a quien le gustó uno? | Sistema de recomendación basado en contenido |
| E2 *(extra)* | ¿El texto contradice la recomendación (sarcasmo o error)? | Detección de discrepancias |
| E3 *(extra)* | ¿La reseña será considerada graciosa? | Clasificación binaria |

## Alumnos

| Código | Nombres y apellidos |
|--------|---------------------|
| u202320608 | Wilber Franz Toledo Mamani |
| u202219719 | Luis Enrique Aguilar Benites |
| u202221068 | Juan Miguel Quijano Calderon |

## Descripción del dataset

- **Nombre:** *Steam Reviews Dataset for NLP & Sentiment Analysis*
- **Fuente:** Kaggle – [akashunikaggle/steam-game-reviews-of-743-games](https://www.kaggle.com/datasets/akashunikaggle/steam-game-reviews-of-743-games)
- **Licencia del dataset:** CC BY-SA 4.0
- **Tamaño:** 730 945 reseñas, 11 columnas, 735 juegos (archivo CSV de 285 MB)
- **Periodo:** reseñas publicadas entre noviembre de 2010 y septiembre de 2025
- **Tipo de dato:** semiestructurado (texto libre de la reseña + metadatos numéricos y booleanos)

| Columna | Descripción |
|---------|-------------|
| `review` | Texto original de la reseña |
| `voted_up` | `True` si el usuario recomienda el juego |
| `votes_up` / `votes_funny` | Votos de “útil” y de “gracioso” recibidos |
| `author_playtime_forever` | Minutos jugados por el autor |
| `word_count` | Número de palabras de la reseña |
| `timestamp_created` | Fecha de publicación (Unix) |
| `name`, `appid`, `price` | Juego, identificador y precio (en centavos de dólar) |

**Dataset final (preparado):** 693 605 reseñas en inglés, tras eliminar duplicados, copypastas, *ASCII art*, reseñas en otros idiomas y reseñas vacías, con columnas nuevas como `review_norm`, `review_clean`, `helpful`, `funny`, `horas_jugadas`, `precio_usd`, `antiguedad_dias` y `votos_por_mes`. Para la modelización se usó una **muestra estratificada de 50 000 reseñas** de 733 juegos, incluida en `data/`.

## Estructura del repositorio

```
CC219-TP-TF-2026-2-CC92/
├── data/
│   ├── README.md                            # descripcion de los datos
│   └── steam_reviews_clean_sample.zip       # muestra estratificada de 50 000 reseñas del dataset final
├── code/
│   ├── README.md                            # descripcion del codigo
│   ├── TP_CC219_Steam_EDA.ipynb             # 1) carga, limpieza, normalizacion, EDA y modelo base (TP)
│   ├── TF_CC219_Steam_Modelos.ipynb         # 2) modelos de P1, P2, P3, E1, E2 y E3 (TF)
│   ├── figuras/                             # graficos generados por el notebook del TF
│   ├── resultados/                          # tablas de metricas en CSV
│   └── app/
│       ├── app.py                           # aplicacion web en Streamlit
│       ├── preprocesamiento.py              # limpieza de texto compartida por el notebook y la app
│       ├── requirements.txt
│       └── modelos/                         # modelos entrenados (.joblib)
└── README.md
```

> **Nota sobre los datos:** GitHub no admite archivos de más de 100 MB. El dataset original (285 MB) y el dataset final completo (636 MB) se obtienen ejecutando el notebook del TP, que descarga el original desde Kaggle con `kagglehub`. El notebook del TF trabaja con la muestra de `data/`.

## Cómo ejecutar

**Notebooks (Google Colab)**

1. `code/TP_CC219_Steam_EDA.ipynb`: **Entorno de ejecución → Ejecutar todas** (unos 15 minutos). Descarga, limpia y explora el dataset.
2. `code/TF_CC219_Steam_Modelos.ipynb`: elegir **Entorno de ejecución → Cambiar tipo de entorno → GPU T4** y ejecutar todas las celdas (unos 15 minutos). Clona el repositorio si hace falta, entrena todos los modelos, ajusta DistilBERT (solo con GPU) y guarda figuras, tablas y modelos.

**Aplicación web (local)**

```bash
cd code/app
pip install -r requirements.txt
streamlit run app.py
```

La aplicación tiene tres pestañas: analizar una reseña nueva (P1, P2, P3 y alerta de incoherencia E2, con las palabras que explican la predicción), recomendar juegos parecidos (E1) y consultar las tablas de resultados.

**Principales librerías:** pandas, NumPy, Matplotlib, scikit-learn, LightGBM, gensim, NLTK, LIME, vaderSentiment, Hugging Face Transformers y Streamlit.

## Resultados

| Pregunta | Mejor modelo | Métrica | Valor | Azar |
|----------|--------------|---------|-------|------|
| P1 – Recomienda | DistilBERT (fine-tuning, 1 época) | F1 macro | 0.856 (AUC-ROC 0.957) | 0.500 |
| P2 – Útil | LightGBM con variables numéricas | AUC-PR | 0.427 | 0.184 |
| P3 – Tema | NMF (k = 8) + regresión logística | F1 macro | 0.879 | 0.125 |
| E1 – Recomendador | TF-IDF + similitud coseno | Hit@5 | 0.953 | 0.007 |
| E2 – Incoherencia | Regresión logística vs. VADER | AUC-ROC | 0.935 | 0.500 |
| E3 – Graciosa | Regresión logística | AUC-PR | 0.243 | 0.119 |

## Conclusiones

- **P1:** el texto predice muy bien si la reseña recomienda el juego. DistilBERT logró el mejor F1 macro (0.856) y AUC-ROC (0.957); entre los modelos clásicos, el SVM lineal sobre TF-IDF llegó a 0.831 entrenando en segundos. Conservar las negaciones fue clave (“not worth”, “not recommend”) y el sarcasmo sigue siendo el principal error, incluso para el transformer.
- **P2:** la utilidad depende más del contexto que de las palabras: si la reseña recomienda el juego (34 % de la importancia), su longitud (25 %) y su antigüedad (24 %). Todos los modelos duplican la referencia del azar, pero el texto solo es el más débil.
- **P3:** NMF superó a LDA en coherencia en todos los casos y encontró 8 temas interpretables. Los **problemas técnicos** (43.6 % de reseñas negativas) y **campaña y multijugador** (25.3 %) concentran la insatisfacción.
- **E1:** el recomendador ubica el juego correcto en el primer puesto en el 79.8 % de los casos y entre los cinco primeros en el 95.3 %, con recomendaciones explicables (Hades → Hades II, Bastion, Dead Cells).
- **E2:** el modelo entrenado marca solo 0.7 % de incoherencias frente al 10.1 % de VADER; 141 reseñas fueron marcadas por ambos.
- **E3:** el humor es lo más difícil de predecir (AUC-PR 0.243, el doble del azar).
- **Trabajo futuro:** entrenar con el corpus completo, ajustar transformers por más épocas y para P2 y P3 (por ejemplo, BERTopic) y validar las reseñas incoherentes con un etiquetado manual.

## Licencia

- **Código** (`code/`): uso académico para el curso CC219 de la UPC, bajo licencia [MIT](https://opensource.org/licenses/MIT).
- **Datos** (`data/`): derivados del dataset de Kaggle de akashunikaggle (2025), distribuidos bajo la misma licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), como exige la licencia original.
