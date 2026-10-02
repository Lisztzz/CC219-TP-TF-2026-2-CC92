# CC219-TP-TF-2026-2-CC92

## Minería de textos sobre reseñas de videojuegos en Steam

Trabajo Parcial y Final del curso **1ACC0219 – Aplicaciones de Data Science** (sección 4896), Universidad Peruana de Ciencias Aplicadas (UPC), ciclo 2026-2.

---

## Objetivo del trabajo

Aplicar técnicas de **minería de textos y procesamiento de lenguaje natural (NLP)** sobre más de 700 mil reseñas de jugadores de Steam para construir modelos que, a partir del texto de una reseña, permitan:

| # | Pregunta | Tipo de tarea |
|---|----------|---------------|
| P1 | ¿La reseña recomienda el juego o no? | Clasificación binaria |
| P2 | ¿La reseña será valorada como útil por otros usuarios? | Clasificación binaria |
| P3 | ¿Qué tema aborda la queja o el elogio (rendimiento, precio, bugs, jugabilidad, historia)? | Modelado de tópicos y clasificación multiclase |
| E1 *(extra)* | ¿Qué otros juegos recomendar a un usuario según lo que dicen las reseñas? | Sistema de recomendación basado en contenido |
| E2 *(extra)* | ¿La reseña es incoherente (el texto contradice la recomendación por sarcasmo o error)? | Detección de discrepancias |
| E3 *(extra)* | ¿La reseña será considerada graciosa por la comunidad? | Clasificación binaria |

## Alumnos

| Código | Nombres y apellidos |
|--------|---------------------|
| U202219719 | Luis Enrique Aguilar Benites |

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

**Dataset final (preparado):** 693 605 reseñas en inglés, tras eliminar duplicados, copypastas, *ASCII art*, reseñas en otros idiomas y reseñas vacías, con columnas nuevas como `review_norm`, `review_clean`, `helpful`, `funny`, `horas_jugadas`, `precio_usd`, `antiguedad_dias` y `votos_por_mes`.

## Estructura del repositorio

```
CC219-TP-TF-2026-2-CC92/
├── data/
│   ├── steam_game_reviews_730945.csv        # dataset original (ver nota)
│   └── steam_reviews_clean_sample.csv       # muestra estratificada de 50 000 reseñas del dataset final
├── code/
│   └── TP_CC219_Steam_EDA.ipynb             # carga, limpieza, normalización, EDA y modelo base
└── README.md
```

> **Nota sobre los datos:** GitHub no admite archivos de más de 100 MB. El dataset original (285 MB) y el dataset final completo (636 MB) se obtienen ejecutando el notebook, que descarga el original directamente desde Kaggle con `kagglehub`. En la carpeta `data/` se incluye una muestra estratificada del dataset final.

## Cómo ejecutar

1. Abrir `code/TP_CC219_Steam_EDA.ipynb` en Google Colab.
2. Ejecutar **Entorno de ejecución → Ejecutar todas** (tarda alrededor de 15 minutos).
3. El notebook descarga el dataset, lo limpia, genera los gráficos y guarda los archivos en `data/`.

**Principales librerías:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, NLTK, py3langid, WordCloud, vaderSentiment y kagglehub.

## Conclusiones

- Tras la limpieza se obtuvo un corpus de **693 605 reseñas en inglés** (94.9 % del original), con un desbalance de **81.4 % positivas y 18.6 % negativas**.
- Las reseñas negativas son **más largas** (mediana de 36 palabras frente a 20), provienen de jugadores con **menos horas de juego** (12.0 h frente a 28.2 h) y reciben **más votos de “útil”**.
- A mayor precio, menor satisfacción: los juegos de hasta USD 20 tienen 84 % de reseñas positivas y los de más de USD 60 solo 62 %.
- Las **copypastas** reciben 5.7 veces más votos de “útil” que una reseña única y el *ASCII art* 4.8 veces más votos de “gracioso”, lo que evidencia el *farming* de votos y premios en Steam y justificó eliminarlos.
- Los votos de “útil” acumulados favorecen a las reseñas antiguas, pero los **votos por mes** caen de 1.11 a 0.03 con la antigüedad: las reseñas reciben casi todos sus votos al inicio.
- Según VADER, alrededor del **10 %** de las reseñas tiene un texto que contradice su recomendación (sarcasmo, humor o reseñas mixtas).
- Un modelo base de **TF-IDF + regresión logística** alcanza un **AUC-ROC de 0.941** y un **F1 macro de 0.826** en la predicción de la recomendación (P1).
- **Trabajo futuro:** entrenar DistilBERT, BERTopic y el sistema de recomendación, controlar el sesgo temporal en P2 y desarrollar una interfaz en Streamlit.
- **Datos** (`data/`): derivados del dataset de Kaggle de akashunikaggle (2025), distribuidos bajo la misma licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), como exige la licencia original.
