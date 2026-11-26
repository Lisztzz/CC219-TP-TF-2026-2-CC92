"""
app.py - Interfaz para stakeholders del proyecto "Mineria de textos sobre reseñas de Steam"
Curso CC219 Aplicaciones de Data Science (UPC)

Ejecutar desde la carpeta code/app:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocesamiento import preparar


# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Analizador de reseñas de Steam', page_icon='🎮', layout='wide')

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')


# ---------------------------------------------------------------------------
# Carga de modelos (una sola vez gracias a la cache de Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    ruta = lambda nombre: os.path.join(DIR_MODELOS, nombre)
    return {
        'p1': joblib.load(ruta('p1_regresion_logistica.joblib')),
        'p2': joblib.load(ruta('p2_lightgbm.joblib')),
        'p3': joblib.load(ruta('p3_temas.joblib')),
        'e1': joblib.load(ruta('e1_recomendador.joblib')),
        'resultados': joblib.load(ruta('resultados.joblib')),
        'vader': SentimentIntensityAnalyzer(),
    }


try:
    M = cargar_modelos()
except FileNotFoundError:
    st.error('No se encontraron los modelos en code/app/modelos. Ejecute primero el notebook TF_CC219_Steam_Modelos.ipynb.')
    st.stop()


# ---------------------------------------------------------------------------
# Funciones de analisis
# ---------------------------------------------------------------------------
def explicar_p1(texto_limpio, n=6):
    """Contribucion de cada termino a la prediccion: valor TF-IDF x coeficiente de la Regresion Logistica."""
    vectorizador = M['p1'].named_steps['tfidfvectorizer']
    regresion = M['p1'].named_steps['logisticregression']
    x = vectorizador.transform([texto_limpio])
    if x.nnz == 0:
        return pd.Series(dtype=float)
    terminos = vectorizador.get_feature_names_out()[x.indices]
    aportes = pd.Series(x.data * regresion.coef_[0][x.indices], index=terminos)
    return aportes.reindex(aportes.abs().sort_values(ascending=False).index[:n])


def analizar(texto, horas, precio):
    """Aplica los modelos de P1, P2, P3 y E2 a una reseña."""
    norm, limpio = preparar(texto)

    # P1: probabilidad de que la reseña recomiende el juego
    prob_recomienda = M['p1'].predict_proba([limpio])[0, 1]
    recomienda = int(prob_recomienda >= 0.5)

    # P2: probabilidad de que reciba 2 o mas votos de "util" (reseña nueva: sin antiguedad)
    datos_p2 = pd.DataFrame([{
        'log_palabras': np.log1p(len(texto.split())),
        'log_horas': np.log1p(horas),
        'recomienda': recomienda,
        'precio_usd': precio,
    }])[M['p2']['columnas']]
    prob_util = M['p2']['modelo'].predict_proba(datos_p2)[0, 1]

    # P3: tema de la reseña
    tema = int(M['p3']['clasificador'].predict([limpio])[0])
    prob_tema = M['p3']['clasificador'].predict_proba([limpio])[0].max()

    # E2: coherencia entre el texto (VADER) y la prediccion del modelo
    vader = M['vader'].polarity_scores(norm)['compound']

    return {
        'limpio': limpio, 'prob_recomienda': prob_recomienda, 'recomienda': recomienda,
        'prob_util': prob_util, 'tema': tema, 'prob_tema': prob_tema, 'vader': vader,
    }


def recomendar(juego, n=5, n_terminos=4):
    """Juegos mas parecidos segun sus reseñas positivas, con los terminos que comparten."""
    e1 = M['e1']
    i = e1['nombres'].index(juego)
    orden = [j for j in np.argsort(e1['similitud'][i])[::-1] if j != i][:n]
    filas = []
    for j in orden:
        compartidos = e1['matriz'][i].multiply(e1['matriz'][j]).toarray().ravel()
        filas.append({
            'Juego recomendado': e1['nombres'][j],
            'Similitud': round(float(e1['similitud'][i, j]), 3),
            'Porque ambos mencionan': ', '.join(e1['terminos'][np.argsort(compartidos)[::-1][:n_terminos]]),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title('🎮 Analizador de reseñas de Steam')
st.caption('Minería de textos sobre 50 000 reseñas de 733 juegos · CC219 Aplicaciones de Data Science · UPC')

with st.sidebar:
    st.header('Sobre la herramienta')
    st.write('Analiza reseñas de videojuegos escritas en inglés con modelos entrenados con reseñas reales de Steam.')
    st.markdown(
        '- **P1:** ¿recomienda el juego?\n'
        '- **P2:** ¿será votada como útil?\n'
        '- **P3:** ¿de qué tema habla?\n'
        '- **E1:** juegos parecidos\n'
        '- **E2:** ¿el texto contradice la recomendación?'
    )
    st.write('Integrantes: Wilber Franz Toledo Mamani · Luis Enrique Aguilar Benites · Juan Miguel Quijano Calderon')

tab_resena, tab_reco, tab_resultados = st.tabs(['📝 Analizar una reseña', '🎯 Recomendar juegos', '📊 Resultados de los modelos'])


# ---- Pestaña 1: analizar una reseña -------------------------------------
with tab_resena:
    ejemplos = {
        'Escribir mi propia reseña': '',
        'Positiva': 'Amazing story and great soundtrack, one of the best games I have ever played. Highly recommend it.',
        'Negativa (problemas técnicos)': 'Not worth the price. The game crashes every ten minutes and the devs never fix anything. I want a refund.',
        'Con negación': 'It is not bad at all, actually pretty fun with friends. Coop mode is great.',
    }
    eleccion = st.selectbox('Cargar un ejemplo', list(ejemplos))
    texto = st.text_area('Reseña (en inglés)', value=ejemplos[eleccion], height=140,
                         placeholder='Write a review of a game...')

    col_h, col_p = st.columns(2)
    horas = col_h.number_input('Horas jugadas por el autor', min_value=0.0, max_value=10000.0, value=20.0, step=1.0)
    precio = col_p.number_input('Precio del juego (USD)', min_value=0.0, max_value=80.0, value=19.99, step=1.0)

    if st.button('Analizar', type='primary', disabled=not texto.strip()):
        r = analizar(texto, horas, precio)

        if not r['limpio']:
            st.warning('La reseña no tiene palabras útiles después de la limpieza (¿está en inglés?).')
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric('P1 · ¿Recomienda el juego?', 'Sí 👍' if r['recomienda'] else 'No 👎',
                      f"{r['prob_recomienda']:.0%} de probabilidad de recomendar", delta_color='off')
            c2.metric('P2 · ¿Será útil para la comunidad?', f"{r['prob_util']:.0%}",
                      'Probablemente útil' if r['prob_util'] >= 0.5 else 'Poco probable', delta_color='off')
            c3.metric('P3 · Tema principal', M['p3']['nombres'][r['tema']],
                      f"{r['prob_tema']:.0%} de confianza", delta_color='off')

            # E2: coherencia entre el lexico (VADER) y el modelo entrenado
            texto_negativo, texto_positivo = r['vader'] < -0.5, r['vader'] > 0.5
            if (texto_negativo and r['recomienda']) or (texto_positivo and not r['recomienda']):
                st.warning(f"E2 · Posible incoherencia: el léxico (VADER = {r['vader']:+.2f}) y el modelo no coinciden. "
                           'Puede tratarse de sarcasmo, humor o una reseña mixta.')
            else:
                st.success(f"E2 · Coherente: el léxico (VADER = {r['vader']:+.2f}) y el modelo coinciden.")

            # Explicacion de P1: palabras que mas pesaron
            st.subheader('¿Por qué? Palabras que más pesaron en la predicción')
            aportes = explicar_p1(r['limpio'])
            if aportes.empty:
                st.info('Ninguna palabra de la reseña está en el vocabulario del modelo.')
            else:
                tabla = pd.DataFrame({
                    'Término': aportes.index,
                    'Aporte': aportes.values.round(3),
                    'Empuja hacia': np.where(aportes.values > 0, 'Recomienda', 'No recomienda'),
                })
                st.dataframe(tabla, hide_index=True, width='stretch')

            st.caption(f"Texto limpio usado por los modelos: {r['limpio']}")


# ---- Pestaña 2: recomendador --------------------------------------------
with tab_reco:
    st.write('Elige un juego que te gustó y el sistema busca los juegos cuyas reseñas positivas hablan de lo mismo.')
    juegos = M['e1']['nombres']
    inicial = juegos.index('Stardew Valley') if 'Stardew Valley' in juegos else 0
    juego = st.selectbox('Juego que te gustó', juegos, index=inicial)
    n = st.slider('Cantidad de recomendaciones', 3, 10, 5)
    st.dataframe(recomendar(juego, n), hide_index=True, width='stretch')


# ---- Pestaña 3: resultados ------------------------------------------------
with tab_resultados:
    resultados = M['resultados']
    st.subheader('Resumen por pregunta')
    st.dataframe(resultados['resumen'], hide_index=True, width='stretch')

    nombres_tablas = {'P1': 'P1 · Recomendación', 'P2': 'P2 · Utilidad', 'E3': 'E3 · Reseñas graciosas',
                      'E1': 'E1 · Evaluación del recomendador', 'E2': 'E2 · Jueces de coherencia'}
    for clave, titulo in nombres_tablas.items():
        st.subheader(titulo)
        st.dataframe(resultados['tablas'][clave].round(3), width='stretch')
