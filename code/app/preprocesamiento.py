"""
preprocesamiento.py
Funciones de normalizacion de texto usadas en el proyecto (las mismas del notebook de EDA,
seccion 5). La app las reutiliza para limpiar una reseña nueva igual que el dataset de entrenamiento.
"""

import re
import html

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ---------------------------------------------------------------------------
# Lematizador (WordNet). Si no esta disponible, se usa la palabra sin cambios.
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lematizador = WordNetLemmatizer()
    _lematizador.lemmatize('games')   # prueba: falla aqui si WordNet no se descargo
except Exception:                      # sin internet o sin nltk
    _lematizador = None


def lematizar(palabra):
    """Palabras terminadas en "ed" se tratan como verbo y el resto como sustantivo."""
    if _lematizador is None:
        return palabra
    if palabra.endswith('ed'):
        return _lematizador.lemmatize(palabra, pos='v')
    return _lematizador.lemmatize(palabra, pos='n')


# ---------------------------------------------------------------------------
# Listas de palabras
# ---------------------------------------------------------------------------
NEGACIONES = {'no', 'not', 'nor', 'never', 'nothing', 'nobody', 'none', 'cannot'}

EXTRA_STOP = {
    'game', 'games', 'play', 'played', 'playing', 'just',
    'does', 'did', 'doing', 've', 'll', 're', 'im',
    'wa', 'ha', 'doe',
}

STOPWORDS = (set(ENGLISH_STOP_WORDS) | EXTRA_STOP) - NEGACIONES


# ---------------------------------------------------------------------------
# Expresiones regulares
# ---------------------------------------------------------------------------
RE_BBCODE = re.compile(r'\[/?[^\]]{1,20}\]')
RE_URL    = re.compile(r'https?://\S+|www\.\S+')
RE_CANT   = re.compile(r"\bcan['’]?t\b")
RE_WONT   = re.compile(r"\bwon['’]t\b|\bwont\b")
RE_AINT   = re.compile(r"\bain['’]?t\b")
RE_NT     = re.compile(r"n['’]t\b")
RE_NT_SIN = re.compile(r"\b(do|does|did|is|was|are|were|could|would|should|have|has|had)nt\b")
RE_REPEAT = re.compile(r'(\w)\1{2,}')
RE_NOALFA = re.compile(r'[^a-z\s]')
RE_ESP    = re.compile(r'\s+')


def normalizar(texto):
    """Pasos 1 a 5: texto natural en minusculas, sin etiquetas, URLs ni simbolos."""
    t = html.unescape(str(texto))
    t = RE_BBCODE.sub(' ', t)
    t = RE_URL.sub(' ', t)
    t = t.lower()

    t = RE_CANT.sub('can not', t)
    t = RE_WONT.sub('will not', t)
    t = RE_AINT.sub('is not', t)
    t = RE_NT.sub(' not', t)
    t = RE_NT_SIN.sub(r'\1 not', t)

    t = RE_REPEAT.sub(r'\1\1', t)
    t = RE_NOALFA.sub(' ', t)
    t = RE_ESP.sub(' ', t)
    return t.strip()


def limpiar(texto_norm):
    """Pasos 6 y 7: sin stopwords (conservando negaciones) y lematizado."""
    palabras = [p for p in texto_norm.split() if len(p) > 1 and p not in STOPWORDS]
    lemas = [lematizar(p) for p in palabras]
    lemas = [l for l in lemas if len(l) > 1 and l not in STOPWORDS]
    return ' '.join(lemas)


def preparar(texto):
    """Devuelve (review_norm, review_clean) de un texto original."""
    norm = normalizar(texto)
    return norm, limpiar(norm)
