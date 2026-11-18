"""
Comparación NLP: NLTK vs spaCy

Entrada:
data/results/movies_unificado.csv

Salida:
data/results/comparacion_spacy_nltk.csv

Una fila por reseña. Se mantienen TODAS las columnas de movies_unificado.csv
(en su mismo orden) y a la derecha se agregan las columnas generadas por
NLTK y spaCy.
"""

import csv
import re
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, cess_esp, conll2002
from nltk.stem import SnowballStemmer
from nltk.tag import (
    DefaultTagger,
    UnigramTagger,
    BigramTagger,
    TrigramTagger,
    ClassifierBasedTagger,
)
from nltk.classify import NaiveBayesClassifier

# =====================================================
# RUTAS
# =====================================================

RAIZ = Path(__file__).resolve().parent.parent

CSV_ENTRADA = RAIZ / "data" / "results" / "movies_unificado.csv"

CSV_SALIDA = (
    RAIZ
    / "data"
    / "results"
    / "comparacion_spacy_nltk.csv"
)

CSV_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

COLUMNA_TEXTO = "review_summary_es"

# =====================================================
# DESCARGA NLTK
# =====================================================

for recurso in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "cess_esp",
    "conll2002",   # corpus en español con entidades (PER, LOC, ORG, MISC) para el NER de NLTK
]:
    nltk.download(recurso, quiet=True)

# =====================================================
# MODELO SPACY
# =====================================================

SPACY_MODEL = "es_core_news_md"

nlp = spacy.load(SPACY_MODEL)

# =====================================================
# NLTK POS TAGGER
# =====================================================

MAPA_POS = {
    "n": "NOUN",
    "a": "ADJ",
    "v": "VERB",
    "r": "ADV",
    "d": "DET",
    "p": "PRON",
    "s": "ADP",
    "c": "CONJ",
    "i": "INTJ",
    "z": "NUM",
    "f": "PUNCT",
}


def simplificar_tag(tag):

    tag = tag.lower()

    if tag.startswith("np"):
        return "PROPN"

    return MAPA_POS.get(tag[0], "X")


print("Entrenando POS Tagger NLTK...")

sents = [
    [
        (w, simplificar_tag(t))
        for w, t in sent
        if t
    ]
    for sent in cess_esp.tagged_sents()
]

default = DefaultTagger("NOUN")
uni = UnigramTagger(sents, backoff=default)
bi = BigramTagger(sents, backoff=uni)
nltk_tagger = TrigramTagger(
    sents,
    backoff=bi
)

# =====================================================
# RECURSOS NLTK
# =====================================================

stop_es = set(
    stopwords.words("spanish")
)

stemmer = SnowballStemmer("spanish")

# =====================================================
# NER NLTK (entrenado en español con CoNLL-2002)
# =====================================================
# nltk.ne_chunk() solo funciona en inglés, por eso salía siempre [].
# Aquí se entrena un reconocedor de entidades en español con el corpus
# CoNLL-2002 (noticias en español etiquetadas con PER, LOC, ORG, MISC).
# Usa las mismas etiquetas que spaCy, así las dos columnas se pueden comparar.

print("Entrenando NER NLTK (CoNLL-2002 español)...")

_conll_train = conll2002.iob_sents("esp.train")

# Palabras que en el corpus aparecen normalmente en minúscula: si van en
# mayúscula al inicio de una oración, casi nunca son entidades.
_comunes_minuscula = Counter(
    w.lower()
    for sent in _conll_train
    for w, _, _ in sent
    if w.islower()
)

_INICIO_ORACION = {".", "!", "?", "'", '"', "(", "|", ":", "¡", "¿", "...", "-"}


def _forma(w):
    if re.fullmatch(r"\d+", w):
        return "num"
    if w.isupper():
        return "MAY"
    if w[:1].isupper():
        return "Cap"
    if w.islower():
        return "min"
    return "otro"


def _features_ner(tokens, i, history):
    w = tokens[i]
    prev = tokens[i - 1] if i > 0 else "<INI>"
    nxt = tokens[i + 1] if i < len(tokens) - 1 else "<FIN>"
    inicio = i == 0 or prev in _INICIO_ORACION
    comun = _comunes_minuscula[w.lower()] >= 3
    return {
        "w": w.lower(),
        "forma": _forma(w),
        "suf3": w[-3:].lower(),
        "pre3": w[:3].lower(),
        "prev": prev.lower(),
        "prev_forma": _forma(prev),
        "next": nxt.lower(),
        "next_forma": _forma(nxt),
        "prev_iob": history[i - 1] if i > 0 else "<INI>",
        "inicio": inicio,
        "comun_en_minuscula": comun,
        "inicio_y_comun": inicio and comun,
    }


nltk_ner = ClassifierBasedTagger(
    train=[[(w, iob) for w, _, iob in sent] for sent in _conll_train],
    feature_detector=_features_ner,
    classifier_builder=NaiveBayesClassifier.train,
)


def entidades_nltk(tokens):
    """Convierte las etiquetas IOB (B-PER, I-PER, O...) en una lista
    [(texto_entidad, tipo)], igual que doc.ents de spaCy."""
    entidades, actual, tipo = [], [], None

    def cerrar():
        if actual:
            # Se descartan:
            #  - entidades formadas solo por stopwords (ej. 'Fui', 'Esta')
            #  - entidades sin ninguna palabra en mayúscula (ej. 'cae', '1990s'):
            #    en español los nombres propios empiezan con mayúscula
            if (
                not all(t.lower() in stop_es for t in actual)
                and any(t[:1].isupper() for t in actual)
            ):
                entidades.append((" ".join(actual), tipo))

    for tok, iob in nltk_ner.tag(tokens):
        if iob.startswith("B-") or (iob.startswith("I-") and iob[2:] != tipo):
            cerrar()
            actual, tipo = [tok], iob[2:]
        elif iob.startswith("I-"):
            actual.append(tok)
        else:
            cerrar()
            actual, tipo = [], None
    cerrar()
    return entidades

# =====================================================
# FORMATO DE ENTIDADES (común a NLTK y spaCy)
# =====================================================

SIN_ENTIDADES = "sin entidades"

def columnas_entidades(entidades, sufijo):
    """
    Devuelve las columnas de NER listas para el CSV:
      entidades_<sufijo>       'Johnny Depp (PER); Rusia (LOC); ONU (ORG)'
      organizaciones_<sufijo>  'ONU'
    """
    # quita duplicados manteniendo el orden
    unicas = list(dict.fromkeys(entidades))

    organizaciones = [texto for texto, tipo in unicas if tipo == "ORG"]

    return {
        f"entidades_{sufijo}": (
            "; ".join(f"{texto} ({tipo})" for texto, tipo in unicas)
            if unicas else SIN_ENTIDADES
        ),
        f"organizaciones_{sufijo}": (
            "; ".join(organizaciones) if organizaciones else SIN_ENTIDADES
        ),
    }

# =====================================================
# NLTK
# =====================================================


def procesar_nltk(texto):

    tokens = word_tokenize(
        texto,
        language="spanish"
    )

    pos_tags = nltk_tagger.tag(tokens)

    tokens_validos = [
        (tok, pos)
        for tok, pos in pos_tags
        if tok.lower() not in stop_es
        and pos != "PUNCT"
        and any(c.isalnum() for c in tok)
    ]

    tokens_minusculas = [
        t.lower()
        for t, _ in tokens_validos
    ]

    lemas = [
        stemmer.stem(t)
        for t in tokens_minusculas
    ]

    texto_lematizado = " ".join(lemas)

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entidades": entidades_nltk(tokens),
        "lemas": lemas,
        "texto_lematizado": texto_lematizado,
    }

# =====================================================
# SPACY
# =====================================================


def procesar_spacy(texto):

    doc = nlp(texto)

    tokens = [
        t.text
        for t in doc
    ]

    pos_tags = [
        (t.text, t.pos_)
        for t in doc
    ]

    entidades = [
        (
            ent.text,
            ent.label_
        )
        for ent in doc.ents
    ]

    tokens_validos = [
        t
        for t in doc
        if not t.is_stop
        and not t.is_punct
    ]

    lemas = [
        t.lemma_.lower()
        for t in tokens_validos
    ]

    texto_lematizado = " ".join(lemas)

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entidades": entidades,
        "lemas": lemas,
        "texto_lematizado": texto_lematizado,
    }

# =====================================================
# CARGA CSV
# =====================================================

print("Leyendo dataset...")

df = pd.read_csv(
    CSV_ENTRADA,
    encoding="utf-8-sig",
    dtype=str,              # mantiene los valores tal como vienen (ej. rating "7.0")
    keep_default_na=False,  # no convierte textos vacíos en NaN
)

if COLUMNA_TEXTO not in df.columns:
    raise ValueError(
        f"No existe {COLUMNA_TEXTO}. Columnas: {list(df.columns)}"
    )

# =====================================================
# COMPARACIÓN
# =====================================================

nuevas_columnas = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Comparando"
):

    texto = str(
        row[COLUMNA_TEXTO]
    )

    nltk_result = procesar_nltk(texto)

    spacy_result = procesar_spacy(texto)

    # Solo las columnas NUEVAS: las originales ya están en df
    nuevas_columnas.append({

        # NLTK
        "tokens_nltk":
            str(nltk_result["tokens"]),

        "pos_tags_nltk":
            str(nltk_result["pos_tags"]),

        # entidades_nltk, organizaciones_nltk
        **columnas_entidades(nltk_result["entidades"], "nltk"),

        "lemas_nltk":
            str(nltk_result["lemas"]),

        "texto_lematizado_nltk":
            nltk_result[
                "texto_lematizado"
            ],

        # spaCy
        "tokens_spacy":
            str(spacy_result["tokens"]),

        "pos_tags_spacy":
            str(spacy_result["pos_tags"]),

        # entidades_spacy, organizaciones_spacy
        **columnas_entidades(spacy_result["entidades"], "spacy"),

        "lemas_spacy":
            str(spacy_result["lemas"]),

        "texto_lematizado_spacy":
            spacy_result[
                "texto_lematizado"
            ],

        # Métricas
        "cantidad_tokens_nltk":
            len(
                nltk_result["tokens"]
            ),

        "cantidad_tokens_spacy":
            len(
                spacy_result["tokens"]
            ),

        "coincidencia_tokens":
            nltk_result["tokens"]
            ==
            spacy_result["tokens"],

        # Entidades que encontraron las DOS librerías (mismo texto y tipo)
        "entidades_en_comun":
            "; ".join(
                f"{t} ({tipo})"
                for t, tipo in dict.fromkeys(nltk_result["entidades"])
                if (t, tipo) in set(spacy_result["entidades"])
            ) or SIN_ENTIDADES,
    })

# =====================================================
# UNIR: columnas originales + columnas nuevas a la derecha
# =====================================================

nuevas_df = pd.DataFrame(
    nuevas_columnas,
    index=df.index
)

# Si alguna columna nueva ya existía en el CSV (por una ejecución anterior),
# se reemplaza por la recién calculada en lugar de duplicarse.
df = df.drop(
    columns=[c for c in nuevas_df.columns if c in df.columns]
)

resultado_df = pd.concat(
    [df, nuevas_df],
    axis=1
)

# =====================================================
# EXPORTAR
# =====================================================

resultado_df.to_csv(
    CSV_SALIDA,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL,  # evita que comas o comillas rompan las columnas
    lineterminator="\n",
)

print("\nArchivo generado:")
print(CSV_SALIDA)

print(
    f"Registros: {len(resultado_df):,}"
)

print(
    f"Columnas ({len(resultado_df.columns)}): {list(resultado_df.columns)}"
)