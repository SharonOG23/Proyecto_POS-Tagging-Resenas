"""
Comparación NLP: NLTK vs spaCy

Salida:
data/results/comparacion_spacy_nltk.csv

Una fila por reseña.
"""

from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, cess_esp
from nltk.stem import SnowballStemmer
from nltk.tag import (
    DefaultTagger,
    UnigramTagger,
    BigramTagger,
    TrigramTagger,
)

# =====================================================
# RUTAS
# =====================================================

RAIZ = Path(__file__).resolve().parent.parent

CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv"

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

# =====================================================
# DESCARGA NLTK
# =====================================================

for recurso in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "cess_esp",
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

    tokens_sin_stopwords = [
        t
        for t, _ in tokens_validos
    ]

    pos_sin_stopwords = tokens_validos

    tokens_minusculas = [
        t.lower()
        for t in tokens_sin_stopwords
    ]

    lemas = [
        stemmer.stem(t)
        for t in tokens_minusculas
    ]

    texto_lematizado = " ".join(lemas)

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entidades": [],
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
    encoding="utf-8-sig"
)

if "review_summary_es" not in df.columns:
    raise ValueError(
        "No existe review_summary_es"
    )

if "id_resena" not in df.columns:
    df["id_resena"] = range(
        1,
        len(df) + 1
    )

# =====================================================
# COMPARACIÓN
# =====================================================

resultados = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Comparando"
):

    texto = str(
        row["review_summary_es"]
    )

    nltk_result = procesar_nltk(texto)

    spacy_result = procesar_spacy(texto)

    resultados.append({

        # originales
        "id_resena":
            row.get("id_resena", ""),

        "movie_id":
            row.get("movie_id", ""),

        "primaryTitle":
            row.get("primaryTitle", ""),

        "review_summary_es":
            texto,

        # NLTK
        "tokens_nltk":
            str(nltk_result["tokens"]),

        "pos_tags_nltk":
            str(nltk_result["pos_tags"]),

        "entidades_nltk":
            str(nltk_result["entidades"]),

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

        "entidades_spacy":
            str(spacy_result["entidades"]),

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
            spacy_result["tokens"]
    })

# =====================================================
# EXPORTAR
# =====================================================

resultado_df = pd.DataFrame(
    resultados
)

resultado_df.to_csv(
    CSV_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado:")
print(CSV_SALIDA)

print(
    f"Registros: {len(resultado_df):,}"
)