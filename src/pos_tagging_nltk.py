"""
Pipeline NLP en español con NLTK sobre la columna review_summary_es.

Orden del pipeline:
1. Tokenización
2. Etiquetado POS
3. Eliminación de stopwords
4. Reconocimiento de entidades (NER)
5. Normalización de mayúsculas/minúsculas
6. Lematización
"""

from pathlib import Path
import pickle

import nltk
import pandas as pd

from nltk.corpus import cess_esp, conll2002, stopwords
from nltk.stem import SnowballStemmer

from nltk.tag import (
    BigramTagger,
    ClassifierBasedTagger,
    DefaultTagger,
    TrigramTagger,
    UnigramTagger,
)

from nltk.tokenize import word_tokenize


# =====================================================
# RUTAS DEL PROYECTO
# =====================================================

# Proyecto_POS-Tagging-Resenas
RAIZ = Path(__file__).resolve().parent.parent

# Entrada
CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv"

# Salida
CSV_SALIDA = RAIZ / "data" / "results" / "pos_analisis_nltk.csv"

# Cache de modelos entrenados
CACHE_DIR = RAIZ / "cache_nltk"


# =====================================================
# SIMPLEMMA (opcional)
# =====================================================

try:
    import simplemma

    HAY_SIMPLEMMA = True

except ImportError:
    HAY_SIMPLEMMA = False


class NLTK_POS_Reviews:
    """
    Pipeline de preprocesamiento lingüístico para texto en español.
    """

    _MAPA_POS = {
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
        "w": "DATE",
        "f": "PUNCT",
    }

    def __init__(
        self,
        ruta_csv=CSV_ENTRADA,
        columna="review_summary_es",
        dir_cache=CACHE_DIR,
        max_sents_ner=None,
    ):

        self.ruta_csv = Path(ruta_csv)
        self.columna = columna
        self.dir_cache = Path(dir_cache)
        self.max_sents_ner = max_sents_ner
        self.df = None

        self.dir_cache.mkdir(parents=True, exist_ok=True)

        self._descargar_recursos()

        self.stop_es = set(stopwords.words("spanish"))

        self.stemmer = SnowballStemmer("spanish")

        self.tagger_pos = self._cargar_o_entrenar(
            "pos.pkl",
            self._entrenar_pos,
        )

        self.tagger_ner = self._cargar_o_entrenar(
            "ner.pkl",
            self._entrenar_ner,
        )

    # =====================================================
    # CONFIGURACIÓN
    # =====================================================

    @staticmethod
    def _descargar_recursos():

        recursos = [
            "punkt",
            "punkt_tab",
            "stopwords",
            "cess_esp",
            "conll2002",
        ]

        for recurso in recursos:
            nltk.download(recurso, quiet=True)

    def _cargar_o_entrenar(
        self,
        nombre,
        funcion_entrenamiento,
    ):

        ruta_modelo = self.dir_cache / nombre

        if ruta_modelo.exists():
            with open(ruta_modelo, "rb") as archivo:
                return pickle.load(archivo)

        modelo = funcion_entrenamiento()

        with open(ruta_modelo, "wb") as archivo:
            pickle.dump(modelo, archivo)

        return modelo

    # =====================================================
    # POS
    # =====================================================

    def _simplificar_tag(self, tag):

        t = tag.lower()

        if t.startswith("np"):
            return "PROPN"

        if t.startswith("va") or t.startswith("vs"):
            return "AUX"

        return self._MAPA_POS.get(t[:1], "X")

    def _entrenar_pos(self):

        sents = [
            [(w, self._simplificar_tag(t)) for w, t in sent if t]
            for sent in cess_esp.tagged_sents()
        ]

        sents = [s for s in sents if s]

        default = DefaultTagger("NOUN")
        unigram = UnigramTagger(sents, backoff=default)
        bigram = BigramTagger(sents, backoff=unigram)
        trigram = TrigramTagger(sents, backoff=bigram)

        return trigram

    # =====================================================
    # NER
    # =====================================================

    @staticmethod
    def _features_ner(tokens, index, history):

        palabra = tokens[index]

        prev = tokens[index - 1] if index > 0 else "<INI>"
        sig = tokens[index + 1] if index < len(tokens) - 1 else "<FIN>"

        return {
            "palabra": palabra.lower(),
            "es_mayus_inicial": palabra[:1].isupper(),
            "es_todo_mayus": palabra.isupper(),
            "es_digito": palabra.isdigit(),
            "prefijo3": palabra[:3].lower(),
            "sufijo3": palabra[-3:].lower(),
            "prev": prev.lower(),
            "prev_mayus": prev[:1].isupper(),
            "sig": sig.lower(),
            "sig_mayus": sig[:1].isupper(),
            "inicio": index == 0,
            "prev_tag": history[index - 1] if index > 0 else "<INI>",
        }

    def _entrenar_ner(self):

        sents = [
            [(w, iob) for (w, _pos, iob) in sent]
            for sent in conll2002.iob_sents("esp.train")
        ]

        if self.max_sents_ner:
            sents = sents[: self.max_sents_ner]

        return ClassifierBasedTagger(
            train=sents,
            feature_detector=self._features_ner,
        )

    # =====================================================
    # CARGA DE DATOS
    # =====================================================

    def cargar_datos(self):

        self.df = pd.read_csv(self.ruta_csv)

        if self.columna not in self.df.columns:
            raise KeyError(
                f"No existe la columna '{self.columna}' en el archivo CSV."
            )

        self.df[self.columna] = (
            self.df[self.columna]
            .fillna("")
            .astype(str)
        )

        return self.df

    # =====================================================
    # PIPELINE
    # =====================================================

    def tokenizar(self, texto):
        return word_tokenize(texto, language="spanish")

    def etiquetar_pos(self, tokens):
        return self.tagger_pos.tag(tokens)

    def quitar_stopwords(self, tokens_pos):

        return [
            (token, pos)
            for token, pos in tokens_pos
            if token.lower() not in self.stop_es
            and pos != "PUNCT"
            and any(c.isalnum() for c in token)
        ]

    def reconocer_entidades(self, tokens):

        if not tokens:
            return []

        etiquetas = self.tagger_ner.tag(tokens)

        entidades = []
        actual = []
        tipo = None

        for token, iob in etiquetas:

            if iob and iob.startswith("B-"):

                if actual:
                    entidades.append(
                        (" ".join(actual), tipo)
                    )

                actual = [token]
                tipo = iob[2:]

            elif iob and iob.startswith("I-") and actual:

                actual.append(token)

            else:

                if actual:
                    entidades.append(
                        (" ".join(actual), tipo)
                    )

                actual = []
                tipo = None

        if actual:
            entidades.append(
                (" ".join(actual), tipo)
            )

        return entidades

    @staticmethod
    def normalizar_caso(tokens):
        return [token.lower() for token in tokens]

    def lematizar(self, tokens):

        if HAY_SIMPLEMMA:
            return [
                simplemma.lemmatize(token, lang="es")
                for token in tokens
            ]

        return [
            self.stemmer.stem(token)
            for token in tokens
        ]

    # =====================================================
    # PROCESAMIENTO DE TEXTO
    # =====================================================

    def procesar_texto(self, texto):

        tokens = self.tokenizar(texto)

        tokens_pos = self.etiquetar_pos(tokens)

        sin_stop = self.quitar_stopwords(tokens_pos)

        tokens_filtrados = [
            token
            for token, _ in sin_stop
        ]

        entidades = self.reconocer_entidades(
            tokens_filtrados
        )

        minusculas = self.normalizar_caso(
            tokens_filtrados
        )

        lemas = self.lematizar(
            minusculas
        )

        return {
            "tokens": tokens,
            "pos_tags": tokens_pos,
            "tokens_sin_stopwords": tokens_filtrados,
            "pos_sin_stopwords": sin_stop,
            "entidades": entidades,
            "tokens_minusculas": minusculas,
            "lemas": lemas,
        }

    # =====================================================
    # EJECUCIÓN
    # =====================================================

    def ejecutar(
        self,
        ruta_salida=CSV_SALIDA,
    ):

        ruta_salida = Path(ruta_salida)

        ruta_salida.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.df is None:
            self.cargar_datos()

        resultados = self.df[self.columna].apply(
            self.procesar_texto
        )

        procesado = pd.DataFrame(
            resultados.tolist(),
            index=self.df.index,
        )

        self.df = pd.concat(
            [self.df, procesado],
            axis=1,
        )

        self.df["texto_lematizado"] = (
            self.df["lemas"].apply(" ".join)
        )

        self.df.to_csv(
            ruta_salida,
            index=False,
            encoding="utf-8-sig",
        )

        return self.df


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    pipeline = NLTK_POS_Reviews(
        ruta_csv=CSV_ENTRADA,
        columna="review_summary_es",
    )

    df = pipeline.ejecutar(
        ruta_salida=CSV_SALIDA
    )

    print(
        "\nArchivo procesado correctamente:"
    )

    print(CSV_SALIDA)

    print(
        df[
            [
                "review_summary_es",
                "pos_tags",
                "entidades",
                "lemas",
            ]
        ].head()
    )