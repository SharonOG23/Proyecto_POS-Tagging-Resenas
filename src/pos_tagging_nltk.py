
Pipeline NLP en español con NLTK sobre la columna `review_summary_es`.

Orden del pipeline:
    1. Tokenización
    2. Etiquetado POS
    3. Borrado de stopwords
    4. NER
    5. Mayúsculas / minúsculas
    6. Lematización
"""

import os #Se utiliza para trabajar con archivos, carpetas, ruta del sistema operativo.
import pickle #Es un módulo nativo de Python que permite serializar y deserializar objetos.

import nltk #Permite analizar, procesar y comprender texto escrito por humanos.
import pandas as pd
from nltk.corpus import cess_esp, conll2002, stopwords
#nltk.corpus:proporciona acceso a una gran colección de corpus lingüísticos y conjuntos de datos para(NLP).
#cess_esp:Entrenamiento y evaluación de modelos NLP en español.
#conll2002:Esta enfocada en Reconocimiento de Entidades Nombradas (NER) para español
#stopwords:Contiene listas de palabras muy frecuentes (palabras vacías)
from nltk.stem import SnowballStemmer
#nltk.stem:utiliza para realizar stemming y lemmatización, técnicas empleadas para reducir las palabras a su forma base o raíz.
#SnowballStemmer:El stemming consiste en reducir las palabras a una raíz común para facilitar el análisis de textos
from nltk.tag import (
    BigramTagger,#Tiene en cuenta la palabra anterior y su etiqueta.
    ClassifierBasedTagger,#Utiliza aprendizaje automático en lugar de reglas simples de frecuencia
    DefaultTagger,#Asigna la misma etiqueta a todas las palabras.
    TrigramTagger, #Utiliza las dos etiquetas anteriores para decidir
    UnigramTagger,#Etiqueta una palabra basándose únicamente en esa palabra.
)
from nltk.tokenize import word_tokenize
#nltk.tokenize:Divide un texto en unidades más pequeñas llamadas tokens.
#word_tokenize:Divide un texto en tokens de palabras y signos de puntuación.

try:
    import simplemma
    _HAY_SIMPLEMMA = True
except ImportError:
    _HAY_SIMPLEMMA = False


class NLTK_POS_Reviews:
    """Pipeline de preprocesamiento lingüístico para texto en español."""

    """
    Etiquetas POS (CESS-ESP / EAGLES)

    ncms000 : Sustantivo común masculino singular
    ncfp000 : Sustantivo común femenino plural
    np00000 : Nombre propio
    vmip3s0 : Verbo principal, indicativo presente, 3ª persona singular
    aq0cs0  : Adjetivo calificativo
    rg      : Adverbio
    sp      : Preposición
    da0ms0  : Artículo determinado masculino singular
    """
# Mapeo de tags EAGLES (cess_esp) a etiquetas legibles
    _MAPA_POS = {
        "n": "NOUN", "a": "ADJ", "v": "VERB", "r": "ADV", "d": "DET",
        "p": "PRON", "s": "ADP", "c": "CONJ", "i": "INTJ", "z": "NUM",
        "w": "DATE", "f": "PUNCT",
    }

    def __init__(self, ruta_csv, columna="review_summary_es",
                 dir_cache="cache_nltk", max_sents_ner=None):
        self.ruta_csv = ruta_csv
        self.columna = columna
        self.dir_cache = dir_cache
        self.max_sents_ner = max_sents_ner
        self.df = None

        os.makedirs(self.dir_cache, exist_ok=True)
        self._descargar_recursos()

        self.stop_es = set(stopwords.words("spanish"))
        self.stemmer = SnowballStemmer("spanish")  # respaldo si no hay simplemma
        self.tagger_pos = self._cargar_o_entrenar("pos.pkl", self._entrenar_pos)
        self.tagger_ner = self._cargar_o_entrenar("ner.pkl", self._entrenar_ner)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _descargar_recursos():
        for recurso in ["punkt", "punkt_tab", "stopwords", "cess_esp", "conll2002"]:
            nltk.download(recurso, quiet=True)

    def _cargar_o_entrenar(self, nombre, funcion_entrenamiento):
        ruta = os.path.join(self.dir_cache, nombre)
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return pickle.load(f)
        modelo = funcion_entrenamiento()
        with open(ruta, "wb") as f:
            pickle.dump(modelo, f)
        return modelo

    def _simplificar_tag(self, tag):
        """'np00000' -> 'PROPN', 'nc0s000' -> 'NOUN', 'vmip3s0' -> 'VERB', etc."""
        t = tag.lower()
        if t.startswith("np"):
            return "PROPN"
        if t.startswith("va") or t.startswith("vs"):
            return "AUX"
        return self._MAPA_POS.get(t[:1], "X")

    # ---------------------------------------------------------- entrenamientos
    def _entrenar_pos(self):
        """Tagger POS en español entrenado con el corpus cess_esp (backoff)."""
        sents = [
            [(w, self._simplificar_tag(t)) for (w, t) in s if t]
            for s in cess_esp.tagged_sents()
        ]
        sents = [s for s in sents if s]
        base = DefaultTagger("NOUN")
        uni = UnigramTagger(sents, backoff=base)
        bi = BigramTagger(sents, backoff=uni)
        return TrigramTagger(sents, backoff=bi)

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
        """Tagger NER (IOB) en español entrenado con conll2002 (esp)."""
        sents = [
            [(w, iob) for (w, _pos, iob) in s]
            for s in conll2002.iob_sents("esp.train")
        ]
        if self.max_sents_ner:
            sents = sents[: self.max_sents_ner]
        return ClassifierBasedTagger(
            train=sents, feature_detector=self._features_ner
        )

    # ------------------------------------------------------------------ pasos
    def cargar_datos(self):
        self.df = pd.read_csv(self.ruta_csv)
        if self.columna not in self.df.columns:
            raise KeyError(f"No existe la columna '{self.columna}' en el CSV")
        self.df[self.columna] = self.df[self.columna].fillna("").astype(str)
        return self.df

    def tokenizar(self, texto):
        return word_tokenize(texto, language="spanish")

    def etiquetar_pos(self, tokens):
        return self.tagger_pos.tag(tokens)

    def quitar_stopwords(self, tokens_pos):
        """Elimina stopwords y signos de puntuación conservando el tag POS."""
        return [
            (tok, pos) for tok, pos in tokens_pos
            if tok.lower() not in self.stop_es
            and pos != "PUNCT"
            and any(c.isalnum() for c in tok)
        ]

    def reconocer_entidades(self, tokens):
        """Devuelve lista de (entidad, tipo) agrupando etiquetas IOB."""
        if not tokens:
            return []
        etiquetas = self.tagger_ner.tag(tokens)
        entidades, actual, tipo = [], [], None
        for tok, iob in etiquetas:
            if iob and iob.startswith("B-"):
                if actual:
                    entidades.append((" ".join(actual), tipo))
                actual, tipo = [tok], iob[2:]
            elif iob and iob.startswith("I-") and actual:
                actual.append(tok)
            else:
                if actual:
                    entidades.append((" ".join(actual), tipo))
                actual, tipo = [], None
        if actual:
            entidades.append((" ".join(actual), tipo))
        return entidades

    @staticmethod
    def normalizar_caso(tokens):
        return [t.lower() for t in tokens]

    def lematizar(self, tokens):
        if _HAY_SIMPLEMMA:
            return [simplemma.lemmatize(t, lang="es") for t in tokens]
        return [self.stemmer.stem(t) for t in tokens]  # respaldo: stem

    # --------------------------------------------------------------- pipeline
    def procesar_texto(self, texto):
        tokens = self.tokenizar(texto)                          # 1. tokenización
        tokens_pos = self.etiquetar_pos(tokens)                 # 2. POS
        sin_stop = self.quitar_stopwords(tokens_pos)            # 3. stopwords
        tokens_filtrados = [t for t, _ in sin_stop]
        entidades = self.reconocer_entidades(tokens_filtrados)  # 4. NER
        minusculas = self.normalizar_caso(tokens_filtrados)     # 5. minúsculas
        lemas = self.lematizar(minusculas)                      # 6. lematización
        return {
            "tokens": tokens,
            "pos_tags": tokens_pos,
            "tokens_sin_stopwords": tokens_filtrados,
            "pos_sin_stopwords": sin_stop,
            "entidades": entidades,
            "tokens_minusculas": minusculas,
            "lemas": lemas,
        }

    def ejecutar(self, ruta_salida="resenas_es_procesado.csv"):
        if self.df is None:
            self.cargar_datos()
        resultados = self.df[self.columna].apply(self.procesar_texto)
        procesado = pd.DataFrame(resultados.tolist(), index=self.df.index)
        self.df = pd.concat([self.df, procesado], axis=1)
        self.df["texto_lematizado"] = self.df["lemas"].apply(" ".join)
        self.df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
        return self.df


if __name__ == "__main__":
    pipeline = NLTK_POS_Reviews("resenas_es 1.csv", columna="review_summary_es")
    df = pipeline.ejecutar("resenas_es_procesado.csv")
    print(df[["review_summary_es", "pos_tags", "entidades", "lemas"]].head())