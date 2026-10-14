"""
Análisis NLP en español con NLTK sobre reseñas de películas.
Un solo script con tres preguntas, separadas por secciones:

  EJE A: COMPARACIÓN ENTRE GÉNEROS
    Pregunta 1: ¿Las reseñas de Acción tienen mayor proporción de verbos que las de Drama?
    Pregunta 2: ¿Qué género presenta mayor diversidad léxica en sus adjetivos?

  EJE B: EMOCIONALIDAD GRAMATICAL
    Pregunta 3: ¿Las reseñas con calificación baja usan más negaciones
                (no, nunca, jamás, nada, ni) que las de calificación alta?

Metodología resumida:
  P1: POS con NLTK (CESS-ESP) sobre el texto completo; proporción = verbos / palabras;
      prueba Mann-Whitney U (Action > Drama).
  P2: adjetivos (ADJ) lematizados por género; TTR, Guiraud y TTR estandarizada
      (bootstrap de reseñas con muestras del mismo tamaño) con IC 95%.
  P3: conteo de negaciones SIN quitar stopwords ("no", "ni", "nada" son stopwords);
      grupos por `rating` (Baja / Alta); Mann-Whitney U y Fisher exacta (Baja > Alta).

Dependencias: pip install nltk pandas numpy scipy matplotlib simplemma (opcional)

Uso:
    python analisis_resenas.py          -> ejecuta las preguntas de PREGUNTAS_A_EJECUTAR
    python analisis_resenas.py 1 3      -> ejecuta solo las preguntas 1 y 3
"""

import os  # Trabajar con archivos, carpetas y rutas del sistema operativo.
import pickle  # Serializar y deserializar objetos (cache de modelos).
import sys
import textwrap
import traceback
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import nltk  # Análisis y procesamiento de texto escrito por humanos.
import numpy as np
import pandas as pd
from nltk.corpus import cess_esp, conll2002, stopwords
# cess_esp: entrenamiento/evaluación de modelos POS en español.
# conll2002: corpus enfocado en Reconocimiento de Entidades Nombradas (NER) en español.
# stopwords: listas de palabras muy frecuentes (palabras vacías).
from nltk.stem import SnowballStemmer  # Stemming: reducir palabras a una raíz común.
from nltk.tag import (
    BigramTagger,           # Considera la palabra anterior y su etiqueta.
    ClassifierBasedTagger,  # Usa aprendizaje automático en lugar de reglas de frecuencia.
    DefaultTagger,          # Asigna la misma etiqueta a todas las palabras.
    TrigramTagger,          # Usa las dos etiquetas anteriores para decidir.
    UnigramTagger,          # Etiqueta basándose únicamente en la palabra.
)
from nltk.tokenize import word_tokenize  # Divide texto en palabras y signos de puntuación.
from scipy.stats import fisher_exact, mannwhitneyu

try:
    import simplemma
    _HAY_SIMPLEMMA = True
except ImportError:
    _HAY_SIMPLEMMA = False


# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================
RUTA_CSV = "resenas_es 1.csv"
COLUMNA = "review_summary_es"
COLUMNA_GENERO = "genre"
COLUMNA_RATING = "rating"

PREGUNTAS_A_EJECUTAR = [1, 2, 3]   # qué preguntas correr por defecto
MOSTRAR_GRAFICOS = True            # False: solo guarda los PNG sin abrir ventanas

# ---- Pregunta 1: verbos Action vs Drama
P1_GENEROS = ["Action", "Drama"]
P1_INCLUIR_AUX = True  # ¿contar auxiliares (ser, estar, haber) como verbos?
P1_VERBOS = {"VERB", "AUX"} if P1_INCLUIR_AUX else {"VERB"}
P1_RUTA_GRAFICO = "comparacion_verbos_action_drama.png"
P1_PREGUNTA = "¿Las reseñas de Acción tienen mayor proporción de verbos que las de Drama?"
P1_COLORES = ["#F28E2B", "#6A4C93"]  # Action = naranja, Drama = morado

# ---- Pregunta 2: diversidad léxica de adjetivos
P2_GENEROS = ["Action", "Adventure", "Animation", "Comedy", "Drama"]  # todos los géneros
P2_EXCLUIR_STOPWORDS = True   # ignora adjetivos que son stopwords (otro, mismo, mucho...)
P2_N_ITERACIONES = 1000       # iteraciones bootstrap para la TTR estandarizada
P2_SEMILLA = 42
P2_RUTA_GRAFICO = "diversidad_lexical_adjetivos.png"
P2_PREGUNTA = "¿Qué género presenta mayor diversidad léxica en sus adjetivos?"
P2_COLORES = ["#2A9D8F", "#E9C46A"]  # panel 1 = verde azulado, panel 2 = amarillo

# ---- Pregunta 3: negaciones según rating
P3_UMBRAL_BAJO = 4    # rating <= 4  -> calificación baja
P3_UMBRAL_ALTO = 7    # rating >= 7  -> calificación alta
P3_NEGACIONES = ["no", "nunca", "jamás", "nada", "ni"]
P3_RUTA_GRAFICO = "negaciones_rating.png"
P3_PREGUNTA = ("¿Las reseñas con calificación baja usan más negaciones "
               "(no, nunca, jamás, nada, ni) que las de calificación alta?")
P3_COLOR_BAJA, P3_COLOR_ALTA = "#C1121F", "#1D3557"  # Baja = rojo, Alta = azul marino


# ============================================================================
# PIPELINE NLP (compartido por las preguntas 1 y 2)
# ============================================================================
class NLTK_POS_Reviews:
    """Pipeline de preprocesamiento lingüístico para texto en español."""

    # Mapeo de tags EAGLES (cess_esp) a etiquetas legibles
    _MAPA_POS = {
        "n": "NOUN", "a": "ADJ", "v": "VERB", "r": "ADV", "d": "DET",
        "p": "PRON", "s": "ADP", "c": "CONJ", "i": "INTJ", "z": "NUM",
        "w": "DATE", "f": "PUNCT",
    }

    def __init__(self, ruta_csv, columna="review_summary_es",
                 dir_cache="cache_nltk", max_sents_ner=None, cargar_ner=True):
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
        # El NER no se necesita para contar verbos; se puede omitir con cargar_ner=False
        self.tagger_ner = (
            self._cargar_o_entrenar("ner.pkl", self._entrenar_ner) if cargar_ner else None
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _descargar_recursos():
        for recurso in ["punkt", "punkt_tab", "stopwords", "cess_esp", "conll2002"]:
            nltk.download(recurso, quiet=True)

    def _cargar_o_entrenar(self, nombre, funcion_entrenamiento):
        ruta = os.path.join(self.dir_cache, nombre)
        if os.path.exists(ruta):
            try:
                with open(ruta, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                # Cache de una versión anterior del código (otro nombre de clase, etc.)
                print(f"Cache '{nombre}' incompatible ({type(e).__name__}); se reentrena.")
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


# ============================================================================
# UTILIDADES COMUNES
# ============================================================================
def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78 + "\n")


def asignar_genero(valor, generos):
    """Devuelve el género si la reseña pertenece a uno solo de los géneros dados."""
    valor = str(valor).lower()
    presentes = [g for g in generos if g.lower() in valor]
    return presentes[0] if len(presentes) == 1 else None


def titulo_figura(fig, numero, pregunta):
    """Pone la pregunta como título de la figura para identificarla."""
    texto = textwrap.fill(f"Pregunta {numero}: {pregunta}", width=85)
    fig.suptitle(texto, fontsize=12, fontweight="bold")
    lineas = texto.count("\n") + 1
    fig.tight_layout(rect=(0, 0, 1, 0.93 if lineas == 1 else 0.88))


def guardar_y_mostrar(fig, ruta):
    fig.savefig(ruta, dpi=150)
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    if COLUMNA not in df.columns:
        raise KeyError(f"No existe la columna '{COLUMNA}' en el CSV")
    df[COLUMNA] = df[COLUMNA].fillna("").astype(str)
    return df


# ============================================================================
# EJE A: COMPARACIÓN ENTRE GÉNEROS
# ============================================================================

# ----------------------------------------------------------------------------
# PREGUNTA 1: ¿Las reseñas de Acción tienen mayor proporción de verbos que las de Drama?
# ----------------------------------------------------------------------------
def p1_contar_verbos(texto, pipeline):
    tokens = pipeline.tokenizar(texto)
    tags = pipeline.etiquetar_pos(tokens)
    pos_palabras = [pos for tok, pos in tags
                    if pos != "PUNCT" and any(c.isalnum() for c in tok)]
    n_pal = len(pos_palabras)
    n_verb = sum(p in P1_VERBOS for p in pos_palabras)
    return pd.Series({
        "n_palabras": n_pal,
        "n_verbos": n_verb,
        "prop_verbos": n_verb / n_pal if n_pal else np.nan,
    })


def p1_graficar(df_cmp, accion, drama, ruta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    medias = df_cmp.groupby("genero_cmp")["prop_verbos"].mean().reindex(P1_GENEROS)
    axes[0].bar(medias.index, medias.values, color=P1_COLORES)
    axes[0].set_title("Proporción media de verbos por reseña")
    axes[0].set_ylabel("Verbos / palabras")
    for i, v in enumerate(medias.values):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    bp = axes[1].boxplot([accion, drama], showmeans=True, patch_artist=True)
    for caja, color in zip(bp["boxes"], P1_COLORES):
        caja.set_facecolor(color)
        caja.set_alpha(0.6)
    for mediana in bp["medians"]:
        mediana.set_color("black")
    axes[1].set_xticklabels(P1_GENEROS)  # compatible con cualquier versión de matplotlib
    axes[1].set_title("Distribución de la proporción de verbos")
    titulo_figura(fig, 1, P1_PREGUNTA)
    guardar_y_mostrar(fig, ruta)


def pregunta_1(df, pipeline):
    titulo(f"PREGUNTA 1: {P1_PREGUNTA}")
    df = df.copy()
    if COLUMNA_GENERO not in df.columns:
        raise KeyError(f"No existe la columna '{COLUMNA_GENERO}' en el CSV")

    print("Valores más frecuentes de 'genre':")
    print(df[COLUMNA_GENERO].value_counts().head(15), "\n")

    df["genero_cmp"] = df[COLUMNA_GENERO].apply(lambda v: asignar_genero(v, P1_GENEROS))
    df_cmp = df.dropna(subset=["genero_cmp"]).copy()
    print("Reseñas por género:")
    print(df_cmp["genero_cmp"].value_counts(), "\n")

    metricas = df_cmp[COLUMNA].apply(lambda t: p1_contar_verbos(t, pipeline))
    df_cmp = pd.concat([df_cmp, metricas], axis=1)

    resumen = df_cmp.groupby("genero_cmp").agg(
        n_resenas=("prop_verbos", "size"),
        prop_media_por_resena=("prop_verbos", "mean"),
        prop_mediana=("prop_verbos", "median"),
        total_verbos=("n_verbos", "sum"),
        total_palabras=("n_palabras", "sum"),
    )
    resumen["prop_global"] = resumen["total_verbos"] / resumen["total_palabras"]
    print("Resumen por género:")
    print(resumen.to_string(), "\n")

    accion = df_cmp.loc[df_cmp["genero_cmp"] == "Action", "prop_verbos"].dropna()
    drama = df_cmp.loc[df_cmp["genero_cmp"] == "Drama", "prop_verbos"].dropna()
    if accion.empty or drama.empty:
        raise ValueError("No hay reseñas suficientes de Action o Drama; revisa P1_GENEROS.")

    u, p = mannwhitneyu(accion, drama, alternative="greater")
    print(f"Action: {accion.mean():.4f} | Drama: {drama.mean():.4f}")
    print(f"Mann-Whitney U = {u:.1f}, p-valor = {p:.4f}")
    if p < 0.05:
        print("Conclusión: Action tiene una proporción de verbos significativamente MAYOR que Drama.")
    else:
        print("Conclusión: no hay evidencia suficiente de que Action tenga mayor proporción de verbos que Drama.")

    p1_graficar(df_cmp, accion, drama, P1_RUTA_GRAFICO)


# ----------------------------------------------------------------------------
# PREGUNTA 2: ¿Qué género presenta mayor diversidad léxica en sus adjetivos?
# ----------------------------------------------------------------------------
def p2_extraer_adjetivos(texto, pipeline):
    """Adjetivos lematizados (minúsculas) de un texto."""
    tokens = pipeline.tokenizar(texto)
    tags = pipeline.etiquetar_pos(tokens)
    adjs = [tok.lower() for tok, pos in tags
            if pos == "ADJ" and any(c.isalpha() for c in tok)]
    if P2_EXCLUIR_STOPWORDS:
        adjs = [a for a in adjs if a not in pipeline.stop_es]
    return pipeline.lematizar(adjs)


def p2_ttr_estandarizada(listas_por_resena, n, n_iter, rng):
    """
    TTR promedio (y percentiles 2.5 / 97.5) con muestras de n adjetivos.
    En cada iteración se remuestrean las reseñas con reemplazo (bootstrap) y de
    ese conjunto se toman n adjetivos sin reemplazo. Así todos los géneros se
    comparan con el mismo tamaño de muestra y el intervalo refleja la variabilidad.
    """
    n_res = len(listas_por_resena)
    valores = []
    for _ in range(n_iter):
        idx = rng.integers(0, n_res, n_res)
        pool = [a for i in idx for a in listas_por_resena[i]]
        if len(pool) < n:
            continue
        muestra = rng.choice(np.array(pool), size=n, replace=False)
        valores.append(len(set(muestra)) / n)
    valores = np.array(valores)
    return valores.mean(), np.percentile(valores, 2.5), np.percentile(valores, 97.5)


def p2_graficar(resumen, ruta):
    resumen = resumen.sort_values("ttr_estandarizada", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    err = [(resumen["ttr_estandarizada"] - resumen["ic95_inf"]).clip(lower=0),
           (resumen["ic95_sup"] - resumen["ttr_estandarizada"]).clip(lower=0)]
    axes[0].bar(resumen.index, resumen["ttr_estandarizada"], yerr=err, capsize=4,
                color=P2_COLORES[0])
    axes[0].set_title("TTR estandarizada de adjetivos (IC 95%)")
    axes[0].set_ylabel("Tipos / tokens (misma muestra)")
    for i, v in enumerate(resumen["ttr_estandarizada"]):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    axes[1].bar(resumen.index, resumen["guiraud"], color=P2_COLORES[1])
    axes[1].set_title("Índice de Guiraud (tipos / √tokens)")
    for i, v in enumerate(resumen["guiraud"]):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    titulo_figura(fig, 2, P2_PREGUNTA)
    guardar_y_mostrar(fig, ruta)


def pregunta_2(df, pipeline):
    titulo(f"PREGUNTA 2: {P2_PREGUNTA}")
    df = df.copy()
    if COLUMNA_GENERO not in df.columns:
        raise KeyError(f"No existe la columna '{COLUMNA_GENERO}' en el CSV")

    df["genero_cmp"] = df[COLUMNA_GENERO].apply(lambda v: asignar_genero(v, P2_GENEROS))
    df_cmp = df.dropna(subset=["genero_cmp"]).copy()
    print("Reseñas por género:")
    print(df_cmp["genero_cmp"].value_counts(), "\n")

    # Adjetivos lematizados de cada reseña
    df_cmp["adjetivos"] = df_cmp[COLUMNA].apply(lambda t: p2_extraer_adjetivos(t, pipeline))

    # Lista de adjetivos por género
    adj_por_genero = {
        g: [a for lista in df_cmp.loc[df_cmp["genero_cmp"] == g, "adjetivos"] for a in lista]
        for g in P2_GENEROS
    }
    sin_datos = [g for g, lista in adj_por_genero.items() if not lista]
    if sin_datos:
        raise ValueError(f"Sin adjetivos para: {sin_datos}. Revisa P2_GENEROS o el CSV.")

    n_min = min(len(lista) for lista in adj_por_genero.values())
    n_std = max(1, int(0.8 * n_min))  # 80% del género más pequeño, deja margen al bootstrap
    rng = np.random.default_rng(P2_SEMILLA)

    filas = {}
    for g, adjs in adj_por_genero.items():
        n_tok, n_tip = len(adjs), len(set(adjs))
        listas = list(df_cmp.loc[df_cmp["genero_cmp"] == g, "adjetivos"])
        media, inf, sup = p2_ttr_estandarizada(listas, n_std, P2_N_ITERACIONES, rng)
        filas[g] = {
            "n_adjetivos": n_tok,
            "adjetivos_unicos": n_tip,
            "ttr": n_tip / n_tok,
            "guiraud": n_tip / np.sqrt(n_tok),
            "ttr_estandarizada": media,
            "ic95_inf": inf,
            "ic95_sup": sup,
        }
    resumen = pd.DataFrame(filas).T
    resumen[["n_adjetivos", "adjetivos_unicos"]] = resumen[["n_adjetivos", "adjetivos_unicos"]].astype(int)

    print(f"Resumen por género (TTR estandarizada con muestras de {n_std} adjetivos):")
    print(resumen.round(4).sort_values("ttr_estandarizada", ascending=False).to_string(), "\n")

    print("Adjetivos más frecuentes por género:")
    for g, adjs in adj_por_genero.items():
        top = ", ".join(f"{a} ({n})" for a, n in Counter(adjs).most_common(8))
        print(f"  {g}: {top}")
    print()

    orden = resumen.sort_values("ttr_estandarizada", ascending=False)
    primero, segundo = orden.index[0], orden.index[1]
    print(f"Conclusión: '{primero}' presenta la mayor diversidad léxica en adjetivos "
          f"(TTR estandarizada = {orden.loc[primero, 'ttr_estandarizada']:.3f}).")
    if orden.loc[primero, "ic95_inf"] <= orden.loc[segundo, "ic95_sup"]:
        print(f"Ojo: su intervalo se solapa con el de '{segundo}', así que la diferencia "
              f"entre ambos no es clara.")
    else:
        print(f"Su intervalo no se solapa con el de '{segundo}', por lo que la ventaja es clara.")

    p2_graficar(resumen, P2_RUTA_GRAFICO)


# ============================================================================
# EJE B: EMOCIONALIDAD GRAMATICAL
# ============================================================================

# ----------------------------------------------------------------------------
# PREGUNTA 3: ¿Las reseñas con calificación baja usan más negaciones
#             (no, nunca, jamás, nada, ni) que las de calificación alta?
# ----------------------------------------------------------------------------
def quitar_acentos(texto):
    """'jamás' -> 'jamas' (así también se detecta si el texto viene sin tilde)."""
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


P3_NEG_NORM = [quitar_acentos(n.lower()) for n in P3_NEGACIONES]  # mismo orden que P3_NEGACIONES


def p3_contar_negaciones(texto):
    """Devuelve (n_palabras, Counter de negaciones) de un texto."""
    tokens = word_tokenize(texto, language="spanish")
    palabras = [quitar_acentos(t.lower()) for t in tokens if any(c.isalnum() for c in t)]
    conteo = Counter(p for p in palabras if p in P3_NEG_NORM)
    return len(palabras), conteo


def p3_clasificar_rating(r):
    if r <= P3_UMBRAL_BAJO:
        return "Baja"
    if r >= P3_UMBRAL_ALTO:
        return "Alta"
    return "Media"


def p3_graficar(resumen, por_palabra, ruta):
    grupos = ["Baja", "Alta"]
    colores = [P3_COLOR_BAJA, P3_COLOR_ALTA]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: % de reseñas con al menos una negación
    vals = resumen.loc[grupos, "pct_resenas_con_negacion"]
    axes[0].bar(grupos, vals, color=colores)
    axes[0].set_title("% de reseñas con al menos una negación")
    axes[0].set_ylabel("% de reseñas")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    # Panel 2: negaciones por 1000 palabras, por palabra
    x = np.arange(len(P3_NEGACIONES))
    ancho = 0.38
    for i, (g, c) in enumerate(zip(grupos, colores)):
        axes[1].bar(x + (i - 0.5) * ancho, por_palabra.loc[g].values, ancho, label=g, color=c)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(P3_NEGACIONES)
    axes[1].set_title("Negaciones por 1000 palabras")
    axes[1].legend(title="Calificación")

    titulo_figura(fig, 3, P3_PREGUNTA)
    guardar_y_mostrar(fig, ruta)


def pregunta_3(df):
    titulo(f"PREGUNTA 3: {P3_PREGUNTA}")
    df = df.copy()
    if COLUMNA_RATING not in df.columns:
        raise KeyError(f"No existe la columna '{COLUMNA_RATING}' en el CSV")

    df[COLUMNA_RATING] = pd.to_numeric(df[COLUMNA_RATING], errors="coerce")
    df = df.dropna(subset=[COLUMNA_RATING]).copy()

    print("Distribución de calificaciones (rating):")
    print(df[COLUMNA_RATING].value_counts().sort_index().to_string(), "\n")

    df["grupo"] = df[COLUMNA_RATING].apply(p3_clasificar_rating)
    print(f"Grupos (Baja <= {P3_UMBRAL_BAJO}, Alta >= {P3_UMBRAL_ALTO}, Media = resto, se excluye):")
    print(df["grupo"].value_counts().to_string(), "\n")

    df_cmp = df[df["grupo"].isin(["Baja", "Alta"])].copy()
    if df_cmp["grupo"].nunique() < 2:
        raise ValueError("Falta uno de los grupos; revisa P3_UMBRAL_BAJO / P3_UMBRAL_ALTO.")

    # ---- Conteo de negaciones por reseña
    filas = []
    for texto in df_cmp[COLUMNA]:
        n_pal, conteo = p3_contar_negaciones(texto)
        fila = {"n_palabras": n_pal, "n_neg": sum(conteo.values())}
        for w in P3_NEG_NORM:
            fila[f"neg_{w}"] = conteo.get(w, 0)
        filas.append(fila)
    df_cmp = pd.concat([df_cmp, pd.DataFrame(filas, index=df_cmp.index)], axis=1)

    df_cmp["prop_neg"] = np.where(df_cmp["n_palabras"] > 0,
                                  df_cmp["n_neg"] / df_cmp["n_palabras"].replace(0, np.nan), np.nan)
    df_cmp["tiene_neg"] = df_cmp["n_neg"] > 0

    # ---- Resumen por grupo
    resumen = df_cmp.groupby("grupo").agg(
        n_resenas=("n_neg", "size"),
        total_negaciones=("n_neg", "sum"),
        total_palabras=("n_palabras", "sum"),
        prop_media_por_resena=("prop_neg", "mean"),
        pct_resenas_con_negacion=("tiene_neg", lambda s: 100 * s.mean()),
    )
    resumen["neg_por_1000_palabras"] = 1000 * resumen["total_negaciones"] / resumen["total_palabras"]
    print("Resumen por grupo:")
    print(resumen.loc[["Baja", "Alta"]].round(4).to_string(), "\n")

    # ---- Desglose por palabra (por 1000 palabras)
    cols = [f"neg_{w}" for w in P3_NEG_NORM]
    por_palabra = df_cmp.groupby("grupo")[cols].sum().div(resumen["total_palabras"], axis=0) * 1000
    por_palabra.columns = P3_NEGACIONES
    print("Negaciones por 1000 palabras, por palabra:")
    print(por_palabra.loc[["Baja", "Alta"]].round(3).to_string(), "\n")

    # ---- Pruebas estadísticas (H1: Baja > Alta)
    baja = df_cmp.loc[df_cmp["grupo"] == "Baja", "prop_neg"].dropna()
    alta = df_cmp.loc[df_cmp["grupo"] == "Alta", "prop_neg"].dropna()
    u, p_mw = mannwhitneyu(baja, alta, alternative="greater")

    t = df_cmp.groupby("grupo")["tiene_neg"].agg(["sum", "count"])
    tabla = [[t.loc["Baja", "sum"], t.loc["Baja", "count"] - t.loc["Baja", "sum"]],
             [t.loc["Alta", "sum"], t.loc["Alta", "count"] - t.loc["Alta", "sum"]]]
    odds, p_fi = fisher_exact(tabla, alternative="greater")

    print(f"Proporción media de negaciones por reseña -> Baja: {baja.mean():.4f} | Alta: {alta.mean():.4f}")
    print(f"Mann-Whitney U = {u:.1f}, p-valor = {p_mw:.4f}")
    print(f"Fisher (reseñas con negación): odds ratio = {odds:.3f}, p-valor = {p_fi:.4f}\n")

    if p_mw < 0.05:
        print("Conclusión: las reseñas con calificación BAJA usan significativamente MÁS negaciones que las de calificación alta.")
    else:
        print("Conclusión: no hay evidencia suficiente de que las reseñas de calificación baja usen más negaciones que las de calificación alta.")

    p3_graficar(resumen, por_palabra, P3_RUTA_GRAFICO)


# ============================================================================
# EJECUCIÓN
# ============================================================================
def main(preguntas=None):
    preguntas = preguntas or PREGUNTAS_A_EJECUTAR

    for recurso in ["punkt", "punkt_tab"]:
        nltk.download(recurso, quiet=True)

    df = cargar_datos()

    # El pipeline POS solo se construye si se necesita (preguntas 1 y 2); sin NER.
    pipeline = None
    if any(p in (1, 2) for p in preguntas):
        pipeline = NLTK_POS_Reviews(RUTA_CSV, columna=COLUMNA, cargar_ner=False)

    tareas = {
        1: lambda: pregunta_1(df, pipeline),
        2: lambda: pregunta_2(df, pipeline),
        3: lambda: pregunta_3(df),
    }
    for p in preguntas:
        if p not in tareas:
            print(f"Pregunta {p} no existe (usa 1, 2 o 3).")
            continue
        try:
            tareas[p]()
        except Exception:
            print(f"\n[ERROR en la pregunta {p}]")
            traceback.print_exc()


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(args or None)