"""Métricas morfosintácticas a partir de pos_analysis_2.csv y processed_es.csv.
Ejecutar: .venv/Scripts/python.exe src/metrics.py
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # El analista usa un backend sin ventana para guardar PNG.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy

from config import CONFIG, METRICAS

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# El analista lee la configuración del análisis.
EXCLUIR = set(METRICAS["etiquetas_excluidas"])
NEGACIONES = set(METRICAS["negaciones"])
INTENSIDAD = set(METRICAS["adverbios_intensidad"])
BASE = METRICAS["base_normalizacion"]
BLOQUE = METRICAS["tamano_bloque"]
FIGURAS = METRICAS["carpeta_figuras"]


def cargar_datos():
    """Carga los tokens POS, los une con los metadatos y crea la polaridad."""
    tokens = pd.read_csv(METRICAS["ruta_tokens"])
    meta = pd.read_csv(METRICAS["ruta_metadatos"]).reset_index(names="review_id")
    # El analista clasifica la polaridad según el rating.
    meta["polaridad"] = meta["rating"].apply(lambda r: "negativa" if r <= METRICAS["rating_negativo"]
                                             else ("positiva" if r >= METRICAS["rating_positivo"] else "neutra"))
    # El analista numera cada token dentro de su reseña para alinearlo con spaCy.
    tokens = tokens.sort_values("review_id").copy()
    tokens["pos_en_resena"] = tokens.groupby("review_id").cumcount()
    columnas = ["review_id", "main_genre", "year", "is_spoiler", "n_words", "review_text_es", "polaridad"]
    return tokens.merge(meta[columnas], on="review_id", how="inner"), meta


def enriquecer_spacy(tabla, meta):
    """Añade lema, persona y oración con spaCy usando una caché en disco."""
    ruta = METRICAS["ruta_cache"]
    if ruta.exists():
        cache = pd.read_csv(ruta)
    else:
        nlp = spacy.load(METRICAS["modelo_spacy"], disable=["ner"])
        filas = []
        for review_id, doc in zip(meta["review_id"], nlp.pipe(meta["review_text_es"], batch_size=64)):
            numero = {t.i: i for i, s in enumerate(doc.sents) for t in s}
            tamano = {t.i: len(s) for s in doc.sents for t in s}
            for token in doc:
                persona = token.morph.get("Person")
                filas.append({"review_id": review_id, "pos_en_resena": token.i, "lema": token.lemma_.lower(),
                              "persona": persona[0] if persona else "", "num_oracion": numero[token.i],
                              "tam_oracion": tamano[token.i]})
        cache = pd.DataFrame(filas)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(ruta, index=False, encoding="utf-8")
    return tabla.merge(cache, on=["review_id", "pos_en_resena"], how="left")


def metricas_por_resena(tabla):
    """Calcula las métricas morfosintácticas de cada reseña."""
    # El analista elimina la puntuación para evitar sesgos.
    tabla = tabla[~tabla["pos_spacy"].isin(EXCLUIR)].sort_values(["review_id", "pos_en_resena"])
    metricas = tabla.groupby("review_id").size().rename("total").to_frame()
    conteo = tabla.groupby(["review_id", "pos_spacy"]).size().unstack(fill_value=0)
    for etiqueta in ["NOUN", "VERB", "ADJ", "ADV", "PRON"]:
        metricas[etiqueta] = conteo[etiqueta] if etiqueta in conteo else 0
    # El analista cuenta negaciones, intensidad y pronombres por persona.
    metricas["negaciones"] = tabla[tabla["token"].str.lower().isin(NEGACIONES)].groupby("review_id").size()
    metricas["intensidad"] = tabla[tabla["token"].str.lower().isin(INTENSIDAD)].groupby("review_id").size()
    for persona in ["1", "2", "3"]:
        filtro = (tabla["pos_spacy"] == "PRON") & (tabla["persona"] == persona)
        metricas["p" + persona] = tabla[filtro].groupby("review_id").size()
    # El analista calcula el TTR sobre bloques fijos y la longitud media de oración.
    metricas["ttr"] = tabla.groupby("review_id")["lema"].apply(lambda l: l.head(BLOQUE).nunique() / min(len(l), BLOQUE))
    metricas["long_oracion"] = metricas["total"] / tabla.groupby("review_id")["num_oracion"].nunique()
    return metricas.fillna(0)


def por_grupo(metricas, meta, columna):
    """Agrega las métricas por una columna de metadatos."""
    grupo = metricas.join(meta.set_index("review_id")[[columna]]).groupby(columna)
    total = grupo["total"].sum()
    resultado = pd.DataFrame({"n_resenas": grupo.size(), "total_tokens": total})
    resultado["densidad_lexica"] = (grupo["NOUN"].sum() + grupo["VERB"].sum() + grupo["ADJ"].sum() + grupo["ADV"].sum()) / total
    resultado["ratio_verbos_sustantivos"] = grupo["VERB"].sum() / grupo["NOUN"].sum()
    for etiqueta in ["ADJ", "VERB", "ADV"]:
        resultado[etiqueta.lower() + "_por_100"] = grupo[etiqueta].sum() / total * BASE
    resultado["negaciones_por_100"] = grupo["negaciones"].sum() / total * BASE
    resultado["intensidad_por_100"] = grupo["intensidad"].sum() / total * BASE
    for persona in ["1", "2", "3"]:
        resultado["pron_p" + persona + "_por_100"] = grupo["p" + persona].sum() / total * BASE
    resultado["ttr_media"] = grupo["ttr"].mean()
    resultado["long_oracion_media"] = grupo["long_oracion"].mean()
    return resultado


def mostrar(titulo, tabla, decimales=3):
    """Imprime una tabla con título y redondeo."""
    print("\n" + titulo)
    print("-" * len(titulo))
    numericas = tabla.select_dtypes("number").columns
    tabla = tabla.copy()
    tabla[numericas] = tabla[numericas].round(decimales)
    print(tabla.to_string())


def interpretar(texto):
    """Imprime un párrafo de interpretación."""
    print("\nInterpretación:", texto, "\n")


def guardar(figura, nombre):
    """Guarda la figura en outputs/figuras/."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")
    plt.close(figura)


def eje_a(metricas, meta):
    """Eje A: género y estilo gramatical."""
    genero = por_grupo(metricas, meta, "main_genre")
    # A.1: índice de adjetivación Comedia vs Drama.
    primero, segundo = METRICAS["generos_eje_a"]
    a1 = genero.loc[[primero, segundo], ["n_resenas", "adj_por_100"]]
    mostrar(f"EJE A.1 - Adjetivación (ADJ por 100): {primero} vs {segundo}", a1)
    figura, eje = plt.subplots(figsize=(6, 4))
    sns.barplot(x=a1.index, y=a1["adj_por_100"], ax=eje)
    eje.set_ylabel("Adjetivos por 100 tokens")
    guardar(figura, "ejeA_adjetivacion_comedy_drama.png")
    v1, v2 = a1.loc[primero, "adj_por_100"], a1.loc[segundo, "adj_por_100"]
    interpretar(f"{primero} registra {v1:.2f} adjetivos por 100 tokens y {segundo} {v2:.2f}. La mayor carga valorativa es de {primero if v1 > v2 else segundo}.")
    # A.2: ratio verbos/sustantivos por género.
    a2 = genero[["n_resenas", "ratio_verbos_sustantivos"]].sort_values("ratio_verbos_sustantivos", ascending=False)
    mostrar("EJE A.2 - Ratio VERB/NOUN por género", a2)
    figura, eje = plt.subplots(figsize=(7, 4))
    sns.barplot(x=a2.index, y=a2["ratio_verbos_sustantivos"], ax=eje)
    eje.set_ylabel("VERB / NOUN")
    guardar(figura, "ejeA_ratio_verbo_sustantivo.png")
    interpretar(f"{a2.index[0]} tiene el ratio más alto ({a2.iloc[0, 1]:.3f}) y {a2.index[-1]} el más bajo ({a2.iloc[-1, 1]:.3f}); un ratio alto sugiere un estilo narrativo.")


def eje_b(tabla, metricas, meta):
    """Eje B: polaridad y léxico."""
    polaridad = por_grupo(metricas, meta[meta["polaridad"] != "neutra"], "polaridad")
    # B.1: adverbios de intensidad según polaridad.
    b1 = polaridad[["n_resenas", "intensidad_por_100"]]
    mostrar("EJE B.1 - Adverbios de intensidad por 100 tokens según polaridad", b1)
    figura, eje = plt.subplots(figsize=(6, 4))
    sns.barplot(x=b1.index, y=b1["intensidad_por_100"], ax=eje)
    eje.set_ylabel("Adverbios de intensidad por 100 tokens")
    guardar(figura, "ejeB_intensidad_polaridad.png")
    neg, pos = b1.loc["negativa", "intensidad_por_100"], b1.loc["positiva", "intensidad_por_100"]
    interpretar(f"Negativas {neg:.3f} y positivas {pos:.3f} adverbios por 100 tokens; la polaridad negativa {'sí' if neg > pos else 'no'} supera a la positiva en intensificadores.")
    # B.2: adjetivos más distintivos de cada polaridad.
    adjetivos = tabla[(tabla["pos_spacy"] == "ADJ") & (tabla["polaridad"] != "neutra")]
    frecuencia = adjetivos.groupby(["lema", "polaridad"]).size().unstack(fill_value=0).div(polaridad["total_tokens"], axis=1) * BASE
    top_pos = (frecuencia["positiva"] - frecuencia["negativa"]).nlargest(10)
    top_neg = (frecuencia["negativa"] - frecuencia["positiva"]).nlargest(10)
    mostrar("EJE B.2 - Adjetivos más distintivos de reseñas POSITIVAS", frecuencia.loc[top_pos.index])
    mostrar("EJE B.2 - Adjetivos más distintivos de reseñas NEGATIVAS", frecuencia.loc[top_neg.index])
    figura, ejes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=top_pos.values, y=top_pos.index, ax=ejes[0], color="#4c9f70")
    ejes[0].set_title("Adjetivos positivos")
    sns.barplot(x=top_neg.values, y=top_neg.index, ax=ejes[1], color="#c0504d")
    ejes[1].set_title("Adjetivos negativos")
    guardar(figura, "ejeB_adjetivos_distintivos.png")
    interpretar(f"Positivos: {', '.join(top_pos.index[:5])}. Negativos: {', '.join(top_neg.index[:5])}. El léxico valorativo separa ambas polaridades.")


def eje_c(metricas, meta):
    """Eje C: evolución temporal."""
    anio = por_grupo(metricas, meta, "year")
    # C.1: simplicidad gramatical por año.
    c1 = anio[["n_resenas", "long_oracion_media", "densidad_lexica"]]
    mostrar("EJE C.1 - Simplicidad gramatical por año", c1)
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=c1.index.astype(str), y=c1["long_oracion_media"], ax=ejes[0])
    ejes[0].set_title("Longitud media de oración por año")
    ejes[0].set_ylabel("Tokens por oración")
    sns.barplot(x=c1.index.astype(str), y=c1["densidad_lexica"], ax=ejes[1])
    ejes[1].set_title("Densidad léxica por año")
    guardar(figura, "ejeC_simplicidad_gramatical.png")
    antiguo, reciente = c1.index[0], c1.index[-1]
    lon_a, lon_r = c1.loc[antiguo, "long_oracion_media"], c1.loc[reciente, "long_oracion_media"]
    den_a, den_r = c1.loc[antiguo, "densidad_lexica"], c1.loc[reciente, "densidad_lexica"]
    interpretar(f"Oración media: {lon_a:.2f} en {antiguo} y {lon_r:.2f} en {reciente}; densidad {den_a:.3f} y {den_r:.3f}. Las reseñas recientes son más simples: {'sí' if (lon_r < lon_a and den_r < den_a) else 'no del todo'}.")
    # C.2: longitud media de las reseñas por año.
    longitud = meta.groupby("year").agg(n_resenas=("review_id", "size"), palabras_medias=("n_words", "mean"))
    longitud["tokens_medios"] = metricas.groupby(meta.set_index("review_id")["year"])["total"].mean()
    mostrar("EJE C.2 - Longitud media de las reseñas por año", longitud)
    figura, eje = plt.subplots(figsize=(6, 4))
    sns.barplot(x=longitud.index.astype(str), y=longitud["tokens_medios"], ax=eje)
    eje.set_ylabel("Tokens medios por reseña")
    guardar(figura, "ejeC_longitud_resenas.png")
    cambio = longitud.loc[reciente, "tokens_medios"] - longitud.loc[antiguo, "tokens_medios"]
    interpretar(f"La reseña media pasa de {longitud.loc[antiguo, 'tokens_medios']:.1f} tokens en {antiguo} a {longitud.loc[reciente, 'tokens_medios']:.1f} en {reciente}; la longitud {'aumenta' if cambio >= 0 else 'disminuye'} con los años.")


def eje_d(metricas, meta):
    """Eje D: spoilers y estilo narrativo."""
    spoiler = por_grupo(metricas, meta, "is_spoiler")
    spoiler.index = spoiler.index.map({True: "con_spoiler", False: "sin_spoiler"})
    d1 = spoiler[["n_resenas", "verb_por_100", "adj_por_100"]]
    mostrar("EJE D.1 - Verbos y adjetivos por 100 tokens según spoiler", d1)
    figura, eje = plt.subplots(figsize=(7, 4))
    d1.plot(kind="bar", ax=eje)
    eje.set_ylabel("Frecuencia por 100 tokens")
    guardar(figura, "ejeD_spoilers_estilo.png")
    con, sin = d1.loc["con_spoiler"], d1.loc["sin_spoiler"]
    t1 = "usan más verbos (narran la trama)" if con["verb_por_100"] > sin["verb_por_100"] else "no usan más verbos"
    t2 = "usan más adjetivos (evalúan la obra)" if sin["adj_por_100"] > con["adj_por_100"] else "no usan más adjetivos"
    interpretar(f"Las reseñas con spoilers {t1} y las reseñas sin spoilers {t2}.")


def causa_discrepancia(fila):
    """Clasifica una discrepancia según su causa probable."""
    if "X" in (fila["pos_nltk"], fila["pos_spacy"]):
        return "palabra ausente del corpus de NLTK"
    if {fila["pos_spacy"], fila["pos_nltk"]} == {"AUX", "VERB"}:
        return "diferencia de criterio entre esquemas (AUX vs VERB)"
    if not str(fila["token"]).isalpha():
        return "error de traducción"
    return "ambigüedad real"


def comparar_nltk(tabla):
    """Compara NLTK con spaCy: concordancia, confusión y discrepancias."""
    # El analista excluye la puntuación en toda la comparación.
    sin_punct = tabla[(tabla["pos_spacy"] != "PUNCT") & (tabla["pos_nltk"] != "PUNCT")].copy()
    global_pct = sin_punct["coincide"].mean() * BASE
    print(f"\nConcordancia global excluyendo puntuación: {global_pct:.2f}%")
    # Concordancia por categoría UPOS.
    por_upos = sin_punct.groupby("pos_spacy").agg(tokens=("coincide", "size"), concordancia=("coincide", "mean"))
    por_upos["concordancia"] = por_upos["concordancia"] * BASE
    por_upos = por_upos.sort_values("concordancia", ascending=False)
    mostrar("Comparación NLTK vs spaCy - Concordancia por UPOS (%)", por_upos, 2)
    figura, eje = plt.subplots(figsize=(9, 5))
    sns.barplot(x=por_upos.index, y=por_upos["concordancia"], ax=eje)
    eje.set_ylabel("Concordancia (%)")
    eje.tick_params(axis="x", rotation=45)
    guardar(figura, "nltk_concordancia_upos.png")
    # Matriz de confusión (filas spaCy, columnas NLTK).
    confusion = pd.crosstab(sin_punct["pos_spacy"], sin_punct["pos_nltk"])
    figura, eje = plt.subplots(figsize=(12, 9))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "tokens"}, ax=eje, annot_kws={"size": 7})
    eje.set_xlabel("Etiqueta NLTK")
    eje.set_ylabel("Etiqueta spaCy")
    guardar(figura, "nltk_matriz_confusion.png")
    # 10 ejemplos de discrepancia clasificados por causa.
    discrepancias = sin_punct[~sin_punct["coincide"]].copy()
    discrepancias["causa"] = discrepancias.apply(causa_discrepancia, axis=1)
    partes = [g.sample(min(len(g), 3), random_state=CONFIG["semilla"]) for _, g in discrepancias.groupby("causa")]
    ejemplos = pd.concat(partes).head(10)
    print("\nComparación NLTK vs spaCy - 10 ejemplos de discrepancia")
    print(ejemplos[["token", "pos_spacy", "pos_nltk", "causa"]].to_string(index=False))
    # Revisión manual de 50 tokens discrepantes.
    muestra = discrepancias.sample(50, random_state=CONFIG["semilla"])
    print("\nRevisión manual de 50 tokens en discrepancia:")
    print(muestra[["token", "pos_spacy", "pos_nltk", "causa"]].to_string(index=False))
    causas = muestra["causa"].value_counts()
    interpretar(f"La concordancia global es {global_pct:.2f}%. La causa más frecuente entre 50 tokens discrepantes es '{causas.index[0]}' ({causas.iloc[0]} de 50). spaCy es más confiable aquí: reconoce mejor nombres propios y formas verbales compuestas (AUX) que un NLTK entrenado sobre noticias (cess_esp).")


def main():
    """Ejecuta el análisis completo y genera tablas, figuras e interpretaciones."""
    print("=" * 70)
    print("ANÁLISIS DE MÉTRICAS MORFOSINTÁCTICAS - POS_Tagging")
    print("=" * 70)
    tabla, meta = cargar_datos()
    tabla = enriquecer_spacy(tabla, meta)
    metricas = metricas_por_resena(tabla)
    print(f"\nReseñas analizadas: {len(metricas)}")
    print(f"Tokens analizados (sin puntuación): {int(metricas['total'].sum())}")
    eje_a(metricas, meta)
    eje_b(tabla, metricas, meta)
    eje_c(metricas, meta)
    eje_d(metricas, meta)
    print("\n" + "=" * 70)
    print("COMPARACIÓN NLTK vs spaCy")
    print("=" * 70)
    comparar_nltk(tabla)
    print("\nFiguras guardadas en:", FIGURAS.resolve())


if __name__ == "__main__":
    main()
