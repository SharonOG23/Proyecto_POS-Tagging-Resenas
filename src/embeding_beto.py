"""Análisis semántico de reseñas de cine y series en español con BETO.

El script desarrolla tres análisis:
1. Polisemia contextual: la misma palabra recibe embeddings distintos según la oración.
2. Búsqueda semántica: encuentra reseñas parecidas a una consulta escrita en lenguaje natural.
3. Masked Language Model: BETO predice palabras ocultas en frases típicas de reseñas.

Los resultados se guardan en data/results/beto/.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("transformers").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"

DIR_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CSV = DIR_PROYECTO / "data" / "processed" / "processed_es_fa.csv"
DIR_RESULTADOS = DIR_PROYECTO / "data" / "results" / "beto"
RUTA_WORD2VEC = DIR_PROYECTO / "data" / "models" / "word2vec_resenas.kv"

MAX_LENGTH = 256
BATCH_SIZE = 8
TOP_K = 5
MAX_ORACIONES_POR_POLARIDAD = 40
N_RESENAS_BUSQUEDA = 300  # None = usar el corpus completo

PALABRAS_POLISEMIA = {
    "oscuro": ["oscuro", "oscura", "oscuros", "oscuras"],
    "brillante": ["brillante", "brillantes"],
    "frío": ["frío", "fría", "fríos", "frías"],
    "duro": ["duro", "dura", "duros", "duras"],
    "pobre": ["pobre", "pobres"],
    "lleno": ["lleno", "llena", "llenos", "llenas"],
}

CONSULTAS_EJEMPLO = [
    "una película lenta pero con una fotografía preciosa",
    "actuaciones brillantes y un guion conmovedor",
    "terror psicológico con un final inquietante",
]

PLANTILLAS_MLM = [
    "El guion se siente [MASK] desde la primera escena.",
    "La fotografía resulta [MASK] en las escenas nocturnas.",
    "El ritmo de la segunda mitad es demasiado [MASK].",
    "La química entre los dos protagonistas es [MASK].",
    "El villano me pareció [MASK] por su forma de actuar.",
    "La banda sonora pasa casi [MASK] durante el clímax.",
    "Salí del cine con una sensación [MASK], la recomiendo.",
    "Salí del cine con una sensación [MASK], no la recomiendo.",
    "La película me mantuvo [MASK] hasta el final, quería ver más.",
    "La película me mantuvo [MASK] hasta el final, miré el reloj varias veces.",
]

RAICES_POSITIVAS = (
    "buen",
    "excelent",
    "brillant",
    "conmovedor",
    "emocionant",
    "emotiv",
    "divertid",
    "inolvidabl",
    "memorabl",
    "perfect",
    "genial",
    "hermos",
    "precios",
    "interesant",
    "solid",
    "redond",
    "eficaz",
    "convincent",
    "bonit",
    "lind",
    "gust",
    "maravillos",
    "espectacular",
    "original",
    "profund",
    "agradabl",
    "fascinant",
    "sorprendent",
    "entretenid",
    "atrapant",
    "cautiv",
    "inspirador",
    "encantador",
    "sublim",
    "magnific",
    "sobresalient",
    "destacabl",
    "fantast",
)

RAICES_NEGATIVAS = (
    "mal",
    "aburrid",
    "lent",
    "pesad",
    "deficient",
    "pobr",
    "floj",
    "confus",
    "predecibl",
    "insoportabl",
    "insufribl",
    "tedios",
    "plan",
    "fri",
    "vaci",
    "decepcionant",
    "olvidabl",
    "ridicul",
    "torp",
    "caotic",
    "incoherent",
    "mediocr",
    "fallid",
    "desastr",
    "molest",
    "irritant",
    "terribl",
    "horribl",
    "espantos",
    "asqu",
    "abominabl",
    "deplorabl",
    "lamentabl",
    "vergonzos",
    "inaceptabl",
    "desagradabl",
    "pesim",
    "trist",
    "sos",
    "inutil",
    "desperdici",
)


# ---------------------------------------------------------------------------
# Carga de datos y del modelo
# ---------------------------------------------------------------------------

def cargar_modelo_beto():
    """Carga el tokenizador y el modelo BETO una sola vez."""
    print(f"Cargando BETO desde HuggingFace ({MODELO_BETO})...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BETO)
    modelo = AutoModel.from_pretrained(MODELO_BETO, add_pooling_layer=False)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dispositivo).eval()

    print("BETO cargado correctamente.")
    print(f"  Dispositivo: {dispositivo}")
    print(f"  Parámetros: {sum(param.numel() for param in modelo.parameters()):,}")
    print(f"  Dimensión del embedding: {modelo.config.hidden_size}")
    print(f"  Capas de atención: {modelo.config.num_hidden_layers}")
    print(f"  Cabezas de atención: {modelo.config.num_attention_heads}\n")
    return tokenizer, modelo, dispositivo


def cargar_resenas():
    """Lee el corpus procesado y muestra un resumen sencillo."""
    print(f"Leyendo reseñas desde {RUTA_CSV}...")
    df = pd.read_csv(RUTA_CSV)
    print(f"  Reseñas cargadas: {len(df):,}")
    print(f"  Columnas disponibles: {', '.join(df.columns)}\n")
    return df


# ---------------------------------------------------------------------------
# Utilidades de texto y tokenización
# ---------------------------------------------------------------------------

def dividir_oraciones(texto):
    """Divide una reseña en oraciones usando la puntuación final."""
    limpio = re.sub(r"\s+", " ", str(texto or "")).strip()
    if not limpio:
        return []
    partes = re.split(r"(?<=[.!?])\s+", limpio)
    return [parte.strip() for parte in partes if parte.strip()]


def _spans_variantes(texto, variantes):
    """Devuelve las posiciones (inicio, fin) de las variantes encontradas."""
    texto_min = texto.lower()
    spans = set()
    for variante in variantes:
        patron = rf"\b{re.escape(variante)}\b"
        for coincidencia in re.finditer(patron, texto_min):
            spans.add((coincidencia.start(), coincidencia.end()))
    return sorted(spans)


def _codificar_lote(tokenizer, textos, **kwargs):
    """Tokeniza un lote y devuelve los offsets cuando el tokenizador lo permite."""
    try:
        enc = tokenizer(textos, return_offsets_mapping=True, **kwargs)
        offsets = enc.pop("offset_mapping")
    except (ValueError, NotImplementedError):
        enc = tokenizer(textos, **kwargs)
        offsets = None
    return enc, offsets


def _indices_palabra(oracion, variantes, tokenizer, input_ids, offsets):
    """Encuentra los tokens que forman la palabra objetivo dentro de una oración."""
    spans = _spans_variantes(oracion, variantes)
    if not spans:
        return []

    if offsets is not None:
        offs = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        for inicio, fin in spans:
            indices = [
                posicion
                for posicion, (a, b) in enumerate(offs)
                if b > a and a < fin and b > inicio
            ]
            if indices:
                return indices
        return []

    # Respaldo para tokenizadores lentos que no entregan offsets.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for variante in variantes:
        indices = []
        for posicion, token in enumerate(tokens):
            if token in tokenizer.all_special_tokens:
                if indices:
                    break
                continue
            limpio = token.replace("##", "").lower()
            if limpio and (limpio == variante or limpio in variante):
                indices.append(posicion)
            elif indices:
                break
        if indices:
            return indices
    return []


# ---------------------------------------------------------------------------
# Embeddings contextuales
# ---------------------------------------------------------------------------

def embeddings_de_palabra(
    oraciones,
    variantes,
    tokenizer,
    modelo,
    dispositivo,
    batch_size=BATCH_SIZE,
):
    """Calcula el embedding contextual de una palabra en cada oración."""
    vectores = []
    validas = []

    for inicio in range(0, len(oraciones), batch_size):
        lote = oraciones[inicio : inicio + batch_size]
        enc, offsets = _codificar_lote(
            tokenizer,
            lote,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        ids_lote = enc["input_ids"]
        entrada = {nombre: tensor.to(dispositivo) for nombre, tensor in enc.items()}

        with torch.no_grad():
            salida = modelo(**entrada)
        ocultos = salida.last_hidden_state.cpu()

        for posicion, oracion in enumerate(lote):
            offs_oracion = offsets[posicion] if offsets is not None else None
            indices = _indices_palabra(
                oracion,
                variantes,
                tokenizer,
                ids_lote[posicion].tolist(),
                offs_oracion,
            )
            if indices:
                vectores.append(ocultos[posicion, indices].mean(dim=0).numpy())
                validas.append(oracion)

    return vectores, validas


def embeddings_promedio(textos, tokenizer, modelo, dispositivo, batch_size=BATCH_SIZE):
    """Genera un vector por texto promediando los tokens (mean pooling)."""
    todos = []

    for inicio in range(0, len(textos), batch_size):
        lote = textos[inicio : inicio + batch_size]
        enc = tokenizer(
            lote,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        entrada = {nombre: tensor.to(dispositivo) for nombre, tensor in enc.items()}

        with torch.no_grad():
            salida = modelo(**entrada)

        mascara = entrada["attention_mask"].unsqueeze(-1).float()
        suma = (salida.last_hidden_state * mascara).sum(dim=1)
        conteo = mascara.sum(dim=1).clamp(min=1e-9)
        vector = suma / conteo
        vector = F.normalize(vector, p=2, dim=1)
        todos.append(vector.cpu().numpy())

    if not todos:
        return np.empty((0, modelo.config.hidden_size), dtype=np.float32)
    return np.vstack(todos)


def _similitud_promedio(vectores):
    """Calcula la similitud coseno promedio, mínima y máxima de un grupo."""
    if len(vectores) < 2:
        return None, None, None
    matriz = np.vstack(vectores)
    matriz = matriz / (np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-12)
    similitudes = matriz @ matriz.T
    indices = np.triu_indices(len(matriz), k=1)
    valores = similitudes[indices]
    return float(valores.mean()), float(valores.min()), float(valores.max())


def _similitud_entre(grupo_a, grupo_b):
    """Calcula la similitud coseno promedio entre dos grupos de vectores."""
    if not grupo_a or not grupo_b:
        return None
    matriz_a = np.vstack(grupo_a)
    matriz_b = np.vstack(grupo_b)
    matriz_a = matriz_a / (np.linalg.norm(matriz_a, axis=1, keepdims=True) + 1e-12)
    matriz_b = matriz_b / (np.linalg.norm(matriz_b, axis=1, keepdims=True) + 1e-12)
    return float((matriz_a @ matriz_b.T).mean())


def _centroide(vectores):
    """Devuelve el centroide normalizado de un grupo de embeddings."""
    if not vectores:
        return None
    centro = np.vstack(vectores).mean(axis=0)
    return centro / (np.linalg.norm(centro) + 1e-12)


def _seleccionar_ejemplos(vectores_pos, oraciones_pos, vectores_neg, oraciones_neg):
    """Elige por cercanía a cada centroide la oración positiva y negativa."""
    if not vectores_pos or not vectores_neg:
        return "", ""

    centro_pos = _centroide(vectores_pos)
    centro_neg = _centroide(vectores_neg)

    def puntaje(vector):
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        return float(vector @ centro_pos - vector @ centro_neg)

    puntajes_pos = [puntaje(vector) for vector in vectores_pos]
    puntajes_neg = [puntaje(vector) for vector in vectores_neg]

    ejemplo_pos = oraciones_pos[int(np.argmax(puntajes_pos))]
    ejemplo_neg = oraciones_neg[int(np.argmin(puntajes_neg))]
    return ejemplo_pos, ejemplo_neg


# ---------------------------------------------------------------------------
# 1. Polisemia contextual
# ---------------------------------------------------------------------------

def recolectar_oraciones(df, variantes, maximo=MAX_ORACIONES_POR_POLARIDAD):
    """Busca oraciones positivas y negativas que contengan alguna variante."""
    positivas = []
    negativas = []
    ratings = pd.to_numeric(df["rating"], errors="coerce")

    for posicion, (_, fila) in enumerate(df.iterrows()):
        rating = ratings.iloc[posicion]
        if pd.isna(rating):
            continue

        for oracion in dividir_oraciones(fila.get("review_text_es")):
            if not _spans_variantes(oracion, variantes):
                continue
            if rating >= 7 and len(positivas) < maximo:
                positivas.append(oracion)
            elif rating <= 4 and len(negativas) < maximo:
                negativas.append(oracion)
            break

        if len(positivas) >= maximo and len(negativas) >= maximo:
            break

    return positivas, negativas


def analizar_polisemia(df, tokenizer, modelo, dispositivo):
    """Compara los embeddings de cada palabra según el contexto y la valoración."""
    print("=== 1. POLISEMIA CONTEXTUAL ===")
    filas = []

    for palabra, variantes in PALABRAS_POLISEMIA.items():
        positivas, negativas = recolectar_oraciones(df, variantes)
        vectores_pos, oraciones_pos = embeddings_de_palabra(
            positivas, variantes, tokenizer, modelo, dispositivo
        )
        vectores_neg, oraciones_neg = embeddings_de_palabra(
            negativas, variantes, tokenizer, modelo, dispositivo
        )
        ejemplo_pos, ejemplo_neg = _seleccionar_ejemplos(
            vectores_pos, oraciones_pos, vectores_neg, oraciones_neg
        )

        todos = vectores_pos + vectores_neg
        sim_promedio, sim_min, sim_max = _similitud_promedio(todos)
        sim_positivas, _, _ = _similitud_promedio(vectores_pos)
        sim_negativas, _, _ = _similitud_promedio(vectores_neg)
        sim_entre = _similitud_entre(vectores_pos, vectores_neg)

        filas.append(
            {
                "palabra": palabra,
                "n_positivas": len(vectores_pos),
                "n_negativas": len(vectores_neg),
                "oracion_positiva": ejemplo_pos,
                "oracion_negativa": ejemplo_neg,
                "sim_promedio_positivas": sim_positivas,
                "sim_promedio_negativas": sim_negativas,
                "sim_promedio_entre_polaridades": sim_entre,
                "sim_promedio_global": sim_promedio,
                "sim_min_global": sim_min,
                "sim_max_global": sim_max,
            }
        )

        valor_mostrado = sim_promedio if sim_promedio is not None else float("nan")
        print(
            f"  {palabra:10s} | positivas={len(vectores_pos):2d} "
            f"| negativas={len(vectores_neg):2d} "
            f"| similitud promedio={valor_mostrado:.4f}"
        )
        print(f"    Ejemplo positivo (BETO): {ejemplo_pos}")
        print(f"    Ejemplo negativo (BETO): {ejemplo_neg}")

    resumen = pd.DataFrame(filas)
    ruta = DIR_RESULTADOS / "polisemia_resumen.csv"
    resumen.to_csv(ruta, index=False)
    print(f"  Resultados guardados en {ruta}\n")
    return resumen


# ---------------------------------------------------------------------------
# 2. Comparación con Word2Vec
# ---------------------------------------------------------------------------

def cargar_word2vec():
    """Carga el modelo Word2Vec del compañero si existe y gensim está instalado."""
    if not RUTA_WORD2VEC.exists():
        print(
            f"[Word2Vec] No se encontró {RUTA_WORD2VEC.name}; "
            "se omite la comparación.\n"
        )
        return None

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("[Word2Vec] gensim no está instalado; se omite la comparación.\n")
        return None

    print(f"[Word2Vec] Cargando {RUTA_WORD2VEC}...")
    return KeyedVectors.load(str(RUTA_WORD2VEC))


def _clave_word2vec(w2v, palabra):
    """Busca la palabra en el vocabulario, con o sin acentos."""
    if palabra in w2v.key_to_index:
        return palabra
    sin_acentos = "".join(
        letra
        for letra in unicodedata.normalize("NFD", palabra)
        if unicodedata.category(letra) != "Mn"
    )
    if sin_acentos in w2v.key_to_index:
        return sin_acentos
    return None


def comparar_word2vec(w2v, resumen):
    """Compara la variación contextual de BETO con el vector único de Word2Vec."""
    if w2v is None:
        return

    print("=== 2. COMPARACIÓN CON WORD2VEC ===")
    filas = []

    for _, fila in resumen.iterrows():
        palabra = fila["palabra"]
        clave = _clave_word2vec(w2v, palabra)
        filas.append(
            {
                "palabra": palabra,
                "sim_promedio_beto": fila["sim_promedio_global"],
                "sim_promedio_word2vec": 1.0 if clave else None,
                "en_vocabulario_word2vec": bool(clave),
                "explicacion": (
                    "Word2Vec asigna un único vector a la palabra; por eso la "
                    "similitud consigo misma es 1.0 sin importar el contexto."
                    if clave
                    else "La palabra no está en el vocabulario del modelo."
                ),
            }
        )

    comparacion = pd.DataFrame(filas)
    ruta = DIR_RESULTADOS / "comparacion_word2vec.csv"
    comparacion.to_csv(ruta, index=False)
    print(comparacion.to_string(index=False))
    print(f"  Resultados guardados en {ruta}\n")


# ---------------------------------------------------------------------------
# 3. Buscador semántico
# ---------------------------------------------------------------------------

def preparar_buscador(df, tokenizer, modelo, dispositivo):
    """Genera los embeddings de las reseñas y guarda el índice para buscar."""
    print("=== 3. BUSCADOR SEMÁNTICO ===")

    if N_RESENAS_BUSQUEDA is not None and N_RESENAS_BUSQUEDA < len(df):
        muestra = df.sample(n=N_RESENAS_BUSQUEDA, random_state=42)
    else:
        muestra = df
    muestra = muestra.reset_index(drop=True)

    textos = [str(texto) for texto in muestra["review_text_es"].fillna("")]
    print(f"  Generando embeddings de {len(textos):,} reseñas...")
    embeddings = embeddings_promedio(textos, tokenizer, modelo, dispositivo)

    ruta_embeddings = DIR_RESULTADOS / "embeddings_resenas.npy"
    np.save(ruta_embeddings, embeddings)

    indice = muestra[
        ["movie_id", "rating", "main_genre", "review_summary", "review_text_es"]
    ].copy()
    ruta_indice = DIR_RESULTADOS / "resenas_indice.csv"
    indice.to_csv(ruta_indice, index=False)

    print(f"  Embeddings guardados en {ruta_embeddings}")
    print(f"  Índice guardado en {ruta_indice}\n")
    return embeddings, indice


def buscar_resenas(
    consulta,
    embeddings,
    indice,
    tokenizer,
    modelo,
    dispositivo,
    top_k=TOP_K,
):
    """Devuelve las reseñas más parecidas a una consulta en lenguaje natural."""
    vector_consulta = embeddings_promedio(
        [consulta], tokenizer, modelo, dispositivo
    )[0]
    similitudes = embeddings @ vector_consulta
    orden = np.argsort(-similitudes)[:top_k]

    print(f'  Consulta: "{consulta}"')
    resultados = []

    for posicion in orden:
        fila = indice.iloc[int(posicion)]
        texto = " ".join(str(fila["review_text_es"]).split())
        fragmento = texto[:180] + ("..." if len(texto) > 180 else "")
        print(
            f"    {similitudes[posicion]:.4f} | rating={fila['rating']} "
            f"| género={fila['main_genre']} | {fragmento}"
        )
        resultados.append(
            {
                "similitud": float(similitudes[posicion]),
                "movie_id": fila["movie_id"],
                "rating": fila["rating"],
                "genero": fila["main_genre"],
                "fragmento": fragmento,
            }
        )

    print()
    return resultados


# ---------------------------------------------------------------------------
# 4. Masked Language Model
# ---------------------------------------------------------------------------

def _clasificar_sentimiento(token):
    """Clasifica una predicción con listas sencillas de raíces positivas y negativas."""
    palabra = token.replace("##", "").lower()
    palabra = "".join(
        letra
        for letra in unicodedata.normalize("NFD", palabra)
        if unicodedata.category(letra) != "Mn"
    )
    for raiz in RAICES_POSITIVAS:
        if palabra.startswith(raiz):
            return "positiva"
    for raiz in RAICES_NEGATIVAS:
        if palabra.startswith(raiz):
            return "negativa"
    return "neutra/otra"


def analizar_mlm(tokenizer, dispositivo):
    """Predice la palabra oculta en varias plantillas de reseñas."""
    print("=== 4. MASKED LANGUAGE MODEL ===")
    print("  Cargando modelo de lenguaje enmascarado de BETO...")
    modelo = AutoModelForMaskedLM.from_pretrained(MODELO_BETO)
    modelo.to(dispositivo).eval()

    filas = []
    for plantilla in PLANTILLAS_MLM:
        enc = tokenizer(plantilla, return_tensors="pt")
        entrada = {nombre: tensor.to(dispositivo) for nombre, tensor in enc.items()}
        posiciones = (entrada["input_ids"][0] == tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[0]

        if posiciones.numel() == 0:
            print(f"  Aviso: la plantilla no contiene [MASK]: {plantilla}")
            continue

        posicion = int(posiciones[0])
        with torch.no_grad():
            logits = modelo(**entrada).logits

        probabilidades = F.softmax(logits[0, posicion], dim=-1)
        valores, indices = torch.topk(probabilidades, k=20)

        print(f'  Plantilla: "{plantilla}"')
        mostradas = 0
        for probabilidad, indice in zip(valores.tolist(), indices.tolist()):
            token = tokenizer.convert_ids_to_tokens(indice)
            if token in tokenizer.all_special_tokens:
                continue

            sentimiento = _clasificar_sentimiento(token)
            prediccion = token.replace("##", "")
            filas.append(
                {
                    "plantilla": plantilla,
                    "ranking": mostradas + 1,
                    "prediccion": prediccion,
                    "probabilidad": probabilidad,
                    "sentimiento_estimado": sentimiento,
                }
            )
            print(
                f"    {mostradas + 1}. {prediccion:15s} "
                f"{probabilidad:.4f}  ({sentimiento})"
            )
            mostradas += 1
            if mostradas >= TOP_K:
                break
        print()

    predicciones = pd.DataFrame(filas)
    ruta = DIR_RESULTADOS / "mlm_predicciones.csv"
    predicciones.to_csv(ruta, index=False)
    print(f"  Resultados guardados en {ruta}")
    print(
        "  Nota: BETO no es un clasificador de sentimiento. La columna "
        "'sentimiento_estimado' solo compara con listas de palabras.\n"
    )
    return predicciones


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def _configurar_consola():
    """Fuerza UTF-8 para que los acentos se vean bien en Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def main():
    """Ejecuta los tres análisis en orden y guarda los resultados."""
    _configurar_consola()
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    tokenizer, modelo, dispositivo = cargar_modelo_beto()

    df = cargar_resenas()
    resumen = analizar_polisemia(df, tokenizer, modelo, dispositivo)

    w2v = cargar_word2vec()
    comparar_word2vec(w2v, resumen)

    embeddings, indice = preparar_buscador(df, tokenizer, modelo, dispositivo)
    for consulta in CONSULTAS_EJEMPLO:
        buscar_resenas(consulta, embeddings, indice, tokenizer, modelo, dispositivo)

    del modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    analizar_mlm(tokenizer, dispositivo)
    print("Análisis completo. Revisa los archivos en data/results/beto/.")


if __name__ == "__main__":
    main()
