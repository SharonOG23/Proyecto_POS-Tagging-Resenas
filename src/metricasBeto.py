"""Métricas morfosintácticas por reseña y global, exportadas a JSON."""

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "results" / "comparacion_spacy_nltk.csv"
SALIDA = RAIZ / "data" / "results" / "beto" / "metricas_beto.json"
NEGACIONES = {
    "no", "ni", "nunca", "jamás", "nada", "nadie",
    "ninguno", "ninguna", "sin", "tampoco",
}


def calcular(total, conteo, negaciones):
    """Devuelve las cinco métricas con división segura."""
    return {
        "num_palabras": int(total),
        "densidad_lexica": round(
            (conteo["NOUN"] + conteo["VERB"] + conteo["ADJ"] + conteo["ADV"]) / total, 2
        ) if total else 0.0,
        "ratio_verbos_sustantivos": round(
            conteo["VERB"] / conteo["NOUN"], 2
        ) if conteo["NOUN"] else 0.0,
        "indice_adjetivacion": round(conteo["ADJ"] / total * 100, 1) if total else 0.0,
        "indice_negacion": round(negaciones / total * 100, 1) if total else 0.0,
    }


def main():
    # Lee los tags POS de cada reseña.
    datos = pd.read_csv(ENTRADA)
    registros = []
    total_global = 0
    conteo_global = Counter()
    negaciones_global = 0

    for id_resena, etiquetas in enumerate(datos["pos_tags_spacy"]):
        pares = [
            (palabra, etiqueta)
            for palabra, etiqueta in ast.literal_eval(str(etiquetas))
            if etiqueta not in {"PUNCT", "SPACE"}
        ]
        conteo = Counter(etiqueta for _, etiqueta in pares)
        negaciones = sum(str(palabra).lower() in NEGACIONES for palabra, _ in pares)

        registros.append(
            {"id_resena": id_resena, "metricas": calcular(len(pares), conteo, negaciones)}
        )
        total_global += len(pares)
        conteo_global.update(conteo)
        negaciones_global += negaciones

    # Agrega la entrada global con las mismas fórmulas.
    registros.append(
        {
            "id_resena": "global",
            "metricas": calcular(total_global, conteo_global, negaciones_global),
        }
    )

    # Guarda el array para MongoDB.
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as archivo:
        json.dump(registros, archivo, ensure_ascii=False, indent=2)
    print(f"Guardado: {SALIDA} ({len(registros):,} objetos)")


if __name__ == "__main__":
    main()
