"""
Pipeline NLP en español utilizando spaCy

Pasos:
1. Tokenización
2. POS Tagging
3. Stopwords
4. NER
5. Minúsculas
6. Lematización

Entrada:
    data/processed/resenas_es.csv

Salida:
    data/results/pos_analisis_spacy.csv
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import spacy
from tqdm.auto import tqdm

# =====================================================
# RUTAS DEL PROYECTO
# =====================================================

RAIZ = Path(__file__).resolve().parent.parent

CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv"

CSV_SALIDA = RAIZ / "data" / "results" / "pos_analisis_spacy.csv"

CSV_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


class POS_Tagging_Spacy:

    def __init__(self, modelo="es_core_news_md"):

        self.columna = "review_summary_es"

        try:
            self.nlp = spacy.load(modelo)

        except OSError:

            print(f"Modelo {modelo} no encontrado.")
            print("Descargando modelo...")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spacy",
                    "download",
                    modelo,
                ],
                check=True,
            )

            self.nlp = spacy.load(modelo)

        print(f"Modelo cargado: {modelo}")

    # =====================================================
    # PROCESAMIENTO DE UNA RESEÑA
    # =====================================================

    def procesar_texto(self, texto):

        doc = self.nlp(str(texto))

        # 1 TOKENIZACIÓN
        tokens = [
            token.text
            for token in doc
        ]

        # 2 POS TAGGING
        pos_tags = [
            (token.text, token.pos_)
            for token in doc
        ]

        # 3 STOPWORDS
        tokens_validos = [
            token
            for token in doc
            if not token.is_stop
            and not token.is_punct
        ]

        tokens_sin_stopwords = [
            token.text
            for token in tokens_validos
        ]

        pos_sin_stopwords = [
            (token.text, token.pos_)
            for token in tokens_validos
        ]

        # 4 NER
        entidades = [
            (ent.text, ent.label_)
            for ent in doc.ents
        ]

        # 5 MINÚSCULAS
        tokens_minusculas = [
            token.text.lower()
            for token in tokens_validos
        ]

        # 6 LEMATIZACIÓN
        lemas = [
            token.lemma_.lower()
            for token in tokens_validos
        ]

        texto_lematizado = " ".join(lemas)

        return {
            "tokens": str(tokens),
            "pos_tags": str(pos_tags),
            "tokens_sin_stopwords": str(tokens_sin_stopwords),
            "pos_sin_stopwords": str(pos_sin_stopwords),
            "entidades": str(entidades),
            "tokens_minusculas": str(tokens_minusculas),
            "lemas": str(lemas),
            "texto_lematizado": texto_lematizado
        }

    # =====================================================
    # PROCESAMIENTO DEL CSV
    # =====================================================

    def etiquetar_csv(
        self,
        ruta_entrada=CSV_ENTRADA,
        ruta_salida=CSV_SALIDA,
    ):

        ruta_entrada = Path(ruta_entrada)
        ruta_salida = Path(ruta_salida)

        if not ruta_entrada.exists():

            raise FileNotFoundError(
                f"No se encontró el archivo:\n{ruta_entrada}"
            )

        print(f"\nLeyendo archivo:\n{ruta_entrada}")

        df = pd.read_csv(
            ruta_entrada,
            encoding="utf-8-sig",
            keep_default_na=False
        )

        print(f"Reseñas encontradas: {len(df):,}")

        if self.columna not in df.columns:

            raise ValueError(
                f"No existe la columna '{self.columna}'"
            )

        resultados = []

        textos = (
            df[self.columna]
            .fillna("")
            .astype(str)
            .tolist()
        )

        for texto in tqdm(
            textos,
            desc="Procesando reseñas"
        ):

            resultados.append(
                self.procesar_texto(texto)
            )

        df_resultados = pd.DataFrame(resultados)

        # Mantener TODAS las columnas originales
        df_final = pd.concat(
            [df, df_resultados],
            axis=1
        )

        ruta_salida.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df_final.to_csv(
            ruta_salida,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 70)
        print("ARCHIVO GENERADO")
        print("=" * 70)
        print(ruta_salida.resolve())
        print("Existe:", ruta_salida.exists())

        if ruta_salida.exists():
            print(
                "Tamaño:",
                ruta_salida.stat().st_size,
                "bytes"
            )

        return df_final


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=" * 70)
    print("POS TAGGING CON SPACY")
    print("=" * 70)

    print("Entrada:")
    print(CSV_ENTRADA)

    print("\nSalida:")
    print(CSV_SALIDA)

    print("=" * 70)

    pipeline = POS_Tagging_Spacy()

    resultado = pipeline.etiquetar_csv()

    print("\nPrimeras filas:")

    print(
        resultado[
            [
                "review_summary_es",
                "pos_tags",
                "entidades",
                "lemas"
            ]
        ].head()
    )