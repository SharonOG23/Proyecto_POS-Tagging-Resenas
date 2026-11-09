"""
POS Tagging en español utilizando spaCy.

Entrada:
    data/processed/resenas_es.csv

Salida:
    data/results/pos_analisis_spacy.csv

Genera un registro por token.
"""

from pathlib import Path
import pandas as pd
import spacy

# =====================================================
# RUTAS DEL PROYECTO
# =====================================================

# Proyecto_POS-Tagging-Resenas/
RAIZ = Path(__file__).resolve().parent.parent

# Archivo de entrada
CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv"

# Archivo de salida
CSV_SALIDA = RAIZ / "data" / "results" / "comparacion_spacy_nltk.csv"

# Crear carpeta de resultados si no existe
CSV_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Mostrar rutas utilizadas
print("\n" + "=" * 60)
print("CONFIGURACIÓN DE RUTAS")
print("=" * 60)
print("RAIZ         :", RAIZ)
print("CSV_ENTRADA  :", CSV_ENTRADA)
print("CSV_SALIDA   :", CSV_SALIDA)
print("EXISTE CSV   :", CSV_ENTRADA.exists())
print("=" * 60)

if not CSV_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo:\n{CSV_ENTRADA}"
    )


class SpacyPOSTagging:
    """
    Procesamiento POS Tagging utilizando spaCy.
    """

    def __init__(
        self,
        ruta_entrada=CSV_ENTRADA,
        ruta_salida=CSV_SALIDA,
        columna="review_summary_es",
        modelo="es_core_news_md"
    ):

        self.ruta_entrada = Path(ruta_entrada)
        self.ruta_salida = Path(ruta_salida)
        self.columna = columna

        print("Cargando modelo spaCy...")

        try:
            self.nlp = spacy.load(modelo)
        except OSError:
            raise OSError(
                f"""
No se pudo cargar el modelo '{modelo}'.

Instálalo con:

python -m spacy download {modelo}
"""
            )

    def cargar_datos(self):

        if not self.ruta_entrada.exists():
            raise FileNotFoundError(
                f"No existe el archivo:\n{self.ruta_entrada}"
            )

        df = pd.read_csv(self.ruta_entrada)

        if self.columna not in df.columns:
            raise KeyError(
                f"No existe la columna '{self.columna}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        df[self.columna] = (
            df[self.columna]
            .fillna("")
            .astype(str)
        )

        return df

    def procesar_review(self, texto, review_id):

        doc = self.nlp(str(texto))

        resultado = []

        for token in doc:

            resultado.append(
                {
                    "review_id": review_id,
                    "token": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "is_stop": token.is_stop,
                    "is_punct": token.is_punct,
                    "is_alpha": token.is_alpha,
                }
            )

        return resultado

    def ejecutar(self):

        print("Leyendo archivo...")

        df = self.cargar_datos()

        filas = []

        total_reviews = len(df)

        print(f"Reseñas encontradas: {total_reviews:,}")

        for idx, row in df.iterrows():

            texto = row[self.columna]

            filas.extend(
                self.procesar_review(
                    texto=texto,
                    review_id=idx
                )
            )

            if (idx + 1) % 100 == 0:
                print(
                    f"Procesadas {idx + 1:,}/{total_reviews:,} reseñas"
                )

        df_resultado = pd.DataFrame(filas)

        print(f"Total registros generados: {len(df_resultado):,}")

        df_resultado.to_csv(
            self.ruta_salida,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nProceso completado")
        print("Archivo generado:")
        print(self.ruta_salida.resolve())

        print(
            "¿Archivo existe?:",
            self.ruta_salida.exists()
        )

        if self.ruta_salida.exists():
            print(
                "Tamaño (bytes):",
                self.ruta_salida.stat().st_size
            )

        return df_resultado


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("POS TAGGING CON SPACY")
    print("=" * 60)

    print("Entrada :", CSV_ENTRADA)
    print("Salida  :", CSV_SALIDA)

    print("=" * 60)

    pipeline = SpacyPOSTagging(
        ruta_entrada=CSV_ENTRADA,
        ruta_salida=CSV_SALIDA
    )

    resultado = pipeline.ejecutar()

    print("\nPrimeros registros:")

    print(resultado.head())