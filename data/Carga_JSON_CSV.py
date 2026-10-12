import json
import pandas as pd
from pathlib import Path


#La clase Carga_Json_CSV permite cargar 2 archivos JSON y luego convertirlo a CSV
#Los datos fueron suministrados del siguiente link:https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset/data?select=IMDB_reviews.json
#Archivo #1: "IMDB_movie_details.json", formato json tradicional {},clave:valor
#Archivo #2: "IMDB_reviews.json", formato json pero con separados por ","
#Nombres de los archivos JSON: "IMDB_movie_details_resumido.json", "IMDB_reviews.json"

class Carga_Json_CSV:
    def __init__(self, input_path: str, csv_output_path: str):
        self.input_path = Path(input_path)
    """
    Clase universal para procesar archivos de IMDb (ya sean CSV disfrazados de JSON
    o archivos JSON reales) y exportarlos limpiamente a formato CSV.
    """

    def __init__(self, input_path: str, csv_output_path: str):
        self.input_path = Path(input_path)
        self.csv_output_path = Path(csv_output_path)
        self.df = None

    def load_file(self):
        """
        Intenta determinar automáticamente el formato del archivo de entrada:
        1. Primero intenta leerlo como JSON real (o JSON Lines).
        2. Si falla, asume que es un archivo con formato CSV y lo lee como tal.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.input_path}")

        # Intento A: Leer como JSON (o JSON Lines)
        try:
            # Intentar lectura estándar de JSON
            try:
                self.df = pd.read_json(self.input_path)
                print(f"[OK] Archivo cargado como JSON estándar: {self.input_path}")
                return
            except ValueError:
                pass

            # Intentar lectura como JSON Lines (.jsonl)
            try:
                self.df = pd.read_json(self.input_path, lines=True)
                print(f"[OK] Archivo cargado como JSON Lines: {self.input_path}")
                return
            except ValueError:
                pass

            # Intentar lectura avanzada si está anidado
            with open(self.input_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, (dict, list)):
                    self.df = pd.json_normalize(data)
                    print(f"[OK] Archivo JSON anidado normalizado: {self.input_path}")
                    return

        except Exception:
            # Si falla todo lo anterior, pasamos al Intento B (CSV)
            pass

        # Intento B: Si no es JSON válido, asumimos que es un CSV (aunque tenga extensión .json)
        try:
            self.df = pd.read_csv(self.input_path, encoding="utf-8")
            print(f"[OK] Archivo cargado como formato CSV: {self.input_path}")
            return
        except Exception as e_csv:
            raise ValueError(
                f"No se pudo procesar el archivo ni como JSON ni como CSV. Error: {e_csv}"
            )

    def export_csv(self):
        """
        Exporta el DataFrame actual a un archivo CSV final.
        """
        if self.df is None or self.df.empty:
            raise ValueError("No hay datos cargados o el DataFrame está vacío.")

        try:
            self.df.to_csv(
                self.csv_output_path,
                index=False,
                encoding="utf-8"
            )
            print(f"[OK] CSV exportado con éxito en: {self.csv_output_path}")
            print(f"[INFO] Total de registros exportados: {len(self.df)}")

        except Exception as e:
            raise RuntimeError(f"Error al exportar el archivo CSV: {e}")

    def process(self):
        """
        Ejecuta el flujo completo: Carga automática y exportación a CSV.
        """
        self.load_file()
        self.export_csv()


# =============================================================================
# Ejemplo de uso para ambos archivos
# =============================================================================

if __name__ == "__main__":
    # 1. Procesar el archivo de reseñas (que internamente era formato CSV)
    print("--- Procesando Reseñas ---")
    converter_reviews = Carga_Json_CSV(
        input_path="IMDB_reviews.json",
        csv_output_path="IMDB_reviews.csv"
    )
    converter_reviews.process()

    print("\n" + "=" * 40 + "\n")

    # 2. Procesar el archivo de detalles (que es JSON real)
    print("--- Procesando Detalles de Películas ---")
    converter_movies = Carga_Json_CSV(
        input_path="IMDB_movie_details_resumido.json",
        csv_output_path="IMDB_movie_details.csv"
    )
    converter_movies.process()