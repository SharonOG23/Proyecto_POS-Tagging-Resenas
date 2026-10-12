import pandas as pd
import os

# Configuración de visualización
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#La clase IMDbDataLoader permite leer archivos tsv.gz obtenidos de la pagina:https://datasets.imdbws.com/
#Dicha pagina ofrece informacion valiosa relacionada con peliculas.

# Fuente oficial:
# https://datasets.imdbws.com/
# Archivos utilizados:
#
# name.basics.tsv.gz
#   Información de personas (actores, actrices, directores, guionistas, etc.).
#
# title.akas.tsv.gz
#   Títulos alternativos y traducciones de películas y series.
#
# title.basics.tsv.gz
#   Información principal de películas, series y episodios:
#   título, tipo, año de estreno, duración y géneros.
#
# title.crew.tsv.gz
#   Directores y guionistas asociados a cada título.
#
# title.episode.tsv.gz
#   Relación entre episodios y series, incluyendo temporada y número.
#
# title.principals.tsv.gz
#   Participantes principales de cada título (actores, directores,
#   productores y otros roles relevantes).
#
# title.ratings.tsv.gz
#   Calificaciones de IMDb: puntuación promedio y cantidad de votos.
#
# Claves principales:
#   tconst -> Identificador único del título.
#   nconst -> Identificador único de la persona.

#La clase tiene los siguientes metodos:
#Cargar_dataset
#
#convertir_tsv_a_csv
#
#convertir_tsv_grande_a_csv
#
#convertir_todos

class IMDbDataLoader:

    def __init__(self):
        """
        Utiliza automáticamente la carpeta raíz del proyecto.
        """
        self.ruta = os.getcwd()
        print(f"Ruta de trabajo: {self.ruta}")

    def cargar_dataset(self, archivo):
        """
        Carga un dataset TSV de IMDb y retorna un DataFrame.
        """

        ruta_completa = os.path.join(self.ruta, archivo)

        df = pd.read_csv(
            ruta_completa,
            sep="\t",
            low_memory=False
        )

        return df

    def convertir_tsv_a_csv(self, archivo_tsv):
        """
        Convierte un archivo TSV.GZ a CSV.
        """

        ruta_completa = os.path.join(
            self.ruta,
            archivo_tsv
        )

        print(f"Cargando {archivo_tsv}...")

        df = pd.read_csv(
            ruta_completa,
            sep="\t",
            low_memory=False
        )

        nombre_csv = archivo_tsv.replace(
            ".tsv.gz",
            ".csv"
        )

        ruta_csv = os.path.join(
            self.ruta,
            nombre_csv
        )

        df.to_csv(
            ruta_csv,
            index=False,
            encoding="utf-8"
        )

        print(f"CSV generado: {nombre_csv}")

    def convertir_tsv_grande_a_csv(
            self,
            archivo_tsv,
            chunksize=100000
    ):
        """
        Convierte archivos grandes por bloques.
        """

        ruta_completa = os.path.join(
            self.ruta,
            archivo_tsv
        )

        nombre_csv = archivo_tsv.replace(
            ".tsv.gz",
            ".csv"
        )

        ruta_csv = os.path.join(
            self.ruta,
            nombre_csv
        )

        print(f"Procesando {archivo_tsv}...")

        primer_bloque = True

        for chunk in pd.read_csv(
                ruta_completa,
                sep="\t",
                chunksize=chunksize,
                low_memory=False
        ):

            chunk.to_csv(
                ruta_csv,
                mode='w' if primer_bloque else 'a',
                header=primer_bloque,
                index=False,
                encoding='utf-8'
            )

            primer_bloque = False

        print(f"CSV generado: {nombre_csv}")

    def convertir_todos(self):
        """
        Convierte todos los datasets IMDb a CSV.
        """

        archivos = [
            "name.basics.tsv.gz",
            "title.akas.tsv.gz",
            "title.basics.tsv.gz",
            "title.crew.tsv.gz",
            "title.episode.tsv.gz",
            "title.principals.tsv.gz",
            "title.ratings.tsv.gz"
        ]

        for archivo in archivos:

            try:

                # Archivos muy grandes
                if archivo in [
                    "name.basics.tsv.gz",
                    "title.akas.tsv.gz",
                    "title.principals.tsv.gz"
                ]:

                    self.convertir_tsv_grande_a_csv(
                        archivo
                    )

                else:

                    self.convertir_tsv_a_csv(
                        archivo
                    )

            except Exception as e:

                print(
                    f"Error procesando {archivo}: {e}"
                )

# ==========================================
# INSTANCIA DE LA CLASE
# ==========================================

loader = IMDbDataLoader()

# Convierte todos los archivos TSV.GZ a CSV
loader.convertir_todos()

print("Proceso finalizado correctamente.")

