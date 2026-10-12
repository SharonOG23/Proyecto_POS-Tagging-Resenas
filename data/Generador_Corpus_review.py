import gc
import re
from datetime import datetime
import pandas as pd

#Archivos CSV que utilizara la clase:
#IMDB_movie_details.csv
#IMDB_reviews.csv
#title.basics.csv

# =============================================================
# CLASE Generador_Corpus_review - DESCRIPCION DE LOS METODOS
# =============================================================
# __init__                          : Guarda las rutas de los tres CSV e inicializa los DataFrames, el log y los atributos de géneros.
# cargar_archivos                   : Lee las reseñas, los detalles y title.basics.csv (por bloques, solo las películas del corpus).
# filtrar_movie_id_validos          : Elimina registros cuyo movie_id no tenga el formato tt + 7 dígitos.
# unir_reviews_detalles             : Une reseñas, detalles y primaryTitle por movie_id y libera la memoria de origen.
# formatear_review_date             : Convierte review_date al formato YYYY-MM-DD, sin depender del idioma del sistema.
# eliminar_user_id                  : Elimina la columna user_id.
# usar_primer_genero                : Deja solo el primer género de la columna genre.
# eliminar_caracteres_especiales    : Quita etiquetas HTML, símbolos raros y espacios repetidos de las columnas de texto.
# recortar_plot_summary_en_punto    : Corta plot_summary en el primer "." y elimina los resúmenes de menos de 30 caracteres.
# verificar_nulos_y_blancos         : Elimina las filas que tengan algún valor nulo o vacío en cualquier columna.
# recortar_review_summary_en_punto  : Corta review_summary en el primer "." y elimina los de menos de 30 caracteres.
# filtrar_longitud_review_summary   : Elimina las filas cuyo review_summary tenga menos de 30 caracteres.
# generar_estadisticas              : Calcula e imprime estadísticas del corpus (registros, películas únicas y rating).
# ordenar_columnas                  : Reordena las columnas según ORDEN_COLUMNAS.
# excluir_columnas_salida           : Quita review_text, plot_synopsis y movie_rating antes de exportar.
# reducir_dataset                   : Reduce a 10,000 filas con 6 géneros de al menos 500 reseñas e informa cuáles se mantienen y cuáles se eliminan.
# guardar_corpus                    : Quita saltos de línea de los textos y guarda el corpus en un CSV.
# resumen_eliminaciones             : Muestra una tabla con todas las eliminaciones registradas en cada paso.
# _registrar                        : Agrega al log una entrada con el paso, motivo y filas eliminadas.
# _verificar_carga                  : Verifica que los archivos ya estén cargados antes de filtrar o unir.
# _corpus_o_error                   : Devuelve el corpus o lanza un error si todavía no se ha creado.
# =============================================================


class Generador_Corpus_review:
    COLUMNA_RESUMEN = "review_summary"

    LONGITUD_MINIMA_RESUMEN = 30
    LONGITUD_MINIMA_PLOT_SUMMARY = 30

    COLUMNAS_TEXTO = [
        "review_text",
        "review_summary",
        "plot_summary",
        "plot_synopsis"
    ]

    ORDEN_COLUMNAS = [
        "movie_id",
        "primaryTitle",
        "review_date",
        "review_text",
        "review_summary",
        "rating",
        "is_spoiler",
        "genre",
        "duration",
        "release_date",
        "movie_rating",
        "plot_summary",
        "plot_synopsis"
    ]

    # Parámetros para reducir el dataset
    TOTAL_REGISTROS = 10000
    NUM_GENEROS = 6
    MINIMO_POR_GENERO = 500

    # Columnas que NO se incluyen en el archivo exportado
    COLUMNAS_EXCLUIR_SALIDA = [
        "review_text",
        "plot_synopsis",
        "movie_rating"
    ]

    # Estructura de title.basics.csv (el archivo no trae encabezado)
    COLUMNAS_BASICS = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "isAdult",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "genres"
    ]

    # Meses en inglés -> número (independiente del idioma de Windows)
    MESES = {
        "January": "01", "February": "02", "March": "03",
        "April": "04", "May": "05", "June": "06",
        "July": "07", "August": "08", "September": "09",
        "October": "10", "November": "11", "December": "12"
    }

    def __init__(
            self,
            ruta_reviews: str,
            ruta_detalles: str,
            ruta_basics: str = "title.basics.csv"
    ):
        self.ruta_reviews = ruta_reviews
        self.ruta_detalles = ruta_detalles
        self.ruta_basics = ruta_basics

        self.df_reviews = None
        self.df_detalles = None
        self.df_basics = None
        self.df_corpus = None
        self.estadisticas = None
        self.generos_mantenidos = {}
        self.generos_eliminados = {}

        self.log_eliminaciones = []

    # ---------------------------------------------------------
    # CARGAR CSV
    # ---------------------------------------------------------
    def cargar_archivos(self):

        self.df_reviews = pd.read_csv(
            self.ruta_reviews,
            dtype=str,
            encoding="utf-8-sig"
        )

        self.df_detalles = pd.read_csv(
            self.ruta_detalles,
            dtype=str,
            encoding="utf-8-sig"
        )

        print(
            f"[cargar] {self.ruta_reviews}: "
            f"{len(self.df_reviews):,} filas"
        )

        print(
            f"[cargar] {self.ruta_detalles}: "
            f"{len(self.df_detalles):,} filas"
        )

        # title.basics.csv: sin encabezado. Se lee por bloques y solo se
        # conservan las películas que aparecen en las reseñas
        # (evita cargar los ~12.8 millones de filas en memoria).
        ids = set(
            self.df_reviews["movie_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        partes = []
        leidas = 0

        for bloque in pd.read_csv(
                self.ruta_basics,
                dtype=str,
                header=None,
                names=self.COLUMNAS_BASICS,
                usecols=["tconst", "primaryTitle"],
                encoding="utf-8-sig",
                chunksize=1_000_000
        ):
            leidas += len(bloque)
            partes.append(
                bloque[bloque["tconst"].isin(ids)]
            )

        self.df_basics = pd.concat(
            partes,
            ignore_index=True
        )

        print(
            f"[cargar] {self.ruta_basics}: "
            f"{leidas:,} filas leídas, "
            f"{len(self.df_basics):,} coinciden con las reseñas"
        )

        return self

    # ---------------------------------------------------------
    # FILTRAR MOVIE_ID
    # ---------------------------------------------------------
    def filtrar_movie_id_validos(self):

        self._verificar_carga()

        patron = r"^tt\d{7}$"

        antes_reviews = len(self.df_reviews)

        filtro_reviews = (
            self.df_reviews["movie_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.match(patron, na=False)
        )

        eliminados_reviews = (
                antes_reviews -
                int(filtro_reviews.sum())
        )

        self.df_reviews = (
            self.df_reviews[filtro_reviews]
            .reset_index(drop=True)
        )

        antes_detalles = len(self.df_detalles)

        filtro_detalles = (
            self.df_detalles["movie_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.match(patron, na=False)
        )

        eliminados_detalles = (
                antes_detalles -
                int(filtro_detalles.sum())
        )

        self.df_detalles = (
            self.df_detalles[filtro_detalles]
            .reset_index(drop=True)
        )

        total = (
                eliminados_reviews +
                eliminados_detalles
        )

        self._registrar(
            "filtrar_movie_id",
            "movie_id",
            "movie_id inválido (tt + 7 dígitos)",
            total,
            antes_reviews + antes_detalles
        )

        print(
            f"[filtrar_movie_id] reviews eliminados: "
            f"{eliminados_reviews:,}"
        )

        print(
            f"[filtrar_movie_id] detalles eliminados: "
            f"{eliminados_detalles:,}"
        )

        return self

    # ---------------------------------------------------------
    # UNIR ARCHIVOS
    # ---------------------------------------------------------
    def unir_reviews_detalles(self):

        self._verificar_carga()

        reviews = self.df_reviews.copy()
        detalles = self.df_detalles.copy()

        reviews["movie_id"] = (
            reviews["movie_id"]
            .astype(str)
            .str.strip()
        )

        detalles["movie_id"] = (
            detalles["movie_id"]
            .astype(str)
            .str.strip()
        )

        detalles = detalles.rename(
            columns={
                "rating": "movie_rating"
            }
        )

        detalles = detalles.drop_duplicates(
            subset=["movie_id"]
        )

        self.df_corpus = reviews.merge(
            detalles,
            on="movie_id",
            how="left"
        )

        # primaryTitle: se agrega cuando movie_id == tconst
        basics = (
            self.df_basics[["tconst", "primaryTitle"]]
            .copy()
        )

        basics["tconst"] = (
            basics["tconst"]
            .astype(str)
            .str.strip()
        )

        # Solo se conservan los títulos de las películas del corpus
        basics = basics[
            basics["tconst"].isin(self.df_corpus["movie_id"])
        ].drop_duplicates(subset=["tconst"])

        self.df_corpus = self.df_corpus.merge(
            basics,
            left_on="movie_id",
            right_on="tconst",
            how="left"
        ).drop(columns=["tconst"])

        print(
            f"[unir_reviews_detalles] "
            f"{len(self.df_corpus):,} filas"
        )

        # Se libera memoria: ya no se necesitan los archivos de origen
        del reviews, detalles, basics
        self.df_reviews = None
        self.df_detalles = None
        self.df_basics = None
        gc.collect()

        return self

    # ---------------------------------------------------------
    # FORMATEAR FECHA
    # ---------------------------------------------------------
    def formatear_review_date(self):

        df = self._corpus_o_error().copy()

        # Se reemplazan los nombres de mes en inglés por su número
        # para no depender del idioma (locale) del sistema.
        texto = (
            df["review_date"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for nombre, numero in self.MESES.items():
            texto = texto.str.replace(
                nombre,
                numero,
                regex=False
            )

        fechas = pd.to_datetime(
            texto,
            format="%d %m %Y",
            errors="coerce"
        )

        df["review_date"] = (
            fechas.dt.strftime("%Y-%m-%d")
        )

        self.df_corpus = df

        print(
            f"[formatear_review_date] fecha convertida "
            f"({int(fechas.isna().sum()):,} fechas no válidas)"
        )

        return self

    # ---------------------------------------------------------
    # ELIMINAR USER_ID
    # ---------------------------------------------------------
    def eliminar_user_id(self):

        df = self._corpus_o_error()

        if "user_id" not in df.columns:
            return self

        self.df_corpus = df.drop(
            columns=["user_id"]
        )

        print(
            "[eliminar_user_id] columna eliminada"
        )

        return self

    # ---------------------------------------------------------
    # USAR SOLO EL PRIMER GENERO
    # ---------------------------------------------------------
    def usar_primer_genero(self):

        df = self._corpus_o_error().copy()

        if "genre" not in df.columns:
            return self

        df["genre"] = (
            df["genre"]
            .fillna("")
            .str.replace(r"[\[\]]", "", regex=True)
            .str.replace("'", "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
        )

        self.df_corpus = df

        print(
            "[usar_primer_genero] proceso completado"
        )

        return self

    # ---------------------------------------------------------
    # ELIMINAR CARACTERES ESPECIALES
    # ---------------------------------------------------------
    def eliminar_caracteres_especiales(self):

        df = self._corpus_o_error()

        for col in self.COLUMNAS_TEXTO:

            if col not in df.columns:
                continue

            # Se limpian solo los textos únicos (plot_summary y
            # plot_synopsis se repiten en cada reseña de la misma
            # película) y por bloques, para no agotar la memoria.
            unicos = pd.Series(
                df[col].dropna().unique(),
                dtype="object"
            )

            limpios = []

            for inicio in range(0, len(unicos), 50_000):
                parte = unicos.iloc[inicio:inicio + 50_000]

                limpios.append(
                    parte
                    .astype(str)
                    .str.replace(
                        r"<[^>]+>",
                        " ",
                        regex=True
                    )
                    .str.replace(
                        r"[^A-Za-zÀ-ÿ0-9\s.,;:!?'\-]",
                        " ",
                        regex=True
                    )
                    .str.replace(
                        r"\s+",
                        " ",
                        regex=True
                    )
                    .str.strip()
                )

            if limpios:
                limpios = pd.concat(limpios)
            else:
                limpios = pd.Series([], dtype="object")

            mapa = dict(zip(unicos, limpios))

            df[col] = (
                df[col]
                .map(mapa)
                .fillna("")
            )

            del unicos, limpios, mapa
            gc.collect()

        self.df_corpus = df

        print(
            "[eliminar_caracteres_especiales] proceso completado"
        )

        return self

    # ---------------------------------------------------------
    # RECORTAR PLOT SUMMARY EN EL PRIMER PUNTO
    # ---------------------------------------------------------
    def recortar_plot_summary_en_punto(self):

        df = self._corpus_o_error().copy()

        if "plot_summary" not in df.columns:
            return self

        # Se conserva solo el texto anterior al primer "."
        # (si no hay punto, el texto queda igual)
        # Se recorta sobre los textos únicos (se repiten por película)
        unicos = df["plot_summary"].dropna().unique()

        mapa = {
            texto: str(texto).split(".", 1)[0].strip()
            for texto in unicos
        }

        df["plot_summary"] = (
            df["plot_summary"]
            .map(mapa)
            .fillna("")
        )

        # Se eliminan los resúmenes con menos de 30 caracteres
        longitud = df["plot_summary"].str.len()

        mascara = (
                longitud >=
                self.LONGITUD_MINIMA_PLOT_SUMMARY
        )

        eliminadas = int((~mascara).sum())

        self._registrar(
            "recortar_plot_summary",
            "plot_summary",
            f"menos de {self.LONGITUD_MINIMA_PLOT_SUMMARY} caracteres "
            f"tras recortar en el primer punto",
            eliminadas,
            len(df)
        )

        self.df_corpus = (
            df[mascara]
            .reset_index(drop=True)
        )

        print(
            f"[recortar_plot_summary_en_punto] "
            f"{eliminadas:,} filas eliminadas"
        )

        return self

    # ---------------------------------------------------------
    # ELIMINAR NULOS Y BLANCOS
    # ---------------------------------------------------------
    def verificar_nulos_y_blancos(self):

        df = self._corpus_o_error()

        filas_iniciales = len(df)

        for col in df.columns:

            mascara = (
                    df[col].isna()
                    |
                    df[col]
                    .astype(str)
                    .str.strip()
                    .eq("")
            )

            eliminadas = int(mascara.sum())

            if eliminadas > 0:
                self._registrar(
                    "nulos_blancos",
                    col,
                    "valor nulo o vacío",
                    eliminadas,
                    len(df)
                )

                df = df[~mascara]

        self.df_corpus = (
            df.reset_index(drop=True)
        )

        print(
            f"[verificar_nulos_y_blancos] "
            f"{filas_iniciales - len(self.df_corpus):,} filas eliminadas"
        )

        return self

    # ---------------------------------------------------------
    # RECORTAR REVIEW SUMMARY EN EL PRIMER PUNTO
    # ---------------------------------------------------------
    def recortar_review_summary_en_punto(self):

        df = self._corpus_o_error().copy()

        # Se conserva solo el texto anterior al primer "."
        # (si no hay punto, el texto queda igual)
        df[self.COLUMNA_RESUMEN] = (
            df[self.COLUMNA_RESUMEN]
            .fillna("")
            .astype(str)
            .str.split(".", n=1)
            .str[0]
            .str.strip()
        )

        longitud = df[self.COLUMNA_RESUMEN].str.len()

        mascara = (
                longitud >=
                self.LONGITUD_MINIMA_RESUMEN
        )

        eliminadas = int((~mascara).sum())

        self._registrar(
            "recortar_review_summary",
            self.COLUMNA_RESUMEN,
            f"menos de {self.LONGITUD_MINIMA_RESUMEN} caracteres "
            f"tras recortar en el primer punto",
            eliminadas,
            len(df)
        )

        self.df_corpus = (
            df[mascara]
            .reset_index(drop=True)
        )

        print(
            f"[recortar_review_summary_en_punto] "
            f"{eliminadas:,} filas eliminadas"
        )

        return self

    # ---------------------------------------------------------
    # LONGITUD MINIMA REVIEW SUMMARY
    # ---------------------------------------------------------
    def filtrar_longitud_review_summary(self):

        df = self._corpus_o_error()

        longitud = (
            df[self.COLUMNA_RESUMEN]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.len()
        )

        mascara = (
                longitud >=
                self.LONGITUD_MINIMA_RESUMEN
        )

        eliminadas = int((~mascara).sum())

        self._registrar(
            "longitud_review_summary",
            self.COLUMNA_RESUMEN,
            f"menos de {self.LONGITUD_MINIMA_RESUMEN} caracteres",
            eliminadas,
            len(df)
        )

        self.df_corpus = (
            df[mascara]
            .reset_index(drop=True)
        )

        print(
            f"[filtrar_longitud_review_summary] "
            f"{eliminadas:,} filas eliminadas"
        )

        return self

    # ---------------------------------------------------------
    # ESTADISTICAS
    # ---------------------------------------------------------
    def generar_estadisticas(self):

        df = self._corpus_o_error()

        rating = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        estadisticas = {
            "total_registros": len(df),
            "total_columnas": len(df.columns),
            "peliculas_unicas": int(
                df["movie_id"].nunique()
            ),
            "rating_promedio": round(
                float(rating.mean()),
                2
            ),
            "rating_mediana": float(
                rating.median()
            ),
            "rating_min": float(
                rating.min()
            ),
            "rating_max": float(
                rating.max()
            ),
            "rating_desviacion": round(
                float(rating.std()),
                2
            ),
        }

        print(
            "\n=== ESTADISTICAS DEL CORPUS ==="
        )

        for k, v in estadisticas.items():
            print(f"{k}: {v}")

        # Se guarda en el objeto y se devuelve self para no romper la cadena
        self.estadisticas = estadisticas

        return self

    # ---------------------------------------------------------
    # ORDENAR COLUMNAS
    # ---------------------------------------------------------
    def ordenar_columnas(self):

        df = self._corpus_o_error()

        existentes = [
            c
            for c in self.ORDEN_COLUMNAS
            if c in df.columns
        ]

        restantes = [
            c
            for c in df.columns
            if c not in existentes
        ]

        self.df_corpus = (
            df[existentes + restantes]
        )

        print(
            "[ordenar_columnas] proceso completado"
        )

        return self

    # ---------------------------------------------------------
    # EXCLUIR COLUMNAS DE LA SALIDA
    # ---------------------------------------------------------
    def excluir_columnas_salida(self):

        df = self._corpus_o_error()

        a_eliminar = [
            c
            for c in self.COLUMNAS_EXCLUIR_SALIDA
            if c in df.columns
        ]

        self.df_corpus = df.drop(
            columns=a_eliminar
        )

        print(
            f"[excluir_columnas_salida] "
            f"columnas omitidas: {a_eliminar}"
        )

        return self

    # ---------------------------------------------------------
    # REDUCIR DATASET POR GENERO
    # ---------------------------------------------------------
    def reducir_dataset(
            self,
            total=TOTAL_REGISTROS,
            num_generos=NUM_GENEROS,
            minimo_por_genero=MINIMO_POR_GENERO,
            semilla=42
    ):

        df = self._corpus_o_error().copy()

        if "genre" not in df.columns:
            raise RuntimeError(
                "La columna 'genre' no existe en el corpus."
            )

        conteo = df["genre"].value_counts()

        # Géneros con al menos el mínimo de reseñas
        elegibles = conteo[conteo >= minimo_por_genero]

        # Se mantienen los num_generos con más reseñas
        mantenidos = elegibles.head(num_generos)

        if mantenidos.empty:
            raise RuntimeError(
                f"Ningún género tiene al menos "
                f"{minimo_por_genero:,} reseñas."
            )

        if len(mantenidos) < num_generos:
            print(
                f"[reducir_dataset] ADVERTENCIA: solo "
                f"{len(mantenidos)} géneros cumplen el mínimo de "
                f"{minimo_por_genero:,} reseñas (se pedían "
                f"{num_generos})."
            )

        # Géneros eliminados y su motivo
        eliminados = {}

        for genero, cantidad in conteo.items():
            if genero in mantenidos.index:
                continue

            if cantidad < minimo_por_genero:
                motivo = (
                    f"menos de {minimo_por_genero:,} reseñas"
                )
            else:
                motivo = (
                    f"fuera de los {num_generos} géneros "
                    f"con más reseñas"
                )

            eliminados[genero] = (int(cantidad), motivo)

        # Se registran y eliminan los géneros descartados
        actual = len(df)

        for genero, (cantidad, motivo) in eliminados.items():
            self._registrar(
                "reducir_dataset",
                "genre",
                f"género '{genero}': {motivo}",
                cantidad,
                actual
            )
            actual -= cantidad

        df = df[df["genre"].isin(mantenidos.index)]

        # Reparto equilibrado entre géneros (cada uno conserva su
        # mínimo; si un género tiene pocas reseñas, el resto
        # se redistribuye entre los demás)
        disponibles = {
            g: int(c) for g, c in mantenidos.items()
        }
        asignado = {g: 0 for g in disponibles}
        pendientes = list(disponibles)
        restante = total

        while restante > 0 and pendientes:

            cuota = max(restante // len(pendientes), 1)

            for g in list(pendientes):

                if restante <= 0:
                    break

                espacio = disponibles[g] - asignado[g]
                dar = min(cuota, espacio, restante)

                asignado[g] += dar
                restante -= dar

                if asignado[g] >= disponibles[g]:
                    pendientes.remove(g)

        partes = [
            df[df["genre"] == g].sample(
                n=asignado[g],
                random_state=semilla
            )
            for g in disponibles
        ]

        reducido = (
            pd.concat(partes)
            .sample(frac=1, random_state=semilla)
            .reset_index(drop=True)
        )

        recortadas = len(df) - len(reducido)

        self._registrar(
            "reducir_dataset",
            "genre",
            f"muestreo a {total:,} registros",
            recortadas,
            len(df)
        )

        self.df_corpus = reducido
        self.generos_mantenidos = dict(asignado)
        self.generos_eliminados = eliminados

        print("\n=== REDUCCION DEL DATASET ===")
        print(f"Registros finales: {len(reducido):,}")

        print("\nGéneros que se mantienen:")
        for g, n in asignado.items():
            print(
                f"  - {g}: {n:,} reseñas "
                f"(disponibles: {disponibles[g]:,})"
            )

        print("\nGéneros que se eliminan:")
        if not eliminados:
            print("  (ninguno)")
        for g, (cantidad, motivo) in eliminados.items():
            print(f"  - {g}: {cantidad:,} reseñas ({motivo})")

        return self

    # ---------------------------------------------------------
    # GUARDAR CORPUS
    # ---------------------------------------------------------
    def guardar_corpus(
            self,
            ruta_salida="corpus_peliculas.csv"
    ):

        df = self._corpus_o_error().copy()

        for col in df.columns:

            if pd.api.types.is_string_dtype(df[col]):
                df[col] = (
                    df[col]
                    .str.replace(
                        r"[\r\n]+",
                        " ",
                        regex=True
                    )
                )

        df.to_csv(
            ruta_salida,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"[guardar_corpus] "
            f"{ruta_salida}"
        )

        return ruta_salida

    # ---------------------------------------------------------
    # RESUMEN ELIMINACIONES
    # ---------------------------------------------------------
    def resumen_eliminaciones(self):

        log = pd.DataFrame(
            self.log_eliminaciones
        )

        print(
            "\n=== RESUMEN ELIMINACIONES ==="
        )

        if log.empty:
            print("No existen registros.")
            return log

        print(
            log.to_string(index=False)
        )

        return log

    # ---------------------------------------------------------
    # AUXILIAR REGISTRO
    # ---------------------------------------------------------
    def _registrar(
            self,
            paso,
            columna,
            motivo,
            eliminadas,
            filas_antes
    ):

        self.log_eliminaciones.append(
            {
                "fecha_hora": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "paso": paso,
                "columna": columna,
                "motivo": motivo,
                "filas_eliminadas": eliminadas,
                "filas_antes": filas_antes,
                "filas_despues": (
                        filas_antes - eliminadas
                )
            }
        )

    # ---------------------------------------------------------
    # VALIDAR CARGA
    # ---------------------------------------------------------
    def _verificar_carga(self):

        if self.df_reviews is None:
            raise RuntimeError(
                "Primero ejecute cargar_archivos()."
            )

        if self.df_detalles is None:
            raise RuntimeError(
                "Primero ejecute cargar_archivos()."
            )

        if self.df_basics is None:
            raise RuntimeError(
                "Primero ejecute cargar_archivos()."
            )

    # ---------------------------------------------------------
    # VALIDAR CORPUS
    # ---------------------------------------------------------
    def _corpus_o_error(self):

        if self.df_corpus is None:
            raise RuntimeError(
                "Primero ejecute unir_reviews_detalles()."
            )

        return self.df_corpus


# -------------------------------------------------------------
# EJECUCION (fuera de la clase, sin indentación)
# -------------------------------------------------------------
if __name__ == "__main__":
    corpus = Generador_Corpus_review(
        "IMDB_reviews.csv",
        "IMDB_movie_details.csv",
        "title.basics.csv"
    )

    (
        corpus
        .cargar_archivos()
        .filtrar_movie_id_validos()
        .unir_reviews_detalles()
        .formatear_review_date()
        .eliminar_user_id()
        .usar_primer_genero()
        .eliminar_caracteres_especiales()
        .recortar_plot_summary_en_punto()
        .verificar_nulos_y_blancos()
        .recortar_review_summary_en_punto()
        .filtrar_longitud_review_summary()
        .generar_estadisticas()
        .ordenar_columnas()
        .excluir_columnas_salida()
        .reducir_dataset()
    )

    corpus.guardar_corpus(
        "corpus_peliculas.csv"
    )

    corpus.resumen_eliminaciones()