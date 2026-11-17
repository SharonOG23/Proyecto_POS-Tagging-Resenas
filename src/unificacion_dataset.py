"""
unificacion_dataset.py
----------------------
Clase para leer los 3 CSV de data/processed, unificarlos y generar un
solo archivo CSV con un esquema común.

Archivos de entrada (Proyecto_POS-Tagging-Resenas/data/processed):
    - peliculas_series.csv   -> Fuente: Kaggle (IMDb Spoiler Dataset)
    - processed_es_fa.csv    -> Fuente: Kaggle (IMDb Spoiler Dataset)
    - resenas_es.csv         -> Fuente: FilmAffinity

Archivo de salida: Proyecto_POS-Tagging-Resenas/data/results/dataset_unificado.csv

Columnas del archivo de salida:
    review_summary_es, review_summary, primaryTitle, genre, rating,
    valoracion, periodo, fecha_resena, Tipo obra, Fuente

Uso:
    python unificacion_dataset.py
    # o desde otro módulo:
    from unificacion_dataset import UnificacionDataset
    df = UnificacionDataset().ejecutar()
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DESCONOCIDO = "desconocido"

FUENTE_KAGGLE = "https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset"
FUENTE_FILMAFFINITY = "https://www.filmaffinity.com"

COLUMNAS_SALIDA = [
    "review_summary_es",
    "review_summary",
    "primaryTitle",
    "genre",
    "rating",
    "valoracion",
    "periodo",
    "fecha_resena",
    "Tipo obra",
    "Fuente",
]

MESES_ES = {
    "enero": "January", "febrero": "February", "marzo": "March",
    "abril": "April", "mayo": "May", "junio": "June", "julio": "July",
    "agosto": "August", "septiembre": "September", "setiembre": "September",
    "octubre": "October", "noviembre": "November", "diciembre": "December",
}


def _normalizar_nombre(nombre: str) -> str:
    """'Género Principal' -> 'generoprincipal' (sin tildes, espacios ni '_')."""
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texto.lower())


class UnificacionDataset:
    """Lee, normaliza y unifica los CSV de reseñas en un solo dataset."""

    # Posibles nombres de columna en los archivos de origen (en orden de prioridad).
    ALIAS = {
        "review_summary_es": [
            "review_summary_es", "review_es", "resena_es", "reseña_es",
            "comentario_es", "critica_es", "texto_es", "review_text_es",
            "summary_es", "traduccion", "traducción",
        ],
        "review_summary": [
            "review_summary", "review_summary_en", "summary", "review_en",
            "comentario_en", "review_text", "review",
        ],
        # Texto genérico: se usa como español si el archivo es de FilmAffinity
        "texto_generico": [
            "resena", "reseña", "comentario", "critica", "crítica", "texto",
            "review", "review_text", "opinion", "opinión",
        ],
        "primaryTitle": [
            "primaryTitle", "primary_title", "title", "titulo", "título",
            "originalTitle", "original_title", "movie_title", "nombre",
            "pelicula", "película", "obra",
        ],
        "genre": ["genre", "genres", "genero", "género", "generos", "géneros"],
        "rating": [
            "rating", "averageRating", "average_rating", "puntuacion",
            "puntuación", "nota", "score", "calificacion", "calificación",
            "rating_usuario", "user_rating",
        ],
        "fecha_resena": [
            "fecha_resena", "fecha_reseña", "review_date", "fecha", "date",
            "fecha_review", "fecha_critica",
        ],
        "anio": [
            "startYear", "start_year", "year", "anio", "año", "release_year",
            "release_date", "fecha_estreno", "estreno",
        ],
        "tipo": ["Tipo obra", "tipo_obra", "titleType", "title_type", "tipo", "type"],
        "id": ["tconst", "movie_id", "imdb_id", "id_pelicula", "titleId", "title_id"],
    }

    def __init__(
        self,
        ruta_processed: str | Path | None = None,
        ruta_results: str | Path | None = None,
        archivo_salida: str = "dataset_unificado.csv",
        archivos_fuente: dict[str, str] | None = None,
        descartar_sin_resena: bool = True,
    ):
        """
        :param ruta_processed: carpeta con los CSV. Si es None se busca
            automáticamente 'data/processed' subiendo desde este script.
        :param ruta_results: carpeta donde se guarda el CSV unificado. Si es
            None se usa 'data/results' (carpeta hermana de 'data/processed').
        :param archivo_salida: nombre (o ruta absoluta) del CSV unificado.
        :param archivos_fuente: {nombre_archivo: url_fuente}.
        :param descartar_sin_resena: elimina filas sin reseña en ningún idioma
            (p. ej. filas de un catálogo de películas sin comentarios).
        """
        self.ruta_processed = Path(ruta_processed) if ruta_processed else self._buscar_processed()
        # Carpeta de salida: Proyecto_POS-Tagging-Resenas/data/results
        self.ruta_results = Path(ruta_results) if ruta_results else self.ruta_processed.parent / "results"
        salida = Path(archivo_salida)
        self.archivo_salida = salida if salida.is_absolute() else self.ruta_results / salida
        self.archivos_fuente = archivos_fuente or {
            "peliculas_series.csv": FUENTE_KAGGLE,
            "processed_es_fa.csv": FUENTE_KAGGLE,
            "resenas_es.csv": FUENTE_FILMAFFINITY,
        }
        self.descartar_sin_resena = descartar_sin_resena
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.df_unificado: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Lectura
    # ------------------------------------------------------------------ #
    @staticmethod
    def _buscar_processed() -> Path:
        inicio = Path(__file__).resolve().parent
        for carpeta in [inicio, *inicio.parents]:
            candidata = carpeta / "data" / "processed"
            if candidata.is_dir():
                return candidata
            candidata = carpeta / "Proyecto_POS-Tagging-Resenas" / "data" / "processed"
            if candidata.is_dir():
                return candidata
        return Path.cwd() / "data" / "processed"

    @staticmethod
    def leer_csv(ruta: Path) -> pd.DataFrame:
        """Lee un CSV detectando separador y codificación."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(ruta, "r", encoding=encoding, newline="") as f:
                    muestra = f.read(50_000)
                try:
                    sep = csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
                except csv.Error:
                    sep = ","
                return pd.read_csv(ruta, sep=sep, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo leer el archivo: {ruta}")

    def cargar_archivos(self) -> dict[str, pd.DataFrame]:
        for nombre in self.archivos_fuente:
            ruta = self.ruta_processed / nombre
            if not ruta.exists():
                print(f"[AVISO] No se encontró {ruta}; se omite.")
                continue
            df = self.leer_csv(ruta)
            self.dataframes[nombre] = df
            print(f"[OK] {nombre}: {len(df):,} filas | columnas: {list(df.columns)}")
        if not self.dataframes:
            raise FileNotFoundError(f"No se encontró ningún CSV en {self.ruta_processed}")
        return self.dataframes

    # ------------------------------------------------------------------ #
    # Utilidades de columnas
    # ------------------------------------------------------------------ #
    def _columna(self, df: pd.DataFrame, clave: str, excluir: set[str] | None = None) -> str | None:
        """Devuelve el nombre real de la columna que coincide con un alias."""
        excluir = excluir or set()
        mapa = {_normalizar_nombre(c): c for c in df.columns if c not in excluir}
        for alias in self.ALIAS[clave]:
            real = mapa.get(_normalizar_nombre(alias))
            if real is not None:
                return real
        return None

    @staticmethod
    def _texto(serie: pd.Series) -> pd.Series:
        s = serie.astype("string").str.strip()
        return s.mask(s.isin(["", "nan", "None", "NaN", "null", r"\N"]))

    # ------------------------------------------------------------------ #
    # Transformaciones de cada campo
    # ------------------------------------------------------------------ #
    @staticmethod
    def primer_genero(valor) -> str | float:
        if pd.isna(valor):
            return np.nan
        texto = str(valor).strip().strip("[]")
        partes = re.split(r"[,|/;]", texto)
        genero = partes[0].strip().strip("'\"").strip() if partes else ""
        return genero if genero and genero not in (r"\N", "nan") else np.nan

    @staticmethod
    def a_rating(serie: pd.Series) -> pd.Series:
        s = serie.astype("string").str.replace(",", ".", regex=False)
        s = s.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def a_fecha(serie: pd.Series) -> pd.Series:
        s = serie.astype("string").str.strip().str.lower()
        for mes_es, mes_en in MESES_ES.items():
            s = s.str.replace(rf"\b{mes_es}\b", mes_en, regex=True)
        s = s.str.replace(r"\bde\b", " ", regex=True).str.replace(r"\s+", " ", regex=True)
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")

    @staticmethod
    def a_anio(serie: pd.Series) -> pd.Series:
        anio = serie.astype("string").str.extract(r"((?:18|19|20)\d{2})", expand=False)
        return pd.to_numeric(anio, errors="coerce")

    @staticmethod
    def tipo_obra(valor) -> str | float:
        if pd.isna(valor):
            return np.nan
        v = _normalizar_nombre(valor)
        if v in ("movie", "tvmovie", "pelicula", "film", "video", "short", "cortometraje"):
            return "Pelicula"
        if "serie" in v or v.startswith("tv") or v in ("miniserie", "tvminiseries", "tvseries", "tvepisode"):
            return "Serie"
        return np.nan

    @staticmethod
    def clasificar_valoracion(rating: pd.Series) -> pd.Series:
        return pd.cut(
            rating,
            bins=[0, 4, 6, 10],
            labels=["negativa", "intermedia", "positiva"],
            include_lowest=True,
        ).astype("string")

    @staticmethod
    def clasificar_periodo(anio: pd.Series) -> pd.Series:
        anio = pd.to_numeric(anio, errors="coerce").astype(float)
        periodo = pd.Series(np.nan, index=anio.index, dtype=object)
        periodo[anio < 2010] = "antes_2010"
        periodo[anio >= 2010] = "2010_o_despues"
        return periodo

    # ------------------------------------------------------------------ #
    # Normalización de un archivo al esquema común
    # ------------------------------------------------------------------ #
    def normalizar(self, df: pd.DataFrame, fuente: str) -> pd.DataFrame:
        es_espanol = fuente == FUENTE_FILMAFFINITY
        out = pd.DataFrame(index=df.index)

        col_es = self._columna(df, "review_summary_es")
        col_en = self._columna(df, "review_summary", excluir={col_es} if col_es else None)
        usadas = {c for c in (col_es, col_en) if c}

        # En FilmAffinity el texto genérico ya está en español
        if es_espanol and col_es is None:
            col_es = self._columna(df, "texto_generico", excluir=usadas)
            if col_es is None and col_en is not None:
                col_es, col_en = col_en, None
            usadas = {c for c in (col_es, col_en) if c}

        out["review_summary_es"] = self._texto(df[col_es]) if col_es else np.nan
        out["review_summary"] = self._texto(df[col_en]) if col_en else np.nan

        col = self._columna(df, "primaryTitle")
        out["primaryTitle"] = self._texto(df[col]) if col else np.nan

        col = self._columna(df, "genre")
        out["genre"] = df[col].map(self.primer_genero) if col else np.nan

        col = self._columna(df, "rating")
        out["rating"] = self.a_rating(df[col]) if col else np.nan

        col = self._columna(df, "fecha_resena")
        fechas = self.a_fecha(df[col]) if col else pd.Series(pd.NaT, index=df.index)
        out["_fecha"] = fechas

        col = self._columna(df, "anio")
        out["_anio"] = self.a_anio(df[col]) if col else np.nan

        col = self._columna(df, "tipo")
        out["Tipo obra"] = df[col].map(self.tipo_obra) if col else np.nan

        col = self._columna(df, "id")
        out["_id"] = self._texto(df[col]) if col else np.nan

        out["Fuente"] = fuente
        return out

    # ------------------------------------------------------------------ #
    # Enriquecimiento: completa título/género/año/tipo usando el id
    # ------------------------------------------------------------------ #
    @staticmethod
    def _enriquecer(df: pd.DataFrame) -> pd.DataFrame:
        if df["_id"].isna().all():
            return df
        campos = ["primaryTitle", "genre", "_anio", "Tipo obra"]
        catalogo = (
            df.dropna(subset=["_id"])
            .groupby("_id")[campos]
            .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        )
        for campo in campos:
            faltantes = df[campo].isna() & df["_id"].notna()
            df.loc[faltantes, campo] = df.loc[faltantes, "_id"].map(catalogo[campo])
        return df

    # ------------------------------------------------------------------ #
    # Proceso completo
    # ------------------------------------------------------------------ #
    def unificar(self) -> pd.DataFrame:
        if not self.dataframes:
            self.cargar_archivos()

        partes = [
            self.normalizar(df, self.archivos_fuente[nombre])
            for nombre, df in self.dataframes.items()
        ]
        df = pd.concat(partes, ignore_index=True)
        df = self._enriquecer(df)

        if self.descartar_sin_resena:
            antes = len(df)
            df = df[df["review_summary_es"].notna() | df["review_summary"].notna()]
            print(f"[INFO] Filas sin reseña descartadas: {antes - len(df):,}")

        # Campos derivados
        df["valoracion"] = self.clasificar_valoracion(df["rating"])
        anio_ref = pd.to_numeric(df["_anio"], errors="coerce").fillna(df["_fecha"].dt.year)  # año de la obra, si no, de la reseña
        df["periodo"] = self.clasificar_periodo(anio_ref)
        df["fecha_resena"] = df["_fecha"].dt.strftime("%Y-%m-%d")

        df = df[COLUMNAS_SALIDA].drop_duplicates().reset_index(drop=True)

        # Valores faltantes -> "desconocido"
        df["rating"] = df["rating"].round(2).astype(object)
        df = df.astype(object).where(df.notna(), DESCONOCIDO)

        self.df_unificado = df
        return df

    def guardar(self, ruta: str | Path | None = None) -> Path:
        if self.df_unificado is None:
            self.unificar()
        ruta = Path(ruta) if ruta else self.archivo_salida
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.df_unificado.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"[OK] Archivo generado: {ruta} ({len(self.df_unificado):,} filas)")
        return ruta

    def resumen(self) -> None:
        if self.df_unificado is None:
            return
        df = self.df_unificado
        print("\n=== Resumen del dataset unificado ===")
        print(f"Total de filas: {len(df):,}")
        for col in ("Fuente", "valoracion", "periodo", "Tipo obra"):
            print(f"\n{col}:\n{df[col].value_counts().to_string()}")
        print("\nRegistros 'desconocido' por columna:")
        print((df == DESCONOCIDO).sum().to_string())

    def ejecutar(self) -> pd.DataFrame:
        self.cargar_archivos()
        self.unificar()
        self.guardar()
        self.resumen()
        return self.df_unificado


if __name__ == "__main__":
    UnificacionDataset().ejecutar()