"""
unificacion_dataset.py
----------------------
Clase para leer los 3 CSV de data/processed, unificarlos y generar un
solo archivo CSV con un esquema común.

Archivos de entrada (Proyecto_POS-Tagging-Resenas/data/processed):
    - processed_es_fa.csv    -> Fuente: Kaggle (IMDb Spoiler Dataset)  | Tipo obra: Pelicula
    - resenas_es.csv         -> Fuente: Kaggle (IMDb Spoiler Dataset)  | Tipo obra: Pelicula
    - peliculas_series.csv   -> Fuente: FilmAffinity                   | Tipo obra: columna "tipo"

Archivo de salida: Proyecto_POS-Tagging-Resenas/data/results/movies_unificado.csv

Columnas del archivo de salida:
    review_summary_es, review_summary, primaryTitle, genre, rating,
    valoracion, Fecha_comentario, periodo, Tipo obra, Fuente

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
SIN_INGLES = "Desconocido, no existe el comentario en el idioma ingles"

FUENTE_KAGGLE = "https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset"
FUENTE_FILMAFFINITY = "https://www.filmaffinity.com"

COLUMNAS_SALIDA = [
    "review_summary_es",
    "review_summary",
    "primaryTitle",
    "genre",
    "rating",
    "valoracion",
    "Fecha_comentario",
    "periodo",
    "Tipo obra",
    "Fuente",
]

# Configuración por archivo: fuente y tipo de obra por defecto
# (tipo_default=None -> se toma de la columna "tipo")
ARCHIVOS_FUENTE = {
    "processed_es_fa.csv":  {"fuente": FUENTE_KAGGLE,       "tipo_default": "Pelicula"},
    "resenas_es.csv":       {"fuente": FUENTE_KAGGLE,       "tipo_default": "Pelicula"},
    "peliculas_series.csv": {"fuente": FUENTE_FILMAFFINITY, "tipo_default": None},
}


def _normalizar_nombre(nombre: str) -> str:
    """'Género Principal' -> 'generoprincipal' (sin tildes, espacios ni '_')."""
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texto.lower())


class UnificacionDataset:
    """Lee, normaliza y unifica los CSV de reseñas en un solo dataset."""

    # Columnas de origen permitidas para cada campo (en orden de prioridad).
    ALIAS = {
        "review_summary_es": ["review_summary_es", "review_text_es", "comentarios"],
        "review_summary":    ["review_summary"],
        "primaryTitle":      ["primaryTitle", "nombre"],
        "genre":             ["genre", "main_genre", "genero"],
        "rating":            ["rating", "ranking"],
        "fecha":             ["fecha_resena", "release_date", "fecha_creacion"],
        "tipo":              ["tipo"],
    }

    def __init__(
        self,
        ruta_processed: str | Path | None = None,
        ruta_results: str | Path | None = None,
        archivo_salida: str = "movies_unificado.csv",
        archivos_fuente: dict[str, dict] | None = None,
        descartar_sin_resena: bool = True,
    ):
        """
        :param ruta_processed: carpeta con los CSV. Si es None se busca
            automáticamente 'data/processed' subiendo desde este script.
        :param ruta_results: carpeta donde se guarda el CSV unificado. Si es
            None se usa 'data/results' (carpeta hermana de 'data/processed').
        :param archivo_salida: nombre (o ruta absoluta) del CSV unificado.
        :param archivos_fuente: {nombre_archivo: {"fuente": url, "tipo_default": str|None}}.
        :param descartar_sin_resena: elimina filas sin reseña en ningún idioma.
        """
        self.ruta_processed = Path(ruta_processed) if ruta_processed else self._buscar_processed()
        self.ruta_results = Path(ruta_results) if ruta_results else self.ruta_processed.parent / "results"
        salida = Path(archivo_salida)
        self.archivo_salida = salida if salida.is_absolute() else self.ruta_results / salida
        self.archivos_fuente = archivos_fuente or ARCHIVOS_FUENTE
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
            for candidata in (
                carpeta / "data" / "processed",
                carpeta / "Proyecto_POS-Tagging-Resenas" / "data" / "processed",
            ):
                if candidata.is_dir():
                    return candidata
        return Path.cwd() / "data" / "processed"

    def _puntaje_columnas(self, columnas) -> int:
        """Cuántas columnas del archivo coinciden con algún alias conocido."""
        conocidos = {_normalizar_nombre(a) for lista in self.ALIAS.values() for a in lista}
        return sum(_normalizar_nombre(c) in conocidos for c in columnas)

    def leer_csv(self, ruta: Path) -> pd.DataFrame:
        """
        Lee un CSV probando codificaciones y separadores, y se queda con el
        separador que reconoce más columnas del mapeo. Así no falla cuando las
        reseñas traen comas, ';', saltos de línea o comillas dentro del texto.
        """
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(ruta, "r", encoding=encoding, newline="") as f:
                    encabezado = f.readline()
            except UnicodeDecodeError:
                continue

            mejor, mejor_puntaje = None, -1
            for sep in (",", ";", "\t", "|"):
                if sep not in encabezado:
                    continue
                try:
                    df = pd.read_csv(
                        ruta, sep=sep, encoding=encoding, dtype=str,
                        quotechar='"', doublequote=True, escapechar=None,
                        on_bad_lines="warn", engine="python",
                    )
                except Exception:
                    continue
                puntaje = self._puntaje_columnas(df.columns)
                if puntaje > mejor_puntaje:
                    mejor, mejor_puntaje = df, puntaje
            if mejor is not None:
                mejor.columns = [str(c).strip() for c in mejor.columns]
                return mejor
            return pd.read_csv(ruta, encoding=encoding, dtype=str)
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

    def mostrar_mapeo(self) -> None:
        """Muestra qué columna de cada archivo se usó para cada campo de salida."""
        if not self.dataframes:
            self.cargar_archivos()
        print("\n=== Mapeo de columnas ===")
        for nombre, df in self.dataframes.items():
            print(f"\n{nombre}:")
            for clave in self.ALIAS:
                col = self._columna(df, clave)
                print(f"  {clave:<18} <- {col if col else '(NO ENCONTRADA)'}")

    # ------------------------------------------------------------------ #
    # Utilidades de columnas
    # ------------------------------------------------------------------ #
    def _columna(self, df: pd.DataFrame, clave: str) -> str | None:
        """Devuelve el nombre real de la primera columna que coincide con un alias
        (sin distinguir mayúsculas, tildes, espacios ni '_')."""
        mapa = {_normalizar_nombre(c): c for c in df.columns}
        for alias in self.ALIAS[clave]:
            real = mapa.get(_normalizar_nombre(alias))
            if real is not None:
                return real
        return None

    @staticmethod
    def _texto(serie: pd.Series) -> pd.Series:
        s = serie.astype("string").str.strip()
        return s.mask(s.isin(["", "nan", "None", "NaN", "null", r"\N"]))

    @staticmethod
    def limpiar_resena(valor) -> str | float:
        """
        Deja la reseña en UNA sola línea y sin comillas dobles, para que el
        CSV final no se rompa al abrirlo en Excel:
          - varias críticas en la misma celda (separadas por saltos de línea)
            se unen con ' | '
          - comillas dobles (" “ ” "") se cambian por comillas simples
          - tabulaciones y espacios repetidos se reducen a un espacio
        """
        if pd.isna(valor):
            return np.nan
        texto = str(valor).replace("\r\n", "\n").replace("\r", "\n")
        texto = re.sub(r'""+', '"', texto)
        texto = re.sub(r'[\"“”„«»]', "'", texto)
        partes = [re.sub(r"\s+", " ", p).strip() for p in texto.split("\n")]
        partes = [p for p in partes if p]
        texto = " | ".join(partes)
        return texto if texto and texto.lower() not in ("nan", "none", "null", r"\n") else np.nan

    # ------------------------------------------------------------------ #
    # Transformaciones de cada campo
    # ------------------------------------------------------------------ #
    # Etiquetas que no son género (FilmAffinity las pone en la columna de género)
    NO_GENERO = {"seriedetv", "miniseriedetv", "serie", "miniserie", "tvseries",
                 "tvminiseries", "pelicula", "pelicuadetv", "peliculadetv", "documental de tv"}

    @classmethod
    def primer_genero(cls, valor) -> str | float:
        """'Drama, Crime' -> 'Drama'; 'Serie de TV. Animación. Western' -> 'Animación'."""
        if pd.isna(valor):
            return np.nan
        texto = str(valor).strip().strip("[]")
        partes = [p.strip().strip("'\"").strip() for p in re.split(r"[,|/;.]", texto)]
        for p in partes:
            if p and p not in (r"\N", "nan") and _normalizar_nombre(p) not in cls.NO_GENERO:
                return p
        return np.nan

    @staticmethod
    def a_rating(serie: pd.Series) -> pd.Series:
        s = serie.astype("string").str.replace(",", ".", regex=False)
        s = s.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def a_anio(serie: pd.Series) -> pd.Series:
        """Extrae solo el año (1800-2099) de cualquier formato de fecha:
        '2015-03-02', '02/03/2015', '12 de marzo de 2015', 'March 2, 2015'..."""
        anio = serie.astype("string").str.extract(r"((?:18|19|20)\d{2})", expand=False)
        return pd.to_numeric(anio, errors="coerce").astype("Int64")

    @staticmethod
    def tipo_obra(valor) -> str | float:
        if pd.isna(valor):
            return np.nan
        v = _normalizar_nombre(valor)
        if v in ("movie", "tvmovie", "pelicula", "peliculas", "film", "video", "short",
                 "cortometraje") or v.startswith("pelic"):
            return "Pelicula"
        if "serie" in v or v.startswith("tv"):
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
    def normalizar(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        col = self._columna(df, "review_summary_es")
        out["review_summary_es"] = df[col].map(self.limpiar_resena) if col else np.nan

        col = self._columna(df, "review_summary")
        out["review_summary"] = df[col].map(self.limpiar_resena) if col else np.nan

        col = self._columna(df, "primaryTitle")
        out["primaryTitle"] = df[col].map(self.limpiar_resena) if col else np.nan

        col = self._columna(df, "genre")
        out["genre"] = df[col].map(self.primer_genero) if col else np.nan

        col = self._columna(df, "rating")
        out["rating"] = self.a_rating(df[col]) if col else np.nan

        col = self._columna(df, "fecha")
        out["Fecha_comentario"] = self.a_anio(df[col]) if col else pd.NA

        if config.get("tipo_default"):
            out["Tipo obra"] = config["tipo_default"]
        else:
            col = self._columna(df, "tipo")
            out["Tipo obra"] = df[col].map(self.tipo_obra) if col else np.nan

        out["Fuente"] = config["fuente"]
        return out

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

        if self.descartar_sin_resena:
            antes = len(df)
            df = df[df["review_summary_es"].notna() | df["review_summary"].notna()]
            print(f"[INFO] Filas sin reseña descartadas: {antes - len(df):,}")

        # Campos derivados
        df["valoracion"] = self.clasificar_valoracion(df["rating"])
        df["periodo"] = self.clasificar_periodo(df["Fecha_comentario"])

        df = df[COLUMNAS_SALIDA].drop_duplicates().reset_index(drop=True)

        # Valores faltantes
        df["rating"] = df["rating"].round(2)
        df = df.astype(object)
        df["review_summary"] = df["review_summary"].where(df["review_summary"].notna(), SIN_INGLES)
        df = df.where(df.notna(), DESCONOCIDO)

        self.df_unificado = df
        return df

    # ------------------------------------------------------------------ #
    # Limpieza final de columnas (se aplica antes de guardar)
    # ------------------------------------------------------------------ #
    MAX_CARACTERES = 100

    @staticmethod
    def eliminar_repetidos(texto: str) -> str:
        """
        - Letras repetidas 3 o más veces -> una sola:  'buenoooo' -> 'bueno'
          (respeta dobles del español como 'll', 'rr', 'cc', 'ee')
        - Signos repetidos -> uno solo:  '!!!' -> '!', '??' -> '?', ',,' -> ','
        - Puntos: 4 o más -> '...' (se respetan los puntos suspensivos)
        - Palabras repetidas seguidas -> una:  'muy muy buena' -> 'muy buena'
        """
        texto = re.sub(r"([^\W\d_])\1{2,}", r"\1", texto)
        texto = re.sub(r"([!¡?¿,;:\-_*'|~])\1+", r"\1", texto)
        texto = re.sub(r"\.{4,}", "...", texto)
        texto = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", texto, flags=re.IGNORECASE)
        return texto

    @staticmethod
    def eliminar_espacios(texto: str) -> str:
        """Quita espacios al inicio/fin, espacios dobles, y espacios mal puestos
        alrededor de signos: 'hola , mundo ( bien )' -> 'hola, mundo (bien)'."""
        texto = re.sub(r"\s+", " ", texto).strip()
        texto = re.sub(r"\s+([,.;:!?)\]])", r"\1", texto)
        texto = re.sub(r"([(\[¡¿])\s+", r"\1", texto)
        return texto

    # Fin de oración: '.', '!' o '?' (no cuenta los '...' ni '(...)'),
    # seguido de espacio, comilla, paréntesis de cierre o fin de texto.
    _FIN_ORACION = re.compile(r"(?<!\.)[.!?](?!\.)['\")]?(?=\s|$)")

    MAX_ORACIONES = 2   # como máximo 2 oraciones si caben en 100 caracteres

    @classmethod
    def recortar(cls, texto: str, maximo: int | None = None, oraciones: int | None = None) -> str:
        """
        Recorta el comentario SIN cortar nunca a media frase:
          1) Si el segundo punto está dentro de los 100 caracteres, se corta ahí.
          2) Si el texto cabe en 100 caracteres, se deja igual.
          3) Si pasa de 100, se toma la oración que está en curso al llegar a
             los 100 caracteres y se continúa hasta su punto final (aunque el
             resultado pase un poco de 100).
          4) Si después de los 100 ya no hay ningún punto, se deja el texto completo.
        Los puntos suspensivos ('...', '(...)', '…') NO cuentan como punto:
        la idea continúa.
        """
        maximo = maximo or cls.MAX_CARACTERES
        oraciones = oraciones or cls.MAX_ORACIONES

        fines = [m.end() for m in cls._FIN_ORACION.finditer(texto)]

        dentro = [f for f in fines if f <= maximo]
        if len(dentro) >= oraciones:
            return texto[: dentro[oraciones - 1]].strip()
        if len(texto) <= maximo:
            return texto

        despues = [f for f in fines if f >= maximo]
        if despues:
            return texto[: despues[0]].strip()
        return texto.strip()

    @staticmethod
    def _cortar_en_palabra(texto: str, maximo: int) -> str:
        """Corta a 'maximo' caracteres sin partir palabras y agrega '...'."""
        if len(texto) <= maximo:
            return texto
        corte = texto[: maximo - 3]
        if " " in corte:
            corte = corte[: corte.rfind(" ")]
        return corte.rstrip(" ,;:|-.") + "..."

    def limpiar_columnas(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Aplica a todas las columnas de texto: 1) quitar caracteres repetidos,
        2) quitar espacios innecesarios, 3) recortar en el segundo punto
        (o a 100 caracteres si no hay puntos)."""
        df = (self.df_unificado if df is None else df).copy()
        contador = {"recortadas": 0}

        columnas_comentario = {"review_summary_es", "review_summary"}

        def limpiar(valor, es_comentario: bool):
            if not isinstance(valor, str):
                return valor
            if re.match(r"https?://", valor.strip()):   # las URLs (Fuente) no se tocan
                return valor.strip()
            texto = self.eliminar_repetidos(valor)
            texto = self.eliminar_espacios(texto)
            if es_comentario:
                recortado = self.recortar(texto)          # 100 caracteres o 2 puntos
            else:
                recortado = self._cortar_en_palabra(texto, self.MAX_CARACTERES)  # solo límite de 100
            if len(recortado) < len(texto):
                contador["recortadas"] += 1
            return recortado if recortado else DESCONOCIDO

        for col in df.columns:
            es_com = col in columnas_comentario
            df[col] = df[col].map(lambda v: limpiar(v, es_com))

        df.columns = [self.eliminar_espacios(str(c)) for c in df.columns]
        df = df.drop_duplicates().reset_index(drop=True)  # al recortar pueden quedar filas iguales
        print(f"[INFO] Limpieza aplicada. Celdas recortadas: {contador['recortadas']:,}")
        self.df_unificado = df
        return df

    def guardar(self, ruta: str | Path | None = None) -> Path:
        if self.df_unificado is None:
            self.unificar()
        self.limpiar_columnas()          # limpieza antes de guardar

        # Encabezado fijo: la primera fila del CSV son SIEMPRE estas columnas,
        # en este orden (la primera instrucción del proyecto).
        df = self.df_unificado.copy()
        df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
        df = df.reindex(columns=COLUMNAS_SALIDA, fill_value=DESCONOCIDO)

        # Si algún archivo de entrada traía su encabezado repetido dentro de los
        # datos, esa fila no es una reseña: se elimina.
        es_encabezado = df.apply(
            lambda fila: sum(str(v).strip() in COLUMNAS_SALIDA or str(v).strip() in
                             {a for lista in self.ALIAS.values() for a in lista}
                             for v in fila) >= 3,
            axis=1,
        )
        if es_encabezado.any():
            print(f"[INFO] Filas de encabezado repetidas eliminadas: {es_encabezado.sum():,}")
            df = df[~es_encabezado].reset_index(drop=True)
        self.df_unificado = df

        ruta = Path(ruta) if ruta else self.archivo_salida
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            ruta, index=False, header=True, encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL, lineterminator="\n",
        )
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
        print(f"\nReseñas sin versión en inglés: {(df['review_summary'] == SIN_INGLES).sum():,}")

    def ejecutar(self) -> pd.DataFrame:
        self.cargar_archivos()
        self.unificar()
        self.guardar()
        self.resumen()
        return self.df_unificado


if __name__ == "__main__":
    unificador = UnificacionDataset()
    unificador.cargar_archivos()
    unificador.mostrar_mapeo()   # revisa aquí qué columna se tomó de cada archivo
    unificador.unificar()
    unificador.resumen()

    # Genera el CSV final en data/results/movies_unificado.csv
    unificador.guardar(unificador.ruta_results / "movies_unificado.csv")