"""
Compara los IDs de processed_es_fa.csv (movie_id) con los de title.basics.csv (tconst)
y, para los IDs en común, agrega el nombre de la película (primaryTitle)
en el mismo archivo processed_es_fa.csv.
"""

from pathlib import Path
import re
import pandas as pd


class TitleMapper:
    def __init__(
        self,
        reviews_path: str | Path = "Proyecto_POS-Tagging-Resenas/data/processed/processed_es_fa.csv",
        titles_path: str | Path = "Proyecto_POS-Tagging-Resenas/data/processed/title.basics.csv",
        reviews_key: str = "movie_id",   # ID en processed_es_fa.csv
        titles_key: str = "tconst",      # ID en title.basics.csv
        title_col: str = "primaryTitle",
    ):
        self.reviews_path = Path(reviews_path)
        self.titles_path = Path(titles_path)
        self.reviews_key = reviews_key
        self.titles_key = titles_key
        self.title_col = title_col
        self._titles: dict[str, str] | None = None  # caché tconst -> primaryTitle

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_sep(path: Path) -> str:
        """IMDb distribuye title.basics como TSV aunque se llame .csv."""
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
        return "\t" if header.count("\t") > header.count(",") else ","

    @staticmethod
    def normalize_id(value) -> str | None:
        """
        Lleva cualquier ID al formato de IMDb 'tt0000000'.
        Ej.: 'tt0111161' -> 'tt0111161', '111161' -> 'tt0111161', '111161.0' -> 'tt0111161'
        """
        if pd.isna(value):
            return None
        s = str(value).strip()
        digits = re.sub(r"\.0$", "", s)          # quita '.0' si vino como número
        digits = re.sub(r"\D", "", digits)       # deja solo los dígitos
        if not digits:
            return None
        return "tt" + digits.zfill(7)

    def _load_titles(self) -> dict[str, str]:
        """Carga solo tconst y primaryTitle (title.basics es muy grande)."""
        if self._titles is None:
            sep = self._detect_sep(self.titles_path)
            df = pd.read_csv(
                self.titles_path,
                sep=sep,
                usecols=[self.titles_key, self.title_col],
                dtype=str,
                na_values="\\N",                 # IMDb usa \N para vacíos
                quoting=3 if sep == "\t" else 0,  # QUOTE_NONE en el TSV de IMDb
                encoding="utf-8",
            )
            df["_id"] = df[self.titles_key].map(self.normalize_id)
            df = df.dropna(subset=["_id"]).drop_duplicates("_id")
            self._titles = dict(zip(df["_id"], df[self.title_col]))
        return self._titles

    def _load_reviews(self) -> tuple[pd.DataFrame, str]:
        sep = self._detect_sep(self.reviews_path)
        reviews = pd.read_csv(self.reviews_path, sep=sep, dtype=str, encoding="utf-8")
        if self.reviews_key not in reviews.columns:
            raise KeyError(
                f"'{self.reviews_key}' no está en {self.reviews_path.name}. "
                f"Columnas disponibles: {list(reviews.columns)}"
            )
        return reviews, sep

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def get_title(self, movie_id) -> str | None:
        """Devuelve el primaryTitle de un ID (o None si no existe)."""
        return self._load_titles().get(self.normalize_id(movie_id))

    def compare_ids(self) -> dict:
        """Compara los IDs de ambos archivos y devuelve un resumen."""
        reviews, _ = self._load_reviews()
        titles = self._load_titles()

        review_ids = set(reviews[self.reviews_key].map(self.normalize_id).dropna())
        common = {i for i in review_ids if i in titles}
        missing = review_ids - common

        summary = {
            "ids_en_resenas": len(review_ids),
            "ids_en_comun": len(common),
            "ids_sin_titulo": len(missing),
            "ejemplos_sin_titulo": sorted(missing)[:10],
        }
        print("---- Comparación de IDs ----")
        print(f"IDs únicos en {self.reviews_path.name}: {summary['ids_en_resenas']}")
        print(f"IDs en común con {self.titles_path.name}: {summary['ids_en_comun']}")
        print(f"IDs sin coincidencia: {summary['ids_sin_titulo']}")
        if missing:
            print(f"Ejemplos sin coincidencia: {summary['ejemplos_sin_titulo']}")
        return summary

    def update_reviews(self, overwrite: bool = True) -> pd.DataFrame:
        """
        Agrega/actualiza la columna primaryTitle en processed_es_fa.csv
        para los IDs que existen en title.basics.csv.

        overwrite=True  -> recalcula todos los títulos (por defecto).
        overwrite=False -> solo rellena los títulos vacíos.
        """
        reviews, sep = self._load_reviews()
        titles = self._load_titles()

        mapped = reviews[self.reviews_key].map(self.normalize_id).map(titles)

        if self.title_col not in reviews.columns or overwrite:
            reviews[self.title_col] = mapped
        else:
            reviews[self.title_col] = reviews[self.title_col].fillna(mapped)

        # Coloca primaryTitle justo después de movie_id para que sea fácil de leer
        cols = [c for c in reviews.columns if c != self.title_col]
        pos = cols.index(self.reviews_key) + 1
        cols.insert(pos, self.title_col)
        reviews = reviews[cols]

        # Escritura segura: temporal y luego reemplazo
        tmp = self.reviews_path.with_suffix(".tmp")
        reviews.to_csv(tmp, sep=sep, index=False, encoding="utf-8")
        tmp.replace(self.reviews_path)

        con_titulo = reviews[self.title_col].notna().sum()
        print(f"Actualizado {self.reviews_path.name}: {len(reviews)} filas, "
              f"{con_titulo} con título, {len(reviews) - con_titulo} sin título.")
        return reviews


if __name__ == "__main__":
    # Ajusta las rutas si ejecutas desde otra carpeta (p. ej. desde src/)
    BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

    mapper = TitleMapper(
        reviews_path=BASE / "processed_es_fa.csv",
        titles_path=BASE / "title.basics.csv",
    )
    mapper.compare_ids()      # 1) muestra cuántos IDs hay en común
    mapper.update_reviews()   # 2) escribe primaryTitle en processed_es_fa.csv