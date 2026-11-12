"""
Importador de reseñas a MongoDB (base `movies_db`, colección `resenas`).

Une dos CSV del proyecto en un solo documento por reseña:
    data/processed/resenas_es.csv            -> datos de la reseña
    data/results/comparacion_spacy_nltk.csv  -> resultados de NLTK y spaCy

Instalación:
    pip install pymongo python-dotenv pandas
    pip install "pymongo[srv]"      # solo si usas una URI "mongodb+srv://" (Atlas)

Configuración (archivo atlas-credentials.env en la raíz del proyecto):
    MONGO_URI=mongodb://usuario:clave@host:27017/
    MONGO_DB=movies_db              # opcional (por defecto movies_db)
    MONGO_COLLECTION=resenas        # opcional (por defecto resenas)

Uso:
    python src/MongoDB_importador.py --diagnostico      # muestra columnas reales de los CSV
    python src/MongoDB_importador.py --dry-run          # solo prepara y muestra un documento
    python src/MongoDB_importador.py --vaciar           # borra la colección y vuelve a importar
    python src/MongoDB_importador.py                    # importa
    python src/MongoDB_importador.py --clave _id        # unir los CSV por una columna común

Es IDEMPOTENTE: la importación hace "upsert" por _id, así que puedes ejecutarla
varias veces sin duplicar documentos.
"""

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import PyMongoError

# --------------------------------------------------------------------------
# Rutas y campos
# --------------------------------------------------------------------------

# Este script vive en <RAIZ>/src/, así que la raíz del proyecto está un nivel arriba.
RAIZ = Path(__file__).resolve().parents[1]
RUTA_ENV = RAIZ / "atlas-credentials.env"
CSV_RESENAS = RAIZ / "data" / "processed" / "resenas_es.csv"
CSV_NLP = RAIZ / "data" / "results" / "comparacion_spacy_nltk.csv"

# Campos que se importan desde resenas_es.csv
CAMPOS_RESENAS = [
    "_id",
    "review_summary_es",
    "review_summary",       # texto original en inglés (solo reseñas traducidas)
    "primaryTitle",
    "genre",
    "release_date",
    "rating",               # escala normalizada 1-10
    "valoracion",           # positiva | negativa | intermedia
    "periodo",              # antes_2010 | 2010_en_adelante
    "fecha_resena",
]

# Campos que se importan desde comparacion_spacy_nltk.csv
CAMPOS_NLP = [
    "tokens_nltk",
    "pos_tags_nltk",
    "lemas_nltk",
    "texto_lematizado_nltk",
    "tokens_spacy",
    "pos_tags_spacy",
    "lemas_spacy",
    "texto_lematizado_spacy",
    "cantidad_tokens_nltk",
    "cantidad_tokens_spacy",
]

# Tipo de cada campo no textual (los demás se guardan como texto)
CAMPOS_ENTEROS = {"release_date", "cantidad_tokens_nltk", "cantidad_tokens_spacy"}
CAMPOS_NUMERICOS = {"rating"}
CAMPOS_FECHA = {"fecha_resena"}
# En el CSV las listas vienen como texto, p. ej. "['el', 'laberinto']"
CAMPOS_LISTA = {
    "tokens_nltk", "pos_tags_nltk", "lemas_nltk",
    "tokens_spacy", "pos_tags_spacy", "lemas_spacy",
}

# Columnas que se prueban (en este orden) para unir ambos CSV si no se indica --clave
CLAVES_POSIBLES = ["_id", "id_resena", "id", "review_id"]


# --------------------------------------------------------------------------
# Conversión de valores del CSV a tipos de Mongo
# --------------------------------------------------------------------------
def _vacio(valor: Any) -> bool:
    """True si el valor es None, NaN o texto en blanco."""
    if valor is None:
        return True
    if isinstance(valor, float) and valor != valor:  # NaN
        return True
    return isinstance(valor, str) and not valor.strip()


def _entero(valor: Any) -> Optional[int]:
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def _numero(valor: Any):
    """Número: devuelve int si no tiene decimales (9.0 -> 9), si no float."""
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def _fecha(valor: Any):
    """Convierte a datetime (ISODate en Mongo); None si no se puede interpretar."""
    f = pd.to_datetime(valor, errors="coerce")
    return None if pd.isna(f) else f.to_pydatetime()


def _listificar(obj: Any) -> Any:
    """Convierte tuplas a listas de forma recursiva (BSON solo maneja listas)."""
    if isinstance(obj, (list, tuple)):
        return [_listificar(x) for x in obj]
    return obj


def _lista(valor: Any):
    """
    Convierte el texto "['a', 'b']" o '[["a","DET"]]' en una lista real.
    Si no se puede interpretar, se conserva el texto original para no perder datos.
    """
    texto = str(valor).strip()
    try:
        return _listificar(ast.literal_eval(texto))
    except (ValueError, SyntaxError):
        try:
            return _listificar(json.loads(texto))
        except ValueError:
            return texto


def _object_id(valor: Any) -> Optional[ObjectId]:
    """Extrae un ObjectId de un texto ('64f...' o "ObjectId('64f...')"); None si no hay."""
    if isinstance(valor, ObjectId):
        return valor
    if _vacio(valor):
        return None
    m = re.search(r"[0-9a-fA-F]{24}", str(valor))
    return ObjectId(m.group(0)) if m else None


def _id_determinista(doc: Dict[str, Any]) -> ObjectId:
    """
    ObjectId estable generado a partir del contenido (título + fecha + reseña).
    Se usa solo si el CSV no trae _id, para que re-importar no duplique documentos.
    """
    base = "|".join(
        str(doc.get(k, "")) for k in ("primaryTitle", "fecha_resena", "review_summary_es")
    )
    return ObjectId(hashlib.sha1(base.encode("utf-8")).digest()[:12])


# --------------------------------------------------------------------------
# Clase principal
# --------------------------------------------------------------------------
class MongoResenas:
    """Conecta a MongoDB (credenciales en .env) e importa las reseñas con su análisis NLP."""

    def __init__(
        self,
        ruta_env: Path = RUTA_ENV,
        ruta_resenas: Path = CSV_RESENAS,
        ruta_nlp: Path = CSV_NLP,
        clave_union: Optional[str] = None,
    ):
        self.ruta_env = Path(ruta_env)
        self.ruta_resenas = Path(ruta_resenas)
        self.ruta_nlp = Path(ruta_nlp)
        self.clave_union = clave_union  # None = autodetectar; si no hay, une por posición
        self.client: Optional[MongoClient] = None
        self.db = None
        self.coleccion = None

    # ---------------- Conexión ----------------
    def conectar(self) -> "MongoResenas":
        """Lee el .env, se conecta y comprueba que el servidor responde."""
        if not self.ruta_env.exists():
            raise FileNotFoundError(f"No se encontró el archivo de credenciales en {self.ruta_env}")
        load_dotenv(self.ruta_env)

        # Se acepta MONGO_URI y algunos nombres alternativos habituales
        uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or os.getenv("ATLAS_URI")
        if not uri:
            raise ValueError(
                f"Falta la variable MONGO_URI en {self.ruta_env.name} "
                "(también se aceptan MONGODB_URI y ATLAS_URI)"
            )
        nombre_db = os.getenv("MONGO_DB", "movies_db")
        nombre_col = os.getenv("MONGO_COLLECTION", "resenas")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")  # MongoClient conecta de forma perezosa: forzamos
        except PyMongoError as e:
            # No se imprime la URI para no filtrar usuario/clave en pantalla o logs
            raise ConnectionError(f"No se pudo conectar a MongoDB: {type(e).__name__}") from e

        self.db = self.client[nombre_db]
        self.coleccion = self.db[nombre_col]
        print(f"[✓] Conectado a MongoDB -> {nombre_db}.{nombre_col}")
        return self

    def cerrar(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "MongoResenas":
        return self.conectar()

    def __exit__(self, *exc):
        self.cerrar()

    # ---------------- Lectura y unión de los CSV ----------------
    @staticmethod
    def _leer_csv(ruta: Path) -> pd.DataFrame:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        # Todo como texto: la conversión de tipos se hace después, campo a campo
        df = pd.read_csv(ruta, encoding="utf-8-sig", dtype=str)
        # Limpia los nombres de columna (espacios, comillas) e ignora mayúsculas/minúsculas
        canonicos = {c.lower(): c for c in CAMPOS_RESENAS + CAMPOS_NLP}
        limpias = [str(c).strip().strip("\"'").strip() for c in df.columns]
        df.columns = [canonicos.get(c.lower(), c) for c in limpias]
        return df

    def _detectar_clave(self, df_res: pd.DataFrame, df_nlp: pd.DataFrame) -> Optional[str]:
        """Columna común para unir ambos CSV (la indicada o la primera candidata común)."""
        if self.clave_union:
            for nombre, df in (("resenas_es", df_res), ("comparacion_spacy_nltk", df_nlp)):
                if self.clave_union not in df.columns:
                    raise ValueError(f"La clave '{self.clave_union}' no existe en {nombre}.csv")
            return self.clave_union
        for c in CLAVES_POSIBLES:
            if c in df_res.columns and c in df_nlp.columns:
                return c
        return None

    def cargar_datos(self, limite: Optional[int] = None) -> pd.DataFrame:
        """Lee los CSV y devuelve un único DataFrame (una fila por reseña) SOLO con los campos pedidos."""
        df_res = self._leer_csv(self.ruta_resenas)
        if self.ruta_nlp.exists():
            df_nlp = self._leer_csv(self.ruta_nlp)
        else:
            print(f"[!] No existe {self.ruta_nlp.name}; se omiten sus columnas.")
            df_nlp = pd.DataFrame()
        print(f"[+] {self.ruta_resenas.name}: {len(df_res)} filas | {self.ruta_nlp.name}: {len(df_nlp)} filas")

        # Columnas NLP: se toman de comparacion_spacy_nltk.csv; si alguna no está ahí
        # pero sí en resenas_es.csv, se toma de resenas_es.csv.
        nlp_en_nlp = [c for c in CAMPOS_NLP if c in df_nlp.columns]
        nlp_en_res = [c for c in CAMPOS_NLP if c not in df_nlp.columns and c in df_res.columns]

        faltan = [c for c in CAMPOS_RESENAS if c not in df_res.columns]
        faltan += [c for c in CAMPOS_NLP if c not in nlp_en_nlp + nlp_en_res]
        if faltan:
            print(f"[!] Columnas no encontradas en ningún CSV (se omiten): {faltan}")
            print("    Ejecuta con --diagnostico para ver los nombres reales de las columnas.")

        clave = self._detectar_clave(df_res, df_nlp) if nlp_en_nlp else None

        # Solo los campos pedidos (el resto de columnas de los CSV NO se importa)
        cols_res = [c for c in CAMPOS_RESENAS + nlp_en_res if c in df_res.columns]
        if clave and clave not in cols_res:
            cols_res.append(clave)
        res = df_res[cols_res].copy()

        if not nlp_en_nlp:
            unido = res
        elif clave:
            # Unión por columna común (left: se conservan todas las reseñas)
            nlp = df_nlp[nlp_en_nlp + ([clave] if clave not in nlp_en_nlp else [])]
            nlp = nlp.drop_duplicates(subset=clave)
            unido = res.merge(nlp, on=clave, how="left")
            sin_nlp = int(unido[nlp_en_nlp].isna().all(axis=1).sum())
            print(f"[+] Unión por '{clave}': {sin_nlp} reseñas sin análisis NLP")
            if clave not in CAMPOS_RESENAS:
                unido = unido.drop(columns=[clave])
        else:
            # Sin columna común: se asume que ambos archivos tienen el MISMO orden de filas
            if len(res) != len(df_nlp):
                raise ValueError(
                    "Los CSV no comparten una columna clave y tienen distinto número de filas "
                    f"({len(res)} vs {len(df_nlp)}). Indica la columna común con --clave."
                )
            print("[!] Sin columna clave común: se unen por posición de fila (mismo orden).")
            unido = pd.concat(
                [res.reset_index(drop=True), df_nlp[nlp_en_nlp].reset_index(drop=True)], axis=1
            )

        return unido.head(limite) if limite else unido

    def diagnostico(self):
        """Muestra columnas reales de cada CSV y cuáles de las pedidas existen (sin conectar a Mongo)."""
        df_res = self._leer_csv(self.ruta_resenas)
        df_nlp = self._leer_csv(self.ruta_nlp) if self.ruta_nlp.exists() else pd.DataFrame()
        for nombre, df, campos in (
            (self.ruta_resenas.name, df_res, CAMPOS_RESENAS),
            (self.ruta_nlp.name, df_nlp, CAMPOS_NLP),
        ):
            print(f"\n== {nombre}: {len(df)} filas ==")
            print("Columnas del archivo :", list(df.columns))
            print("Pedidas que SÍ están :", [c for c in campos if c in df.columns])
            print("Pedidas que FALTAN   :", [c for c in campos if c not in df.columns])
        print("\nColumnas NLP presentes en", self.ruta_resenas.name, ":",
              [c for c in CAMPOS_NLP if c in df_res.columns])
        try:
            clave = self._detectar_clave(df_res, df_nlp)
        except ValueError as e:
            clave = f"ERROR: {e}"
        print("Clave de unión       :", clave or "ninguna (se uniría por posición de fila)")

    def vaciar(self) -> bool:
        """Borra TODOS los documentos de la colección (pide confirmación). Requiere conectar()."""
        n = self.coleccion.count_documents({})
        resp = input(f"Se borrarán {n} documentos de '{self.coleccion.name}'. Escribe SI para confirmar: ")
        if resp.strip().upper() != "SI":
            print("[!] Cancelado, no se borró nada.")
            return False
        self.coleccion.delete_many({})
        print(f"[✓] Colección vaciada ({n} documentos borrados)")
        return True

    # ---------------- Preparación de documentos ----------------
    @staticmethod
    def _convertir(campo: str, valor: Any):
        if _vacio(valor):
            return None
        if campo in CAMPOS_ENTEROS:
            return _entero(valor)
        if campo in CAMPOS_NUMERICOS:
            return _numero(valor)
        if campo in CAMPOS_FECHA:
            return _fecha(valor)
        if campo in CAMPOS_LISTA:
            return _lista(valor)
        return str(valor)

    def preparar_documentos(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convierte cada fila en un documento con los tipos correctos (omite campos vacíos)."""
        documentos = []
        for fila in df.to_dict("records"):
            doc: Dict[str, Any] = {}
            for campo, valor in fila.items():
                if campo == "_id":
                    continue  # el _id se resuelve aparte
                convertido = self._convertir(campo, valor)
                if convertido is not None:
                    doc[campo] = convertido
            doc["_id"] = _object_id(fila.get("_id")) or _id_determinista(doc)
            documentos.append(doc)
        return documentos

    # ---------------- Importación ----------------
    def importar(self, lote: int = 1000, limite: Optional[int] = None) -> Dict[str, int]:
        """Importa los documentos con upsert por _id, en lotes. Requiere conectar() antes."""
        if self.coleccion is None:
            raise RuntimeError("Llama a conectar() (o usa 'with') antes de importar.")

        documentos = self.preparar_documentos(self.cargar_datos(limite))
        total = {"insertados": 0, "actualizados": 0}

        for i in range(0, len(documentos), lote):
            operaciones = [
                ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in documentos[i : i + lote]
            ]
            res = self.coleccion.bulk_write(operaciones, ordered=False)
            total["insertados"] += res.upserted_count
            total["actualizados"] += res.modified_count
            print(f"[{min(i + lote, len(documentos))}/{len(documentos)}] documentos procesados")

        print(
            f"[✓] Insertados: {total['insertados']} | actualizados: {total['actualizados']} | "
            f"total en colección: {self.coleccion.count_documents({})}"
        )
        ejemplo = self.coleccion.find_one()
        if ejemplo:
            print("[+] Campos de un documento:", list(ejemplo.keys()))
        return total


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Importa reseñas y análisis NLP a MongoDB")
    p.add_argument("--env", default=str(RUTA_ENV), help="ruta del archivo .env")
    p.add_argument("--resenas", default=str(CSV_RESENAS))
    p.add_argument("--nlp", default=str(CSV_NLP))
    p.add_argument("--clave", default=None, help="columna común para unir ambos CSV")
    p.add_argument("--lote", type=int, default=1000, help="documentos por escritura")
    p.add_argument("--limite", type=int, default=None, help="importar solo las N primeras (pruebas)")
    p.add_argument("--diagnostico", action="store_true",
                   help="muestra las columnas reales de los CSV y cuáles de las pedidas faltan")
    p.add_argument("--vaciar", action="store_true",
                   help="borra los documentos de la colección antes de importar (pide confirmación)")
    p.add_argument("--dry-run", action="store_true",
                   help="no se conecta a Mongo: prepara los datos y muestra un documento")
    args = p.parse_args()

    importador = MongoResenas(args.env, args.resenas, args.nlp, args.clave)

    if args.diagnostico:
        importador.diagnostico()
        return

    if args.dry_run:
        docs = importador.preparar_documentos(importador.cargar_datos(args.limite or 3))
        print(f"[✓] {len(docs)} documentos preparados. Ejemplo:")
        print(json.dumps(docs[0], indent=2, ensure_ascii=False, default=str)[:3000])
        return

    with importador:
        if args.vaciar and not importador.vaciar():
            return
        importador.importar(lote=args.lote, limite=args.limite)


if __name__ == "__main__":
    main()