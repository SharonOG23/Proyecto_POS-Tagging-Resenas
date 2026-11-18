"""
Importador de reseñas a MongoDB (base `movies_db`, colección `resenas`).

Lee un solo CSV que ya trae las columnas de movies_unificado.csv + el análisis
de NLTK y spaCy, y guarda un documento por reseña:
    data/results/comparacion_spacy_nltk.csv

Campos de cada documento (26 + _id):
    review_summary_es, review_summary, primaryTitle, genre, rating, valoracion,
    Fecha_comentario, periodo, Tipo obra, Fuente,
    tokens_nltk, pos_tags_nltk, entidades_nltk, organizaciones_nltk,
    lemas_nltk, texto_lematizado_nltk,
    tokens_spacy, pos_tags_spacy, entidades_spacy, organizaciones_spacy,
    lemas_spacy, texto_lematizado_spacy,
    cantidad_tokens_nltk, cantidad_tokens_spacy, coincidencia_tokens, entidades_en_comun

Instalación:
    pip install pymongo python-dotenv pandas
    pip install "pymongo[srv]"      # solo si usas una URI "mongodb+srv://" (Atlas)

Configuración (archivo atlas-credentials.env en la raíz del proyecto):
    MONGO_URI=mongodb://usuario:clave@host:27017/
    MONGO_DB=movies_db              # opcional (por defecto movies_db)
    MONGO_COLLECTION=resenas        # opcional (por defecto resenas)

Uso:
    python src/MongoDB_importador.py --diagnostico      # muestra qué campos tiene el CSV
    python src/MongoDB_importador.py --dry-run          # solo prepara y muestra un documento
    python src/MongoDB_importador.py --vaciar           # borra la colección y vuelve a importar
    python src/MongoDB_importador.py                    # importa

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
CSV_ENTRADA = RAIZ / "data" / "results" / "comparacion_spacy_nltk.csv"

# Campos que se importan (en este orden)
CAMPOS = [
    # --- Datos de la reseña (movies_unificado.csv) ---
    "review_summary_es",
    "review_summary",
    "primaryTitle",
    "genre",
    "rating",
    "valoracion",              # positiva | intermedia | negativa
    "Fecha_comentario",        # año
    "periodo",                 # antes_2010 | 2010_o_despues
    "Tipo obra",               # Pelicula | Serie
    "Fuente",
    # --- NLTK ---
    "tokens_nltk",
    "pos_tags_nltk",
    "entidades_nltk",
    "organizaciones_nltk",
    "lemas_nltk",
    "texto_lematizado_nltk",
    # --- spaCy ---
    "tokens_spacy",
    "pos_tags_spacy",
    "entidades_spacy",
    "organizaciones_spacy",
    "lemas_spacy",
    "texto_lematizado_spacy",
    # --- Comparación ---
    "cantidad_tokens_nltk",
    "cantidad_tokens_spacy",
    "coincidencia_tokens",
    "entidades_en_comun",
]

# Tipo de cada campo no textual (los demás se guardan como texto)
CAMPOS_ENTEROS = {
    "Fecha_comentario",
    "cantidad_tokens_nltk", "cantidad_tokens_spacy",
}
CAMPOS_NUMERICOS = {"rating"}
CAMPOS_BOOLEANOS = {"coincidencia_tokens"}

# En el CSV las listas vienen como texto, p. ej. "['el', 'laberinto']"
CAMPOS_LISTA = {
    "tokens_nltk", "pos_tags_nltk", "lemas_nltk",
    "tokens_spacy", "pos_tags_spacy", "lemas_spacy",
}

# Entidades: en el CSV vienen como "Johnny Depp (PER); Rusia (LOC)"
#   -> en Mongo: [{"texto": "Johnny Depp", "tipo": "PER"}, {"texto": "Rusia", "tipo": "LOC"}]
CAMPOS_ENTIDADES = {"entidades_nltk", "entidades_spacy", "entidades_en_comun"}

# Organizaciones: en el CSV vienen como "ONU; Netflix"
#   -> en Mongo: ["ONU", "Netflix"]
CAMPOS_LISTA_TEXTO = {"organizaciones_nltk", "organizaciones_spacy"}

# Valores que significan "no hay dato"
DESCONOCIDO = {"desconocido", "nan", "none", "null", ""}
SIN_ENTIDADES = "sin entidades"


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
        return None  # 'desconocido' -> null


def _numero(valor: Any):
    """Número: devuelve int si no tiene decimales (9.0 -> 9), si no float."""
    try:
        f = float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return None  # 'desconocido' -> null
    return int(f) if f.is_integer() else f


def _booleano(valor: Any) -> Optional[bool]:
    texto = str(valor).strip().lower()
    if texto in ("true", "1", "si", "sí", "verdadero"):
        return True
    if texto in ("false", "0", "no", "falso"):
        return False
    return None


def _listificar(obj: Any) -> Any:
    """Convierte tuplas a listas de forma recursiva (BSON solo maneja listas)."""
    if isinstance(obj, (list, tuple)):
        return [_listificar(x) for x in obj]
    return obj


def _lista(valor: Any):
    """
    Convierte el texto "['a', 'b']" o "[('el', 'DET')]" en una lista real.
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


def _entidades(valor: Any) -> List[Dict[str, str]]:
    """'Johnny Depp (PER); Rusia (LOC)' -> [{'texto': 'Johnny Depp', 'tipo': 'PER'}, ...]"""
    texto = str(valor).strip()
    if texto.lower() in DESCONOCIDO or texto.lower() == SIN_ENTIDADES:
        return []
    resultado = []
    for parte in texto.split(";"):
        parte = parte.strip()
        if not parte:
            continue
        m = re.match(r"^(.*)\s\(([A-Z]+)\)$", parte)
        if m:
            resultado.append({"texto": m.group(1).strip(), "tipo": m.group(2)})
        else:
            resultado.append({"texto": parte, "tipo": None})
    return resultado


def _lista_texto(valor: Any) -> List[str]:
    """'Johnny Depp; Ana Torrent' -> ['Johnny Depp', 'Ana Torrent']; 'sin entidades' -> []"""
    texto = str(valor).strip()
    if texto.lower() in DESCONOCIDO or texto.lower() == SIN_ENTIDADES:
        return []
    return [p.strip() for p in texto.split(";") if p.strip()]


def _id_determinista(doc: Dict[str, Any]) -> ObjectId:
    """
    ObjectId estable generado a partir del contenido de la reseña.
    Así re-importar el mismo CSV no duplica documentos.
    """
    base = "|".join(
        str(doc.get(k, ""))
        for k in ("primaryTitle", "Fecha_comentario", "review_summary_es", "Fuente")
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
        ruta_csv: Path = CSV_ENTRADA,
        campos: Optional[List[str]] = None,
    ):
        self.ruta_env = Path(ruta_env)
        self.ruta_csv = Path(ruta_csv)
        self.campos = campos or CAMPOS
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

    # ---------------- Lectura del CSV ----------------
    def _leer_csv(self) -> pd.DataFrame:
        if not self.ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.ruta_csv}")
        # Todo como texto: la conversión de tipos se hace después, campo a campo
        df = pd.read_csv(self.ruta_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        # Limpia los nombres de columna (BOM, espacios, comillas) e ignora mayúsculas/minúsculas
        canonicos = {c.lower(): c for c in self.campos}
        limpias = [str(c).replace("﻿", "").strip().strip("\"'").strip() for c in df.columns]
        df.columns = [canonicos.get(c.lower(), c) for c in limpias]
        return df

    def cargar_datos(self, limite: Optional[int] = None) -> pd.DataFrame:
        """Lee el CSV y devuelve SOLO los campos configurados (en su orden)."""
        df = self._leer_csv()
        print(f"[+] {self.ruta_csv.name}: {len(df):,} filas")

        faltan = [c for c in self.campos if c not in df.columns]
        if faltan:
            print(f"[!] Campos que no están en el CSV (se guardan como null): {faltan}")
            print("    Ejecuta con --diagnostico para ver los nombres reales de las columnas.")
            for c in faltan:
                df[c] = None

        df = df[self.campos]
        return df.head(limite) if limite else df

    def diagnostico(self):
        """Muestra las columnas reales del CSV y cuáles de las pedidas existen (sin conectar a Mongo)."""
        df = self._leer_csv()
        print(f"\n== {self.ruta_csv.name}: {len(df):,} filas ==")
        print("Columnas del archivo  :", list(df.columns))
        print("Pedidas que SÍ están  :", [c for c in self.campos if c in df.columns])
        print("Pedidas que FALTAN    :", [c for c in self.campos if c not in df.columns])
        print("Columnas que NO se importan:", [c for c in df.columns if c not in self.campos])

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
        """Convierte un valor del CSV al tipo correcto para Mongo."""
        if campo in CAMPOS_ENTIDADES:
            return [] if _vacio(valor) else _entidades(valor)
        if campo in CAMPOS_LISTA_TEXTO:
            return [] if _vacio(valor) else _lista_texto(valor)
        if _vacio(valor):
            return None
        if campo in CAMPOS_ENTEROS:
            return _entero(valor)
        if campo in CAMPOS_NUMERICOS:
            return _numero(valor)
        if campo in CAMPOS_BOOLEANOS:
            return _booleano(valor)
        if campo in CAMPOS_LISTA:
            return _lista(valor)
        return str(valor).strip()

    def preparar_documentos(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convierte cada fila en un documento con los 26 campos y los tipos correctos.
        Todos los documentos tienen los mismos campos (los vacíos quedan como null o []).
        """
        documentos = []
        for fila in df.to_dict("records"):
            doc: Dict[str, Any] = {campo: self._convertir(campo, fila.get(campo)) for campo in self.campos}
            doc = {"_id": _id_determinista(doc), **doc}
            documentos.append(doc)
        return documentos

    # ---------------- Importación ----------------
    def importar(self, lote: int = 1000, limite: Optional[int] = None) -> Dict[str, int]:
        """Importa los documentos con upsert por _id, en lotes. Requiere conectar() antes."""
        if self.coleccion is None:
            raise RuntimeError("Llama a conectar() (o usa 'with') antes de importar.")

        documentos = self.preparar_documentos(self.cargar_datos(limite))

        # Reseñas idénticas generan el mismo _id: se deja una sola
        unicos = list({d["_id"]: d for d in documentos}.values())
        if len(unicos) < len(documentos):
            print(f"[!] Reseñas duplicadas omitidas: {len(documentos) - len(unicos):,}")
        documentos = unicos

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
            print(f"[+] Campos de un documento ({len(ejemplo)}):", list(ejemplo.keys()))
        return total


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Importa reseñas y análisis NLP a MongoDB")
    p.add_argument("--env", default=str(RUTA_ENV), help="ruta del archivo .env")
    p.add_argument("--csv", default=str(CSV_ENTRADA), help="ruta de comparacion_spacy_nltk.csv")
    p.add_argument("--lote", type=int, default=1000, help="documentos por escritura")
    p.add_argument("--limite", type=int, default=None, help="importar solo las N primeras (pruebas)")
    p.add_argument("--diagnostico", action="store_true",
                   help="muestra las columnas reales del CSV y cuáles de las pedidas faltan")
    p.add_argument("--vaciar", action="store_true",
                   help="borra los documentos de la colección antes de importar (pide confirmación)")
    p.add_argument("--dry-run", action="store_true",
                   help="no se conecta a Mongo: prepara los datos y muestra un documento")
    args = p.parse_args()

    importador = MongoResenas(args.env, args.csv)

    if args.diagnostico:
        importador.diagnostico()
        return

    if args.dry_run:
        docs = importador.preparar_documentos(importador.cargar_datos(args.limite or 3))
        print(f"[✓] {len(docs)} documentos preparados. Ejemplo:")
        print(json.dumps(docs[0], indent=2, ensure_ascii=False, default=str)[:4000])
        return

    with importador:
        if args.vaciar and not importador.vaciar():
            return
        importador.importar(lote=args.lote, limite=args.limite)


if __name__ == "__main__":
    main()