"""
Pipeline para Spacy

    1. Tokenización
    2. POS Tagging
    3. Stopwords
    4. NER
    5. Mayúsculas y minúsculas
    6. Lematización
"""

import subprocess  # sirve para descargar el modelo de spaCy si no esta instalado
import sys  # sirve para saber cual Python se esta usando
from pathlib import Path  # sirve para armar las rutas de las carpetas del proyecto
import pandas as pd  # libreria para leer y guardar los CSV
import spacy  # libreria que hace el POS Tagging
from tqdm.auto import tqdm  # muestra la barra de progreso

#-----------------------------------------------------------------------------------------#

# La carpeta principal del proyecto es la que esta un nivel arriba de src/
RAIZ = Path(__file__).resolve().parent.parent
# De aqui se lee el CSV que ya esta traducido
CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv"
# Aqui se guarda el resultado del etiquetado (un token por fila)
CSV_SALIDA = RAIZ / "data" / "results" / "pos_analisis_spacy.csv"

#-----------------------------------------------------------------------------------------#

class POS_Tagging_Spacy:

    def __init__(self, modelo="es_core_news_md"):
        # Columna del CSV traducido que se va a etiquetar (solo las resenas)
        self.columnas = ["review_summary_es"]

        try:
            # Se carga el modelo completo (incluye parser y ner)
            self.nlp = spacy.load(modelo)
        except OSError:
            # Si el modelo no esta instalado, se descarga y despues se carga
            print("Modelo no encontrado. Instalando...")
            subprocess.run([sys.executable, "-m", "spacy", "download", modelo], check=True)
            self.nlp = spacy.load(modelo)
        print(f"Modelo de spaCy cargado: {modelo}")

#------------------------------------------------------------------------------------------#
# Pasos del Pipeline
#------------------------------------------------------------------------------------------#

    # 1. Tokenizacion: separar el texto en palabras y signos
    def tokenizar(self, texto):
        doc = self.nlp(texto)  # nlp recibe el texto y devuelve un Doc con los tokens
        tokens = [token.text for token in doc]  # se guarda el texto de cada token
        print("TOKENS:", tokens)
        return tokens

#-----------------------------------------------------------------------------------------#

    # 2. Etiquetado POS: categoria gramatical de cada token
    def etiquetar_pos(self, texto):
        doc = self.nlp(texto)
        for token in doc:
            # pos_ es la categoria general y tag_ es la etiqueta detallada
            print(f"{token.text:15} -> {token.pos_} ({token.tag_})")
        return [(token.text, token.pos_, token.tag_) for token in doc]

#-----------------------------------------------------------------------------------------#

    # 3. Borrado de stopwords: quitar las palabras muy comunes (de, la, que...)
    def quitar_stopwords(self, texto):
        doc = self.nlp(texto)
        # is_stop es True si spaCy considera que el token es una stopword
        sin_stop = [token.text for token in doc if not token.is_stop]
        print("CON stopwords: ", [token.text for token in doc])
        print("SIN stopwords: ", sin_stop)
        return sin_stop

#-----------------------------------------------------------------------------------------#

    # 4. NER: encontrar nombres de personas, lugares, organizaciones, etc.
    def entidades(self, texto):
        doc = self.nlp(texto)
        for ent in doc.ents:  # ents son las entidades que encontro spaCy
            print(f"{ent.text:25} -> {ent.label_}")
        return [(ent.text, ent.label_) for ent in doc.ents]

#-----------------------------------------------------------------------------------------#

    # 5. Mayusculas y minusculas: pasar los tokens a minuscula
    # (se hace DESPUES de etiquetar, porque spaCy necesita las mayusculas para etiquetar bien)
    def a_minusculas(self, texto):
        doc = self.nlp(texto)
        minusculas = [token.text.lower() for token in doc]
        print("ORIGINAL:   ", [token.text for token in doc])
        print("MINUSCULAS: ", minusculas)
        return minusculas

#-----------------------------------------------------------------------------------------#

    # 6. Lematizacion: llevar cada palabra a su forma base (corrieron -> correr)
    def lematizar(self, texto):
        doc = self.nlp(texto)
        for token in doc:
            # lemma_ es el lema que calcula spaCy
            print(f"{token.text:15} -> {token.lemma_.lower()}")
        return [token.lemma_.lower() for token in doc]

#----------------------------------------------------------------------------------------#
# Aplicarmos los metodos al corpus
#----------------------------------------------------------------------------------------#

    def etiquetar_corpus(self, df, batch_size=64):
        filas = []  # aqui se va guardando una fila por cada token

        for col in self.columnas:
            # Se pasan los textos como lista; los vacios o NaN se vuelven ""
            textos = df[col].fillna("").astype(str).tolist()
            # nlp.pipe procesa los textos por lotes, que es mas rapido que uno por uno
            docs = self.nlp.pipe(textos, batch_size=batch_size)

            for id_resena, doc in tqdm(zip(df["id_resena"], docs), total=len(df), desc=col):
                # Se recorre cada oracion del texto (el parser es el que las separa)
                for n_oracion, oracion in enumerate(doc.sents):
                    for token in oracion:
                        filas.append({
                            "id_resena": id_resena,
                            "columna": col,
                            "oracion": n_oracion,
                            "token": token.text,  # 1. tokenizacion
                            "upos_spacy": token.pos_,  # 2. etiquetado POS
                            "tag_spacy": token.tag_,
                            "morf": str(token.morph),
                            "es_stopword": token.is_stop,  # 3. stopwords (solo se marcan)
                            "entidad": token.ent_type_,  # 4. NER (vacio si no es entidad)
                            "token_min": token.text.lower(),  # 5. minusculas
                            "lema": token.lemma_.lower(),  # 6. lematizacion
                        })
        return pd.DataFrame(filas)

#-----------------------------------------------------------------------------------------#

    def sin_stopwords(self, tokens):
        return tokens[~tokens["es_stopword"]]

    # Leer el CSV traducido, aplicar el pipeline y guardar pos_analysis.csv
    def etiquetar_csv(self, ruta_entrada=CSV_ENTRADA, ruta_salida=CSV_SALIDA, batch_size=64):
        ruta_entrada = Path(ruta_entrada)
        ruta_salida = Path(ruta_salida)

        # Si el CSV de entrada no existe, se avisa con la ruta que se busco
        if not ruta_entrada.exists():
            raise FileNotFoundError(f"No encuentro el CSV en: {ruta_entrada}")

        # Se lee el CSV (keep_default_na=False evita que textos como "N/A" se vuelvan vacios)
        df = pd.read_csv(ruta_entrada, encoding="utf-8-sig", keep_default_na=False)
        print(f"Leidas {len(df):,} resenas de {ruta_entrada}")

        # Se revisa que existan las columnas que se van a etiquetar
        faltan = [c for c in self.columnas if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan las columnas {faltan}. Columnas: {list(df.columns)}")

        # Se aplica el pipeline a todo el corpus
        tokens = self.etiquetar_corpus(df, batch_size=batch_size)

        # Se crea la carpeta de salida (data/results) si no existe y se guarda el CSV
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        tokens.to_csv(ruta_salida, index=False, encoding="utf-8")
        print(f"Guardado: {ruta_salida} ({len(tokens):,} tokens)")
        return tokens
