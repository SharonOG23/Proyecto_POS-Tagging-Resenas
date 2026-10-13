import re  # Sirve para quitar las etiquetas HTML con expresiones regulares
import time  # Sirve para medir cuanto tarda la traduccion
from pathlib import Path  # Sirve para armar las rutas de las carpetas del proyecto

import nltk  # Libreria que se usa para dividir los textos en oraciones
import pandas as pd  # Libreria para leer y guardar los CSV
import torch  # Libreria que el modelo de traduccion usa por dentro
from nltk.tokenize import sent_tokenize  # Funcion que separa un texto en oraciones
from tqdm.auto import tqdm  # Muestra la barra de progreso
from transformers import MarianMTModel, MarianTokenizer  # El modelo de traduccion y su tokenizador

RAIZ = Path(__file__).resolve().parent.parent
CSV_ENTRADA = RAIZ / "data" / "raw" / "muestra_3200.csv" # De aqui se lee la muestra que se creo con ComentariosAlea
CSV_SALIDA = RAIZ / "data" / "processed" / "resenas_es.csv" # Aqui se guarda el CSV ya traducido


class Traduccion_CSV:

    def __init__(self, lote_oraciones=32, max_length=256, num_beams=4): # Cantidad de oraciones que el modelo traduce al mismo tiempo
        self.lote = lote_oraciones
        self.max_length = max_length # Largo maximo (en tokens) de cada oracion
        self.num_beams = num_beams # num_beams=4 da mejor calidad y num_beams=1 es mas rapido
        self.columnas = ["review_summary", "plot_summary"] # Columnas del CSV que se van a traducir

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Se usa la GPU si hay una disponible, si no se usa el procesador (cpu)

        # Se descargan los recursos de nltk para separar oraciones
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        modelo = "Helsinki-NLP/opus-mt-en-es" # Nombre del modelo Helsinki
        # El tokenizador convierte el texto en numeros que el modelo entiende
        self.tok = MarianTokenizer.from_pretrained(modelo)
        # Se carga el modelo y se manda al dispositivo (cpu o gpu)
        self.mt = MarianMTModel.from_pretrained(modelo).to(self.device)
        # eval() pone el modelo en modo traduccion (no en modo entrenamiento)
        self.mt.eval()
        print(f"Modelo cargado: {modelo} | dispositivo: {self.device}")

    # 1. Limpiamos el texto antes de traducirlo
    @staticmethod #Usamos @staticmethod en limpiar porque solo limpia un texto y no necesita datos de la clase
    def limpiar(texto):
        # Se quitan las etiquetas HTML como <br />
        texto = re.sub(r"<[^>]+>", " ", str(texto))
        # Se quitan los espacios repetidos y los de los extremos
        return re.sub(r"\s+", " ", texto).strip()

    # 2. Traducir una lista de oraciones por lotes
    def traducir_oraciones(self, oraciones):
        # Si la lista esta vacia no hay nada que traducir
        if not oraciones:
            return []
        # Se ordenan las posiciones de la oracion mas corta a la mas larga (asi se desperdicia menos)
        orden = sorted(range(len(oraciones)), key=lambda i: len(oraciones[i]))
        # Lista donde se guardan las traducciones en su lugar original
        resultado = [None] * len(oraciones)

        # Se recorre la lista de a un lote a la vez
        for i in range(0, len(orden), self.lote):
            # Posiciones de las oraciones que van en este lote
            idx = orden[i:i + self.lote]
            # El tokenizador prepara el lote para el modelo
            entrada = self.tok([oraciones[j] for j in idx], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=self.max_length).to(self.device)
            # no_grad() hace que el modelo gaste menos memoria porque no esta entrenando
            with torch.no_grad():
                # El modelo genera la traduccion del lote
                generado = self.mt.generate(**entrada, num_beams=self.num_beams,
                                            max_length=self.max_length)
            # Se convierten los numeros que devolvio el modelo otra vez a texto
            traducidas = self.tok.batch_decode(generado, skip_special_tokens=True)
            # Cada traduccion se guarda en la posicion que le corresponde
            for j, t in zip(idx, traducidas):
                resultado[j] = t
        return resultado

    # 3. Traducir varios textos completos
    def traducir_textos(self, textos):
        # Cada texto se divide en oraciones porque el modelo corta los textos largos
        oraciones_por_texto = [sent_tokenize(self.limpiar(t)) for t in textos]
        # Se juntan todas las oraciones en una sola lista para traducirlas por lotes
        todas = [o for lista in oraciones_por_texto for o in lista]
        # Se traducen todas las oraciones
        traducidas = self.traducir_oraciones(todas)

        # Aqui se vuelve a armar cada texto uniendo sus oraciones traducidas
        salida = []
        pos = 0
        for lista in oraciones_por_texto:
            n = len(lista)
            salida.append(" ".join(traducidas[pos:pos + n]))
            pos += n
        return salida

    # 4. Leer el CSV, traducirlo y guardar el resultado
    def traducir_csv(self, ruta_entrada=CSV_ENTRADA, ruta_salida=CSV_SALIDA,
                     tamano_chunk=200):
        # Se convierten las rutas a Path para poder usarlas facilmente
        ruta_entrada = Path(ruta_entrada)
        ruta_salida = Path(ruta_salida)

        # Si el CSV de entrada no existe, se avisa con la ruta que se busco
        if not ruta_entrada.exists():
            raise FileNotFoundError(f"No encuentro el CSV en: {ruta_entrada}")

        # Se lee el CSV (keep_default_na=False evita que textos como "N/A" se vuelvan vacios)
        df = pd.read_csv(ruta_entrada, encoding="utf-8-sig", keep_default_na=False)
        print(f"Leidas {len(df):,} resenas de {ruta_entrada}")

        # Se revisa que existan las columnas que se van a traducir
        faltan = [c for c in self.columnas if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan las columnas {faltan}. Columnas: {list(df.columns)}")

        # Si el CSV no tiene id_resena, se crea con el numero de fila
        if "id_resena" not in df.columns:
            df.insert(0, "id_resena", df.index)

        # Se juntan los textos unicos de las dos columnas (asi no se traduce lo repetido)
        unicos = set()
        for col in self.columnas:
            df[col] = df[col].fillna("").astype(str)
            for texto in df[col]:
                unicos.add(self.limpiar(texto))
        unicos.discard("")  # los textos vacios no se traducen
        print(f"Textos a traducir: {len(unicos):,} unicos "
              f"(de {len(df) * len(self.columnas):,} celdas en {len(self.columnas)} columnas)")

        # Las traducciones se guardan en un diccionario: {texto en ingles: texto en espanol}
        traducciones = {}

        # Los textos se ordenan del mas corto al mas largo para que el modelo trabaje mas rapido
        pendientes = sorted(unicos, key=len)

        inicio = time.time()  # se guarda la hora de inicio para medir el tiempo

        # Se traduce de a bloques de tamano_chunk textos y se muestra la barra de progreso
        for i in tqdm(range(0, len(pendientes), tamano_chunk), desc="Chunks"):
            bloque = pendientes[i:i + tamano_chunk]
            traducidos = self.traducir_textos(bloque)
            # Cada texto traducido se guarda en el diccionario
            traducciones.update(zip(bloque, traducidos))

        print(f"Traduccion lista en {(time.time() - inicio) / 60:.1f} min")

        # Se crea una columna nueva (_es) por cada columna traducida
        for col in self.columnas:
            df[f"{col}_es"] = df[col].map(lambda t: traducciones.get(self.limpiar(t), ""))

        # Se crea la carpeta de salida (data/processed) si no existe y se guarda el CSV
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_salida, index=False, encoding="utf-8")
        print(f"Guardado: {ruta_salida} ({len(df):,} filas)")
        return df
