import subprocess  # Sirve para descargar el modelo de spaCy si no esta instalado
import sys  # Sirve para saber cual Python se esta usando
from collections import Counter  # Sirve para contar cuantas veces aparece cada categoria
from pathlib import Path  # Sirve para armar las rutas de las carpetas del proyecto

import matplotlib.pyplot as plt  # Libreria para hacer la grafica
import pandas as pd  # Libreria para leer y guardar los CSV
import spacy  # Libreria que hace el POS Tagging
from tqdm.auto import tqdm  # Muestra la barra de progreso

RAIZ = Path(__file__).resolve().parent.parent
CSV_ENTRADA = RAIZ / "data" / "processed" / "resenas_es.csv" # De aqui se lee el CSV que ya esta traducido
CSV_SALIDA = RAIZ / "data" / "results" / "pos_analysis_spacy.csv" # Aqui se guarda el resultado del etiquetado


class POS_Tagging_Spacy:

    def __init__(self, modelo="es_core_news_md"): # Utilizamos el modelo mediano de spaCy para español
        self.columnas = ["review_summary_es", "plot_summary_es"] # Columnas del CSV traducido que se van a etiquetar

        try:
            # Se carga el modelo; el ner se desactiva para que vaya mas rapido
            self.nlp = spacy.load(modelo, disable=["ner"])
        except OSError:
            # Si el modelo no esta instalado, se descarga y despues se carga
            print("Modelo no encontrado. Instalando...")
            subprocess.run([sys.executable, "-m", "spacy", "download", modelo], check=True) # Se descarga el modelo
            self.nlp = spacy.load(modelo, disable=["ner"]) # Se carga el modelo descargado
        print(f"Modelo de spaCy cargado: {modelo}") # Se indica que el modelo se cargo correctamente

    # Metodo 1. Etiquetas de un texto: token, POS, Fine POS, lema y morfologia
    def mostrar_etiquetas(self, texto):
        # nlp recibe un texto y devuelve un Doc que tiene los tokens y sus atributos
        doc = self.nlp(texto)
        print(f'ORACION ORIGINAL:\n"{texto}"\n')
        print("POS TAGGING CON SPACY:")
        print(f"{'Token':15}  {'POS':6}  {'Fine POS':8}  {'Lema':12}  Morfologia")
        print("-" * 80)
        # Se recorre cada token del Doc
        for token in doc:
            # pos_ es la categoria general, tag_ es la etiqueta detallada, lemma_ es el lema
            print(f"{token.text:15}  {token.pos_:6}  {token.tag_:8}  {token.lemma_:12}  {token.morph}")
        return doc

    # Metodo 2. Etiquetas de varias oraciones
    def mostrar_varias_oraciones(self, oraciones):
        # Se recorre la lista de oraciones una por una
        for oracion in oraciones:
            doc = self.nlp(oracion) # Se procesa la oracion con el modelo de spaCy
            print(f"Oracion: {oracion}") # Se imprime la oracion original
            for token in doc: # Se recorre cada token de la oracion
                print(f"  {token.text:15} -> {token.pos_} ({token.tag_})") # Se imprime el token, su categoria y su etiqueta detallada
            print() # Se imprime una linea en blanco entre oraciones

    # Metodo 3. Distribucion de las palabras por categoria gramatical
    def distribucion_pos(self, texto):
        doc = self.nlp(texto)
        # Diccionario donde cada categoria guarda la lista de sus palabras
        distribucion = {} # Se recorre cada token del Doc
        for token in doc:
            # No se cuentan los signos de puntuacion ni los espacios
            if not token.is_punct and not token.is_space:
                if token.pos_ not in distribucion:
                    distribucion[token.pos_] = []
                distribucion[token.pos_].append(token.text)

        print("ANALISIS POR CATEGORIA GRAMATICAL\n") # Imprime el encabezado del analisis
        # Se muestran las categorias en orden alfabetico
        for pos, palabras in sorted(distribucion.items()): # Se ordenan las categorias alfabeticamente
            unicas = sorted(set(palabras)) # Se obtienen las palabras unicas de cada categoria
            print(f"{pos}: ({len(unicas)} unicas)") # Se imprime la categoria y el numero de palabras unicas
            print(f"   Ejemplos: {', '.join(unicas[:5])}\n") # Se imprimen los primeros 5 ejemplos de palabras unicas de cada categoria
        return distribucion

    # Metodo 4. Grafica de barras de la distribucion de categorias
    def grafica_pos(self, texto):
        doc = self.nlp(texto)
        # Counter cuenta cuantas veces aparece cada categoria
        conteo = Counter([t.pos_ for t in doc if not t.is_punct and not t.is_space]) # Se cuentan las categorias de los tokens que no son signos de puntuacion ni espacios
        nombres = list(conteo.keys()) # Se obtienen los nombres de las categorias
        valores = list(conteo.values()) # Se obtienen los valores de las categorias

        fig, ax = plt.subplots(figsize=(10, 6)) # Se crea la figura y los ejes de la grafica
        colores = plt.cm.Set3(range(len(nombres))) # Se asignan colores diferentes a cada barra de la grafica
        barras = ax.bar(nombres, valores, color=colores, edgecolor="black", linewidth=1.5) # Se crea la grafica de barras con los nombres y valores de las categorias

        # Se escribe el valor encima de cada barra
        for barra in barras:
            altura = barra.get_height() # Se obtiene la altura de la barra
            ax.text(barra.get_x() + barra.get_width() / 2., altura, f"{int(altura)}", # Se escribe el valor encima de la barra
                    ha="center", va="bottom", fontweight="bold") # Se centra el texto y se pone en negrita

        ax.set_xlabel("Categoria Gramatical (POS)", fontsize=12, fontweight="bold") # Se pone el titulo del eje x
        ax.set_ylabel("Frecuencia", fontsize=12, fontweight="bold") # Se pone el titulo del eje y
        ax.set_title("Distribucion de Etiquetas POS en el Texto", fontsize=14, fontweight="bold") # Se pone el titulo de la grafica
        ax.grid(axis="y", alpha=0.3, linestyle="--") # Creamos una linea de cuadrícula en el eje y para facilitar la lectura de los valores
        plt.tight_layout() # Ajusta el diseño de la grafica para que no se solapen los elementos
        plt.show() # Muestra la grafica en pantalla

        print(f"Total de palabras (sin puntuacion): {sum(valores)}") # Se imprime el total de palabras sin contar los signos de puntuacion
        return conteo

    # Metodo 5. Dependencias sintacticas de cada token
    def mostrar_dependencias(self, texto):
        doc = self.nlp(texto) # Se procesa el texto con el modelo de spaCy
        print(f"Oracion: {texto}\n") # Se imprime la oracion original
        print(f"{'Token':14} {'POS':6} {'Dependencia':12} Cabeza") # Se imprime el encabezado de la tabla
        print("-" * 50) # Se imprime una linea de separacion
        for token in doc: # Se recorre cada token del Doc
            # dep_ es la relacion con otra palabra y head es esa otra palabra (la cabeza)
            print(f"{token.text:14} {token.pos_:6} {token.dep_:12} {token.head.text}")
        return doc

    # Metodo 6. Arbol de dependencias
    def imprimir_arbol(self, token, nivel=0): # Se imprime el arbol de dependencias de manera recursiva
        # La sangria depende del nivel en el que esta la palabra
        prefijo = "" if nivel == 0 else "  " * (nivel - 1) + "|__ " # Se crea un prefijo para la sangria dependiendo del nivel de profundidad en el arbol
        # Si la cabeza del token es el mismo token, entonces es la raiz
        if token.head == token:
            print(f"{prefijo}ROOT: {token.text} ({token.pos_})") # Se imprime la raiz del arbol
        else:
            print(f"{prefijo}{token.text} ({token.pos_}) [dep: {token.dep_}]") # Se imprime el token, su categoria y su relacion con la cabeza
        # Se llama a la misma funcion con cada hijo
        for hijo in token.children: # Se recorre cada hijo del token
            self.imprimir_arbol(hijo, nivel + 1) # Se llama a la funcion recursivamente para imprimir los hijos del token, aumentando el nivel de profundidad en el arbol

    def arbol_dependencias(self, texto): # Se imprime el arbol de dependencias de cada oracion del texto
        doc = self.nlp(texto) # Se procesa el texto con el modelo de spaCy
        print(f"Oracion: {texto}\n") # Se imprime la oracion original
        # Cada oracion del texto tiene su propia raiz, y de ahi se arma el arbol
        for oracion in doc.sents:
            self.imprimir_arbol(oracion.root)

    # Metodo 7. Caso de uso: Frases Nominales
    def frases_nominales(self, texto):
        doc = self.nlp(texto) # Se procesa el texto con el modelo de spaCy
        print(f"Texto: {texto}\n") # Se imprime el texto original
        print("Frases nominales encontradas:")
        # noun_chunks devuelve los grupos de sustantivo con sus palabras que lo acompañan
        for frase in doc.noun_chunks:
            print(f"  - {frase.text} - Raiz: {frase.root.text} ({frase.root.pos_})") # Se imprime la frase nominal, su palabra raiz y la categoria de esa palabra
        return list(doc.noun_chunks) # Se devuelve la lista de frases nominales encontradas

    # 8. Caso de uso: Palabras clave por Categoria
    def palabras_por_categoria(self, texto): # Se buscan las palabras de interes en el texto y se agrupan por categoria
        doc = self.nlp(texto) # Se procesa el texto con el modelo de spaCy
        # Se guardan las palabras de cada categoria que le interesa al analisis
        resultado = {
            "NOUN": [t.text for t in doc if t.pos_ == "NOUN"],
            "VERB": [t.text for t in doc if t.pos_ == "VERB"],
            "ADJ": [t.text for t in doc if t.pos_ == "ADJ"],
        }
        print(f"Texto: {texto}\n") # Se imprime el texto original
        print("Palabras clave por categoria:") # Se imprimen las palabras encontradas por categoria
        print(f"  Sustantivos (NOUN): {resultado['NOUN']}") # Se imprimen los sustantivos encontrados
        print(f"  Verbos (VERB): {resultado['VERB']}") # Se imprimen los verbos encontrados
        print(f"  Adjetivos (ADJ): {resultado['ADJ']}") # Se imprimen los adjetivos encontrados
        return resultado

    # Metodo 9. Caso de uso: Adjetivos y Adverbios
    def adjetivos_y_adverbios(self, texto): # Se buscan los adjetivos y adverbios en el texto, que pueden indicar valoracion o intensidad
        doc = self.nlp(texto) # Se procesa el texto con el modelo de spaCy
        adjetivos = [t.text for t in doc if t.pos_ == "ADJ"] # Se guardan los adjetivos encontrados
        adverbios = [t.text for t in doc if t.pos_ == "ADV"] # Se guardan los adverbios encontrados
        print(f"Texto: {texto}\n") # Se imprime el texto original
        print(f"Adjetivos (potencial valoracion): {adjetivos}") # Se imprimen los adjetivos encontrados
        print(f"Adverbios (intensificadores): {adverbios}") # Se imprimen los adverbios encontrados
        print(f"\nAnalisis: el texto tiene {len(adjetivos)} adjetivos y {len(adverbios)} adverbios.") # Se imprime un resumen del analisis
        return adjetivos, adverbios

    # Metodo 10. Etiquetar todo el corpus
    def etiquetar_corpus(self, df, columnas=None, batch_size=64): # Se etiquetan todas las columnas del DataFrame y se devuelve un DataFrame con los tokens y sus atributos
        columnas = columnas or self.columnas # Si no se pasan columnas, se usan las columnas por defecto
        filas = []  # aqui se va guardando una fila por cada token

        for col in columnas:
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
                            "token": token.text,
                            "lema": token.lemma_,
                            "upos_spacy": token.pos_,
                            "tag_spacy": token.tag_,
                            "morf": str(token.morph),
                        })
        return pd.DataFrame(filas)

    # Metodo 11. Leer el CSV traducido, etiquetarlo y guardar pos_analysis_spacy.csv
    def etiquetar_csv(self, ruta_entrada=CSV_ENTRADA, ruta_salida=CSV_SALIDA, batch_size=64):
        ruta_entrada = Path(ruta_entrada)
        ruta_salida = Path(ruta_salida)

        if not ruta_entrada.exists(): # Si el CSV de entrada no existe, se avisa con la ruta que se busco
            raise FileNotFoundError(f"No encuentro el CSV en: {ruta_entrada}")

        df = pd.read_csv(ruta_entrada, encoding="utf-8-sig", keep_default_na=False) # Se lee el CSV (keep_default_na=False evita que textos como "N/A" se vuelvan vacios)
        print(f"Leidas {len(df):,} resenas de {ruta_entrada}")

        faltan = [c for c in self.columnas if c not in df.columns] # Se revisa que existan las columnas que se van a etiquetar
        if faltan:
            raise ValueError(f"Faltan las columnas {faltan}. Columnas: {list(df.columns)}")

        tokens = self.etiquetar_corpus(df, batch_size=batch_size) # Se etiqueta todo el corpus

        ruta_salida.parent.mkdir(parents=True, exist_ok=True) # Se crea la carpeta de salida (data/results) si no existe y se guarda el CSV
        tokens.to_csv(ruta_salida, index=False, encoding="utf-8") # Se guarda el CSV con los tokens y sus atributos
        print(f"Guardado: {ruta_salida} ({len(tokens):,} tokens)") # Se indica que se guardo el CSV con la cantidad de tokens
        return tokens
