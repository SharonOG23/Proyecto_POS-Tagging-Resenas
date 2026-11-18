"""
Embeddings - Word2Vec

Análisis Requerido:

• Campos semánticos por género y polaridad: entrenar Word2Vec sobre el corpus y
descubrir los clusters temáticos dominantes (guion y trama, actuación y reparto,
aspectos técnicos como fotografía, música y efectos, ritmo y duración, emociones del
espectador). Usar most_similar() para explorar los vecinos semánticos de palabras
clave como "guion", "actuación", "aburrida", "obra maestra" o "predecible".

• Analogías vectoriales: probar analogías dentro del dominio cinematográfico (por
ejemplo, actor − hombre + mujer ≈ ?, terror − miedo + risa ≈ ?, película − cine +
televisión ≈ ?, excelente − buena + mala ≈ ?). Documentar al menos 5 analogías y
analizar cuáles funcionan y cuáles no, y por qué.

• Similitud entre categorías: calcular el vector promedio de cada género (Drama,
Comedia, Acción, Terror, Animación, etc.), de cada polaridad (positiva / negativa) y de
cada origen (traducido / nativo), y medir la distancia coseno entre ellos. ¿Qué géneros
son semánticamente más cercanos? ¿Se parecen más las reseñas traducidas entre sí
que a las nativas?

• Vocabulario exclusivo: identificar las palabras más características de cada género o
polaridad según su posición en el espacio vectorial. ¿Qué términos aparecen solo cerca
del centroide de las reseñas de Terror, o solo en las reseñas negativas? ¿Aparecen
calcos de traducción en el vecindario de las reseñas traducidas?

"""

import re
import json
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
 
 
def _hash_estable(texto):
    """Hash fijo para que Word2Vec dé los mismos vectores en cada ejecución."""
    return zlib.adler32(texto.encode("utf-8"))
 
 
class Clase_W2V:
 
    # Campos semánticos que pide el enunciado (palabras semilla)
    CAMPOS = {
        "Guion y trama":      ["guion", "historia", "trama", "argumento", "final"],
        "Actuación y reparto": ["actuación", "actor", "actriz", "reparto", "personajes"],
        "Aspectos técnicos":  ["fotografía", "música", "efectos", "banda", "sonido"],
        "Ritmo y duración":   ["ritmo", "lenta", "larga", "aburrida", "duración"],
        "Emociones":          ["emocionante", "triste", "divertida", "miedo", "llorar"],
    }
 
    # Analogías: (positivas, negativas, descripción)
    # "actor - hombre + mujer"  ->  positivas=[actor, mujer], negativas=[hombre]
    ANALOGIAS = [
        (["actor", "mujer"], ["hombre"], "actor - hombre + mujer"),
        (["terror", "risa"], ["miedo"],  "terror - miedo + risa"),
        (["película", "televisión"], ["cine"],   "película - cine + televisión"),
        (["excelente", "mala"], ["buena"],  "excelente - buena + mala"),
        (["director", "actriz"], ["actor"],  "director - actor + actriz"),
        (["aburrida", "emocionante"], ["lenta"],  "aburrida - lenta + emocionante"),
    ]
 
    # Posibles calcos de traducción (inglés -> español literal)
    CALCOS = ["remolque", "complot", "escenario", "caracteres", "realmente",
              "definitivamente", "ciertamente", "actualmente", "eventualmente",
              "teatro", "teatros", "gráficos"]
 
    # ------------------------------------------------------------------ #
    # 1. Carga y entrenamiento
    # ------------------------------------------------------------------ #
    def __init__(self, ruta_csv, col_texto="review_summary_es",
                 dim=100, ventana=5, min_count=5, epocas=30, semilla=42):
        self.ruta_csv = ruta_csv
        self.col_texto = col_texto
        self.dim = dim
        self.ventana = ventana
        self.min_count = min_count
        self.epocas = epocas
        self.semilla = semilla
        self.df = None
        self.modelo = None
        self.wv = None
        self.doc_vecs = None
 
    def cargar_y_limpiar(self):
        """Lee el CSV, crea la columna 'origen' y tokeniza el texto."""
        nltk.download("stopwords", quiet=True)
        stop = set(stopwords.words("spanish"))
 
        self.df = pd.read_csv(self.ruta_csv)
 
        # Origen: kaggle/IMDB = traducido, filmaffinity = nativo
        # (si el criterio es otro, se cambia aquí)
        self.df["origen"] = np.where(
            self.df["Fuente"].str.contains("filmaffinity"), "nativo", "traducido")
 
        def tokenizar(texto):
            palabras = re.findall(r"[a-záéíóúüñ]+", str(texto).lower())
            return [p for p in palabras if len(p) > 2 and p not in stop]
 
        self.df["tokens"] = self.df[self.col_texto].apply(tokenizar)
        print(f"Reseñas: {len(self.df)} | Tokens totales: {self.df['tokens'].str.len().sum()}")
        return self
 
    def entrenar(self):
        """Entrena Word2Vec (Skip-Gram, mejor para corpus pequeños)."""
        self.modelo = Word2Vec(
            sentences=self.df["tokens"].tolist(),
            vector_size=self.dim, window=self.ventana,
            min_count=self.min_count, sg=1,
            epochs=self.epocas, seed=self.semilla, workers=1,
            hashfxn=_hash_estable)
        self.wv = self.modelo.wv
        print(f"Word2Vec entrenado: {len(self.wv)} palabras x {self.dim} dimensiones")
        return self
 
    def guardar_kv(self, ruta="word2vec_peliculas.kv"):
        """Guarda los vectores en formato gensim KeyedVectors (.kv)."""
        self.wv.save(ruta)
        print(f"Vectores guardados en: {ruta}")
        print("  Para cargarlos: KeyedVectors.load('%s')" % ruta)
 
    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def _vector(self, palabra):
        """Vector de una palabra; si es una frase ('obra maestra') promedia."""
        partes = [p for p in palabra.lower().split() if p in self.wv]
        if not partes:
            return None
        return np.mean([self.wv[p] for p in partes], axis=0)
 
    def _vectores_documentos(self):
        """Embedding de documento = promedio de vectores de sus palabras."""
        if self.doc_vecs is None:
            vecs = []
            for toks in self.df["tokens"]:
                v = [self.wv[t] for t in toks if t in self.wv]
                vecs.append(np.mean(v, axis=0) if v else np.zeros(self.dim))
            self.doc_vecs = np.array(vecs)
        return self.doc_vecs
 
    # ------------------------------------------------------------------ #
    # 2. Campos semánticos y vecinos (most_similar)
    # ------------------------------------------------------------------ #
    def vecinos(self, palabras=("guion", "actuación", "aburrida",
                                "obra maestra", "predecible"), topn=8):
        """Vecinos semánticos de palabras clave (most_similar)."""
        print("\n=== VECINOS SEMÁNTICOS ===")
        for p in palabras:
            vec = self._vector(p)
            if vec is None:
                print(f"\n'{p}': no está en el vocabulario")
                continue
            excluir = set(p.lower().split())
            res = [(w, s) for w, s in self.wv.similar_by_vector(vec, topn=topn + 3)
                   if w not in excluir][:topn]
            print(f"\n'{p}':")
            for w, s in res:
                print(f"   {w:<18} {s:.3f}")
 
    def campos_semanticos(self, topn=6):
        """Vecinos de las palabras semilla de cada campo temático."""
        print("\n=== CAMPOS SEMÁNTICOS ===")
        for campo, semillas in self.CAMPOS.items():
            print(f"\n[{campo}]")
            for s in semillas:
                if s in self.wv:
                    vec = [w for w, _ in self.wv.most_similar(s, topn=topn)]
                    print(f"   {s:<14} -> {', '.join(vec)}")
                else:
                    print(f"   {s:<14} -> (no está en el vocabulario)")
 
    def graficar_campos(self):
        """Gráfico 1: t-SNE 2D de las palabras semilla de cada campo."""
        palabras, etiquetas = [], []
        for campo, semillas in self.CAMPOS.items():
            for s in semillas:
                if s in self.wv:
                    palabras.append(s)
                    etiquetas.append(campo)
        X = np.array([self.wv[p] for p in palabras])
        X2 = TSNE(n_components=2, random_state=self.semilla,
                  perplexity=min(8, len(palabras) - 1)).fit_transform(X)
 
        plt.figure(figsize=(11, 8))
        for campo in self.CAMPOS:
            idx = [i for i, e in enumerate(etiquetas) if e == campo]
            plt.scatter(X2[idx, 0], X2[idx, 1], s=120, label=campo, alpha=0.8)
            for i in idx:
                plt.annotate(palabras[i], X2[i], xytext=(5, 5),
                             textcoords="offset points", fontweight="bold")
        plt.title("Campos semánticos del corpus (t-SNE)", fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.show()
 
    # ------------------------------------------------------------------ #
    # 3. Analogías
    # ------------------------------------------------------------------ #
    def analogias(self, lista=None, topn=5):
        """a - b + c, usando most_similar(positive, negative)."""
        print("\n=== ANALOGÍAS VECTORIALES ===")
        for positivas, negativas, desc in (lista or self.ANALOGIAS):
            faltan = [p for p in positivas + negativas if p not in self.wv]
            print(f"\n{desc}")
            if faltan:
                print(f"   No se puede: faltan en el vocabulario -> {faltan}")
                continue
            for w, s in self.wv.most_similar(positive=positivas,
                                             negative=negativas, topn=topn):
                print(f"   {w:<18} {s:.3f}")
 
    # ------------------------------------------------------------------ #
    # 4. Similitud entre categorías (centroides)
    # ------------------------------------------------------------------ #
    def centroides(self, columna):
        """Vector promedio de cada categoría de la columna (genre, valoracion, origen)."""
        docs = self._vectores_documentos()
        validos = np.linalg.norm(docs, axis=1) > 0
        cent = {}
        for cat in sorted(self.df[columna].unique()):
            mascara = (self.df[columna] == cat).values & validos
            cent[cat] = docs[mascara].mean(axis=0)
        return cent
 
    def _tabla_similitud(self, columna):
        """Matriz de similitud coseno entre los centroides de una columna."""
        cent = self.centroides(columna)
        nombres = list(cent)
        sim = cosine_similarity(np.array([cent[n] for n in nombres]))
        return pd.DataFrame(sim, index=nombres, columns=nombres)
 
    def similitud_categorias(self, columna):
        """Imprime similitud y distancia coseno (1 - similitud) entre centroides."""
        sim = self._tabla_similitud(columna)
        dist = (1 - sim).clip(lower=0)
        nombres = list(sim.index)
 
        print(f"\n=== CENTROIDES POR {columna.upper()} ===")
        print("\nSimilitud coseno:")
        print(sim.round(3).to_string())
        print("\nDistancia coseno (1 - similitud):")
        print(dist.round(4).to_string())
 
        pares = [(nombres[i], nombres[j], dist.iloc[i, j])
                 for i in range(len(nombres)) for j in range(i + 1, len(nombres))]
        pares.sort(key=lambda x: x[2])
        print(f"\n   Más cercanos : {pares[0][0]} - {pares[0][1]} (distancia {pares[0][2]:.4f})")
        print(f"   Más lejanos  : {pares[-1][0]} - {pares[-1][1]} (distancia {pares[-1][2]:.4f})")
        return sim
 
    def graficar_similitud(self, columna="genre"):
        """Gráfico 2: mapa de calor de similitud coseno entre géneros."""
        tabla = self._tabla_similitud(columna)
        plt.figure(figsize=(8, 7))
        sns.heatmap(tabla, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.3)
        plt.title(f"Similitud coseno entre centroides: {columna}", fontweight="bold")
        plt.tight_layout()
        plt.show()
 
    # ------------------------------------------------------------------ #
    # 5. Vocabulario exclusivo
    # ------------------------------------------------------------------ #
    def vocabulario_exclusivo(self, columna, topn=10, min_freq=20): # Palabras más características de cada categoría según su posición en el espacio vectorial
       
        cent = self.centroides(columna)
        palabras = [w for w in self.wv.index_to_key
                    if self.wv.get_vecattr(w, "count") >= min_freq]
        M = np.array([self.wv[w] for w in palabras])
        sims = {c: cosine_similarity(M, v.reshape(1, -1)).ravel()
                for c, v in cent.items()}
 
        print(f"\n=== VOCABULARIO CARACTERÍSTICO POR {columna.upper()} ===")
        resultado = {}
        for c in cent:
            otras = np.mean([sims[o] for o in cent if o != c], axis=0)
            score = sims[c] - otras
            top = np.argsort(-score)[:topn]
            resultado[c] = [palabras[i] for i in top]
            print(f"\n[{c}]  {', '.join(resultado[c])}")
        return resultado
 
    def calcos(self, topn=5): # Posibles calcos de traducción
        
        print("\n=== POSIBLES CALCOS DE TRADUCCIÓN ===")
        totales = {o: self.df.loc[self.df["origen"] == o, "tokens"].str.len().sum()
                   for o in self.df["origen"].unique()}
        print(f"{'palabra':<16}" + "".join(f"{o:>12}" for o in totales) + "   vecinos")
        for p in self.CALCOS:
            fila = f"{p:<16}"
            for o, tot in totales.items():
                n = sum(t.count(p) for t in self.df.loc[self.df["origen"] == o, "tokens"])
                fila += f"{n / tot * 10000:>12.2f}"
            vec = ", ".join(w for w, _ in self.wv.most_similar(p, topn=topn)) \
                if p in self.wv else "(fuera del vocabulario)"
            print(fila + "   " + vec)
        print("(frecuencia por cada 10.000 tokens)")
 
    # ------------------------------------------------------------------ #
    # 6. Exportar embeddings (campo embeddings.word2vec_avg de MongoDB)
    # ------------------------------------------------------------------ #
    def exportar_embeddings(self, ruta="embeddings.word2vec_avg.csv"): # Guarda un CSV
        
        docs = self._vectores_documentos()
        salida = pd.DataFrame({
            "indice_csv": self.df.index,
            "texto_es": self.df[self.col_texto],
            "embeddings.word2vec_avg": [
                json.dumps([round(float(x), 6) for x in v]) for v in docs],
        })
        salida.to_csv(ruta, index=False, encoding="utf-8")
        vacios = int((np.linalg.norm(docs, axis=1) == 0).sum())
        print(f"\nEmbeddings guardados en: {ruta} ({len(salida)} reseñas, {self.dim} dims)")
        print(f"  Reseñas sin vector (ninguna palabra en el vocabulario): {vacios}")
 
    # ------------------------------------------------------------------ #
    # Ejecutar todo
    # ------------------------------------------------------------------ #
    def ejecutar_todo(self):
        # 5.1 Campos semánticos
        self.vecinos()
        self.campos_semanticos()
        self.graficar_campos()                      # gráfico 1: t-SNE
        # Analogías
        self.analogias()
        # Similitud entre categorías
        for col in ["genre", "valoracion", "origen"]:
            self.similitud_categorias(col)
        self.graficar_similitud("genre")            # gráfico 2: heatmap de géneros
        # Vocabulario exclusivo
        for col in ["genre", "valoracion", "origen"]:
            self.vocabulario_exclusivo(col)
        self.calcos()
 
 
if __name__ == "__main__":
    w2v = Clase_W2V("data/processed/movies_unificado.csv")
    w2v.cargar_y_limpiar()
    w2v.entrenar()
    w2v.guardar_kv("word2vec_peliculas.kv")
    w2v.exportar_embeddings("data/results/embeddings.word2vec_avg.csv")
    w2v.ejecutar_todo()