"""
embeddings_w2v.py
-----------------
Clase W2V: entrena y analiza modelos Word2Vec (CBOW y Skip-Gram) sobre un CSV
de reseñas de cine en español.
 
Uso rápido:
    from src.embeddings_w2v import W2V
 
    w2v = W2V("data/raw/resenas_es.csv")
    w2v.cargar_datos()
    w2v.preprocesar()
    w2v.entrenar()                       # entrena CBOW y Skip-Gram
 
    w2v.vecinos("película", modelo="skipgram")
    w2v.analogia("hombre", "actor", "mujer")
    w2v.matriz_similitud_categorias("genero")
"""
 
import re
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
 
 
class W2V:
    """Entrenamiento y análisis de Word2Vec sobre un corpus de reseñas."""
 
    def __init__(
        self,
        ruta_csv,
        columnas_texto=("review_summary_es",),
        vector_size=100,
        window=5,
        min_count=2,
        epochs=50,
        seed=42,
        workers=1,
    ):
        """
        ruta_csv       : CSV con las reseñas (ej. resenas_es.csv).
        columnas_texto : columna(s) cuyo texto se usa para entrenar. Si pasas
                         varias, se concatenan (más texto = mejores vectores).
        vector_size    : dimensión de los vectores (100-300).
        window         : tamaño de la ventana de contexto.
        min_count      : frecuencia mínima para entrar al vocabulario.
        epochs         : épocas de entrenamiento (corpus pequeño -> más épocas).
        seed           : semilla (usa el número de tu equipo).
        workers        : 1 + seed fija = resultados reproducibles.
        """
        self.ruta_csv = Path(ruta_csv)
        self.columnas_texto = list(columnas_texto)
        self.params = dict(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            seed=seed,
            workers=workers,
        )
        self.df = None
        self.modelos = {}  # {"cbow": Word2Vec, "skipgram": Word2Vec}
 
    # ------------------------------------------------------------------
    # 1. Datos y preprocesamiento
    # ------------------------------------------------------------------
    def cargar_datos(self):
        """Lee el CSV y elimina filas sin texto."""
        self.df = pd.read_csv(self.ruta_csv)
        faltantes = [c for c in self.columnas_texto if c not in self.df.columns]
        if faltantes:
            raise ValueError(f"Columnas no encontradas en el CSV: {faltantes}")
        self.df = self.df.dropna(subset=self.columnas_texto).reset_index(drop=True)
        print(f"Reseñas cargadas: {len(self.df)}")
        return self.df
 
    @staticmethod
    def _tokenizar(texto, stopwords=None):
        """Minúsculas, solo letras (con tildes y ñ), sin stopwords si se indican."""
        tokens = re.findall(r"[a-záéíóúüñ]+", str(texto).lower())
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        return [t for t in tokens if len(t) > 1]
 
    def preprocesar(self, quitar_stopwords=False):
        """
        Crea la columna 'tokens' en el DataFrame.
 
        quitar_stopwords=False conserva palabras funcionales (útil para el
        contexto de Word2Vec); True suele dar vecindarios más "temáticos" en
        corpus pequeños. Conviene probar ambas y comparar.
        """
        stopwords = None
        if quitar_stopwords:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as sw
            stopwords = set(sw.words("spanish"))
 
        texto = self.df[self.columnas_texto].astype(str).agg(" ".join, axis=1)
        self.df["tokens"] = texto.apply(lambda t: self._tokenizar(t, stopwords))
        n_tokens = int(self.df["tokens"].str.len().sum())
        print(f"Tokens totales: {n_tokens:,}")
        return self.df["tokens"]
 
    # ------------------------------------------------------------------
    # 2. Entrenamiento
    # ------------------------------------------------------------------
    def entrenar(self, arquitecturas=("cbow", "skipgram")):
        """Entrena CBOW (sg=0) y/o Skip-Gram (sg=1) con los mismos parámetros."""
        if "tokens" not in self.df.columns:
            raise RuntimeError("Ejecuta preprocesar() antes de entrenar().")
        oraciones = self.df["tokens"].tolist()
        for nombre in arquitecturas:
            sg = 1 if nombre == "skipgram" else 0
            self.modelos[nombre] = Word2Vec(sentences=oraciones, sg=sg, **self.params)
            print(f"{nombre}: vocabulario = {len(self.modelos[nombre].wv):,} palabras")
        return self.modelos
 
    def guardar(self, carpeta="models"):
        Path(carpeta).mkdir(parents=True, exist_ok=True)
        for nombre, modelo in self.modelos.items():
            modelo.save(str(Path(carpeta) / f"w2v_{nombre}.model"))
 
    def cargar_modelos(self, carpeta="models"):
        for nombre in ("cbow", "skipgram"):
            ruta = Path(carpeta) / f"w2v_{nombre}.model"
            if ruta.exists():
                self.modelos[nombre] = Word2Vec.load(str(ruta))
        return self.modelos
 
    def _wv(self, modelo):
        if modelo not in self.modelos:
            raise KeyError(f"Modelo '{modelo}' no entrenado. Disponibles: {list(self.modelos)}")
        return self.modelos[modelo].wv
 
    # ------------------------------------------------------------------
    # 3. Exploración: vecinos, similitud y analogías
    # ------------------------------------------------------------------
    def vecinos(self, palabra, modelo="skipgram", topn=10):
        """Palabras más similares (equivale a most_similar)."""
        wv = self._wv(modelo)
        if palabra not in wv:
            print(f"'{palabra}' no está en el vocabulario.")
            return []
        return wv.most_similar(palabra, topn=topn)
 
    def similitud(self, p1, p2, modelo="skipgram"):
        wv = self._wv(modelo)
        if p1 not in wv or p2 not in wv:
            return None
        return float(wv.similarity(p1, p2))
 
    def analogia(self, a, b, c, modelo="skipgram", topn=5):
        """a es a b como c es a ???   ->   vec(b) - vec(a) + vec(c)."""
        wv = self._wv(modelo)
        fuera = [p for p in (a, b, c) if p not in wv]
        if fuera:
            print(f"Fuera del vocabulario: {fuera}")
            return []
        return wv.most_similar(positive=[b, c], negative=[a], topn=topn)
 
    # ------------------------------------------------------------------
    # 4. Vectores de documento y similitud entre categorías
    # ------------------------------------------------------------------
    def vector_documento(self, tokens, modelo="skipgram"):
        """Promedio de los vectores de las palabras (las desconocidas se ignoran)."""
        wv = self._wv(modelo)
        vectores = [wv[t] for t in tokens if t in wv]
        if vectores:
            return np.mean(vectores, axis=0)
        return np.zeros(wv.vector_size)
 
    def matriz_documentos(self, modelo="skipgram"):
        """Matriz (n_reseñas x vector_size), lista para clasificación/clustering."""
        return np.array([self.vector_documento(t, modelo) for t in self.df["tokens"]])
 
    def centroides(self, columna, modelo="skipgram"):
        """Vector promedio por categoría (ej. columna='genero' o 'valoracion')."""
        X = self.matriz_documentos(modelo)
        categorias = self.df[columna].values
        return {c: X[categorias == c].mean(axis=0) for c in sorted(set(categorias))}
 
    def matriz_similitud_categorias(self, columna, modelo="skipgram"):
        """Similitud coseno entre los centroides de cada categoría."""
        cent = self.centroides(columna, modelo)
        nombres = list(cent)
        sim = cosine_similarity(np.array([cent[n] for n in nombres]))
        return pd.DataFrame(sim, index=nombres, columns=nombres).round(4)
 
    def palabras_caracteristicas(self, columna, categoria, modelo="skipgram", topn=15):
        """Palabras del vocabulario más cercanas al centroide de una categoría."""
        cent = self.centroides(columna, modelo)
        if categoria not in cent:
            raise KeyError(f"Categoría '{categoria}' no existe en '{columna}'.")
        return self._wv(modelo).similar_by_vector(cent[categoria], topn=topn)
 
 
if __name__ == "__main__":
    w2v = W2V("data/processed/resenas_es.csv")
    w2v.cargar_datos()
    w2v.preprocesar()
    w2v.entrenar()
 
    for palabra in ["película", "actuación", "aburrida"]:
        print(f"\nVecinos de '{palabra}':")
        for p, s in w2v.vecinos(palabra, topn=5):
            print(f"  {p:<15} {s:.3f}")
 
    print("\nSimilitud entre géneros:")
    print(w2v.matriz_similitud_categorias("genero"))