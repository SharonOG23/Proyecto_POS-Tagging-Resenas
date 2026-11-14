# Proyecto de Minería de Textos

<h1 align="center">Primera Parte - Análisis Morfosintáctico de Reseñas de Cine y Series en Español 🎬📽️🎞️</h1>

Esta primera parte corresponde al **Proyecto #1**, en el que se construye un corpus de al menos **3.000 reseñas** de cine y series en español y se aplican técnicas de **POS Tagging** con **NLTK** y **spaCy** para analizar su estructura morfosintáctica.

El análisis se orienta a tres ejes:

- **Comparación entre géneros cinematográficos** (Drama, Comedia, Acción, Terror, Animación, etc.)
- **Emocionalidad gramatical**: relación entre estructura gramatical y valoración
- **Evolución temporal** de la complejidad gramatical de las reseñas

[![NLTK](https://img.shields.io/badge/NLP-NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)]()
[![Kaggle](https://img.shields.io/badge/Recolecci%C3%B3n%20de%20datos-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]()
[![Traducciones](https://img.shields.io/badge/Traducciones-Multilenguaje-4285F4?style=for-the-badge&logo=googletranslate&logoColor=white)]()
[![Limpieza](https://img.shields.io/badge/Limpieza-Preprocesamiento-4CAF50?style=for-the-badge&logo=pandas&logoColor=white)]()
[![POS Tagging](https://img.shields.io/badge/POS%20Tagging-Etiquetado%20gramatical-FF5722?style=for-the-badge&logo=readthedocs&logoColor=white)]()
[![Análisis](https://img.shields.io/badge/An%C3%A1lisis-Texto-9C27B0?style=for-the-badge&logo=jupyter&logoColor=white)]()
[![Visualización](https://img.shields.io/badge/Visualizaci%C3%B3n-Plotly%20Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)]()

| Módulo | Herramienta | Objetivo |
| ------ | ----------- | -------- |
| 🌐 **Traducción automática** | **Helsinki-NLP** | Traducir al español las reseñas originales en ingles |
| 🧹 **Limpieza** | **pandas** | Eliminación de HTML, duplicados y textos vacíos; normalización de espacios |
| 🏷️ **POS Tagging con spaCy** | **es_core_news_md** | Etiquetas **Universal POS** (`NOUN`, `VERB`, `ADJ`) |
| 🏷️ **POS Tagging con NLTK** | **pos_tag** | Etiquetas **Penn Treebank** (`NN`, `VB`, `JJ`) |
| ⚖️ **Comparación NLTK vs spaCy** | **Matriz de confusión** | Medir el acuerdo entre ambas herramientas |
| 📈 **Visualización y Dashboard** | **Plotly Dash** | Crear un dashboard en donde podamos tener filtros interactivos |

---

## 📂 Estructura del proyecto

```
proyecto-pos-tagging-resenas/
|
|-- README.md # Descripcion del proyecto
|-- requirements.txt # Dependencias
|-- USO_DE_IA.md # Documentacion del uso de IA (OBLIGATORIO)
|-- .gitignore
|
|-- data/
| |-- raw/ # Dataset original de Kaggle (no se sube)
| |-- processed/
| | |-- resenas_es.csv # Muestra traducida
| | |-- resenas_limpias.csv
| |-- results/
| |-- pos_analysis.csv
|
|-- notebooks/
Página 13
| |-- 01_carga_traduccion_limpieza.ipynb
| |-- 02_pos_tagging.ipynb
| |-- 03_analisis_metricas.ipynb
| |-- 04_visualizaciones.ipynb
|
|-- src/
| |-- pos_tagging.py # Funciones de traduccion, limpieza y etiquetado
| |-- metrics.py # Funciones de calculo de metricas
| |-- visualizations.py # Funciones de graficos
|
|-- outputs/
|-- figures/ # Graficos PNG
|-- dashboard.html
```

---

## 📄 Corpus utilizados - Proyecto #1

| Corpus | URL |
|--------|-----|
| IMDB Spoiler Dataset | https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset |
| Reviews of IMDB Movies | https://www.kaggle.com/datasets/thedevastator/reviews-of-imdb-movies |
| IMDB Dataset of 50K Movie Reviews (Spanish) | https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish |

---

## 🏆 Rúbrica de evaluación

| Peso | Criterio |
|------|----------|
| 30% | Implementación Técnica POS Tagging |
| 30% | Profundidad del Análisis Morfológico |
| 20% | Comparación NLTK vs spaCy |
| 15% | Visualización y Dashboard |
| 5%  | Colaboración y Repositorio GitHub |
| **TOTAL** | 100% |

---

<h1 align="center">Segunda Parte - Análisis Semántico de Reseñas de Cine y Series en Español 🎬📽️🎞️</h1>

Esta segunda parte corresponde al **Proyecto #2**, que extiende el **Proyecto #1 (POS Tagging)** incorporando representaciones semánticas del lenguaje. Se enriquece el corpus con reseñas nativas en español obtenidas por Web Scraping, se almacena todo en MongoDB y se comparan 2 representaciones: Word2Vec y BETO (Spanish BERT).

El corpus final combina al menos 6.000 reseñas: 3.000 del Proyecto 1 (traducidas automáticamente) y 3.000 nuevas por scraping (nativas en español).

El análisis se orienta a tres ejes:

Géneros cinematográficos (Drama, Comedia, Acción, Terror, Animación, etc.)
Polaridad de las reseñas (positiva, intermedia, negativa)
Origen del texto (traducido vs. nativo)

[![MongoDB](https://img.shields.io/badge/Base%20de%20datos-MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)]()
[![Scraping](https://img.shields.io/badge/Web%20Scraping-BeautifulSoup%20%7C%20Selenium-43B02A?style=for-the-badge&logo=python&logoColor=white)]()
[![Word2Vec](https://img.shields.io/badge/Embeddings-Word2Vec-FF9800?style=for-the-badge&logo=python&logoColor=white)]()
[![BETO](https://img.shields.io/badge/Embeddings-BETO-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)]()
[![Visualización](https://img.shields.io/badge/Visualizaci%C3%B3n-Plotly%20Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)]()

| Módulo | Herramienta | Objetivo |
| ------ | ----------- | -------- |
| 🗄️ **Migración** | **pymongo, pandas** | Migrar el corpus del Proyecto 1 (con POS Tags) a MongoDB |
| 🕷️ **Web Scraping** | **requests, BeautifulSoup, Selenium** | Extraer reseñas nativas en español respetando `robots.txt` y *rate limiting* |
| 🔤 **Word2Vec** | **gensim** | Embeddings estáticos (CBOW y Skip-Gram): campos semánticos, analogías y similitud entre categorías |
| 🤖 **BETO** | **transformers, torch** | Embeddings contextuales: polisemia, búsqueda semántica y Masked Language Model |
| ⚖️ **Comparación W2V vs. BETO** | **Logistic Regression / KNN, K-Means, t-SNE** | Clasificación, clustering (Silhouette Score) y proyecciones 2D |
| 📈 **Visualización y Dashboard** | **Plotly Dash** | Dashboard interactivo para explorar embeddings y similitudes |

---

## 🧠 Análisis realizados

### Word2Vec
- Entrenamiento de modelos **CBOW** y **Skip-Gram** sobre el corpus combinado.
- Campos semánticos por género y polaridad con `most_similar()` (*guion*, *actuación*, *aburrida*, *obra maestra*, *predecible*).
- Al menos **5 analogías vectoriales** documentadas, indicando cuáles funcionan y por qué.
- Similitud coseno entre vectores promedio de género, polaridad y origen (traducido / nativo).
- Vocabulario característico por categoría y detección de calcos de traducción.

### BETO (`dccuchile/bert-base-spanish-wwm-cased`)
- **Polisemia contextual**: al menos 5 palabras (*pesada*, *plano*, *papel*, *corto*, *final*…) comparadas contra el vector único de Word2Vec.
- **Búsqueda semántica** de reseñas a partir de consultas en lenguaje natural.
- **Masked Language Model** por categoría (ej. *"Esta película de terror fue [MASK]"*).

### Comparación BoW vs. Word2Vec vs. BETO
- **Clasificación** por polaridad, género principal y origen del texto.
- **Clustering** con K-Means y evaluación con **Silhouette Score**.
- **t-SNE** comparativo de las tres representaciones.
- Misma partición train/test (`random_state` = número de equipo) para una comparación justa.

---

## 🗃️ Esquema de MongoDB

Cada reseña se guarda como un documento con campos flexibles (el detalle completo está en [`docs/esquema_mongodb.md`](docs/esquema_mongodb.md)):

```json
{
  "texto_es": "...",
  "texto_en": "... (solo reseñas traducidas)",
  "origen_texto": "traducido | nativo",
  "titulo_obra": "El laberinto del fauno",
  "tipo_obra": "pelicula | serie",
  "generos": ["Drama", "Fantasía"],
  "genero_principal": "Drama",
  "anio_estreno": 2006,
  "calificacion": 9,
  "polaridad": "positiva | negativa | intermedia",
  "periodo": "antes_2010 | 2010_en_adelante",
  "fuente": "imdb_kaggle | filmaffinity | sensacine | ...",
  "url_fuente": "https://...",
  "fecha_resena": "ISODate",
  "fecha_recopilacion": "ISODate",
  "pos_tags": { "nltk": [], "spacy": [] },
  "embeddings": { "word2vec_avg": [], "beto_cls": [] },
  "metricas": {
    "num_palabras": 120,
    "densidad_lexica": 0.54,
    "ratio_verbos_sustantivos": 0.81,
    "indice_adjetivacion": 7.9,
    "indice_negacion": 1.6
  }
}
```

> Los embeddings de BETO (768 dimensiones) se guardan con precisión reducida (4 decimales) o en una colección separada vinculada por `_id`, para respetar el límite de 512 MB de Atlas.

---

## 📂 Estructura del proyecto

```
proyecto2-analisis-semantico-resenas-cine/
|
|-- README.md                      # Descripcion del proyecto
|-- requirements.txt               # Dependencias
|-- USO_DE_IA.md                   # Documentacion del uso de IA (OBLIGATORIO)
|-- .gitignore
|-- .env                           # Cadena de conexion a MongoDB (NO se sube)
|
|-- data/
|   |-- raw/                       # CSV del Proyecto 1 (resenas_limpias.csv, pos_analysis.csv)
|
|-- notebooks/
|   |-- 01_migracion_mongodb.ipynb
|   |-- 02_web_scraping.ipynb
|   |-- 03_word2vec_analisis.ipynb
|   |-- 04_beto_analisis.ipynb
|   |-- 05_comparacion_final.ipynb
|
|-- src/
|   |-- db_manager.py              # Conexion y operaciones MongoDB
|   |-- scraper.py                 # Web scraping de resenas
|   |-- preprocessing.py           # Limpieza, tokenizacion y POS (del Proyecto 1)
|   |-- embeddings_w2v.py          # Funciones Word2Vec
|   |-- embeddings_beto.py         # Funciones BETO
|
|-- dashboard/
|   |-- app.py                     # Dashboard con Plotly Dash
|
|-- docs/
    |-- esquema_mongodb.md         # Documentacion del esquema
```

---

## ✅ Buenas prácticas de scraping

- Revisión de `robots.txt` y términos de uso del sitio.
- `time.sleep()` de al menos 1–2 segundos entre solicitudes.
- `User-Agent` identificatorio y manejo de errores HTTP.
- No se almacenan nombres de usuario ni datos personales de los autores.
- Se documenta la fuente y la fecha de recolección (`fecha_recopilacion`).
- Los datos se usan **únicamente con fines educativos**.

---

## 📄 Corpus utilizado - Proyecto #2

| Corpus | Origen del texto | URL |
|--------|------------------|-----|
| FilmAffinity | Nativo | https://www.filmaffinity.com |

---

## 🛠️ Tecnologías

| Herramienta | Versión / Modelo | Uso |
|-------------|------------------|-----|
| MongoDB | Community 7.x o Atlas | Almacenamiento del corpus |
| pymongo | 4.x | Conexión Python–MongoDB |
| gensim | 4.x | Word2Vec |
| transformers / torch | 4.x / 2.x | BETO |
| scikit-learn | 1.x | TF-IDF, clasificadores, K-Means, t-SNE |
| BeautifulSoup4 / requests / Selenium | 4.x / 2.x / 4.x | Web scraping |
| spaCy / NLTK | 3.x (`es_core_news_md`) / 3.x | Preprocesamiento y POS |
| Plotly / Dash | 5.x / 2.x | Dashboard interactivo |

---

## 📝 Rúbrica de evaluación

| Peso | Criterio |
|------|----------|
| 20% | Web Scraping e Integración MongoDB |
| 25% | Análisis con Word2Vec |
| 25% | Análisis con BETO |
| 15% | Comparación W2V vs. BETO |
| 10% | Visualización y Dashboard |
| 5%  | Colaboración y Repositorio GitHub |
| **TOTAL** | 100% |

---

## 👥 Equipo

- Marco Álvarez Quirós.
- Sharon Obando Gómez.
- Fabián Brenes Loría.

**Profesor:** Osvaldo Gónzalez Chaves  
**Curso:** Minería de Textos 2026  
**Fecha de entrega:** 18 de noviembre de 2026

---

## 📜 Licencia

Este proyecto es de uso académico. Corpus utilizados respetan sus licencias originales.

---

> Proyecto académico del Colegio Universitario de Cartago (CUC) · Curso de Minería de Textos
