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

## 📄 Corpus utilizados

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

## 🤝 Equipo

- Marco Álvarez Quirós.
- Sharon Obando Gómez.

**Profesor:** Osvaldo Gónzalez Chaves  
**Curso:** Minería de Textos 2026  
**Fecha de entrega:** 14 de octubre de 2026

---

## 📜 Licencia

Este proyecto es de uso académico. Corpus utilizados respetan sus licencias originales.

---

> Proyecto académico del Colegio Universitario de Cartago (CUC) · Curso de Minería de Textos
