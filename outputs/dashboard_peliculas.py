# --- PRIMERA PARTE ---
import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, dash_table, Input, Output, State
 
# --- SEGUNDA PARTE (Embeddings Word2Vec y BETO): librerías extra ---
import io
import os
import re
import sys
import textwrap
import unicodedata
import numpy as np
import plotly.express as px
from contextlib import redirect_stdout
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
 
#--PRIMER PARTE DEL PROYECTO--#
 
# La carpeta principal del proyecto es la que está un nivel arriba de este archivo
RAIZ = Path(__file__).resolve().parent.parent
# CSV traducido, que está en data/processed
CSV = RAIZ / "data" / "processed" / "resenas_es.csv"
 
# Colores del dashboard adaptados al estilo de la imagen (Modo Oscuro / Cine)
FONDO = "#1e1e1e"        # Fondo general oscuro que combina con la imagen
BLANCO = "#2a2a2a"       # Fondo para las tarjetas/contenedores (gris muy oscuro)
TEXTO_BLANCO = "#f5f6fa" # Texto principal claro
MORADO = "#ff4757"       # Cambiamos a un rojo/coral vibrante estilo "palomitas" para destacar
TEXTO = "#f5f6fa"
GRIS = "#b2bec3"         # Gris claro para subtítulos y textos secundarios
 
COLORES_VALORACION = {"positiva": "#00b894", "intermedia": "#fdcb6e", "negativa": "#ff6b81"}
MENSAJE_INICIAL = "Haz clic en una película de la tabla para ver sus reseñas."
 
# --- SEGUNDA PARTE (Embeddings): configuración ---
# Corpus que usa Word2Vec (el mismo de src/embeddings_w2v.py)
CSV_UNIFICADO = RAIZ / "data" / "processed" / "movies_unificado.csv"
# Carpeta donde src/embeding_beto.py deja sus resultados
DIR_BETO = RAIZ / "data" / "results" / "beto"
# Código de la clase Clase_W2V
RUTA_SRC = RAIZ / "src"
# Vectores de Word2Vec: se usa el primero que exista. Si no hay ninguno, se entrena y se guarda en KV_GUARDAR
KV_GUARDAR = RAIZ / "data" / "results" / "word2vec_peliculas.kv"
KV_CANDIDATOS = [KV_GUARDAR,
                 RAIZ / "data" / "models" / "word2vec_resenas.kv",
                 RAIZ / "word2vec_peliculas.kv"]
# Columnas del corpus por las que se comparan categorías
COLUMNAS_W2V = {"genre": "Género", "valoracion": "Polaridad", "origen": "Origen (nativo / traducido)"}
COLOR_SENTIMIENTO = {"positiva": "#00b894", "negativa": "#ff6b81", "neutra/otra": "#b2bec3"}
 
 
class DashboardPeliculas:
    def __init__(self, ruta_csv=CSV):
        self.df = pd.read_csv(ruta_csv)
        self.generos = sorted(self.df["genero"].unique())
        self.preparar_embeddings()   # NUEVO: sección Embeddings
 
        self.app = Dash(__name__)
        self.app.title = "Dashboard de Películas"
        self.app.layout = self.crear_layout()
        self.crear_callbacks()
        self.crear_callbacks_embeddings()   # NUEVO: sección Embeddings
 
    # ---------- Datos ----------
    def filtrar(self, genero, periodo, valoracion, spoilers):
        datos = self.df
        if genero != "Todos":
            datos = datos[datos["genero"] == genero]
        if periodo != "Todos":
            datos = datos[datos["periodo"] == periodo]
        if valoracion != "Todas":
            datos = datos[datos["valoracion"] == valoracion]
        if spoilers == "sin":
            datos = datos[~datos["is_spoiler"]]
        return datos
 
    def resumen_peliculas(self, datos, minimo, orden, top):
        """Una fila por película: cantidad de reseñas y rating promedio."""
        tabla = datos.groupby(["movie_id", "primaryTitle"]).agg(
            resenas=("id_resena", "count"),
            rating=("rating", "mean"),
            positivas=("valoracion", lambda x: (x == "positiva").mean() * 100),
        ).reset_index()
 
        tabla = tabla[tabla["resenas"] >= minimo]
        tabla = tabla.sort_values(orden, ascending=False)
        if top != "todas":
            tabla = tabla.head(int(top))
 
        tabla["rating"] = tabla["rating"].round(2)
        tabla["positivas"] = tabla["positivas"].round(0)
        return tabla
 
    # ---------- Diseño ----------
    def caja(self, hijos, extra=None):
        estilo = {"background": BLANCO, "padding": "20px", "borderRadius": "14px",
                  "boxShadow": "0 4px 12px rgba(0,0,0,0.3)", "color": TEXTO_BLANCO}
        estilo.update(extra or {})
        return html.Div(hijos, style=estilo)
 
    def tarjeta(self, titulo, id_valor):
        return self.caja([
            html.P(titulo, style={"margin": 0, "color": GRIS, "fontSize": "14px"}),
            html.H2(id=id_valor, style={"margin": 0, "color": MORADO}),
        ], {"flex": 1})
 
    def filtro(self, titulo, componente):
        return html.Div([
            html.Label(titulo, style={"fontWeight": "600", "color": TEXTO_BLANCO, "display": "block", "marginBottom": "5px"}),
            componente
        ], style={"flex": 1, "minWidth": "180px"})
 
    def crear_layout(self):
        # Nota: Para los Dropdowns nativos de dcc, Dash aplica estilos por defecto,
        # pero con este contenedor oscuro encajarán mucho mejor.
        filtros = self.caja([
            self.filtro("Género", dcc.Dropdown(
                id="f-genero", options=["Todos"] + self.generos,
                value="Todos", clearable=False, style={"color": "#333"})),
 
            self.filtro("Periodo de la reseña", dcc.Dropdown(
                id="f-periodo", clearable=False, value="Todos",
                options=[{"label": "Todos", "value": "Todos"},
                         {"label": "Antes de 2010", "value": "antes_2010"},
                         {"label": "2010 o después", "value": "2010_o_despues"}], style={"color": "#333"})),
 
            self.filtro("Valoración", dcc.Dropdown(
                id="f-valoracion", clearable=False, value="Todas",
                options=["Todas", "positiva", "intermedia", "negativa"], style={"color": "#333"})),
 
            self.filtro("Spoilers", dcc.Dropdown(
                id="f-spoilers", clearable=False, value="con",
                options=[{"label": "Incluir spoilers", "value": "con"},
                         {"label": "Sin spoilers", "value": "sin"}], style={"color": "#333"})),
        ], {"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "20px"})
 
        filtros2 = self.caja([
            self.filtro("Mostrar", dcc.Dropdown(
                id="f-top", clearable=False, value=5,
                options=[{"label": "Top 5", "value": 5},
                         {"label": "Top 10", "value": 10},
                         {"label": "Top 20", "value": 20},
                         {"label": "Todas", "value": "todas"}], style={"color": "#333"})),
 
            self.filtro("Ordenar por", dcc.Dropdown(
                id="f-orden", clearable=False, value="resenas",
                options=[{"label": "Más reseñas", "value": "resenas"},
                         {"label": "Mejor rating promedio", "value": "rating"},
                         {"label": "Más % de reseñas positivas", "value": "positivas"}], style={"color": "#333"})),
 
            self.filtro("Mínimo de reseñas por película", dcc.Slider(
                id="f-minimo", min=1, max=10, step=1, value=1,
                marks={i: {"label": str(i), "style": {"color": GRIS}} for i in range(1, 11)})),
        ], {"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "20px"})
 
        kpis = html.Div([
            self.tarjeta("Películas", "kpi-peliculas"),
            self.tarjeta("Reseñas", "kpi-resenas"),
            self.tarjeta("Rating promedio", "kpi-rating"),
            self.tarjeta("% reseñas positivas", "kpi-positivas"),
        ], style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap"})
 
        tabla = self.caja([
            html.H3("Detalle de películas", style={"marginTop": 0, "color": TEXTO_BLANCO}),
            dash_table.DataTable(
                id="tabla", page_size=10, sort_action="native",
                columns=[{"name": "Película", "id": "primaryTitle"},
                         {"name": "Reseñas", "id": "resenas"},
                         {"name": "Rating promedio", "id": "rating"},
                         {"name": "% positivas", "id": "positivas"}],
                style_header={"backgroundColor": MORADO, "color": "white", "fontWeight": "bold", "border": "none"},
                style_cell={"padding": "10px", "textAlign": "left", "backgroundColor": "#333333", "color": "#f5f6fa", "border": "1px solid #444"},
                style_data_conditional=[
                    {
                        'if': {'state': 'active'},
                        'backgroundColor': '#ff4757',
                        'color': 'white',
                        'border': '1px solid #ff6b81'
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#ff4757',
                        'color': 'white',
                        'border': '1px solid #ff6b81'
                    }
                ],
                style_as_list_view=False),
        ], {"marginTop": "20px"})
 
        resenas = self.caja([
            html.H3("Reseñas de la película", style={"marginTop": 0, "color": TEXTO_BLANCO}),
            html.Div(id="resenas-pelicula",
                     children=html.P(MENSAJE_INICIAL, style={"color": GRIS})),
        ], {"marginTop": "20px"})
 
        return html.Div([
            html.H1("🎬 Dashboard de Reseñas de Películas",
                    style={"color": TEXTO_BLANCO, "marginBottom": 0}),
            html.P("Explora las reseñas por género, periodo y valoración",
                   style={"color": GRIS}),
            filtros, filtros2, kpis, tabla, resenas,
            self.seccion_embeddings(),   # NUEVO: sección Embeddings
        ], style={"background": FONDO, "minHeight": "100vh", "padding": "30px 6%",
                  "fontFamily": "Segoe UI, sans-serif"})
 
    # ---------- Interacción ----------
    def crear_callbacks(self):
        entradas = [Input("f-genero", "value"), Input("f-periodo", "value"),
                    Input("f-valoracion", "value"), Input("f-spoilers", "value"),
                    Input("f-top", "value"), Input("f-orden", "value"),
                    Input("f-minimo", "value")]
 
        @self.app.callback(
            Output("tabla", "data"),
            Output("tabla", "active_cell"),
            Output("kpi-peliculas", "children"),
            Output("kpi-resenas", "children"),
            Output("kpi-rating", "children"),
            Output("kpi-positivas", "children"),
            entradas,
        )
        def actualizar(genero, periodo, valoracion, spoilers, top, orden, minimo):
            datos = self.filtrar(genero, periodo, valoracion, spoilers)
            tabla = self.resumen_peliculas(datos, minimo, orden, top)
 
            total_peliculas = datos["movie_id"].nunique()
            rating = round(datos["rating"].mean(), 2) if len(datos) else 0
            positivas = round((datos["valoracion"] == "positiva").mean() * 100) if len(datos) else 0
 
            return (tabla.to_dict("records"), None,
                    total_peliculas, len(datos), rating, f"{positivas}%")
 
        @self.app.callback(
            Output("resenas-pelicula", "children"),
            Input("tabla", "active_cell"),
            State("tabla", "derived_viewport_data"),
            State("f-genero", "value"), State("f-periodo", "value"),
            State("f-valoracion", "value"), State("f-spoilers", "value"),
        )
        def mostrar_resenas(celda, filas_visibles, genero, periodo, valoracion, spoilers):
            if not celda or not filas_visibles:
                return html.P(MENSAJE_INICIAL, style={"color": GRIS})
 
            movie_id = filas_visibles[celda["row"]]["movie_id"]
            datos = self.filtrar(genero, periodo, valoracion, spoilers)
            datos = datos[datos["movie_id"] == movie_id]
            if datos.empty:
                return html.P("Esa película no tiene reseñas con los filtros actuales.",
                              style={"color": GRIS})
 
            primera = datos.iloc[0]
            lista = [html.H4(primera["primaryTitle"], style={"marginBottom": 0, "color": TEXTO_BLANCO}),
                     html.P(primera["plot_summary_es"], style={"color": GRIS})]
            for _, fila in datos.iterrows():
                color = COLORES_VALORACION[fila["valoracion"]]
                lista.append(html.Div([
                    html.B(f"⭐ {fila['rating']}/10  ", style={"color": TEXTO_BLANCO}),
                    html.Span(fila["review_summary_es"], style={"color": "#dfe6e9"}),
                    html.Small(f"  ({fila['review_date']})", style={"color": GRIS}),
                ], style={"borderLeft": f"4px solid {color}", "padding": "8px 12px",
                          "margin": "8px 0", "background": "#222222", "borderRadius": "4px"}))
            return lista
 
 #---------------------------------------------------------------------------------------------------------------------------------------------------#
 
    # SEGUNDA PARTE DEL PROYECTO
    # EMBEDDINGS: WORD2VEC Y BETO
    # =====================================================================
    # Esta parte NO depende de los filtros de arriba: Word2Vec trabaja sobre
    # movies_unificado.csv y BETO lee los resultados de data/results/beto/.
 
    # ---------- Carga de datos ----------
    def preparar_embeddings(self):
        self.cargar_word2vec()
        self.cargar_beto()
 
    def cargar_word2vec(self):
        """Usa la clase Clase_W2V de src/embeddings_w2v.py y precalcula los resultados."""
        self.w2v, self.w2v_error, self.w2v_origen = None, None, ""
        try:
            os.environ.setdefault("MPLBACKEND", "Agg")   # la clase importa matplotlib; aquí no se grafica con él
            sys.path.insert(0, str(RUTA_SRC))
            from embeddings_w2v import Clase_W2V
            from gensim.models import KeyedVectors
 
            w2v = Clase_W2V(str(CSV_UNIFICADO))
            with redirect_stdout(io.StringIO()):          # la clase imprime mucho; se silencia
                w2v.cargar_y_limpiar()
 
            kv = next((ruta for ruta in KV_CANDIDATOS if ruta.exists()), None)
            if kv:
                w2v.wv = KeyedVectors.load(str(kv))
                w2v.dim = w2v.wv.vector_size
                self.w2v_origen = f"vectores cargados de {kv.name}"
            else:
                with redirect_stdout(io.StringIO()):
                    w2v.entrenar()
                KV_GUARDAR.parent.mkdir(parents=True, exist_ok=True)
                w2v.wv.save(str(KV_GUARDAR))
                self.w2v_origen = "modelo entrenado al iniciar (guardado en data/results/word2vec_peliculas.kv)"
            self.w2v = w2v
            self.calcular_resultados_w2v()
        except Exception as error:   # si algo falla, el resto del dashboard sigue funcionando
            self.w2v = None
            self.w2v_error = f"{type(error).__name__}: {error}"
            print(f"[Word2Vec] No se pudo preparar: {self.w2v_error}")
 
    def calcular_resultados_w2v(self):
        w2v = self.w2v
        self.w2v_sim, self.w2v_excl = {}, {}
        with redirect_stdout(io.StringIO()):
            for columna in COLUMNAS_W2V:
                self.w2v_sim[columna] = w2v.similitud_categorias(columna)     # matriz de similitud coseno
                self.w2v_excl[columna] = w2v.vocabulario_exclusivo(columna)   # {categoría: [palabras]}
 
        self.w2v_analogias = []
        for positivas, negativas, descripcion in w2v.ANALOGIAS:
            faltan = [p for p in positivas + negativas if p not in w2v.wv]
            resultado = [] if faltan else w2v.wv.most_similar(positive=positivas, negative=negativas, topn=5)
            self.w2v_analogias.append({"desc": descripcion, "faltan": faltan, "res": resultado})
 
        self.w2v_calcos = self.tabla_calcos()
        self.w2v_fig_campos = self.figura_campos()
 
    def cargar_beto(self):
        """Lee los CSV y el .npy que genera src/embeding_beto.py (no carga el modelo)."""
        def leer(nombre):
            ruta = DIR_BETO / nombre
            return pd.read_csv(ruta) if ruta.exists() else None
 
        self.beto_pol = leer("polisemia_resumen.csv")
        self.beto_busq = leer("busqueda_resenas.csv")
        self.beto_mlm = leer("mlm_predicciones.csv")
        self.beto_indice = leer("resenas_indice.csv")
        ruta_emb = DIR_BETO / "embeddings_resenas.npy"
        self.beto_emb = np.load(ruta_emb) if ruta_emb.exists() else None
 
        # El .npy y el índice deben tener la misma cantidad de filas
        if self.beto_emb is not None and (self.beto_indice is None or len(self.beto_indice) != len(self.beto_emb)):
            print("[BETO] embeddings_resenas.npy y resenas_indice.csv no coinciden; no se dibuja el mapa.")
            self.beto_emb = None
        if self.beto_emb is not None:
            self.beto_xy = PCA(n_components=2, random_state=42).fit_transform(self.beto_emb)
        self.preparar_comparacion()
 
    # ---------- Utilidades de diseño ----------
    def estilo_figura(self, fig, alto=420, ejes=True):
        fig.update_layout(template="plotly_dark", paper_bgcolor=BLANCO, plot_bgcolor="#222222",
                          font={"color": TEXTO_BLANCO}, height=alto,
                          margin={"l": 10, "r": 10, "t": 40, "b": 10}, hoverlabel={"align": "left"})
        if not ejes:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
        return fig
 
    def figura_vacia(self, mensaje):
        fig = px.scatter()
        fig.update_layout(annotations=[{"text": mensaje, "showarrow": False,
                                        "font": {"color": GRIS, "size": 16}}])
        return self.estilo_figura(fig, 250, ejes=False)
 
    def aviso(self, texto):
        return html.P(texto, style={"color": GRIS, "fontStyle": "italic"})
 
    def bloque(self, titulo, descripcion, hijos):
        """Bloque con título y descripción corta dentro de una pestaña."""
        return html.Div([
            html.H3(titulo, style={"marginTop": 0, "marginBottom": "4px", "color": TEXTO_BLANCO}),
            html.P(descripcion, style={"color": GRIS, "fontSize": "13px", "marginTop": 0}),
            *hijos,
        ], style={"background": "#222222", "padding": "16px", "borderRadius": "10px", "marginTop": "16px"})
 
    def tarjeta_texto(self, titulo, texto, color=MORADO, extra=""):
        return html.Div([
            html.B(f"{titulo}  ", style={"color": TEXTO_BLANCO}),
            html.Span(extra, style={"color": GRIS, "fontSize": "13px"}),
            html.Div(texto, style={"color": "#dfe6e9", "marginTop": "4px"}),
        ], style={"borderLeft": f"4px solid {color}", "padding": "8px 12px",
                  "margin": "8px 0", "background": "#2f2f2f", "borderRadius": "4px"})
 
    def tabla_simple(self, datos):
        return dash_table.DataTable(
            data=datos.to_dict("records"),
            columns=[{"name": c, "id": c} for c in datos.columns],
            style_header={"backgroundColor": MORADO, "color": "white", "fontWeight": "bold", "border": "none"},
            style_cell={"padding": "10px", "textAlign": "left", "backgroundColor": "#333333",
                        "color": "#f5f6fa", "border": "1px solid #444", "whiteSpace": "normal", "height": "auto"},
            style_table={"overflowX": "auto"})
 
    def color_rating(self, rating):
        rating = pd.to_numeric(rating, errors="coerce")
        if pd.isna(rating):
            return GRIS
        return COLORES_VALORACION["positiva" if rating >= 7 else "negativa" if rating <= 4 else "intermedia"]
 
    # ---------- Cálculos de Word2Vec ----------
    def tabla_calcos(self):
        """Frecuencia (por 10.000 tokens) de posibles calcos en cada origen y sus vecinos."""
        w2v = self.w2v
        grupos = dict(tuple(w2v.df.groupby("origen")))
        filas = []
        for palabra in w2v.CALCOS:
            fila = {"Palabra": palabra}
            for origen, g in grupos.items():
                total = g["tokens"].str.len().sum()
                n = sum(t.count(palabra) for t in g["tokens"])
                fila[f"{origen.capitalize()} (por 10 mil)"] = round(n / total * 10000, 2)
            fila["Vecinos más cercanos"] = (", ".join(w for w, _ in w2v.wv.most_similar(palabra, topn=5))
                                            if palabra in w2v.wv else "(fuera del vocabulario)")
            filas.append(fila)
        return pd.DataFrame(filas)
 
    def figura_campos(self):
        """t-SNE de las palabras semilla de cada campo semántico."""
        w2v = self.w2v
        palabras, campos = [], []
        for campo, semillas in w2v.CAMPOS.items():
            for s in semillas:
                if s in w2v.wv:
                    palabras.append(s)
                    campos.append(campo)
        X = np.array([w2v.wv[p] for p in palabras])
        xy = TSNE(n_components=2, random_state=42, perplexity=min(8, len(palabras) - 1)).fit_transform(X)
        graf = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "palabra": palabras, "campo": campos})
        fig = px.scatter(graf, x="x", y="y", color="campo", text="palabra",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(marker={"size": 12}, textposition="top center")
        fig.update_layout(legend={"title": {"text": "Campo semántico"}})
        return self.estilo_figura(fig, 520, ejes=False)
 
    def figura_vecinos(self, palabra):
        palabra = (palabra or "").strip().lower()
        vector = self.w2v._vector(palabra) if palabra else None
        if vector is None:
            return self.figura_vacia("Esa palabra no está en el vocabulario"), "Prueba con otra palabra (por ejemplo: guion, actuación, aburrida)."
        excluir = set(palabra.split())
        vecinos = [(w, s) for w, s in self.w2v.wv.similar_by_vector(vector, topn=15) if w not in excluir][:10]
        graf = pd.DataFrame(vecinos, columns=["palabra", "similitud"]).iloc[::-1]
        fig = px.bar(graf, x="similitud", y="palabra", orientation="h", text=graf["similitud"].round(3))
        fig.update_traces(marker_color=MORADO, textposition="outside")
        fig.update_layout(xaxis_title="Similitud coseno", yaxis_title=None)
        return self.estilo_figura(fig, 380), f"Las 10 palabras más cercanas a «{palabra}» en el espacio vectorial."
 
    def figura_similitud(self, columna):
        tabla = self.w2v_sim[columna]
        nombres = list(tabla.index)
        pares = [(nombres[i], nombres[j], 1 - tabla.iloc[i, j])
                 for i in range(len(nombres)) for j in range(i + 1, len(nombres))]
        pares.sort(key=lambda p: p[2])
        fuera_diagonal = tabla.values[~np.eye(len(tabla), dtype=bool)]
        fig = px.imshow(tabla, text_auto=".3f", color_continuous_scale="YlOrRd",
                        zmin=float(fuera_diagonal.min()) - 0.005 if len(fuera_diagonal) else None, zmax=1)
        fig.update_layout(coloraxis_colorbar={"title": "Similitud"})
        nota = (f"Más cercanos: {pares[0][0]} y {pares[0][1]} (distancia coseno {pares[0][2]:.4f}) · "
                f"Más lejanos: {pares[-1][0]} y {pares[-1][1]} (distancia {pares[-1][2]:.4f}). "
                "El color se ajusta al rango real para que se noten las diferencias.") if pares else ""
        return self.estilo_figura(fig, 460), nota
 
    def resultado_analogia(self, indice):
        analogia = self.w2v_analogias[int(indice)]
        if analogia["faltan"]:
            return self.aviso(f"No se puede calcular: faltan en el vocabulario {analogia['faltan']}.")
        return [self.tarjeta_texto(f"{i}. {palabra}", "", MORADO, f"similitud {sim:.3f}")
                for i, (palabra, sim) in enumerate(analogia["res"], start=1)]
 
    def vocabulario_caracteristico(self, columna):
        tarjetas = []
        for categoria, palabras in self.w2v_excl[columna].items():
            chips = [html.Span(p, style={"display": "inline-block", "background": "#3a3a3a", "color": TEXTO_BLANCO,
                                         "padding": "3px 10px", "borderRadius": "12px", "margin": "3px",
                                         "fontSize": "13px"}) for p in palabras]
            tarjetas.append(html.Div([html.B(categoria, style={"color": MORADO}), html.Div(chips)],
                                     style={"flex": "1 1 240px", "background": "#2f2f2f", "padding": "10px 12px",
                                            "borderRadius": "8px"}))
        return html.Div(tarjetas, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})
 
    # ---------- Cálculos de BETO ----------
    def figura_polisemia(self):
        datos = self.beto_pol.rename(columns={
            "sim_promedio_positivas": "Entre contextos positivos",
            "sim_promedio_negativas": "Entre contextos negativos",
            "sim_promedio_entre_polaridades": "Positivo vs negativo"})
        largo = datos.melt(id_vars="palabra", value_vars=["Entre contextos positivos", "Entre contextos negativos",
                                                          "Positivo vs negativo"],
                           var_name="comparación", value_name="similitud").dropna()
        fig = px.bar(largo, x="palabra", y="similitud", color="comparación", barmode="group",
                     color_discrete_sequence=["#00b894", "#ff6b81", "#74b9ff"])
        fig.update_layout(xaxis_title=None, yaxis_title="Similitud coseno promedio", legend={"title": None})
        return self.estilo_figura(fig, 400)
 
    def ejemplos_polisemia(self, palabra):
        fila = self.beto_pol[self.beto_pol["palabra"] == palabra].iloc[0]
 
        def texto(valor):
            return valor if pd.notna(valor) and str(valor).strip() else "(sin ejemplo en el corpus)"
 
        return [self.tarjeta_texto("Contexto positivo", texto(fila["oracion_positiva"]), COLORES_VALORACION["positiva"],
                                   f"({int(fila['n_positivas'])} oraciones analizadas)"),
                self.tarjeta_texto("Contexto negativo", texto(fila["oracion_negativa"]), COLORES_VALORACION["negativa"],
                                   f"({int(fila['n_negativas'])} oraciones analizadas)")]
 
    def resultados_busqueda_beto(self, consulta):
        filas = self.beto_busq[self.beto_busq["consulta"] == consulta]
        return [self.tarjeta_texto(f"{i}. {fila['genero']}", fila["fragmento"], self.color_rating(fila["rating"]),
                                   f"⭐ {fila['rating']}/10 · similitud {fila['similitud']:.3f}")
                for i, (_, fila) in enumerate(filas.iterrows(), start=1)]
 
    def figura_mlm(self, plantilla):
        datos = self.beto_mlm[self.beto_mlm["plantilla"] == plantilla].sort_values("ranking")
        fig = px.bar(datos.iloc[::-1], x="probabilidad", y="prediccion", orientation="h",
                     color="sentimiento_estimado", color_discrete_map=COLOR_SENTIMIENTO,
                     text=datos.iloc[::-1]["probabilidad"].round(3))
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Probabilidad", yaxis_title=None, legend={"title": {"text": "Sentimiento estimado"}})
        return self.estilo_figura(fig, 340)
 
    def figura_mapa_beto(self, color):
        indice = self.beto_indice
        graf = indice.assign(
            x=self.beto_xy[:, 0], y=self.beto_xy[:, 1], pos=range(len(indice)),
            rating=pd.to_numeric(indice["rating"], errors="coerce"),
            resumen=indice["review_summary"].fillna("").astype(str).map(
                lambda t: textwrap.fill(t[:200], 55).replace("\n", "<br>")))
        opciones = ({"color_continuous_scale": "Plasma"} if color == "rating"
                    else {"color_discrete_sequence": px.colors.qualitative.Bold})
        fig = px.scatter(graf, x="x", y="y", color=color,
                         custom_data=["pos", "main_genre", "rating", "resumen"], **opciones)
        fig.update_traces(marker={"size": 9, "opacity": 0.8},
                          hovertemplate="<b>%{customdata[1]}</b> · ⭐ %{customdata[2]}/10<br>%{customdata[3]}<extra></extra>")
        return self.estilo_figura(fig, 520, ejes=False)
 
    def resenas_similares_beto(self, pos, cantidad=5):
        similitud = self.beto_emb @ self.beto_emb[pos]   # los vectores de BETO ya vienen normalizados
        orden = [i for i in np.argsort(-similitud) if i != pos][:cantidad]
 
        def tarjeta(i, extra=""):
            fila = self.beto_indice.iloc[i]
            texto = " ".join(str(fila["review_text_es"]).split())
            return self.tarjeta_texto(str(fila["main_genre"]), texto[:300] + ("..." if len(texto) > 300 else ""),
                                      self.color_rating(fila["rating"]), f"⭐ {fila['rating']}/10 {extra}")
 
        return ([html.H4("Reseña seleccionada", style={"marginBottom": 0, "color": TEXTO_BLANCO}), tarjeta(pos),
                 html.H4(f"Las {cantidad} reseñas más parecidas (según BETO)", style={"marginBottom": 0, "color": TEXTO_BLANCO})]
                + [tarjeta(i, f"· similitud {similitud[i] * 100:.0f} %") for i in orden])
 
    # ---------- Comparación BETO vs Word2Vec ----------
    def vector_w2v(self, texto):
        """Vector promedio de un texto con Word2Vec (las stopwords no están en el vocabulario)."""
        wv = self.w2v.wv
        vectores = [wv[p] for p in re.findall(r"[a-záéíóúüñ]+", str(texto).lower()) if p in wv]
        return np.mean(vectores, axis=0) if vectores else None
 
    def preparar_comparacion(self):
        """Hace la misma búsqueda semántica con Word2Vec sobre las reseñas del índice de BETO."""
        self.cmp_busqueda = {}
        if self.w2v is None or self.beto_busq is None or self.beto_indice is None:
            return
 
        textos = [" ".join(str(t).split()) for t in self.beto_indice["review_text_es"].fillna("")]
        vectores = [self.vector_w2v(t) for t in textos]
        matriz = np.array([v if v is not None else np.zeros(self.w2v.wv.vector_size) for v in vectores])
        normas = np.linalg.norm(matriz, axis=1)
        matriz_norm = matriz / np.where(normas == 0, 1, normas)[:, None]
 
        def resumen(i, similitud):
            fila = self.beto_indice.iloc[i]
            return {"idx": i, "similitud": float(similitud), "rating": fila["rating"], "genero": fila["main_genre"],
                    "fragmento": textos[i][:180] + ("..." if len(textos[i]) > 180 else "")}
 
        for consulta in self.beto_busq["consulta"].unique():
            vector = self.vector_w2v(consulta)
            if vector is None:
                continue
            similitudes = matriz_norm @ (vector / np.linalg.norm(vector))
            similitudes[normas == 0] = -1
            w2v_top = [resumen(i, similitudes[i]) for i in np.argsort(-similitudes)[:5]]
 
            beto_top = []
            for _, fila in self.beto_busq[self.beto_busq["consulta"] == consulta].iterrows():
                inicio = str(fila["fragmento"]).removesuffix("...")
                idx = next((i for i, t in enumerate(textos) if t.startswith(inicio)), None)
                beto_top.append({"idx": idx, "similitud": float(fila["similitud"]), "rating": fila["rating"],
                                 "genero": fila["genero"], "fragmento": fila["fragmento"]})
 
            coinciden = len({r["idx"] for r in beto_top if r["idx"] is not None} & {r["idx"] for r in w2v_top})
            self.cmp_busqueda[consulta] = {"beto": beto_top, "w2v": w2v_top, "coinciden": coinciden}
 
    def figura_comparacion_polisemia(self):
        wv = self.w2v.wv if self.w2v is not None else None
 
        def en_vocabulario(palabra):
            if wv is None:
                return None
            sin_acento = "".join(c for c in unicodedata.normalize("NFD", palabra) if unicodedata.category(c) != "Mn")
            return palabra in wv or sin_acento in wv
 
        filas = []
        for _, fila in self.beto_pol.iterrows():
            filas.append({"palabra": fila["palabra"], "modelo": "BETO (varía con el contexto)",
                          "similitud promedio": fila["sim_promedio_global"]})
            if en_vocabulario(fila["palabra"]):
                filas.append({"palabra": fila["palabra"], "modelo": "Word2Vec (un solo vector)",
                              "similitud promedio": 1.0})
        fig = px.bar(pd.DataFrame(filas).dropna(), x="palabra", y="similitud promedio", color="modelo",
                     barmode="group", color_discrete_sequence=[MORADO, "#74b9ff"])
        fig.update_layout(xaxis_title=None, legend={"title": None})
        return self.estilo_figura(fig, 400)
 
    def resultado_comparacion_busqueda(self, consulta):
        datos = self.cmp_busqueda[consulta]
 
        def columna(titulo, filas):
            tarjetas = [self.tarjeta_texto(f"{i}. {r['genero']}", r["fragmento"], self.color_rating(r["rating"]),
                                           f"⭐ {r['rating']}/10 · similitud {r['similitud']:.3f}")
                        for i, r in enumerate(filas, start=1)]
            return html.Div([html.H4(titulo, style={"color": MORADO, "marginBottom": 0}), *tarjetas],
                            style={"flex": "1 1 380px"})
 
        return [html.P(f"Coinciden {datos['coinciden']} de 5 reseñas entre los dos modelos.",
                       style={"color": TEXTO_BLANCO, "fontWeight": "600"}),
                html.Div([columna("BETO (contexto)", datos["beto"]),
                          columna("Word2Vec (promedio de palabras)", datos["w2v"])],
                         style={"display": "flex", "gap": "20px", "flexWrap": "wrap"})]
 
    def tabla_resumen_modelos(self):
        dim_beto = self.beto_emb.shape[1] if self.beto_emb is not None else 768
        dim_w2v = self.w2v.wv.vector_size if self.w2v is not None else 100
        return pd.DataFrame([
            {"Característica": "Tipo de vector", "Word2Vec": "Estático: un vector por palabra",
             "BETO": "Contextual: el vector cambia según la oración"},
            {"Característica": "Entrenamiento", "Word2Vec": "Entrenado con el corpus del proyecto",
             "BETO": "Preentrenado (dccuchile/bert-base-spanish-wwm-cased)"},
            {"Característica": "Dimensión", "Word2Vec": str(dim_w2v), "BETO": str(dim_beto)},
            {"Característica": "Polisemia", "Word2Vec": "No la distingue", "BETO": "Sí la distingue"},
            {"Característica": "Vector de una reseña", "Word2Vec": "Promedio de los vectores de sus palabras",
             "BETO": "Promedio de los tokens (mean pooling)"},
            {"Característica": "Costo de cómputo", "Word2Vec": "Muy bajo", "BETO": "Alto (conviene GPU)"},
        ])
 
    # ---------- Diseño ----------
    def seccion_embeddings(self):
        estilo = {"backgroundColor": "#333333", "color": GRIS, "border": "none", "padding": "10px"}
        estilo_activo = {**estilo, "backgroundColor": MORADO, "color": "white", "fontWeight": "bold"}
        pestanas = [("Word2Vec", "w2v"), ("BETO", "beto"), ("Comparación", "cmp")]
 
        return self.caja([
            html.H2("Embeddings: Word2Vec y BETO", style={"marginTop": 0, "color": TEXTO_BLANCO}),
            html.P("Word2Vec asigna un vector fijo a cada palabra; BETO genera vectores que dependen del contexto. "
                   "Esta sección no usa los filtros de arriba.", style={"color": GRIS}),
            dcc.Tabs(id="emb-tabs", value="w2v", style={"marginTop": "10px"},
                     children=[dcc.Tab(label=etiqueta, value=valor, style=estilo, selected_style=estilo_activo)
                               for etiqueta, valor in pestanas]),
            html.Div(id="emb-contenido"),
        ], {"marginTop": "20px"})
 
    def contenido_w2v(self):
        if self.w2v is None:
            return self.aviso(f"No se pudo preparar Word2Vec ({self.w2v_error}). "
                              f"Revisa que existan {CSV_UNIFICADO.name} y src/embeddings_w2v.py.")
        w2v = self.w2v
        estilo_input = {"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #444",
                        "background": "#333333", "color": TEXTO_BLANCO}
        grafico = {"displaylogo": False}
 
        return html.Div([
            html.P(f"{len(w2v.wv):,} palabras · {w2v.wv.vector_size} dimensiones · {len(w2v.df):,} reseñas de "
                   f"{CSV_UNIFICADO.name} · {self.w2v_origen}", style={"color": GRIS, "fontSize": "13px"}),
 
            self.bloque("Vecinos semánticos", "Escribe una palabra (o una frase como «obra maestra») y pulsa Enter.", [
                dcc.Input(id="w2v-palabra", type="text", value="guion", debounce=True, style=estilo_input),
                html.P(id="w2v-vecinos-nota", style={"color": GRIS, "fontSize": "13px"}),
                dcc.Graph(id="w2v-vecinos-grafico", config=grafico)]),
 
            self.bloque("Campos semánticos", "Palabras semilla de cada tema proyectadas en 2D (t-SNE). "
                        "Si un tema está bien capturado, sus palabras quedan juntas.",
                        [dcc.Graph(figure=self.w2v_fig_campos, config=grafico)]),
 
            self.bloque("Analogías vectoriales", "a − b + c: se muestran las 5 palabras más cercanas al resultado.", [
                dcc.Dropdown(id="w2v-analogia-sel", clearable=False, value=0, style={"color": "#333"},
                             options=[{"label": a["desc"], "value": i} for i, a in enumerate(self.w2v_analogias)]),
                html.Div(id="w2v-analogia-res")]),
 
            self.bloque("Similitud entre categorías", "Similitud coseno entre los vectores promedio (centroides) de cada categoría.", [
                dcc.Dropdown(id="w2v-sim-col", clearable=False, value="genre", style={"color": "#333"},
                             options=[{"label": n, "value": c} for c, n in COLUMNAS_W2V.items()]),
                html.P(id="w2v-sim-nota", style={"color": GRIS, "fontSize": "13px"}),
                dcc.Graph(id="w2v-sim-grafico", config=grafico)]),
 
            self.bloque("Vocabulario característico", "Palabras más cercanas al centroide de cada categoría "
                        "y más lejanas del resto.", [
                dcc.Dropdown(id="w2v-excl-col", clearable=False, value="genre", style={"color": "#333"},
                             options=[{"label": n, "value": c} for c, n in COLUMNAS_W2V.items()]),
                html.Div(id="w2v-excl-res", style={"marginTop": "10px"})]),
 
            self.bloque("Posibles calcos de traducción", "Palabras que suenan a traducción literal del inglés: "
                        "frecuencia en cada origen y vecinos en el espacio vectorial.",
                        [self.tabla_simple(self.w2v_calcos)]),
        ])
 
    def contenido_beto(self):
        def falta(archivo):
            return self.aviso(f"No se encontró {archivo} en data/results/beto/. Ejecuta primero src/embeding_beto.py.")
 
        grafico = {"displaylogo": False}
 
        if self.beto_pol is None:
            polisemia = [falta("polisemia_resumen.csv")]
        else:
            palabras = list(self.beto_pol["palabra"])
            polisemia = [dcc.Graph(figure=self.figura_polisemia(), config=grafico),
                         dcc.Dropdown(id="beto-pol-sel", clearable=False, value=palabras[0], options=palabras,
                                      style={"color": "#333"}),
                         html.Div(id="beto-pol-ejemplos")]
 
        if self.beto_busq is None:
            busqueda = [falta("busqueda_resenas.csv")]
        else:
            consultas = list(self.beto_busq["consulta"].unique())
            busqueda = [dcc.Dropdown(id="beto-busq-sel", clearable=False, value=consultas[0], options=consultas,
                                     style={"color": "#333"}),
                        html.Div(id="beto-busq-res")]
 
        if self.beto_mlm is None:
            mlm = [falta("mlm_predicciones.csv")]
        else:
            plantillas = list(self.beto_mlm["plantilla"].unique())
            mlm = [dcc.Dropdown(id="beto-mlm-sel", clearable=False, value=plantillas[0], options=plantillas,
                                style={"color": "#333"}),
                   dcc.Graph(id="beto-mlm-grafico", config=grafico)]
 
        if self.beto_emb is None:
            mapa = [falta("embeddings_resenas.npy y resenas_indice.csv")]
        else:
            mapa = [dcc.Dropdown(id="beto-mapa-color", clearable=False, value="main_genre", style={"color": "#333"},
                                 options=[{"label": "Género", "value": "main_genre"},
                                          {"label": "Rating", "value": "rating"}]),
                    dcc.Graph(id="beto-mapa-grafico", config=grafico),
                    html.Div(id="beto-mapa-similares",
                             children=self.aviso("Haz clic en un punto para ver las reseñas más parecidas."))]
 
        return html.Div([
            self.bloque("Polisemia contextual", "La misma palabra recibe vectores distintos según la oración. "
                        "Si la similitud entre contextos positivos y negativos es baja, BETO distingue los significados.",
                        polisemia),
            self.bloque("Búsqueda semántica", "Reseñas más parecidas a una consulta en lenguaje natural "
                        "(resultados guardados por src/embeding_beto.py).", busqueda),
            self.bloque("Masked Language Model", "Palabras que BETO predice para el hueco [MASK] de cada frase.", mlm),
            self.bloque("Mapa de reseñas (BETO)", "Cada punto es una reseña proyectada en 2D con PCA. "
                        "Puntos cercanos = significado parecido.", mapa),
        ])
 
    def contenido_comparacion(self):
        grafico = {"displaylogo": False}
        if self.beto_pol is None:
            polisemia = [self.aviso("Falta polisemia_resumen.csv de BETO.")]
        else:
            polisemia = [dcc.Graph(figure=self.figura_comparacion_polisemia(), config=grafico)]
 
        if not self.cmp_busqueda:
            busqueda = [self.aviso("Para comparar la búsqueda se necesitan Word2Vec y los archivos de BETO "
                                   "(busqueda_resenas.csv y resenas_indice.csv).")]
        else:
            consultas = list(self.cmp_busqueda)
            busqueda = [dcc.Dropdown(id="cmp-busq-sel", clearable=False, value=consultas[0], options=consultas,
                                     style={"color": "#333"}),
                        html.Div(id="cmp-busq-res")]
 
        return html.Div([
            self.bloque("Polisemia: ¿cambia el vector según el contexto?",
                        "Word2Vec devuelve siempre el mismo vector para una palabra (similitud 1.0). "
                        "BETO produce vectores distintos en cada oración, por eso su similitud promedio es menor.", polisemia),
            self.bloque("Búsqueda semántica: BETO vs Word2Vec",
                        "La misma consulta se busca en las mismas reseñas con ambos modelos "
                        "(Word2Vec compara el promedio de los vectores de las palabras).", busqueda),
            self.bloque("Resumen de los dos enfoques", "", [self.tabla_simple(self.tabla_resumen_modelos())]),
        ])
 
    # ---------- Interacción ----------
    def crear_callbacks_embeddings(self):
        # El contenido de las pestañas se crea al elegirlas, así que Dash no puede ver esos ids de antemano
        self.app.config.suppress_callback_exceptions = True
 
        @self.app.callback(Output("emb-contenido", "children"), Input("emb-tabs", "value"))
        def cambiar_pestana(pestana):
            contenido = {"w2v": self.contenido_w2v, "beto": self.contenido_beto, "cmp": self.contenido_comparacion}
            return contenido[pestana]()
 
        # --- Word2Vec ---
        @self.app.callback(Output("w2v-vecinos-grafico", "figure"), Output("w2v-vecinos-nota", "children"),
                           Input("w2v-palabra", "value"))
        def vecinos(palabra):
            return self.figura_vecinos(palabra)
 
        @self.app.callback(Output("w2v-analogia-res", "children"), Input("w2v-analogia-sel", "value"))
        def analogia(indice):
            return self.resultado_analogia(indice)
 
        @self.app.callback(Output("w2v-sim-grafico", "figure"), Output("w2v-sim-nota", "children"),
                           Input("w2v-sim-col", "value"))
        def similitud(columna):
            return self.figura_similitud(columna)
 
        @self.app.callback(Output("w2v-excl-res", "children"), Input("w2v-excl-col", "value"))
        def exclusivo(columna):
            return self.vocabulario_caracteristico(columna)
 
        # --- BETO ---
        @self.app.callback(Output("beto-pol-ejemplos", "children"), Input("beto-pol-sel", "value"))
        def polisemia(palabra):
            return self.ejemplos_polisemia(palabra)
 
        @self.app.callback(Output("beto-busq-res", "children"), Input("beto-busq-sel", "value"))
        def busqueda_beto(consulta):
            return self.resultados_busqueda_beto(consulta)
 
        @self.app.callback(Output("beto-mlm-grafico", "figure"), Input("beto-mlm-sel", "value"))
        def mlm(plantilla):
            return self.figura_mlm(plantilla)
 
        @self.app.callback(Output("beto-mapa-grafico", "figure"), Input("beto-mapa-color", "value"))
        def mapa_beto(color):
            return self.figura_mapa_beto(color)
 
        @self.app.callback(Output("beto-mapa-similares", "children"), Input("beto-mapa-grafico", "clickData"))
        def similares_beto(click):
            if not click:
                return self.aviso("Haz clic en un punto para ver las reseñas más parecidas.")
            return self.resenas_similares_beto(int(click["points"][0]["customdata"][0]))
 
        # --- Comparación ---
        @self.app.callback(Output("cmp-busq-res", "children"), Input("cmp-busq-sel", "value"))
        def comparar_busqueda(consulta):
            return self.resultado_comparacion_busqueda(consulta)
 
    def run(self):
        self.app.run(debug=True)
 
 
if __name__ == "__main__":
    DashboardPeliculas().run()