import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, dash_table, Input, Output, State
 
# --- NUEVO (sección Embeddings): librerías extra ---
import textwrap
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
 
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
 
# --- NUEVO (sección Embeddings): configuración ---
# Si ya tienen los embeddings guardados, pongan aquí el nombre del archivo .npy (una fila por reseña,
# en el mismo orden que resenas_es.csv). Si no existe, se usa una aproximación con TF-IDF.
ARCHIVOS_EMBEDDINGS = [RAIZ / "data" / "processed" / "embeddings.npy",
                       RAIZ / "data" / "processed" / "embeddings_resenas.npy"]
MAX_PUNTOS_EMBEDDINGS = 4000   # máximo de reseñas dibujadas (para que el gráfico sea rápido)
MENSAJE_EMBEDDINGS = "Haz clic en un punto del gráfico para ver las reseñas más parecidas."
 
 
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
 
    # =====================================================================
    # SECCIÓN EMBEDDINGS
    # =====================================================================
 
    # ---------- Datos de embeddings ----------
    def cargar_matriz_embeddings(self):
        """Devuelve (matriz, texto) con un vector por reseña, en el orden de self.df."""
        # 1. Columnas del propio CSV que empiecen con "emb_"
        columnas = [c for c in self.df.columns if str(c).startswith("emb_")]
        if columnas:
            return self.df[columnas].to_numpy(dtype=float), f"Embeddings del CSV ({len(columnas)} dimensiones)"
 
        # 2. Archivo .npy con una fila por reseña
        for archivo in ARCHIVOS_EMBEDDINGS:
            if archivo.exists():
                matriz = np.load(archivo)
                if matriz.ndim == 2 and matriz.shape[0] == len(self.df):
                    return matriz.astype(float), f"Embeddings de {archivo.name} ({matriz.shape[1]} dimensiones)"
                print(f"Aviso: {archivo.name} tiene {matriz.shape[0]} filas y el CSV {len(self.df)}; no se usa.")
 
        # 3. Si no hay embeddings guardados, se aproximan con TF-IDF + SVD sobre el texto de la reseña
        texto = self.df["review_summary_es"].fillna("").astype(str)
        tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True).fit_transform(texto)
        componentes = max(2, min(100, tfidf.shape[1] - 1))
        matriz = TruncatedSVD(n_components=componentes, random_state=42).fit_transform(tfidf)
        return matriz, "Aproximación con TF-IDF + SVD (no se encontró un archivo de embeddings)"
 
    def preparar_embeddings(self):
        matriz, self.emb_origen = self.cargar_matriz_embeddings()
 
        # Muestra fija si hay demasiadas reseñas
        if len(self.df) > MAX_PUNTOS_EMBEDDINGS:
            self.emb_df = self.df.sample(MAX_PUNTOS_EMBEDDINGS, random_state=42)
        else:
            self.emb_df = self.df
        filas = self.df.index.get_indexer(self.emb_df.index)
        self.emb_matriz = matriz[filas]
 
        # Versión normalizada para medir la similitud coseno
        normas = np.linalg.norm(self.emb_matriz, axis=1, keepdims=True)
        self.emb_norm = self.emb_matriz / np.where(normas == 0, 1, normas)
 
        self.emb_proyecciones = {}   # aquí se guardan las proyecciones ya calculadas
        self.emb_varianza = None
 
    def proyectar(self, metodo):
        """Lleva los vectores a 2 dimensiones (PCA o t-SNE). Se calcula una sola vez."""
        if metodo not in self.emb_proyecciones:
            if metodo == "PCA":
                pca = PCA(n_components=2, random_state=42)
                self.emb_proyecciones[metodo] = pca.fit_transform(self.emb_matriz)
                self.emb_varianza = pca.explained_variance_ratio_.sum()
            else:
                n = len(self.emb_matriz)
                perplexity = max(5, min(30, (n - 1) // 3))
                self.emb_proyecciones[metodo] = TSNE(
                    n_components=2, init="pca", perplexity=perplexity, random_state=42
                ).fit_transform(self.emb_matriz)
        return self.emb_proyecciones[metodo]
 
    # ---------- Gráfico ----------
    def figura_vacia(self, mensaje):
        fig = px.scatter()
        fig.update_layout(template="plotly_dark", paper_bgcolor=BLANCO, plot_bgcolor="#222222", height=300,
                          xaxis={"visible": False}, yaxis={"visible": False},
                          annotations=[{"text": mensaje, "showarrow": False, "font": {"color": GRIS, "size": 16}}])
        return fig
 
    def graficar_embeddings(self, genero, periodo, valoracion, spoilers, metodo, color):
        datos = self.filtrar(genero, periodo, valoracion, spoilers)
        datos = datos[datos.index.isin(self.emb_df.index)]
        if datos.empty:
            return self.figura_vacia("No hay reseñas con los filtros actuales."), "Sin reseñas para mostrar."
 
        xy = self.proyectar(metodo)
        filas = self.emb_df.index.get_indexer(datos.index)
        graf = datos.assign(
            x=xy[filas, 0], y=xy[filas, 1], pos=filas,
            resumen_corto=datos["review_summary_es"].astype(str).map(
                lambda t: textwrap.fill(t[:200], 55).replace("\n", "<br>")),
        )
 
        opciones = {}
        if color == "valoracion":
            opciones["color_discrete_map"] = COLORES_VALORACION
        elif color == "rating":
            opciones["color_continuous_scale"] = "Plasma"
        else:
            opciones["color_discrete_sequence"] = px.colors.qualitative.Bold
 
        fig = px.scatter(graf, x="x", y="y", color=color,
                         custom_data=["pos", "primaryTitle", "rating", "valoracion", "resumen_corto"],
                         **opciones)
        fig.update_traces(marker={"size": 8, "opacity": 0.8},
                          hovertemplate="<b>%{customdata[1]}</b><br>⭐ %{customdata[2]}/10 · %{customdata[3]}"
                                        "<br>%{customdata[4]}<extra></extra>")
        fig.update_layout(template="plotly_dark", paper_bgcolor=BLANCO, plot_bgcolor="#222222",
                          font={"color": TEXTO_BLANCO}, height=560, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                          xaxis={"visible": False}, yaxis={"visible": False},
                          legend={"title": {"text": color}}, hoverlabel={"align": "left"})
 
        info = f"Mostrando {len(graf)} de {len(self.emb_df)} reseñas · {self.emb_origen}"
        if metodo == "PCA" and self.emb_varianza is not None:
            info += f" · Las 2 dimensiones de PCA explican el {self.emb_varianza * 100:.0f} % de la información"
        return fig, info
 
    # ---------- Reseñas parecidas ----------
    def tarjeta_resena(self, fila, extra=""):
        color = COLORES_VALORACION[fila["valoracion"]]
        return html.Div([
            html.B(f"{fila['primaryTitle']}  ", style={"color": TEXTO_BLANCO}),
            html.Span(f"⭐ {fila['rating']}/10 {extra}", style={"color": GRIS}),
            html.Div(fila["review_summary_es"], style={"color": "#dfe6e9", "marginTop": "4px"}),
        ], style={"borderLeft": f"4px solid {color}", "padding": "8px 12px",
                  "margin": "8px 0", "background": "#222222", "borderRadius": "4px"})
 
    def resenas_similares(self, pos, cantidad=5):
        similitud = self.emb_norm @ self.emb_norm[pos]
        orden = [i for i in np.argsort(-similitud) if i != pos][:cantidad]
 
        lista = [html.H4("Reseña seleccionada", style={"marginBottom": 0, "color": TEXTO_BLANCO}),
                 self.tarjeta_resena(self.emb_df.iloc[pos]),
                 html.H4(f"Las {cantidad} reseñas más parecidas", style={"marginBottom": 0, "color": TEXTO_BLANCO})]
        for i in orden:
            lista.append(self.tarjeta_resena(self.emb_df.iloc[i], f"· similitud {similitud[i] * 100:.0f} %"))
        return lista
 
    # ---------- Diseño ----------
    def seccion_embeddings(self):
        controles = html.Div([
            self.filtro("Proyección", dcc.Dropdown(
                id="emb-proyeccion", clearable=False, value="PCA",
                options=[{"label": "PCA (rápida)", "value": "PCA"},
                         {"label": "t-SNE (más detalle, tarda un poco)", "value": "t-SNE"}],
                style={"color": "#333"})),
 
            self.filtro("Colorear por", dcc.Dropdown(
                id="emb-color", clearable=False, value="valoracion",
                options=[{"label": "Valoración", "value": "valoracion"},
                         {"label": "Género", "value": "genero"},
                         {"label": "Periodo de la reseña", "value": "periodo"},
                         {"label": "Rating", "value": "rating"}],
                style={"color": "#333"})),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"})
 
        return self.caja([
            html.H2("Embeddings", style={"marginTop": 0, "color": TEXTO_BLANCO}),
            html.P("Cada punto es una reseña. Las reseñas con significado parecido quedan cerca. "
                   "El gráfico responde a los filtros de arriba (género, periodo, valoración y spoilers).",
                   style={"color": GRIS}),
            controles,
            html.P(id="emb-info", style={"color": GRIS, "fontSize": "13px"}),
            dcc.Loading(dcc.Graph(id="emb-grafico", config={"displaylogo": False}),
                        type="circle", color=MORADO),
            html.Div(id="emb-similares", children=html.P(MENSAJE_EMBEDDINGS, style={"color": GRIS})),
        ], {"marginTop": "20px"})
 
    # ---------- Interacción ----------
    def crear_callbacks_embeddings(self):
        @self.app.callback(
            Output("emb-grafico", "figure"),
            Output("emb-info", "children"),
            Input("f-genero", "value"), Input("f-periodo", "value"),
            Input("f-valoracion", "value"), Input("f-spoilers", "value"),
            Input("emb-proyeccion", "value"), Input("emb-color", "value"),
        )
        def actualizar_embeddings(genero, periodo, valoracion, spoilers, metodo, color):
            return self.graficar_embeddings(genero, periodo, valoracion, spoilers, metodo, color)
 
        @self.app.callback(
            Output("emb-similares", "children"),
            Input("emb-grafico", "clickData"),
        )
        def mostrar_similares(click):
            if not click:
                return html.P(MENSAJE_EMBEDDINGS, style={"color": GRIS})
            return self.resenas_similares(int(click["points"][0]["customdata"][0]))
 
    def run(self):
        self.app.run(debug=True)
 
 
if __name__ == "__main__":
    DashboardPeliculas().run()