from pathlib import Path  # sirve para armar las rutas de las carpetas del proyecto
import matplotlib.pyplot as plt  # libreria para hacer los graficos
import pandas as pd  # libreria para leer los CSV y hacer las tablas

#-------------------------------------------------------------------------------------------#

# La carpeta principal del proyecto es la que esta un nivel arriba de src/
RAIZ = Path(__file__).resolve().parent.parent
# Resultado del etiquetado con spaCy (un token por fila)
CSV_POS = RAIZ / "data" / "results" / "pos_analisis_spacy.csv"
# CSV traducido, de aqui se sacan genero, valoracion, periodo y fecha de cada resena
CSV_RESENAS = RAIZ / "data" / "processed" / "resenas_es.csv"
# Carpeta donde se guardan los graficos
CARPETA_FIGURAS = RAIZ / "outputs" / "figures"

#-------------------------------------------------------------------------------------------#

class Metricas_Spacy:

    def __init__(self, ruta_pos=CSV_POS, ruta_resenas=CSV_RESENAS, columna="review_summary_es"):
        self.ruta_pos = Path(ruta_pos)
        self.ruta_resenas = Path(ruta_resenas)
        self.columna = columna  # columna que se etiqueto con spaCy
        self.datos = None  # aqui quedan los tokens (sin puntuacion) con los datos de cada resena
        self.por_resena = None  # aqui queda una fila por resena con sus conteos

#-------------------------------------------------------------------------------------------#
# 1. Cargar los datos y dejarlos listos

    def cargar(self):
        # Se revisa que existan los dos archivos
        for ruta in (self.ruta_pos, self.ruta_resenas):
            if not ruta.exists():
                raise FileNotFoundError(f"No encuentro el archivo: {ruta}")

        # Se leen los CSV (keep_default_na=False evita que palabras como "NA" se vuelvan vacias)
        tokens = pd.read_csv(self.ruta_pos, keep_default_na=False)
        resenas = pd.read_csv(self.ruta_resenas, keep_default_na=False)

        # Se deja solo la columna que se etiqueto (por si el CSV trae mas de una)
        tokens = tokens[tokens["columna"] == self.columna]

        # De cada resena solo se necesitan estos datos
        datos_resena = resenas[["id_resena", "genero", "periodo", "valoracion", "rating", "review_date"]]

        # Se une cada token con los datos de su resena usando id_resena
        datos = tokens.merge(datos_resena, on="id_resena")

        # Segun el PDF, las metricas se calculan sin signos de puntuacion
        datos = datos[~datos["upos_spacy"].isin(["PUNCT", "SPACE"])].copy()

        # Año de la resena (para la pregunta C2)
        datos["anio"] = pd.to_datetime(datos["review_date"], errors="coerce").dt.year

        # Columnas de verdadero/falso que facilitan los conteos
        datos["es_verbo"] = datos["upos_spacy"] == "VERB"
        datos["es_sust"] = datos["upos_spacy"] == "NOUN"
        datos["es_adj"] = datos["upos_spacy"] == "ADJ"
        self.datos = datos

        # Una fila por resena: cuantos tokens y cuantos adjetivos tiene
        por_resena = datos.groupby("id_resena").agg(
            tokens=("token", "count"),
            adjetivos=("es_adj", "sum"),
            anio=("anio", "first"),
            periodo=("periodo", "first"),
        )
        # Indice de adjetivacion: adjetivos por cada 100 tokens
        por_resena["adj_100"] = por_resena["adjetivos"] / por_resena["tokens"] * 100
        self.por_resena = por_resena

        print(f"Resenas analizadas: {datos['id_resena'].nunique():,} | tokens sin puntuacion: {len(datos):,}")
        return datos

    # Guarda un grafico en outputs/figures y lo muestra
    def guardar_figura(self, fig, nombre):
        CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)  # se crea la carpeta si no existe
        ruta = CARPETA_FIGURAS / nombre
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"Grafico guardado en: {ruta}")
        plt.show()

#-------------------------------------------------------------------------------------------#
# 2. Pregunta A3: ¿Que genero tiene el ratio verbos/sustantivos mas alto?

    def pregunta_A3(self):
        print("PREGUNTA A3: ¿Qué género tiene el ratio verbos/sustantivos más alto?\n")

        # Por cada genero se suman los verbos, los sustantivos y los tokens
        tabla = self.datos.groupby("genero").agg(
            verbos=("es_verbo", "sum"),
            sustantivos=("es_sust", "sum"),
            tokens=("token", "count"),
        )
        # Ratio = verbos / sustantivos (se usan los totales del genero porque hay
        # resenas sin sustantivos y dividir entre cero daria error)
        tabla["ratio_verbos_sustantivos"] = (tabla["verbos"] / tabla["sustantivos"]).round(3)
        tabla = tabla.sort_values("ratio_verbos_sustantivos", ascending=False)
        print(tabla)

        # Grafico de barras con el ratio de cada genero
        fig, ax = plt.subplots(figsize=(8, 5))
        barras = ax.bar(tabla.index, tabla["ratio_verbos_sustantivos"], color="#ff4757", edgecolor="black")
        # Se escribe el valor encima de cada barra
        for barra in barras:
            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                    f"{barra.get_height():.2f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title("Ratio verbos/sustantivos por género")
        ax.set_xlabel("Género")
        ax.set_ylabel("Ratio verbos / sustantivos")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        self.guardar_figura(fig, "EjeA-3_ratio_verbos_sustantivos.png")

#-------------------------------------------------------------------------------------------#
# 3. Pregunta B4: ¿Que adjetivos (lemas) son mas caracteristicos de las positivas y de las negativas?

    def pregunta_B4(self, minimo=5, top=10):
        print("PREGUNTA B4: ¿Qué adjetivos (lemas) son más característicos de las reseñas positivas y cuáles de las negativas?\n")

        # Se queda solo con los adjetivos (aqui NO se quitan las stopwords porque
        # adjetivos como "buena" son stopwords para spaCy y se perderian)
        adjetivos = self.datos[self.datos["es_adj"]]

        # Cuantas veces aparece cada lema en las positivas y en las negativas
        positivas = adjetivos[adjetivos["valoracion"] == "positiva"]["lema"].value_counts()
        negativas = adjetivos[adjetivos["valoracion"] == "negativa"]["lema"].value_counts()

        # Tabla con una fila por lema (si no aparece en un grupo, queda en 0)
        tabla = pd.DataFrame({"positivas": positivas, "negativas": negativas}).fillna(0)
        tabla["total"] = tabla["positivas"] + tabla["negativas"]
        # Se quitan los lemas que casi no aparecen, porque no sirven para comparar
        tabla = tabla[tabla["total"] >= minimo]

        # Porcentaje de cada lema dentro de los adjetivos de su grupo
        # (asi se compara bien aunque haya mas resenas positivas que negativas)
        tabla["pct_positivas"] = tabla["positivas"] / positivas.sum() * 100
        tabla["pct_negativas"] = tabla["negativas"] / negativas.sum() * 100
        tabla["diferencia"] = tabla["pct_positivas"] - tabla["pct_negativas"]

        # Los mas positivos tienen la diferencia mas alta (mayor que 0)
        # y los mas negativos la mas baja (menor que 0)
        top_pos = tabla[tabla["diferencia"] > 0].sort_values("diferencia", ascending=False).head(top)
        top_neg = tabla[tabla["diferencia"] < 0].sort_values("diferencia").head(top)
        print("Adjetivos más característicos de las POSITIVAS:")
        print(top_pos[["positivas", "negativas", "diferencia"]].round(2))
        print("\nAdjetivos más característicos de las NEGATIVAS:")
        print(top_neg[["positivas", "negativas", "diferencia"]].round(2))

        # Dos graficos de barras horizontales, uno al lado del otro
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.barh(top_pos.index, top_pos["diferencia"], color="#00b894", edgecolor="black")
        ax1.set_title("Más característicos de reseñas positivas")
        ax2.barh(top_neg.index, -top_neg["diferencia"], color="#ff6b81", edgecolor="black")
        ax2.set_title("Más característicos de reseñas negativas")
        for ax in (ax1, ax2):
            ax.invert_yaxis()  # el mas caracteristico queda arriba
            ax.set_xlabel("Diferencia de porcentaje entre grupos")
            ax.set_ylabel("Adjetivo (lema)")
            ax.grid(axis="x", alpha=0.3, linestyle="--")
        fig.suptitle("Adjetivos característicos por valoración", fontweight="bold")
        self.guardar_figura(fig, "EjeB-4_adjetivos_caracteristicos.png")
        return top_pos, top_neg

#-------------------------------------------------------------------------------------------#
# 4. Pregunta C2: ¿Ha cambiado la proporción de adjetivos a lo largo del tiempo?

    def pregunta_C2(self, min_resenas=30):
        print("PREGUNTA C2: ¿Ha cambiado la proporción de adjetivos a lo largo del tiempo?\n")

        # Promedio del indice de adjetivacion (adjetivos por 100 tokens) de cada año
        por_anio = self.por_resena.groupby("anio").agg(
            resenas=("adj_100", "count"),
            adj_100=("adj_100", "mean"),
        ).round(2)
        # Se quitan los años con muy pocas resenas, porque su promedio no es confiable
        por_anio = por_anio[por_anio["resenas"] >= min_resenas]
        print(por_anio)

        # Promedio por periodo (antes de 2010 y 2010 o despues)
        por_periodo = self.por_resena.groupby("periodo")["adj_100"].mean().round(2)
        por_periodo = por_periodo.reindex(["antes_2010", "2010_o_despues"])
        print("\nAdjetivos por 100 tokens según el periodo:")
        print(por_periodo)

        # Dos graficos: linea por año y barras por periodo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(por_anio.index, por_anio["adj_100"], marker="o", color="#ff4757")
        ax1.set_title("Adjetivos por 100 tokens a lo largo de los años")
        ax1.set_xticks(por_anio.index[::2])  # se muestran solo años enteros (uno por medio)
        ax1.set_xlabel("Año de la reseña")
        ax1.set_ylabel("Adjetivos por 100 tokens")
        ax1.grid(alpha=0.3, linestyle="--")

        barras = ax2.bar(por_periodo.index, por_periodo.values, color=["#fdcb6e", "#00b894"], edgecolor="black")
        for barra in barras:
            ax2.text(barra.get_x() + barra.get_width() / 2, barra.get_height(),
                     f"{barra.get_height():.2f}", ha="center", va="bottom", fontweight="bold")
        ax2.set_title("Adjetivos por 100 tokens según el periodo")
        ax2.set_xlabel("Periodo")
        ax2.set_ylabel("Adjetivos por 100 tokens")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        self.guardar_figura(fig, "EjeC-2_adjetivos_en_el_tiempo.png")

        return por_anio, por_periodo

    #-------------------------------------------------------------------------------------------#
# 5. Ejecutamos las metricas

    def ejecutar_todo(self):
        self.cargar()
        self.pregunta_A3()
        self.pregunta_B4()
        self.pregunta_C2()
