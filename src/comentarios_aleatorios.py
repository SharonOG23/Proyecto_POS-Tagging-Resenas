# ---------------------------CLASE PARA COMENTARIOS ALEATORIOS---------------------------#

from pathlib import Path  # Libreria para manejar rutas de archivos y directorios
import pandas as pd  # Libreria para manejar y analizar datos en estructuras de datos como DataFrames

# Rutas de archivos
RAIZ = Path(__file__).resolve().parent.parent  # Ruta principal del proyecto
CARPETA_RAW = RAIZ / "data" / "raw"  # Carpeta donde se encuentran los archivos CSV originales
CSV_ORIGINAL = CARPETA_RAW / "corpus_peliculas 2.csv"  # Ruta del archivo CSV original con las reseñas de películas

#-----------------------------------------------------------------------------------------#

# Creamos la clase ComentariosAlea para manejar el muestreo de reseñas de películas
class ComentariosAlea:

    def __init__(self, ruta_csv=CSV_ORIGINAL, n_total=3200,
                 semilla=123):  # Creamos el constructor de la clase, que recibe la ruta del CSV, el número total de reseñas a muestrear y una semilla para la aleatoriedad
        self.ruta_csv = Path(ruta_csv)  # Convertimos la ruta del CSV a un objeto Path para manejarla más fácilmente
        self.n_total = n_total  # Guardamos el número total de reseñas a muestrear
        self.semilla = semilla  # Guardamos la semilla para la aleatoriedad
        self.generos = ["Drama", "Comedy", "Action", "Animation", "Adventure"]  # Definimos los géneros de películas
        self.anio_corte = 2010  # Definimos el año de corte para separar las reseñas en dos periodos: antes de 2010 y 2010 o después
        self.max_por_pelicula = 20  # Definimos el número máximo de reseñas por película
        self.periodos = ["antes_2010", "2010_o_despues"]  # Definimos los periodos de tiempo para las reseñas

        self.muestra = None  # Inicializamos la variable que contendrá la muestra final de reseñas

#-----------------------------------------------------------------------------------------#

    # Método 1. Cargar y limpiar el CSV
    def cargar(
            self):  # Cargamos el CSV original y limpiamos los datos para quedarnos solo con las reseñas que cumplen los requisitos
        if not self.ruta_csv.exists():  # Verificamos si el archivo CSV existe en la ruta especificada
            raise FileNotFoundError(
                f"No encuentro el CSV en: {self.ruta_csv}")  # Si no existe, lanzamos un error indicando que no se encuentra el archivo
        df = pd.read_csv(self.ruta_csv, encoding="utf-8-sig",
                         keep_default_na=False)  # Leemos nuestro CSV, le indicamos que la codificación es utf-8-sig y que no queremos que se interpreten valores vacíos como NaN
        print(f"Corpus cargado: {len(df)} reseñas")  # Imprimimos la cantidad de reseñas cargadas del CSV original

        # Quitar HTML y espacios repetidos en las columnas de texto
        for col in ["review_summary", "plot_summary"]:  # Iteramos sobre las columnas de texto que queremos limpiar
            df[col] = (df[col].str.replace(r"<[^>]+>", " ",
                                           regex=True)  # Eliminamos etiquetas HTML usando una expresión regular
                       .str.replace(r"\s+", " ", regex=True)  # Reemplazamos múltiples espacios por un solo espacio
                       .str.strip())  # Eliminamos espacios al inicio y al final del texto

        df = df[(df["review_summary"] != "") & (df["plot_summary"] != "")]  # Nos quedamos solo con las filas donde las columnas de texto no están vacías

        df["fecha_resena"] = pd.to_datetime(df["review_date"],
                                            errors="coerce")  # Convertimos la columna de fecha de reseña a tipo datetime
        df = df.dropna(
            subset=["fecha_resena"])  # Eliminamos las filas donde la fecha de reseña no se pudo convertir a datetime

        df = df[df["genre"].isin(
            self.generos)]  # Nos quedamos solo con las filas donde el género de la película está en la lista de géneros definidos

        df["palabras_en"] = df[
            "review_summary"].str.split().str.len()  # Contamos el número de palabras en la columna de resumen de reseña
        df = df[df[
                    "palabras_en"] >= 1]  # Nos quedamos solo con las filas donde el resumen de reseña tiene al menos una palabra

        df = df.drop_duplicates(subset=["movie_id",
                                        "review_summary"])  # Eliminamos filas duplicadas basándonos en el ID de la película y el resumen de reseña

        df["genero"] = df["genre"]  # Creamos una nueva columna "genero" que es igual a la columna "genre"
        df["periodo"] = "2010_o_despues"  # Inicializamos la columna "periodo" con el valor "2010_o_despues"
        df.loc[df["fecha_resena"].dt.year < self.anio_corte, "periodo"] = "antes_2010"
        df["valoracion"] = pd.cut(df["rating"], bins=[0, 4, 6, 10],
                                  labels=["negativa", "intermedia", "positiva"]).astype(
            str)  # Indicamos por medio de una nueva columna "valoracion" si la reseña es negativa, intermedia o positiva

        self.df = df.reset_index(
            drop=True)  # Reiniciamos el índice del DataFrame y eliminamos la columna de índice original
        print(
            f"Reseñas disponibles después de filtrar: {len(self.df)}")  # Imprimimos la cantidad de reseñas disponibles después de aplicar todos los filtros
        return self.df

#-----------------------------------------------------------------------------------------#

    # Método 2. Tomar n reseñas de un grupo
    def tomar(self, grupo, n):  # Tomamos n reseñas de un grupo, asegurándonos de no exceder el máximo por película
        grupo = grupo.sample(frac=1, random_state=self.semilla)  # Mezclar aleatoriamente el grupo
        grupo["exceso"] = grupo.groupby(
            "movie_id").cumcount() >= self.max_por_pelicula  # Marcamos las reseñas que exceden el máximo por película
        grupo = grupo.sort_values("exceso", kind="stable").drop(
            columns="exceso")  # Ordenamos el grupo para que las reseñas que exceden el máximo por película queden al final y luego eliminamos la columna "exceso"
        return grupo.head(
            n)  # Devolvemos las primeras n reseñas del grupo, que son las que cumplen con el máximo por película

#-----------------------------------------------------------------------------------------#

    # Método 3. Muestreo: mismo número de reseñas por género y por periodo
    def muestrear(self):
        por_genero = self.n_total // len(self.generos)  # Aqui indicamos cuántas reseñas sacar por género (3200/5 = 640)
        mitad = por_genero // 2  # Aqui indicamos cuántas reseñas sacar por periodo (640/2 = 320)
        partes = []  # Creamos una lista vacía para almacenar las partes de la muestra que vamos a tomar de cada género y periodo

        for genero in self.generos:  # Iteramos sobre cada género para tomar las reseñas correspondientes
            de_genero = self.df[self.df[
                                    "genero"] == genero]  # Nos quedamos solo con las filas donde el género es igual al género actual del bucle
            disponibles = de_genero["periodo"].value_counts()  # Contamos cuántas reseñas hay disponibles por periodo

            cuota = {p: min(mitad, disponibles.get(p, 0)) for p in
                     self.periodos}  # Calculamos las reseñas a tomar por periodo, asegurándonos de no exceder la cantidad disponible
            faltan = por_genero - sum(
                cuota.values())  # Calculamos cuántas reseñas faltan por tomar para completar la cuota por género
            for p in self.periodos:  # Si un periodo no alcanza, se completa con el otro
                extra = min(faltan, disponibles.get(p, 0) - cuota[
                    p])  # Calculamos cuántas reseñas extra podemos tomar del periodo actual sin exceder la cantidad disponible
                cuota[p] += extra  # Sumamos las reseñas extra a la cuota del periodo actual
                faltan -= extra  # Restamos las reseñas extra tomadas de las que faltan por tomar

            for p in self.periodos:  # Tomamos las reseñas correspondientes a cada periodo y las agregamos a la lista de partes
                grupo = de_genero[de_genero[
                                      "periodo"] == p]  # Nos quedamos solo con las filas donde el periodo es igual al periodo actual del bucle
                partes.append(
                    self.tomar(grupo, cuota[p]))  # Agregamos a la lista de partes las reseñas tomadas del grupo actual

        muestra = pd.concat(partes).sample(frac=1,
                                           random_state=self.semilla)  # Concatenamos todas las partes de la muestra y las mezclamos aleatoriamente
        muestra = muestra.reset_index(
            drop=True)  # Indicamos que queremos reiniciar el índice del DataFrame resultante y eliminar la columna de índice original
        muestra.insert(0, "id_resena",
                       muestra.index)  # Creamos una nueva columna "id_resena" que contiene el índice del DataFrame como identificador único para cada reseña
        self.muestra = muestra  # Guardamos la muestra final en el atributo de la clase
        return muestra  # Devolvemos la muestra final de reseñas

#-----------------------------------------------------------------------------------------#

    # Método 4. Para validar que la muestra cumple con los requisitos de cantidad por género y periodo
    def validar(self):  # Validamos que la muestra cumple con los requisitos de cantidad por género y periodo
        m = self.muestra  # Nos quedamos con la muestra final de reseñas
        print(
            "\nReseñas por género y periodo:")  # Imprimimos un mensaje indicando que vamos a mostrar la cantidad de reseñas por género y periodo
        print(pd.crosstab(m["genero"], m["periodo"],
                          margins=True))  # Mostramos una tabla cruzada con la cantidad de reseñas por género y periodo, incluyendo totales

        assert len(
            m) >= 3000, "Hay menos de 3000 reseñas"  # Verificamos que la cantidad total de reseñas sea al menos 3000
        assert m[
                   "genero"].nunique() == 5, "No hay 5 géneros"  # Verificamos que haya exactamente 5 géneros en la muestra
        assert (m[
                    "genero"].value_counts() >= 400).all(), "Un género tiene menos de 400"  # Verificamos que cada género tenga al menos 400 reseñas
        assert (m[
                    "periodo"].value_counts() >= 1000).all(), "Un periodo tiene menos de 1000"  # Verificamos que cada periodo tenga al menos 1000 reseñas
        print(
            "\nTotal, géneros y periodos: OK")  # Imprimimos un mensaje indicando que la validación de la muestra fue exitosa
        print(
            "AVISO: las reseñas son cortas (review_summary), no cumplen 30-400 palabras.")  # Imprimimos un mensaje de aviso indicando que las reseñas son cortas y no cumplen con el rango de 30-400 palabras en la columna "review_summary"

#-----------------------------------------------------------------------------------------#

    # Método 5. Guardar el CSV
    def guardar(self, ruta_salida=None):  # Guardamos la muestra final de reseñas en un archivo CSV
        # Se guarda en la ruta de data/raw/muestra_n.csv
        ruta_salida = Path(
            ruta_salida) if ruta_salida else CARPETA_RAW / f"muestra_{self.n_total}.csv"  # Si no se especifica una ruta de salida, se guarda en la carpeta raw con el nombre "muestra_n.csv", donde n es el número total de reseñas
        ruta_salida.parent.mkdir(parents=True,
                                 exist_ok=True)  # Creamos la carpeta de salida si no existe, incluyendo los directorios padres necesarios
        self.muestra.to_csv(ruta_salida, index=False,
                            encoding="utf-8")  # Guardamos la muestra final de reseñas en un archivo CSV sin incluir el índice y con codificación utf-8
        print(
            f"Muestra guardada en {ruta_salida}")  # Imprimimos un mensaje indicando la ruta donde se guardó la muestra final de reseñas
