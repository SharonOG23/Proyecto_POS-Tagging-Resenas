import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


#La clase CargadorCSV lo que nos permite es visualizar los archivos CSV
#y generar una previsualizacion de los datos

#Metodos utilizados de la clase:
"""1. cargar_csv()
Carga un archivo CSV desde una ruta especificada y lo almacena en un DataFrame de Pandas para su posterior procesamiento y análisis.
2. obtener_dataframe()
Retorna el DataFrame cargado en memoria, permitiendo acceder a los datos desde otros métodos o procesos.
3. informacion()
Proporciona información básica del conjunto de datos, incluyendo número de filas, columnas y nombres de las columnas disponibles.
4. tipos_datos()
Muestra el tipo de dato asociado a cada columna del DataFrame, facilitando la validación y análisis de la estructura de los datos.
5. estadisticas()
Genera estadísticas descriptivas de las variables del conjunto de datos, como promedio, desviación estándar, valores mínimos y máximos.
6. valores_nulos()
Calcula y devuelve la cantidad de valores nulos presentes en cada columna del DataFrame.
7. porcentaje_nulos()
Calcula el porcentaje de valores nulos por columna con respecto al total de registros del conjunto de datos.
8. columnas_con_nulos()
Identifica y devuelve únicamente las columnas que contienen uno o más valores nulos.
9. reporte_calidad_datos()
Genera un reporte de calidad de datos que incluye tipo de dato, cantidad de valores nulos, porcentaje de nulos y cantidad de valores únicos por columna.
10. resumen_completo()
Presenta un resumen general del dataset, incluyendo dimensiones, columnas numéricas, columnas de texto y total de valores nulos encontrados.
"""

class CargadorCSV:
    """
    Clase para cargar y analizar archivos CSV utilizando Pandas.
    """

    def __init__(self, ruta_archivo):
        """
        Constructor de la clase.

        Args:
            ruta_archivo (str): Ruta del archivo CSV.
        """
        self.ruta_archivo = ruta_archivo
        self.df = None

    def cargar_csv(self, separador=",", encoding="utf-8"):
        """
        Carga el archivo CSV en un DataFrame.
        """
        try:
            self.df = pd.read_csv(
                self.ruta_archivo,
                sep=separador,
                encoding=encoding
            )

            print(" Archivo cargado correctamente.")
            return self.df

        except Exception as e:
            print(f" Error al cargar el archivo: {e}")
            return None

    def obtener_dataframe(self):
        """
        Retorna el DataFrame cargado.
        """
        return self.df

    def informacion(self):
        """
        Retorna información básica del DataFrame.
        """
        if self.df is not None:
            return {
                "filas": self.df.shape[0],
                "columnas": self.df.shape[1],
                "nombres_columnas": list(self.df.columns)
            }

        return "No hay datos cargados."

    def tipos_datos(self):
        """
        Retorna los tipos de datos de cada columna.
        """
        if self.df is not None:
            return self.df.dtypes

        return "No hay datos cargados."

    def estadisticas(self):
        """
        Retorna estadísticas descriptivas.
        """
        if self.df is not None:
            return self.df.describe(include="all")

        return "No hay datos cargados."

    def valores_nulos(self):
        """
        Retorna la cantidad de valores nulos por columna.
        """
        if self.df is not None:
            return self.df.isnull().sum()

        return "No hay datos cargados."

    def porcentaje_nulos(self):
        """
        Retorna el porcentaje de nulos por columna.
        """
        if self.df is not None:
            return (self.df.isnull().sum() / len(self.df) * 100).round(2)

        return "No hay datos cargados."

    def columnas_con_nulos(self):
        """
        Retorna únicamente las columnas que tienen valores nulos.
        """
        if self.df is not None:
            nulos = self.df.isnull().sum()
            return nulos[nulos > 0]

        return "No hay datos cargados."

    def reporte_calidad_datos(self):
        """
        Genera un reporte de calidad de datos.
        """
        if self.df is None:
            return "No hay datos cargados."

        reporte = pd.DataFrame({
            "tipo_dato": self.df.dtypes,
            "nulos": self.df.isnull().sum(),
            "porcentaje_nulos": round(self.df.isnull().mean() * 100, 2),
            "valores_unicos": self.df.nunique()
        })

        return reporte

    def resumen_completo(self):
        """
        Genera un resumen completo del dataset.
        """
        if self.df is None:
            return "No hay datos cargados."

        return {
            "filas": self.df.shape[0],
            "columnas": self.df.shape[1],
            "columnas_numericas":
                self.df.select_dtypes(include="number").columns.tolist(),

            # Corregido para evitar el warning de Pandas
            "columnas_texto":
                self.df.select_dtypes(
                    include=["object", "string"]
                ).columns.tolist(),

            "nulos_totales":
                int(self.df.isnull().sum().sum())
        }


# ====================================================
# INSTANCIA DE LA CLASE
# ====================================================

ruta = "corpus_peliculas.csv"

cargador = CargadorCSV(ruta)

df = cargador.cargar_csv()

if df is not None:
    print("\n===== PRIMERAS FILAS =====")
    print(df.head())

    print("\n===== INFORMACIÓN =====")
    print(cargador.informacion())

    print("\n===== TIPOS DE DATOS =====")
    print(cargador.tipos_datos())

    print("\n===== ESTADÍSTICAS =====")
    print(cargador.estadisticas())

    print("\n===== VALORES NULOS =====")
    print(cargador.valores_nulos())

    print("\n===== PORCENTAJE DE NULOS =====")
    print(cargador.porcentaje_nulos())

    print("\n===== COLUMNAS CON NULOS =====")




