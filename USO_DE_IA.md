# Uso de IA

• Qué herramientas de IA utilizaron y para qué tareas específicas.
• Ejemplos de prompts utilizados (mínimo 3 a 5 ejemplos representativos).
• Reflexión sobre cómo la IA ayudó en su aprendizaje (1 a 2 párrafos).
• Qué modificaciones hicieron al código o al análisis generado por IA
----------------------------------------------------------------------------------

### Herramienta utilizada: Claude.
### Tarea: Clase de Traduccion.

**Ejemplo del prompt:** 'Estoy haciendo un proyecto de minería de textos en Python y necesito traducir 
un corpus de reseñas de películas del inglés al español (3.200 registros). En el proyecto nos indican
usar el modelo Helsinki-NLP/opus-mt-en-es, pero me gustaría saber si el modelo de Google es bueno y 
cuál de los dos sería mejor utilizar, si alguno tiene ventajas sobre el otro y cuál traduciría más 
rápido. Trabajo en una laptop por lo que todo correría en CPU, y no puedo usar servicios de pago. 
También quisiera saber qué métodos o ajustes harían que la traducción sea más rápida'

------

**Respuesta:**
Para este proyecto te recomiendo quedarte con Helsinki-NLP/opus-mt-en-es. Primero hay que aclarar a qué te refieres con “el modelo de Google”, porque son dos cosas distintas. Si hablas de Google Translate como servicio, normalmente da traducciones más fluidas que un modelo pequeño abierto, pero no lo he comparado con tu corpus. Para saberlo de verdad tendrías que traducir unas 50 reseñas con cada uno y compararlas a ojo. La versión oficial es la API de Cloud Translation, y los primeros 500.000 caracteres al mes son gratis (se aplican como un crédito de $10) y después cuesta $20 por millón. Pero incluso con ese plan gratis se exige una cuenta de facturación activa con tarjeta de crédito, y tú dices que no puedes usar servicios de pago. Existen librerías gratuitas no oficiales que usan la página web de Google, pero no son un servicio oficial, se pueden bloquear en cualquier momento y no corren en tu equipo, así que tampoco las recomiendo. 
google
zoho

La otra opción son los modelos abiertos de Google. Existe TranslateGemma, que viene en tamaños de 4B, 12B y 27B parámetros, y Google dice que el de 12B supera al modelo base Gemma 3 de 27B en el benchmark WMT24++. Sí podría traducir mejor, pero son modelos de miles de millones de parámetros, mientras que Helsinki ronda los 75 millones. En una laptop sin tarjeta gráfica serían mucho más lentos (mi estimación, sin haberlo medido, es de horas en vez de minutos) y el de 12B probablemente no cabe en 16 GB de RAM sin cuantizarlo. Para un corpus de 3.200 textos cortos, esa mejora de calidad no compensa. 
neowin
neowin

Para traducir más rápido con Helsinki en CPU, lo que más ayuda es:

- num_beams=1: usar búsqueda greedy en lugar de beam search da traducciones casi iguales mucho más rápido.
- Traducir por oraciones y por lotes, ordenando las oraciones por longitud para que el modelo desperdicie menos.
- Traducir solo los textos únicos, porque si hay repetidos (por ejemplo, la misma sinopsis en varias reseñas) no tiene sentido traducirlos varias veces.
- Usar textos más cortos: el tiempo crece más o menos con la cantidad total de caracteres, así que acortar o descartar reseñas muy largas reduce el tiempo. 
- CTranslate2 con cuantización int8: convierte el mismo modelo a un formato más rápido. Sus autores hablan de aceleraciones de entre 2x y 8x con int8, y en su benchmark con OPUS-MT en CPU pasó de unos 345 a unos 696 tokens por segundo con casi la misma calidad. Esa comparación es contra Marian original y no contra transformers, así que en tu caso la ganancia puede variar.

-----

**Reflexión:** Usar IA en esta parte del proyecto me ayudo a entender mejor como funciona el
modelo de traduccion Helsinki-NLP/opus-mt-en-es con el que traducimos las 3.200 reseñas.
Claude me recomendo usar num_beams=1, y como no sabia que significaba lo busque por internet, lo que
me indicaba era que con num_beams=4 el modelo compara varias traducciones posibles y se queda con la mejor, y con
num_beams=1 elige directamente la mas probable, por eso es mas rápido aunque la calidad puede
bajar un poco. Con ese ajuste la traduccion tardo aproximadamente 20 minutos. Tambien aprendi
que conviene traducir por oraciones, porque el modelo corta los textos largos, y traducir solo
los textos que no se repiten, porque las sinopsis aparecen en varias reseñas.

-----

### Herramienta utilizada: Claude.
### Tarea: Dashboard con Plotly.

**Ejemplo del prompt:** 'Para generar el dashboard ya no puedo utilizar streamlit, ahora debo de usar 
Plotly dash, pero me gustaria saber algunas diferencia entre ellos y saber que cambia en cuanto a codigo 
para poder realizarlo.'

------

**Respuesta:**
La diferencia principal es cómo reacciona cada herramienta cuando el usuario cambia un filtro. Streamlit vuelve a ejecutar el script completo de arriba hacia abajo cada vez que alguien toca algo, y los componentes te devuelven su valor directamente. Por eso es muy rápido de escribir: casi no hay estructura que aprender. Plotly Dash separa el programa en dos partes. Una es el diseño de la página (layout), que es un árbol de componentes con un id cada uno. La otra son los callbacks, funciones que dicen “cuando cambie este componente, recalcula aquel otro”. Eso exige más código y más estructura, pero da más control sobre el diseño, que se maneja con estilos tipo CSS.

En el código, lo que más cambia es dónde vive la lógica. Mira el mismo filtro en las dos:

```
# Streamlit: el valor llega directo y todo corre de arriba a abajo
genero = st.selectbox("Género", ["Todos"] + generos)
datos = df if genero == "Todos" else df[df["genero"] == genero]
st.plotly_chart(px.histogram(datos, x="valoracion"))

------------------------------------------------------------------

# Dash: primero el diseño, luego el callback que conecta ids
app.layout = html.Div([
    dcc.Dropdown(id="f-genero", options=["Todos"] + generos, value="Todos"),
    dcc.Graph(id="grafico"),
])

@app.callback(Output("grafico", "figure"), Input("f-genero", "value"))
def actualizar(genero):
    datos = df if genero == "Todos" else df[df["genero"] == genero]
    return px.histogram(datos, x="valoracion")
```

Lo que se reescribe es la interfaz: st.selectbox pasa a dcc.Dropdown, st.slider a dcc.Slider, st.dataframe a dash_table.DataTable, st.plotly_chart a dcc.Graph, st.title y st.markdown a html.H1 y html.P, y st.columns o st.sidebar se arman con html.Div y estilos (por ejemplo display: flex). Además, los valores que en Streamlit eran variables ahora son parámetros de la función del callback, y cada resultado se devuelve como Output. Se instala con pip install dash y se ejecuta con python app.py (el servidor queda en http://127.0.0.1:8050), en lugar de streamlit run

------

**Reflexión:** El usar la IA para el dashboard me ayud a ver que no solo me debo quedar con
Streamlit o Power BI, sino que existe otra herramienta, Plotly Dash, que a pesar de que es
mas diferente y pide mas estructura, tambien sirve para hacer un dashboard con filtros,
tablas y tarjetas. Al principio me costó entenderla porque funciona distinto a Streamlit.
Con las explicaciones de la IA pude comparar las dos herramientas y entender
por que en Dash se escribe mas código, pero se tiene mas control sobre el diseño. Esto me ayudo a 
darme cuenta de que no existe una sola herramienta para todo, y que conviene conocer varias para 
escoger la que mejor se adapte al proyecto.

### Documentacion: https://dash.plotly.com/