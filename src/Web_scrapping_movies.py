"""
Scraper masivo de FilmAffinity (Costa Rica): películas y series.

Instalación:
    pip install requests beautifulsoup4
    pip install lxml                   # opcional, pero acelera el parseo ~2-3x
    pip install pandas openpyxl        # solo para --excel

Uso:
    python filmaffinity_masivo.py --objetivo 10000 --hilos 3 --excel

    # Si te sigue limitando (HTTP 429), ve más despacio y sin la 2.ª petición de críticas:
    python filmaffinity_masivo.py --hilos 2 --intervalo 3 6 --sin-criticas-usuarios

    # Opcional: más puntos de partida (páginas de géneros, tops, listas...)
    python filmaffinity_masivo.py --semillas https://www.filmaffinity.com/cr/topgen.php

Es REANUDABLE: si lo detienes (Ctrl+C) o se corta, ejecuta el mismo comando
y continúa donde quedó, sin duplicar registros.

Archivos que genera (las carpetas se crean solas si no existen):
  En la raíz del proyecto, RAIZ/data/results/:
    peliculas_series_scrapping.csv   -> resultado final
    peliculas_series_scrapping.xlsx  -> solo con --excel (junto al CSV)
  En RAIZ/webscrapping/data/:
    urls_filmaffinity.txt  -> todas las URLs descubiertas (para reanudar)
    fallidas.txt           -> URLs que no se pudieron procesar

Nota: revisa los términos de uso y el robots.txt del sitio antes de hacer
un rastreo masivo, y mantén un ritmo de peticiones moderado.
"""

import argparse
import csv
import os
import random
import re
import string
import threading
import time
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Constantes globales
# --------------------------------------------------------------------------

# Parser HTML: lxml es bastante más rápido que html.parser; si no está
# instalado se usa el parser incluido en Python sin que el script falle.
try:
    import lxml  # noqa: F401

    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"

# Raíz del proyecto: este script vive en <RAIZ>/src/, así que subimos un nivel.
# Si lo mueves a otra profundidad, ajusta el índice de parents[...].
RAIZ = Path(__file__).resolve().parents[1]

# Resultado final (CSV y, con --excel, el XLSX): en la raíz del proyecto
CSV_SALIDA = RAIZ / "data" / "results" / "peliculas_series_scrapping.csv"

# Archivos de apoyo para reanudar (urls_filmaffinity.txt y fallidas.txt)
DIRECTORIO_AUXILIAR = RAIZ / "webscrapping" / "data"

# Códigos HTTP que indican "te estoy limitando" (se espera y se reintenta)
CODIGOS_BLOQUEO = (403, 429, 503)

# Caracteres de control que Excel/openpyxl no acepta en una celda
CARACTERES_ILEGALES = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Los comentarios pueden ser largos; evitamos el límite por defecto del módulo csv
csv.field_size_limit(10_000_000)


class Bloqueado(Exception):
    """El sitio respondió 403/429/503 repetidamente para esta URL (o se pidió parar)."""


# --------------------------------------------------------------------------
# Limitador de ritmo global
# --------------------------------------------------------------------------
class LimitadorTasa:
    """
    Limita el ritmo GLOBAL de peticiones (compartido por todos los hilos) y se adapta:
      - Tras un 429 hace una pausa global que crece (60s, 2min, 4min... hasta 30min).
      - Además enlentece el ritmo (`factor`) y lo recupera poco a poco cuando
        las peticiones vuelven a salir bien.
    """

    def __init__(self, minimo: float, maximo: float, pausa_base: float = 60.0):
        self.minimo, self.maximo = minimo, maximo  # rango del intervalo entre peticiones
        self.pausa_base = pausa_base               # pausa inicial tras un bloqueo
        self.factor = 1.0                          # multiplicador del intervalo (1 = normal)
        self.racha = 0                             # nº de bloqueos "recientes" (baja con éxitos)
        self._exitos = 0                           # peticiones buenas seguidas
        self._pausa_hasta = 0.0                    # instante (monotonic) en que acaba la pausa actual
        self._siguiente = 0.0                      # instante en que se puede lanzar la próxima petición
        self._lock = threading.Lock()              # protege los campos anteriores
        self.parar = threading.Event()             # al activarse, todos los hilos salen de inmediato

    def esperar(self):
        """Bloquea al hilo hasta que le toque su turno. Lanza Bloqueado si se pidió parar."""
        if self.parar.is_set():
            raise Bloqueado("detenido")
        with self._lock:
            ahora = time.monotonic()
            # Reservamos un "turno" y desplazamos el siguiente un intervalo aleatorio
            turno = max(ahora, self._siguiente)
            self._siguiente = turno + random.uniform(self.minimo, self.maximo) * self.factor
            espera = turno - ahora
        # La espera se hace fuera del lock y es interrumpible con Ctrl+C / parar
        if espera > 0 and self.parar.wait(espera):
            raise Bloqueado("detenido")

    def penalizar(self, segundos: float):
        """Retrasa a TODOS los hilos `segundos` (errores de red, descansos programados)."""
        with self._lock:
            self._siguiente = max(self._siguiente, time.monotonic()) + segundos

    def bloqueo(self, retry_after: float = 0.0) -> Tuple[float, bool]:
        """Registra un 429/403. Devuelve (segundos de pausa, ¿es un bloqueo nuevo?)."""
        with self._lock:
            ahora = time.monotonic()
            # Varios hilos pueden recibir 429 a la vez: solo el primero escala la pausa
            if ahora < self._pausa_hasta:
                return self._pausa_hasta - ahora, False
            # Pausa exponencial (60, 120, 240...) con tope de 30 min; respeta Retry-After
            pausa = max(
                min(self.pausa_base * 2 ** self.racha, 1800.0), min(retry_after, 1800.0)
            )
            self.racha += 1
            self.factor = min(self.factor * 1.5, 8.0)  # además, ir más lento en adelante
            self._exitos = 0
            self._pausa_hasta = ahora + pausa
            self._siguiente = max(self._siguiente, ahora) + pausa
            return pausa, True

    def exito(self):
        """Registra una petición correcta; cada 30 seguidas relaja el ritmo un poco."""
        with self._lock:
            self._exitos += 1
            if self._exitos >= 30:
                self._exitos = 0
                self.racha = max(0, self.racha - 1)
                self.factor = max(1.0, self.factor * 0.9)


# --------------------------------------------------------------------------
# Scraper
# --------------------------------------------------------------------------
class FilmAffinityMasivo:
    BASE_URL = "https://www.filmaffinity.com"
    FILM_RE = re.compile(r"/cr/film(\d+)\.html")  # detecta enlaces a fichas
    LETRAS = ["0-9"] + list(string.ascii_uppercase)
    MAX_PAG_LISTADO = 60

    # Columnas del CSV (en este orden)
    CAMPOS = [
        "nombre",
        "genero",
        "duracion",
        "pais",
        "direccion",
        "fecha_creacion",
        "ranking",
        "votos",
        "comentarios",
        "url",
        "tipo",
    ]

    def __init__(
        self,
        salida=CSV_SALIDA,
        archivo_urls: str = "urls_filmaffinity.txt",
        archivo_fallidas: str = "fallidas.txt",
        hilos: int = 3,
        intervalo: Tuple[float, float] = (1.5, 3.0),
        reintentos: int = 3,
        max_comentarios: int = 10,
        incluir_criticas_usuarios: bool = True,
        semillas: Optional[List[str]] = None,
        lote: int = 200,
        descanso: Tuple[float, float] = (90, 180),
        pausa_bloqueo: float = 60.0,
        max_racha: int = 6,
        reintentar_fallidas: bool = True,
        directorio=DIRECTORIO_AUXILIAR,
    ):
        # Los nombres sueltos se guardan dentro de `directorio`; las rutas que ya
        # traen carpeta (como CSV_SALIDA) se respetan tal cual.
        self.salida = self._ruta(directorio, salida)
        self.archivo_urls = self._ruta(directorio, archivo_urls)
        self.archivo_fallidas = self._ruta(directorio, archivo_fallidas)
        # Crea las carpetas de salida (y las intermedias) si no existen
        for ruta in (self.salida, self.archivo_urls, self.archivo_fallidas):
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
        self.hilos = hilos
        self.reintentos = reintentos                      # errores de red/5xx tolerados por URL
        self.max_comentarios = max_comentarios
        self.incluir_criticas_usuarios = incluir_criticas_usuarios
        self.semillas = semillas or []
        self.reintentar_fallidas = reintentar_fallidas

        self.lote = lote              # cada `lote` fichas guardadas, descanso largo (0 = sin descansos)
        self.descanso = descanso
        self.max_racha = max_racha    # bloqueos seguidos tras los cuales se detiene el script
        self.limitador = LimitadorTasa(*intervalo, pausa_base=pausa_bloqueo)
        self._local = threading.local()  # una Session de requests por hilo

        # Listados alfabéticos: solo se usan como refuerzo si la cola se vacía
        self._gen_listados = self._generador_listados()
        self._fallos_listado = 0
        self._listados_activos = True

    @staticmethod
    def _ruta(directorio, archivo) -> str:
        """Une directorio + archivo, salvo que el archivo ya traiga su propia carpeta."""
        archivo = str(archivo)
        return archivo if os.path.dirname(archivo) else os.path.join(str(directorio), archivo)

    @property
    def main_url(self) -> str:
        return f"{self.BASE_URL}/cr/main.html"

    # ------------------------------------------------------------------
    # Red
    # ------------------------------------------------------------------
    def _session(self) -> requests.Session:
        """Session propia de cada hilo (reutiliza la conexión TCP/TLS = menos latencia)."""
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                    "Accept": "text/html,application/xhtml+xml",
                }
            )
            self._local.s = s
        return self._local.s

    @staticmethod
    def _retry_after(resp) -> float:
        """Lee la cabecera Retry-After (en segundos); 0 si no existe o no es numérica."""
        try:
            return float(resp.headers.get("Retry-After", 0))
        except (TypeError, ValueError):
            return 0.0

    def _get_soup(self, url: str, reintentos_bloqueo: int = 3) -> Optional[BeautifulSoup]:
        """
        Descarga y parsea una URL.
          - Devuelve el BeautifulSoup si todo va bien.
          - Devuelve None si la página no existe (404) o falla por otros motivos.
          - Lanza Bloqueado si el sitio sigue respondiendo 403/429/503; en ese caso
            la URL NO se descarta: se reintenta más tarde.
        """
        errores = bloqueos = 0
        while errores < self.reintentos and bloqueos < reintentos_bloqueo:
            self.limitador.esperar()  # respeta el ritmo global entre hilos
            try:
                resp = self._session().get(url, timeout=25)
            except requests.RequestException as e:
                errores += 1
                print(f"[!] Error de red en {url}: {e} (intento {errores})")
                self.limitador.penalizar(3 * errores)  # backoff lineal global
                continue

            if resp.status_code == 200:
                self.limitador.exito()
                return BeautifulSoup(resp.text, PARSER)

            if resp.status_code in CODIGOS_BLOQUEO:
                bloqueos += 1
                pausa, nuevo = self.limitador.bloqueo(self._retry_after(resp))
                if nuevo:  # solo el primer hilo en detectarlo imprime el aviso
                    print(
                        f"[!] HTTP {resp.status_code}: pausa global de {pausa:.0f}s "
                        f"(ritmo reducido x{self.limitador.factor:.1f}, "
                        f"bloqueo {self.limitador.racha}/{self.max_racha})"
                    )
                continue

            if resp.status_code >= 500:  # otros 5xx: fallo temporal del servidor, reintentar
                errores += 1
                print(f"[!] HTTP {resp.status_code} en {url} (intento {errores})")
                self.limitador.penalizar(3 * errores)
                continue

            if resp.status_code != 404:  # 404 es normal (ficha inexistente): sin ruido
                print(f"[!] HTTP {resp.status_code} en {url}")
            return None

        if bloqueos >= reintentos_bloqueo:
            raise Bloqueado(url)
        return None

    # ------------------------------------------------------------------
    # Descubrimiento de URLs
    # ------------------------------------------------------------------
    def _enlaces(self, soup: BeautifulSoup, propia: str = "") -> List[str]:
        """Devuelve todas las fichas de película/serie enlazadas desde una página."""
        id_propio = None
        m0 = self.FILM_RE.search(propia)
        if m0:
            id_propio = m0.group(1)  # para no volver a encolar la propia ficha

        vistos, urls = set(), []
        for a in soup.find_all("a", href=True):
            m = self.FILM_RE.search(a["href"])
            if m and m.group(1) != id_propio and m.group(1) not in vistos:
                vistos.add(m.group(1))
                urls.append(self.BASE_URL + m.group(0))  # URL normalizada (sin parámetros)
        return urls

    def _generador_listados(self):
        """Genera URLs del listado A-Z (allfilms_<letra>_<pagina>.html)."""
        for pagina in range(1, self.MAX_PAG_LISTADO + 1):
            for letra in self.LETRAS:
                yield f"{self.BASE_URL}/cr/allfilms_{letra}_{pagina}.html"

    def _agregar(self, nuevas, frontera: List[str], vistas: Set[str], f_urls):
        """Añade a la cola las URLs no vistas y las persiste en el archivo de URLs."""
        for u in nuevas:
            if u not in vistas:
                vistas.add(u)
                frontera.append(u)
                f_urls.write(u + "\n")
        f_urls.flush()

    def _rellenar_desde_listados(self, frontera, vistas, f_urls) -> bool:
        """Refuerzo: si la cola se agota, intenta traer títulos del listado A-Z."""
        while self._listados_activos:
            url = next(self._gen_listados, None)
            if url is None:  # se agotaron todas las páginas del listado
                self._listados_activos = False
                return False
            try:
                soup = self._get_soup(url)
            except Bloqueado:
                return False
            nuevas = [u for u in (self._enlaces(soup) if soup else []) if u not in vistas]
            if nuevas:
                self._fallos_listado = 0
                self._agregar(nuevas, frontera, vistas, f_urls)
                return True
            # 15 páginas seguidas sin títulos nuevos: el listado no sirve, se abandona
            self._fallos_listado += 1
            if self._fallos_listado >= 15:
                print("[!] El listado A-Z no aporta títulos; se desactiva el refuerzo.")
                self._listados_activos = False
        return False

    @staticmethod
    def _leer_lineas(ruta: str) -> List[str]:
        """Lee un archivo de texto y devuelve sus líneas no vacías ([] si no existe)."""
        if not os.path.exists(ruta):
            return []
        with open(ruta, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    # ------------------------------------------------------------------
    # Extracción de una ficha
    # ------------------------------------------------------------------
    @staticmethod
    def _texto(el) -> str:
        """Texto de un elemento con espacios normalizados ('' si el elemento es None)."""
        return re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el else ""

    @staticmethod
    def _pick(info: Dict, *claves):
        """Devuelve el primer valor existente entre varias claves posibles (con/sin tilde)."""
        for c in claves:
            if c in info:
                return info[c]
        return None

    @staticmethod
    def _unicos(items) -> List[str]:
        """Elimina vacíos y duplicados conservando el orden."""
        return list(dict.fromkeys(i for i in items if i))

    def _extraer_info(self, soup: BeautifulSoup) -> Dict:
        """Convierte la lista <dl class="movie-info"> en un dict {título_en_minúscula: <dd>}."""
        info = {}
        dl = soup.select_one("dl.movie-info")
        if dl:
            for dt in dl.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                if dd is not None:
                    info[dt.get_text(strip=True).lower()] = dd
        return info

    def _extraer_comentarios(self, soup: BeautifulSoup, url: str) -> List[str]:
        """
        Primero intenta las críticas de prensa de la propia ficha. Si no hay y está
        permitido, hace UNA petición extra a la página de críticas de usuarios.
        """
        comentarios = []

        # 1) Críticas profesionales (ya están en la página, sin petición extra)
        for li in soup.select("ul.pro-reviews li, .pro-review"):
            cuerpo = li.select_one("[itemprop='reviewBody'], .review-text")
            autor = li.select_one(".review-author, [itemprop='author']")
            texto = self._texto(cuerpo)
            if texto:
                a = self._texto(autor)
                comentarios.append(f"{texto} ({a})" if a else texto)

        # 2) Críticas de usuarios (cuesta una petición adicional por ficha)
        if not comentarios and self.incluir_criticas_usuarios:
            m = self.FILM_RE.search(url)
            if m:
                soup_rev = self._get_soup(
                    f"{self.BASE_URL}/cr/reviews/1/film{m.group(1)}.html"
                )
                if soup_rev:
                    for item in soup_rev.select("div.rw-item"):
                        cuerpo = item.select_one(
                            ".review-text1, [itemprop='reviewBody'], .rw-text"
                        )
                        texto = self._texto(cuerpo)
                        if texto:
                            comentarios.append(texto)

        return comentarios[: self.max_comentarios]

    def _parsear(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extrae todos los campos de una ficha y los devuelve como dict (columnas CAMPOS)."""
        info = self._extraer_info(soup)

        # Título y tipo. El tipo se deduce del texto que acompaña al nombre en el <h1>,
        # p. ej. "Dark (Miniserie de TV)" -> "Serie", "(C)" -> cortometraje.
        h1 = soup.select_one("h1#main-title")
        nombre = self._texto(soup.select_one("h1#main-title [itemprop='name']") or h1)

        sufijo = self._texto(h1).replace(nombre, "", 1).lower()
        if "serie" in sufijo:
            tipo = "Serie"
        elif "(c)" in sufijo or "cortometraje" in sufijo:
            tipo = "Cortometraje"
        else:
            tipo = "Película"

        # Género: enlaces <a> dentro del <dd>; si no hay, el texto plano
        genero = ""
        dd = self._pick(info, "género", "genero")
        if dd is not None:
            genero = ", ".join(
                self._unicos(a.get_text(strip=True) for a in dd.find_all("a"))
            ) or self._texto(dd)

        # Dirección (en series suele llamarse "Creador")
        direccion = ""
        dd = self._pick(info, "dirección", "direccion", "creador")
        if dd is not None:
            nombres = self._unicos(
                s.get_text(strip=True) for s in dd.select("[itemprop='name']")
            )
            direccion = ", ".join(nombres) if nombres else self._texto(dd)

        return {
            "nombre": nombre,
            "genero": genero,
            "duracion": self._texto(self._pick(info, "duración", "duracion")),
            "pais": self._texto(self._pick(info, "país", "pais")),
            "direccion": direccion,
            "fecha_creacion": self._texto(self._pick(info, "año", "ano")),
            "ranking": self._texto(soup.select_one("#movie-rat-avg")),
            "votos": self._texto(soup.select_one("#movie-count-rat span")),
            "comentarios": "\n".join(self._extraer_comentarios(soup, url)),
            "url": url,
            "tipo": tipo,
        }

    def _procesar(self, url: str):
        """
        Trabajo de cada hilo: descarga y parsea UNA ficha. NO escribe en disco
        (eso lo hace solo el hilo principal, evitando locks sobre el CSV).
        Devuelve (url, datos, enlaces, estado) con estado:
          "ok"        -> ficha extraída
          "fallo"     -> no existe o error no recuperable
          "bloqueado" -> el sitio limita; se reencola para más tarde
        """
        try:
            soup = self._get_soup(url)
            if soup is None:
                return url, None, [], "fallo"
            datos = self._parsear(soup, url)
            return url, datos, self._enlaces(soup, url), "ok"
        except Bloqueado:
            return url, None, [], "bloqueado"
        except Exception as e:  # noqa: BLE001  (un error en una ficha no debe tumbar el script)
            print(f"[!] Error procesando {url}: {e}")
            return url, None, [], "fallo"

    # ------------------------------------------------------------------
    # Orquestación (rastreo + reanudación)
    # ------------------------------------------------------------------
    def _urls_ya_guardadas(self) -> Set[str]:
        """URLs que ya están en el CSV (sirve para reanudar sin duplicar)."""
        if not os.path.exists(self.salida) or os.path.getsize(self.salida) == 0:
            return set()
        with open(self.salida, newline="", encoding="utf-8-sig") as f:
            return {fila["url"] for fila in csv.DictReader(f) if fila.get("url")}

    def ejecutar(self, objetivo: int = 10000):
        # ---- Estado previo (reanudación) ----
        hechas = self._urls_ya_guardadas()
        fallidas_previas = self._leer_lineas(self.archivo_fallidas)
        descubiertas = self._leer_lineas(self.archivo_urls)
        vistas: Set[str] = set(descubiertas) | hechas  # todo lo que ya conocemos

        # Cola inicial = descubiertas - ya guardadas. Las fallidas previas se
        # reintentan por defecto (muchas fueron 429), salvo que se pida lo contrario.
        excluir = set(hechas)
        if not self.reintentar_fallidas:
            excluir |= set(fallidas_previas)
        frontera = [u for u in descubiertas if u not in excluir]
        total_ok = len(hechas)

        print(
            f"[+] Guardados: {total_ok} | en cola: {len(frontera)} "
            f"(fallidas previas: {len(fallidas_previas)}, "
            f"{'se reintentan' if self.reintentar_fallidas else 'se omiten'}) "
            f"| objetivo: {objetivo}"
        )
        if total_ok >= objetivo:
            print("[✓] Objetivo ya alcanzado.")
            return

        nuevo = not os.path.exists(self.salida) or os.path.getsize(self.salida) == 0
        inicio, procesadas, desde_descanso = time.time(), 0, 0
        detenido_por_bloqueo = False

        # Si se omiten las fallidas previas hay que conservarlas (modo "a");
        # si se reintentan, el archivo se regenera con las que sigan fallando ("w").
        modo_fallidas = "w" if self.reintentar_fallidas else "a"

        with open(self.salida, "a", newline="", encoding="utf-8-sig") as f, open(
            self.archivo_urls, "a", encoding="utf-8"
        ) as fu, open(self.archivo_fallidas, modo_fallidas, encoding="utf-8") as ff:
            writer = csv.DictWriter(f, fieldnames=self.CAMPOS)
            if nuevo:
                writer.writeheader()

            # ---- Semillas: portada + las que indique el usuario ----
            for semilla in [self.main_url] + self.semillas:
                try:
                    soup = self._get_soup(semilla)
                except Bloqueado:
                    soup = None
                if soup:
                    self._agregar(self._enlaces(soup), frontera, vistas, fu)
            print(f"[+] Cola inicial: {len(frontera)} títulos")

            ex = ThreadPoolExecutor(max_workers=self.hilos)
            en_proceso: set = set()
            try:
                while total_ok < objetivo and not detenido_por_bloqueo:
                    # ---- 1) Mantener a los hilos ocupados (2 tareas por hilo en vuelo) ----
                    while len(en_proceso) < self.hilos * 2 and total_ok + len(en_proceso) < objetivo:
                        if not frontera:
                            # Cola vacía: pedir refuerzo al listado A-Z; si no hay, salir
                            if not self._rellenar_desde_listados(frontera, vistas, fu):
                                break
                        if not frontera:
                            break
                        # Extracción aleatoria O(1): intercambia con el último y hace pop.
                        # Da variedad de géneros/épocas y evita patrones secuenciales.
                        i = random.randrange(len(frontera))
                        frontera[i], frontera[-1] = frontera[-1], frontera[i]
                        en_proceso.add(ex.submit(self._procesar, frontera.pop()))

                    if not en_proceso:
                        print("[!] La cola se agotó (o el sitio está limitando) antes del objetivo.")
                        print("    Usa --semillas con más páginas de partida (géneros, tops, listas).")
                        break

                    # ---- 2) Esperar a que termine al menos una tarea ----
                    terminados, _ = wait(en_proceso, return_when=FIRST_COMPLETED)
                    for fut in terminados:
                        en_proceso.discard(fut)
                        url, datos, enlaces, estado = fut.result()
                        procesadas += 1

                        if estado == "bloqueado":
                            frontera.append(url)  # NO es un fallo: se reintenta más tarde
                        elif estado == "ok" and datos and datos["nombre"]:
                            writer.writerow(datos)
                            total_ok += 1
                            desde_descanso += 1
                            self._agregar(enlaces, frontera, vistas, fu)  # crece el grafo
                        else:
                            ff.write(url + "\n")  # ficha inexistente o sin título
                            self._agregar(enlaces, frontera, vistas, fu)

                        # Volcado a disco periódico (si se corta, se pierde muy poco)
                        if procesadas % 20 == 0:
                            f.flush()
                            ff.flush()

                        # Progreso y estimación de tiempo restante
                        if procesadas % 50 == 0:
                            vel = procesadas / (time.time() - inicio)
                            falta = max(objetivo - total_ok, 0) / vel / 60 if vel else 0
                            print(
                                f"[{total_ok}/{objetivo}] {vel:.2f} fichas/s | "
                                f"cola: {len(frontera)} | faltan ~{falta:.0f} min"
                            )

                        # Descanso largo periódico para no despertar el límite del sitio
                        if self.lote and desde_descanso >= self.lote:
                            desde_descanso = 0
                            seg = random.uniform(*self.descanso)
                            print(f"[~] Descanso programado de {seg:.0f}s...")
                            self.limitador.penalizar(seg)

                        # Demasiados bloqueos seguidos: parar y dejar el progreso guardado
                        if self.limitador.racha >= self.max_racha:
                            detenido_por_bloqueo = True
                            break
            except KeyboardInterrupt:
                print("\n[!] Interrumpido. Guardando progreso...")
            finally:
                self.limitador.parar.set()  # despierta y termina los hilos en espera
                ex.shutdown(wait=False, cancel_futures=True)
                f.flush()
                ff.flush()

        print(f"[✓] Registros en {self.salida}: {total_ok}")
        if detenido_por_bloqueo:
            print(
                "[!] Detenido: el sitio sigue limitando las peticiones. El progreso está "
                "guardado.\n    Espera 1-2 horas (o hasta mañana) y ejecuta el mismo comando; "
                "reanuda solo.\n    Considera subir --intervalo y bajar --hilos."
            )
        elif total_ok < objetivo:
            print("    Vuelve a ejecutar el mismo comando para continuar (reanuda solo).")

    # ------------------------------------------------------------------
    # Excel (opcional)
    # ------------------------------------------------------------------
    def exportar_excel(self, ruta: str = "peliculas_series_scrapping.xlsx"):
        # Imports locales: pandas/openpyxl solo se exigen si se usa --excel
        import pandas as pd
        from openpyxl.styles import Alignment

        df = pd.read_csv(self.salida, encoding="utf-8-sig", dtype=str).fillna("")
        # Excel limita cada celda a 32767 caracteres
        df["comentarios"] = df["comentarios"].str.slice(0, 32000)
        # Quita caracteres de control que harían fallar a openpyxl (IllegalCharacterError)
        df = df.apply(lambda col: col.str.replace(CARACTERES_ILEGALES, "", regex=True))

        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Peliculas_Series")
            ws = writer.sheets["Peliculas_Series"]
            # Columna de comentarios ancha y con ajuste de texto
            col = self.CAMPOS.index("comentarios") + 1
            ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = 80
            for fila in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                for celda in fila:
                    celda.alignment = Alignment(wrap_text=True, vertical="top")
        print(f"[✓] Excel guardado en {ruta} ({len(df)} filas)")


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Scraper masivo de FilmAffinity")
    p.add_argument("--objetivo", type=int, default=10000, help="registros deseados")
    p.add_argument("--hilos", type=int, default=3)
    p.add_argument(
        "--intervalo", type=float, nargs=2, default=(1.5, 3.0), metavar=("MIN", "MAX"),
        help="segundos entre peticiones (global, entre todos los hilos)",
    )
    p.add_argument(
        "--lote", type=int, default=200,
        help="cada N fichas guardadas hace un descanso largo (0 = desactivado)",
    )
    p.add_argument(
        "--descanso", type=float, nargs=2, default=(90, 180), metavar=("MIN", "MAX"),
        help="segundos de descanso entre lotes",
    )
    p.add_argument(
        "--pausa-bloqueo", type=float, default=60.0,
        help="pausa inicial (s) tras un HTTP 429; se duplica en cada bloqueo seguido",
    )
    p.add_argument(
        "--max-racha", type=int, default=6,
        help="bloqueos seguidos tras los cuales el script se detiene",
    )
    p.add_argument("--max-comentarios", type=int, default=5)
    p.add_argument("--sin-criticas-usuarios", action="store_true",
                   help="no hace la 2.ª petición a la página de críticas de usuarios")
    p.add_argument("--no-reintentar-fallidas", action="store_true",
                   help="omite las URLs de fallidas.txt en lugar de reintentarlas")
    p.add_argument("--semillas", nargs="*", default=[], help="URLs extra de partida")
    p.add_argument("--salida", default=str(CSV_SALIDA),
                   help="ruta del CSV final")
    p.add_argument("--directorio", default=str(DIRECTORIO_AUXILIAR),
                   help="carpeta de urls_filmaffinity.txt y fallidas.txt")
    p.add_argument("--excel", action="store_true", help="exportar también a .xlsx")
    args = p.parse_args()

    scraper = FilmAffinityMasivo(
        salida=args.salida,
        hilos=args.hilos,
        intervalo=tuple(args.intervalo),
        max_comentarios=args.max_comentarios,
        incluir_criticas_usuarios=not args.sin_criticas_usuarios,
        semillas=args.semillas,
        lote=args.lote,
        descanso=tuple(args.descanso),
        pausa_bloqueo=args.pausa_bloqueo,
        max_racha=args.max_racha,
        reintentar_fallidas=not args.no_reintentar_fallidas,
        directorio=args.directorio,
    )
    scraper.ejecutar(objetivo=args.objetivo)

    if args.excel:
        scraper.exportar_excel(os.path.splitext(scraper.salida)[0] + ".xlsx")


if __name__ == "__main__":
    main()