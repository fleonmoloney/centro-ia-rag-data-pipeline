# centro-ia-rag-data-pipeline

Pipeline RAG de referencia en Python: ingesta con metadata, chunking jerárquico, almacenes
vectoriales **aislados** con upsert/borrado reales, recuperación (padre, multi-query, self-query),
respuesta con citas y abstención, y evals de recuperación reproducibles.

Es un prototipo de arquitectura probado de forma automática, no un producto. Más abajo está dicho,
sin adornos, qué se midió y qué no.

## Qué incluye

| Pieza | Qué hace | Código |
|---|---|---|
| **Ingesta** | .md/.txt/.pdf/.csv/.json. `doc_id` estable por ruta, `content_hash` para saltar lo que no cambió, metadata filtrable (`year`, `institution`, `document_type`) desde front matter o sidecar `.meta.json`. Un PDF escaneado falla con un error claro (no hay OCR). | `ingest.py` |
| **Chunking** | Secciones por headers Markdown (o por página en PDF) y luego chunks con solapamiento. Cada chunk conoce su sección y su documento. | `ingest.py` |
| **Almacenes** | `MemoryBackend` (numpy, JSON opcional) y `ChromaBackend` con las mismas pruebas. Colecciones aisladas, upsert idempotente, borrado sin chunks huérfanos, `sync(prune=True)`. | `store.py` |
| **Recuperación** | `DenseRetriever`, `ParentRetriever` (sección o documento completo), `MultiQueryRetriever` (reformulaciones + RRF), `SelfQueryRetriever` (filtros del LLM validados contra lista blanca; el alcance del llamador no se puede ampliar). | `retrieval.py` |
| **QA** | Si el mejor puntaje es menor que `min_score` (**obligatorio: depende del embedder**) no se llama al LLM; el modelo puede abstenerse; una respuesta sin cita válida `[n]` se rechaza. | `qa.py` |
| **Embeddings** | `hashing` (offline, baseline léxico), `local` (sentence-transformers, multilingüe) y `watsonx` (solo con `sensitivity="public"`). | `embeddings.py` |
| **Evals** | 30 consultas de recuperación en dos grupos (`léxico` y `parafraseo`), 21 casos deterministas (13 de abstención, 10 de ellos cercanos al dominio), umbrales de regresión y **calibración del umbral de abstención**. | `evals.py` |

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # agregá ".[local]" o ".[watsonx]" si los vas a usar
pytest                                  # 144 tests
python -m centro_rag.evals              # evals con el corpus sintético (embedder: hashing)

# Indexar y consultar tus documentos (sin generación)
rag ingest --dir ./mis_docs --store .rag_store --embedder local
rag search "plazo de conservación" --store .rag_store --embedder local --where "year>=2022" -k 4
```

Comparar embedders con los mismos casos: `python -m centro_rag.evals --embedder local`
(descarga el modelo) o `--embedder watsonx --sensitivity public` (el corpus de ejemplo es ficticio,
por eso se puede enviar a la nube; **con tus datos reales no**).

## Qué se midió

Corpus sintético de **9 documentos ficticios** (ver `data/README.md`), embedder `hashing`
(baseline léxico, sin semántica), k = 3:

| Grupo | Estrategia | hit@3 | MRR | Tokens de contexto (prom.) |
|---|---|---|---|---|
| léxico (20 consultas) | chunks | 1,0 | 1,0 | 143,8 |
| léxico | padre-sección | 1,0 | 1,0 | 143,8 |
| léxico | padre-documento | 1,0 | 1,0 | 492,6 |
| parafraseo (10 consultas) | chunks | 0,9 | 0,7 | 149,7 |
| parafraseo | padre-sección | 0,9 | 0,7 | 149,7 |
| parafraseo | padre-documento | 0,9 | 0,7 | 511,5 |

Con el modelo local `paraphrase-multilingual-MiniLM-L12-v2` (corrida del autor en Windows,
Python 3.14, 3/11/2026; yo no pude descargar el modelo desde mi entorno):

| Grupo | Estrategia | hit@3 | MRR | Tokens de contexto (prom.) |
|---|---|---|---|---|
| léxico (20 consultas) | chunks | 1,0 | 1,0 | 147,3 |
| léxico | padre-sección | 1,0 | 1,0 | 147,3 |
| léxico | padre-documento | 1,0 | 1,0 | 505,2 |
| parafraseo (10 consultas) | chunks | 1,0 | 0,95 | 148,1 |
| parafraseo | padre-sección | 1,0 | 0,95 | 148,1 |
| parafraseo | padre-documento | 1,0 | 0,95 | 473,5 |

En `parafraseo` el modelo multilingüe mejora al baseline (MRR 0,7 → 0,95; hit@3 0,9 → 1,0). Son
10 consultas: un solo caso mueve hit@3 en 0,1, así que es una señal a favor, no una
demostración. Con un corpus real y más consultas puede cambiar.

Cómo leerlo:

- **El grupo `léxico` es fácil a propósito** (la consulta repite palabras del texto) y se satura:
  sirve para detectar regresiones, no para comparar modelos.
- **El grupo `parafraseo` usa sinónimos** y es el que separa un baseline léxico de un modelo
  semántico. Con `hashing` falla 1 de 10 (`p02-concurso-abierto`) y el MRR baja a 0,7. Un
  modelo multilingüe debería mejorar ahí, **pero no lo medí**: desde mi entorno no se puede
  descargar ningún modelo. Corré `--embedder local` y mirá esa tabla.
- **Padre-sección da lo mismo que chunks** porque en este corpus casi todas las secciones caben en
  un chunk. El beneficio del retriever padre solo se ve con secciones largas; no está demostrado
  aquí.
- **Padre-documento devuelve ~3,4 veces más contexto** (492,6 vs 143,8 tokens) sin mejorar
  la recuperación en este set.
- Los tokens son una estimación (~4 caracteres/token), no el conteo de un proveedor.

## Abstención: el umbral depende del embedder

`qa.answer` exige `min_score` (no hay valor por defecto) porque no existe un valor universal.
Dos observaciones medidas:

- **Baseline léxico con 0,15:** abstiene bien en las 3 consultas sin respuesta, pero rechaza
  indebidamente **6 de las 10** consultas parafraseadas válidas (sus sinónimos comparten pocas
  palabras con el texto y el puntaje cae por debajo de 0,15).
- **Modelo local con 0,15:** falló en 2 de las 3 consultas sin respuesta (`a02`, `a03`): contestó
  a "resultados del campeonato de fútbol" y "cómo configurar un router". Un modelo neuronal da
  similitudes apreciables incluso entre textos sin relación, así que un umbral pensado para el
  baseline no se traslada.

- **Preguntas cercanas al dominio (baseline léxico):** con 10 consultas sin respuesta que
  comparten vocabulario con el corpus ("licencia por maternidad del Ministerio de Salud",
  "plazo de conservación de datos biométricos"), el mejor puntaje de las que no tienen respuesta
  llega a 0,304 y el de las válidas baja hasta 0,0: **ningún umbral separa**. Con 0,15 el baseline
  contesta 7 de esas 10. Esos casos figuran como `LÍMITE CONOCIDO` para `hashing-2048` (no rompen
  el CI, pero se ven en el reporte). Para cualquier otro embedder cuentan como fallas reales.
- **Modelo local:** el reporte con solo 3 consultas lejanas dio margen 0,03 (0,275 contra 0,305);
  con las cercanas **no está medido** (hay que correrlo en una máquina con acceso al modelo).

`python -m centro_rag.evals --embedder <el_tuyo>` imprime una sección **Calibración**: el mejor
puntaje de las consultas con respuesta, el de las que no la tienen, si se separan, un umbral
sugerido y cuántas consultas válidas se rechazarían con él. `--min-score` fija uno a mano.
Está calibrado y evaluado sobre las mismas consultas: confirma que existe una separación, no que
generalice. Para tu corpus real armá tu propio set de preguntas sin respuesta y calibrá con él.

## Qué NO está demostrado

- **Economía de la inferencia / ventana de contexto.** La versión anterior de este repo imprimía
  cifras fijas ("10x", "40%") que no salían de ninguna medición. Se eliminaron. Con este corpus
  (todo cabe en ~1.500 tokens) el argumento de la ventana de contexto no se puede demostrar;
  habría que medirlo con un corpus real, más grande que la ventana del modelo elegido.
- **El umbral de abstención en tu corpus real.** Se calibró con 13 consultas sin respuesta
  inventadas por mí sobre un corpus ficticio: sirve para mostrar el método, no para fijar el valor.
- **Calidad de respuestas.** No hay LLM real en este repo. Los controles de `qa.py` son
  mecánicos (hay evidencia, hay cita válida); que la cita respalde de verdad la afirmación
  requiere evaluar con personas o con otro modelo.
- **Multi-query y self-query con un LLM real.** Los tests usan un LLM de guion: verifican la
  fusión RRF, la validación de planes y que el alcance del llamador no se amplía. No miden qué
  tan buenas son las reformulaciones o los planes de un modelo real.
- **Resistencia a inyección de instrucciones en los documentos.** El prompt marca las fuentes
  como no confiables, lo cual es una mitigación y no una defensa completa.
- **Escala.** `MemoryBackend` busca por fuerza bruta (O(n) por consulta). Chroma escala mejor, pero
  no se probó con volúmenes reales.
- **`WatsonxEmbedder` y `SentenceTransformerEmbedder` no se probaron contra sus servicios.** El
  cableado de watsonx se verifica con un SDK simulado; confirmá que el `model_id` esté disponible
  en tu región.

## Datos sensibles

- **Aislamiento:** cada colección tiene su backend. La versión anterior usaba el cliente efímero de
  Chroma, que comparte estado entre instancias del mismo proceso: un segundo usuario recuperaba
  fragmentos del primero. Hay tests que lo impiden en ambos backends.
- **Nube:** un embedder en la nube se rechaza salvo que declares `sensitivity="public"`, y la
  política queda fijada en la colección. Con watsonx, cada fragmento y cada consulta viajan a la
  nube de IBM; para datos personales o de salud hay que revisar la normativa aplicable (en
  Uruguay, la Ley 18.331) antes de usarlo. Esto no es asesoramiento legal.
- **Borrado:** `delete_document` elimina chunks y padres. No se probó el borrado físico a nivel
  de disco de Chroma, ni la política de retención del almacenamiento subyacente.
- **Sin autenticación ni control de acceso por usuario:** el parámetro `where` permite acotar el
  alcance, pero quien lo arma es el llamador.

## Estructura

```
src/centro_rag/   ingest · store · embeddings · retrieval · qa · evals · cli · llm
data/             corpus sintético (ficticio)
evals/            casos dorados y umbrales de regresión
tests/            144 tests (pytest); los de almacenamiento corren contra Memory y Chroma
docs/             decisiones de diseño (ADR)
```

Por qué no hay Gradio: la interfaz anterior reconstruía el índice completo en cada pregunta y
dependía de un entorno de laboratorio de IBM. Prefiero una CLI que persista el índice y sea
verificable. Una interfaz web se puede construir encima de `Collection` y `qa.answer`.

Licencia MIT.
