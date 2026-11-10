# Decisiones de arquitectura (ADR)

## ADR-001: Recuperación componible sobre un contrato mínimo
Todas las estrategias implementan `retrieve(query, k, where)` y se apilan (por ejemplo,
self-query sobre multi-query sobre denso). Cada una se prueba aislada y combinada.
**Costo:** un contrato mínimo no expone parámetros finos de cada estrategia.

## ADR-002: El "padre" es configurable y se mide, no se asume
La versión anterior afirmaba que el retriever padre inyectaba "el documento completo" y que
eliminaba "el 90% de las alucinaciones". Lo primero era falso (partía en bloques de ~2.000
caracteres) y lo segundo no tenía medición. Ahora hay dos niveles reales (`section` y `document`)
y el eval reporta cuánto contexto devuelve cada uno. En el corpus de ejemplo, padre-sección da el
mismo resultado que chunks y padre-documento devuelve unas 3,4 veces más contexto sin mejorar la
recuperación: la ventaja del padre queda **sin demostrar** hasta medirla con secciones largas.

## ADR-003: Aislamiento por colección; Chroma persistente, nunca efímero
El cliente efímero de Chroma comparte estado entre instancias del mismo proceso. Un servicio con
varios usuarios o corpus mezclaría datos. Decisión: cada `Collection` tiene su propio backend, y
Chroma se usa siempre con `PersistentClient`, con colecciones de nombres distintos. Los tests de
aislamiento corren contra ambos backends.

## ADR-004: Upsert idempotente con `content_hash`; borrado sin huérfanos
`doc_id` sale de la ruta relativa y `content_hash` del contenido, los metadatos y la configuración
de chunking. Re-ingestar lo mismo no hace nada; un documento modificado reemplaza todos sus chunks
(incluso si se acortó) y sus padres. `sync(prune=True)` borra lo que ya no está en la fuente.
**Límite:** dos archivos que generen el mismo `doc_id` hacen fallar la ingesta, en lugar de
sobrescribirse en silencio.

## ADR-005: Los embeddings por defecto son multilingües y locales
El corpus objetivo está en español; modelos como `slate-125m-english` o `all-mpnet-base-v2` están
pensados para inglés. Por defecto: `paraphrase-multilingual-MiniLM-L12-v2` (local). Watsonx
(`granite-embedding-278m-multilingual`) queda como opción para datos públicos: el texto sale a la
nube de IBM. La elección entre modelos **no se midió** en este repo (no hay acceso a descarga);
`evals --embedder` permite compararlos con los mismos casos.

## ADR-006: Guarda de sensibilidad fijada en la colección
Un embedder en la nube exige `sensitivity="public"`, y esa política y el nombre del embedder se
guardan en la colección: reabrirla con otro embedder o con otra política falla. Evita mezclar
espacios vectoriales y evita "blanquear" una colección sensible cambiando un parámetro.
**Límite:** es una barrera contra errores, no un control de seguridad contra un actor malicioso
con acceso al proceso.

## ADR-007: El LLM propone, el código valida
Self-query: el LLM devuelve un plan y el código lo valida contra una lista blanca de campos,
tipos, operadores y (opcionalmente) un vocabulario de valores; solo entran filtros válidos, con
reintento acotado. El `where` del llamador se conserva siempre: el LLM puede restringir más, no
ampliar. QA: se abstiene sin evidencia sin llamar al LLM, y rechaza respuestas sin cita válida.

## ADR-008: Evals que no se saturan
Un set donde la consulta repite el texto da 100 % con cualquier embedder. Se separan dos grupos:
`léxico` (detecta regresiones) y `parafraseo` (usa sinónimos y discrimina). Los umbrales de CI
están **por debajo** de lo medido: son una guarda de regresión, no un objetivo de calidad. El
corpus es sintético y chico: los números no se extrapolan a documentos reales.

## ADR-009: Sin interfaz web
La interfaz Gradio anterior reconstruía el índice en cada pregunta. Se reemplazó por una CLI que
persiste el índice. Una UI se puede construir sobre `Collection` y `qa.answer`.

## ADR-010: El umbral de abstención se calibra por embedder; no hay valor por defecto
Un umbral de similitud fijo (0,15) se eligió con el baseline léxico, donde lo no relacionado
puntúa ~0. Con el modelo multilingüe real dejó de abstener (2 de 3 consultas sin respuesta
fueron contestadas), y con el baseline rechazaba 6 de 10 consultas válidas redactadas con
sinónimos. Decisión: `min_score` es obligatorio en `qa.answer`, y los evals miden por embedder
el mejor puntaje de consultas con y sin respuesta, informan si se separan y qué umbral las
separa, y cuántas consultas válidas se rechazarían con el umbral usado.
Al sumar 10 consultas sin respuesta cercanas al dominio, el baseline léxico dejó de ser
separable (superposición 0,304): un umbral por similitud no alcanza para ese tipo de pregunta, y
el control de fondo pasa a ser el `NO_SE` del LLM más la exigencia de cita válida. Los casos que
un embedder no puede resolver se marcan `known_limit` por embedder: se muestran, no tapan nada y
no rompen el CI; en cualquier otro embedder fallan de verdad.
**Límite:** la calibración usa las mismas consultas con las que se evalúa; confirma que existe
una separación, no que generalice. En un corpus real hay que armar un set propio de preguntas
sin respuesta.
