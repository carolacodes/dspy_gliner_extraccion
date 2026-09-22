# dspy_gliner_extraccion

Experimento para estudiar si DSPy + GEPA pueden optimizar las **descripciones de entidades** utilizadas por GLiNER para extraer información de documentos jurídicos de embargo.

El objetivo no es entrenar ni modificar los pesos de GLiNER.

DSPy + GEPA optimizan las instrucciones de un programa que genera un **schema global de descripciones** para GLiNER.

GLiNER sigue siendo siempre el modelo que realiza la extracción de entidades.

Las únicas entidades del experimento son:

- `nombre_embargado`
- `dni`
- `cuit_cuil`

No se extraen montos, cuentas bancarias, CBU, CVU ni otras entidades.

---

## Concepto general

El proyecto tiene dos etapas diferentes.

### 1. Optimización

Durante esta etapa intervienen:

- DSPy
- GEPA
- un LLM
- GLiNER
- dataset supervisado

Flujo conceptual:

```text
schema inicial
      ↓
programa DSPy generador de descripciones
      ↓
GEPA optimiza sus instrucciones
      ↓
schema candidato global
      ↓
GLiNER ejecuta extracción
      ↓
comparación contra ground truth
      ↓
score + feedback
      ↓
GEPA reflexiona y genera mejores instrucciones
      ↓
nuevo schema candidato
```

El resultado final es:

- `schemas/schema_optimizado.json`

con las tres descripciones seleccionadas.

Después se busca también el mejor threshold de GLiNER sobre validation.

### 2. Inferencia

Una vez terminada la optimización:

```text
documento
    ↓
schema optimizado
+
threshold optimizado
    ↓
GLiNER
    ↓
postproceso
    ↓
nombre_embargado
dni
cuit_cuil
```

En esta etapa:

- no se usa DSPy
- no se usa GEPA
- no se usa ningún LLM.

Sólo se utiliza GLiNER.

## Qué optimiza GEPA

GEPA no modifica directamente GLiNER.

Optimiza las instrucciones del programa DSPy encargado de producir las descripciones globales.

Por ejemplo, el schema inicial puede contener:

```json
{
  "nombre_embargado": "Nombre y apellido de persona física o razón social embargada o demandada",
  "dni": "Documento Nacional de Identidad de 7 u 8 dígitos del embargado",
  "cuit_cuil": "Clave Única de Identificación Tributaria o Laboral del embargado con formato XX-XXXXXXXX-X o 11 dígitos"
}
```

GEPA intenta encontrar instrucciones que permitan al LLM producir mejores descripciones.

Esas descripciones son posteriormente probadas con GLiNER.

La calidad del schema se determina exclusivamente por cómo funciona GLiNER sobre datos supervisados.

## Arquitectura

```text
                         DSPy + GEPA
                             │
                             ▼
                 Generador global de schema
                             │
                             ▼
                    schema candidato
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
             doc 1          doc 2          doc N
               │             │             │
               └────────── GLiNER ─────────┘
                             │
                             ▼
                     predicciones
                             │
                             ▼
                  ground truth / gold
                             │
                             ▼
                 métrica por persona
                             │
                             ▼
                     score + feedback
                             │
                             └──────────► GEPA
```

El documento jurídico no se utiliza para generar un schema específico.

El schema es global y reutilizable.

## Estructura del proyecto

```text
dspy_gliner_extraccion/
├── config/
│   └── gliner_embargo.yaml
├── src/
│   ├── __init__.py
│   ├── gliner_module.py
│   ├── schema_optimizer.py
│   ├── dataset.py
│   ├── metricas.py
│   ├── normalizacion.py
│   ├── postproceso.py
│   └── utils.py
├── entrenar.py
├── evaluar.py
├── benchmark.py
├── ejecutar.py
├── schemas/
│   ├── schema_inicial.json
│   └── schema_optimizado.json
├── output/
├── requirements.txt
├── README.md
└── test_unitarios.py
```

## Responsabilidad de cada archivo

### `src/gliner_module.py`

Wrapper de GLiNER.

Responsabilidades:

```text
texto
+
schema
+
threshold
    ↓
GLiNER
    ↓
spans
    ↓
postproceso
```

Devuelve:

- nombre_embargado
- dni
- cuit_cuil
- spans_raw
- entidades

El modelo GLiNER se carga una sola vez por instancia.

Las descripciones completas del schema son enviadas como labels a GLiNER.

### `src/schema_optimizer.py`

Contiene la lógica DSPy + GEPA.

Responsabilidades principales:

- construir el programa DSPy generador de schema
- convertir ejemplos supervisados a dspy.Example
- crear la métrica compatible con GEPA
- ejecutar GLiNER dentro de la evaluación
- devolver score + feedback
- ejecutar GEPA.compile(...)
- obtener un único schema global final
- comparar el schema optimizado contra el baseline
- optimizar posteriormente el threshold.

GEPA utiliza:

```text
TRAIN
→ reflexión y optimización
```

```text
VALIDATION
→ selección del programa/schema
```

TEST nunca participa en esta función.

### `src/dataset.py`

Carga el dataset supervisado.

Contrato de cada ejemplo:

```python
{
    "id": str,
    "text": str,
    "group": str | None,
    "expected": {
        "nombre_embargado": str,
        "dni": str,
        "cuit_cuil": str
    }
}
```

También realiza la división:

- train
- validation
- test

manteniendo documentos relacionados dentro del mismo grupo.

### `src/metricas.py`

Evalúa predicciones contra ground truth.

La evaluación se realiza a nivel de persona:

```text
persona
├── nombre_embargado
├── dni
└── cuit_cuil
```

Cada persona utiliza:

```text
nombre:    50%
DNI:       25%
CUIT/CUIL: 25%
```

Los nombres utilizan similitud fuzzy.

DNI y CUIT/CUIL requieren coincidencia exacta después de normalización.

Esto permite detectar errores como:

Gold:

```text
Juan  → DNI 123
María → DNI 456
```

Predicción:

```text
Juan  → DNI 456
María → DNI 123
```

Aunque los mismos nombres y DNI aparezcan, la asociación incorrecta reduce el score.

La función principal es:

```python
evaluar(expected, prediction)
```

y devuelve:

- nombre_embargado
- dni
- cuit_cuil
- global
- errores
- matching

con scores entre 0.0 y 1.0.

### `src/normalizacion.py`

Normaliza valores únicamente para comparación.

Ejemplos:

```text
JUÁN Pérez
→ juan perez
```

```text
30.123.456
→ 30123456
```

```text
20-30123456-7
→ 20301234567
```

No modifica el texto jurídico original.

### `src/postproceso.py`

Procesa los spans devueltos por GLiNER.

Ordena entidades por posición start y realiza una agrupación ordinal:

```text
primer nombre
↔ primer DNI
↔ primer CUIT
```

```text
segundo nombre
↔ segundo DNI
↔ segundo CUIT
```

Los valores múltiples se representan mediante:

```text
Juan Pérez | María López
```

Esta agrupación es una simplificación experimental.

Si GLiNER omite una entidad intermedia, puede producirse un desalineamiento ordinal.

### `src/utils.py`

Contiene:

- contratos comunes
- validación del schema
- validación del YAML
- configuración de GEPA
- configuración del LLM
- configuración de GLiNER
- lectura/escritura JSON
- lectura/escritura CSV
- hashes
- paths
- selección persistida
- generación de grilla de thresholds.

## Dataset supervisado

El proyecto espera un CSV supervisado.

Ejemplo:

```csv
texto,nombre_embargado,dni,cuit_cuil
"Se ordena embargo de Juan Pérez, DNI 30.123.456, CUIT 20-30123456-7.",Juan Pérez,30123456,20301234567
```

Valores múltiples:

```text
Juan Pérez | María López
```

Campos sin entidad:

vacíos

Los DNI y CUIT/CUIL se manejan como strings.

## Train / Validation / Test

La base supervisada se divide por defecto:

```text
60% train
20% validation
20% test
```

Configuración:

```yaml
datos:
  split:
    train: 0.60
    validation: 0.20
    test: 0.20

  seed: 42
```

### Train

Se utiliza durante la optimización GEPA.

Sirve para que GEPA reciba ejemplos y feedback sobre los errores del schema.

### Validation

Se utiliza para seleccionar:

- el mejor programa generado por GEPA
- el mejor schema
- el mejor threshold.

### Test

Se reserva exclusivamente para evaluación final.

No participa en:

- generación del schema
- reflexión
- selección del schema
- selección del threshold.

### Seed

El seed controla la aleatoriedad de la división del dataset.

Por ejemplo:

```yaml
seed: 42
```

permite que una misma base produzca siempre los mismos:

- train
- validation
- test

si el dataset no cambia.

El número 42 no representa cantidad de documentos ni cantidad de mezclas.

Es simplemente la referencia utilizada por el generador aleatorio.

## Control de leakage

El proyecto intenta evitar que documentos relacionados aparezcan en splits distintos.

Dos documentos se mantienen juntos cuando:

- comparten group
- tienen exactamente el mismo texto después de normalización ligera.

Ejemplo:

```text
expediente 123 → train
```

todos los documentos asociados al mismo group deben permanecer en train.

Esto evita resultados artificialmente optimistas.

## GLiNER

El modelo configurado inicialmente es:

```yaml
gliner:
  modelo: "urchade/gliner_multi-v2.1"
```

Configuración inicial:

```yaml
device: "cpu"

threshold_inicial: 0.45

flat_ner: true

multi_label: false

chunk_words: 160

overlap_words: 40
```

En una máquina con GPU compatible puede utilizarse:

```yaml
device: "cuda"
```

### Descripciones GLiNER

El wrapper envía a GLiNER las descripciones completas:

```python
list(schema.values())
```

y no únicamente:

- nombre_embargado
- dni
- cuit_cuil

Esto es fundamental porque las descripciones son precisamente lo que GEPA intenta optimizar.

Después de la inferencia, las descripciones son remapeadas a las claves canónicas.

## LLM utilizado por DSPy

El ejemplo de configuración utiliza Ollama:

```yaml
llm:
  provider: "ollama_chat"
  modelo: "llama3.2"
  api_base: "http://localhost:11434"
```

Este LLM genera las descripciones del schema.

No extrae entidades directamente.

## LLM de reflexión GEPA

GEPA puede utilizar el mismo modelo o uno distinto:

```yaml
llm_reflexion:
  provider: "ollama_chat"
  modelo: "llama3.2"
```

Este modelo analiza:

```text
score
+
feedback
```

y ayuda a GEPA a modificar las instrucciones del programa.

## Recursos y presupuesto GEPA

GEPA puede realizar muchas evaluaciones.

En este proyecto se limita explícitamente su presupuesto.

Configuración inicial:

```yaml
gepa:
  auto: null
  max_full_evals: null
  max_metric_calls: 20

  num_threads: 1

  seed: 42

  candidate_selection_strategy: "pareto"

  reflection_minibatch_size: 3

  skip_perfect_score: true
```

GEPA requiere utilizar exactamente uno de:

- auto
- max_full_evals
- max_metric_calls

Para las primeras pruebas se utiliza:

```yaml
max_metric_calls: 20
```

para controlar el consumo de recursos.

Con un LLM local esto ayuda a limitar:

- tiempo
- CPU
- RAM
- VRAM.

Con una API también limita indirectamente:

- cantidad de llamadas
- tokens
- costo.

El número debe ajustarse según el tamaño real del dataset.

## Optimización del threshold

Después de elegir el schema final, el proyecto busca un mejor threshold de GLiNER.

Configuración:

```yaml
optimizacion:
  optimizar_threshold: true

  threshold_min: 0.30

  threshold_max: 0.70

  threshold_paso: 0.05
```

Ejemplo:

```text
0.30
0.35
0.40
0.45
0.50
...
0.70
```

Cada threshold se evalúa exclusivamente sobre validation.

TEST no participa.

## Instalación

Crear entorno virtual:

```powershell
python -m venv .venv
```

Activar:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar:

```powershell
python -m pip install -r requirements.txt
```

Dependencias principales:

- gliner
- dspy>=3.0
- PyYAML

## Tests livianos

Los tests unitarios no utilizan:

- GLiNER real
- Ollama
- GPU
- APIs externas.

Utilizan stubs y mocks.

Ejecutar:

```powershell
python -m unittest -v test_unitarios
```

Estado validado:

```text
Ran 24 tests
OK
```

Los tests cubren:

- configuración GEPA
- presupuesto GEPA
- normalización
- métricas por persona
- DNI intercambiados
- dataset
- seed
- splits
- prevención de leakage
- postproceso
- uso de descripciones GLiNER
- métrica GEPA score + feedback
- schema global GEPA
- optimización de threshold
- benchmark externo.

## Verificación de sintaxis

Ejecutar:

```powershell
python -m compileall -q src entrenar.py evaluar.py ejecutar.py benchmark.py test_unitarios.py
```

Si no muestra errores, todos los archivos compilaron correctamente.

Este comando:

- no ejecuta GLiNER
- no utiliza Ollama
- no ejecuta GEPA
- no consume recursos pesados.

## Smoke test GLiNER

Primera integración real:

```powershell
python entrenar.py --eval-only --limit 2
```

Este comando:

```text
dataset
↓
split
↓
2 documentos de validation
↓
GLiNER real
↓
métricas
```

No utiliza:

- DSPy
- GEPA
- LLM
- Ollama

Su objetivo es comprobar:

- que GLiNER cargue
- que el modelo pueda descargarse/cargarse
- que las descripciones funcionen como labels
- que el postproceso funcione
- que las métricas puedan calcularse sobre predicciones reales.

## Baseline real

Después del smoke test:

```powershell
python entrenar.py --eval-only
```

Evalúa el schema inicial sobre todo validation.

No usa GEPA ni LLM.

Genera:

- `output/baseline.json`
- `output/baseline.csv`

## Optimización GEPA

Una optimización real utiliza:

```text
DSPy
+
GEPA
+
LLM
+
GLiNER
```

Ejemplo con el presupuesto configurado en YAML:

```powershell
python entrenar.py
```

También puede sobrescribirse desde CLI.

Por ejemplo:

```powershell
python entrenar.py --max-metric-calls 20
```

o:

```powershell
python entrenar.py --max-full-evals 1
```

o:

```powershell
python entrenar.py --gepa-auto light
```

No utilizar más de uno al mismo tiempo.

## Salida de GEPA

Después de la optimización se guarda:

- `schemas/schema_optimizado.json`

y metadata en:

- `output/seleccion.json`

También se generan:

- `output/historial.json`
- `output/thresholds.json`

## Evaluación final

Una vez congelados schema y threshold:

```powershell
python evaluar.py --split test
```

También puede evaluarse validation:

```powershell
python evaluar.py --split validation
```

El proyecto comprueba que el dataset y los splits coincidan con los utilizados durante la optimización.

## Inferencia final

Para aplicar el sistema a documentos que ya no necesitan ground truth:

```bash
python ejecutar.py \
  --input nuevos_documentos.csv \
  --output output/predicciones.csv
```

Este proceso utiliza únicamente:

```text
GLiNER
+
schema optimizado
+
threshold optimizado
```

No utiliza DSPy ni GEPA.

## Inferencia antes de optimizar

Mientras schema_optimizado.json todavía sea una copia del inicial:

```bash
python ejecutar.py \
  --input nuevos_documentos.csv \
  --output output/baseline_inferencia.csv \
  --schema schemas/schema_inicial.json
```

## Benchmark

Permite comparar:

```text
GLiNER baseline
vs
GLiNER optimizado con GEPA
vs
otro sistema externo
```

Ejecutar:

```powershell
python benchmark.py --split test
```

Comparación externa:

```bash
python benchmark.py \
  --split test \
  --external-csv resultados_externos.csv \
  --external-name otro_sistema
```

Formato externo:

```csv
id,nombre_embargado,dni,cuit_cuil,latencia_segundos
```

Las predicciones externas se evalúan contra el mismo ground truth utilizando las mismas métricas.

## Outputs

### Smoke test

- `output/smoke.json`
- `output/smoke.csv`

### Baseline

- `output/baseline.json`
- `output/baseline.csv`

### Optimización

- `output/historial.json`
- `output/thresholds.json`
- `output/seleccion.json`
- `schemas/schema_optimizado.json`

### Evaluación

- `output/evaluacion_<schema>_<split>.csv`
- `output/evaluacion_<schema>_<split>.json`

### Benchmark

- `output/benchmark_<split>.csv`

## Reproducibilidad

El proyecto guarda:

- seed
- schema
- hash del schema
- threshold
- configuración GLiNER
- configuración GEPA
- configuración del LLM
- versiones instaladas
- manifiesto de splits.

El seed hace reproducible la división del dataset.

No garantiza determinismo completo de:

- LLM
- GEPA
- hardware
- kernels de GLiNER.

## Estado actual

La arquitectura está validada mediante:

```powershell
python -m compileall -q src entrenar.py evaluar.py ejecutar.py benchmark.py test_unitarios.py
```

y:

```powershell
python -m unittest -v test_unitarios
```

Resultado actual:

```text
Ran 24 tests
OK
```

Estos tests validan la lógica mediante simulaciones.

Todavía no representan resultados experimentales reales.

Falta realizar:

- smoke test con GLiNER real
- baseline real
- integración con LLM
- optimización GEPA real
- evaluación final sobre test.

No se deben reportar métricas experimentales de GLiNER + GEPA hasta completar esas etapas.
