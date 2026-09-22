# DSPy + GEPA + GLiNER2 — Extracción de personas embargadas

Proyecto experimental para optimizar automáticamente las descripciones de un schema de extracción usando:

- GLiNER2 → realiza la extracción de entidades.
- DSPy → define el programa de optimización.
- GEPA → busca mejores descripciones para el schema.
- LLM → propone/reflexiona sobre nuevas descripciones.
- Gold supervisado → permite medir si las extracciones son correctas.

El objetivo es extraer de documentos jurídicos:

- `nombre_embargado`
- `dni`
- `cuit_cuil`

---

# 1. Idea general

El proyecto parte de un schema inicial:

```text
nombre_embargado → descripción
dni              → descripción
cuit_cuil        → descripción
```

Estas descripciones se convierten internamente en un schema structured de GLiNER2:

```
persona_embargada
├── nombre_embargado
├── dni_embargado
└── cuit_cuil_embargado
```

GLiNER2 utiliza ese schema para extraer información de los documentos.

DSPy + GEPA intenta modificar las descripciones del schema para mejorar las métricas contra una base supervisada.

# 2. Flujo general

```
Gold supervisado +
Documentos para extracción
↓
dataset.py
↓
train / validation / test
↓
schema inicial
↓
GLiNER2
↓
predicciones
↓
metricas.py
↓
comparación contra gold
↓
DSPy + GEPA
↓
nuevo schema candidato
↓
GLiNER2 vuelve a ejecutarse
↓
validation decide si mejoró
↓
schema_optimizado.json
```

El split test no participa de la optimización.

Se utiliza únicamente al final para medir el resultado definitivo.

# 3. Estructura del proyecto

```
dspy_gliner_extraccion/
│
├── config/
│   └── gliner_embargo.yaml
│
├── data/
│   ├── bd_supervisada_embargos_actualizada.csv
│   └── fragmentos.csv
│
├── schemas/
│   ├── schema_inicial.json
│   └── schema_optimizado.json
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── gliner_module.py
│   ├── schema_optimizer.py
│   ├── metricas.py
│   ├── normalizacion.py
│   ├── postproceso.py
│   └── utils.py
│
├── output/
│
├── entrenar.py
├── evaluar.py
├── ejecutar.py
├── benchmark.py
├── test_unitarios.py
├── requirements.txt
└── README.md
```

# 4. Archivos principales

config/gliner_embargo.yaml

Es el punto central de configuración.

Aquí se define:

```
modelo GLiNER2
threshold global
thresholds por campo
LLM de DSPy
LLM de reflexión
presupuesto GEPA
rutas de los CSV
modo de entrada
split train/validation/test
schema inicial
schema optimizado
output
```

### Ejemplo

```
gliner:
  modelo: "fastino/gliner2-multi-v1"
  threshold_inicial: 0.55

  field_thresholds:
    nombre_embargado: 0.60
    dni: 0.50
    cuit_cuil: 0.50

datos:
  modo_entrada: "texto_completo"

  split:
    train: 0.60
    validation: 0.20
    test: 0.20

  seed: 42
```

### Para usar fragmentos:

`modo_entrada: "fragmentos"`

# 5. Archivos de src/

`dataset.py`

Prepara los datos del experimento.

Lee dos archivos:

```
bd_supervisada_embargos_actualizada.csv
        ↓
gold

fragmentos.csv
        ↓
texto_completo + fragmentos
```

Los une mediante:

`id`

También consolida documentos que tienen más de una persona embargada.

Finalmente crea:

```
train
validation
test
```

según las proporciones y seed definidos en el `YAML.`

`gliner_module.py`

Es el adaptador entre el proyecto y `GLiNER2.`

Carga el modelo:

`AutoExtractor.from_pretrained(...)`

y construye un schema structured:

```
create_schema()
    ↓
structure("persona_embargada")
    ↓
field("nombre_embargado")
field("dni_embargado")
field("cuit_cuil_embargado")
```

Luego ejecuta:

`model.extract(...)`

y transforma la respuesta de `GLiNER2` al formato interno:

```
{
  "nombre_embargado": "...",
  "dni": "...",
  "cuit_cuil": "..."
}
```

`schema_optimizer.py`

Contiene la lógica principal de `DSPy + GEPA.`

Sus responsabilidades son:

```
evaluar un schema
↓
ejecutar GLiNER2
↓
calcular métricas
↓
generar feedback
↓
permitir que GEPA proponga nuevas descripciones
↓
volver a evaluar
↓
seleccionar el mejor schema
```

También contiene la lógica para:

```
texto_completo
vs
fragmentos
```

Si se usan fragmentos:

```
fragmento 1 → GLiNER2
fragmento 2 → GLiNER2
fragmento 3 → GLiNER2
              ↓
       consolidación
              ↓
     resultado documento
```

`metricas.py`

Compara:

```
predicción GLiNER2
vs
gold supervisado
```

Calcula scores para:

```
nombre_embargado
dni
cuit_cuil
global
```

La métrica también tiene en cuenta la asociación entre personas y sus identificadores.

`normalizacion.py`

Normaliza valores antes de compararlos.

Por ejemplo:

```
30.123.456
↓
30123456
```

o:

```
JUAN PÉREZ
↓
juan perez
```

Esto evita penalizar diferencias puramente de formato.

`postproceso.py`

Contiene utilidades de postprocesado de predicciones.

La extracción structured actual ya devuelve los campos principales normalizados al contrato interno, mientras que otras operaciones de consolidación —como combinar resultados de fragmentos— se realizan desde schema_optimizer.py.

`utils.py`

Contiene funciones compartidas por todo el proyecto.

Entre otras cosas:

```
carga gliner_embargo.yaml
valida configuración
valida schemas
crea GLiNERModule
lee/escribe JSON
lee/escribe CSV
maneja thresholds
resuelve rutas
guarda selección final
```

Es el archivo por el que entra gran parte de la configuración del `YAML.`

# 6. ¿Cómo entra la configuración `YAML` al proyecto?

El flujo principal es:

```
config/gliner_embargo.yaml
        ↓
utils.cargar_config()
        ↓
config
```

Ese objeto config se comparte con los demás módulos.

Por ejemplo:

```
config["datos"]
        ↓
dataset.py
```

define los CSV y los splits.

```
config["gliner"]
        ↓
utils.crear_extractor()
        ↓
gliner_module.py
```

define modelo y thresholds.

```
config["llm"]
config["llm_reflexion"]
config["gepa"]
        ↓
schema_optimizer.py
```

define cómo se ejecuta `DSPy + GEPA.`

# 7. ¿Cómo funciona `DSPy + GEPA`?

`DSPy` no extrae directamente las entidades.

`GLiNER2` sigue siendo el extractor.

El flujo es:

```
GEPA propone nuevas descripciones
        ↓
se construye un nuevo schema GLiNER2
        ↓
GLiNER2 procesa documentos
        ↓
se compara contra gold
        ↓
se calcula score
        ↓
GEPA recibe score + feedback
        ↓
propone otro schema
```

Por ejemplo:

```
Schema candidato A
→ score validation 0.69
```

```
Schema candidato B
→ score validation 0.74
```

```
Schema candidato C
→ score validation 0.71
```

El sistema conserva el candidato que tenga mejor rendimiento en validation.

`GEPA` no modifica los pesos de `GLiNER2.`

Optimiza las instrucciones/descripciones del schema.

# 8. Train / Validation / Test

Por defecto:

```
60 % train
20 % validation
20 % test
```

### Train

Se utiliza durante la optimización de `DSPy/GEPA.`

### Validation

Se utiliza para decidir:

```
qué schema es mejor
qué threshold es mejor
```

### Test

No participa de la optimización.

Se utiliza solamente al final para medir el resultado final.

# 9. Scripts principales

Hay cuatro comandos principales.

`entrenar.py`

Es el comando de experimentación y optimización.

Puede utilizarse solamente para evaluar el baseline:

`python entrenar.py --eval-only`

O para ejecutar `DSPy + GEPA:`

`python entrenar.py --max-metric-calls 5`

Flujo:

```
train + validation
↓
schema inicial
↓
GLiNER2
↓
DSPy + GEPA
↓
schema candidato
↓
validation
↓
schema_optimizado.json
```

`evaluar.py`  
Evalúa un schema ya congelado.

No modifica el schema y no ejecuta GEPA.

#### Ejemplo:

` python evaluar.py --schema schemas/schema_optimizado.json --split test`

Sirve para responder:

### ¿Qué rendimiento tiene el schema optimizado sobre datos que no participaron de la optimización?

`ejecutar.py`

Es inferencia pura.

No necesita gold.

No utiliza:

```
DSPy
GEPA
LLM
train
validation
test
```

Solamente hace:

```
documentos nuevos
↓
schema final
↓
GLiNER2
↓
predicciones
```

Ejemplo:

`python ejecutar.py --input datos_nuevos.csv --output predicciones.csv`

`benchmark.py`

Sirve para comparar sistemas utilizando exactamente el mismo conjunto de documentos.

Por ejemplo:

```
GLiNER2 baseline
vs
GLiNER2 optimizado
vs
otro sistema externo
```

Permite comparar:

```
nombre_embargado
dni
cuit_cuil
score global
latencia
```

No optimiza ningún sistema.

# 10. Diferencia entre los cuatro comandos

## Script Función

`entrenar.py`: Busca y selecciona un mejor schema con DSPy + GEPA  
`evaluar.py`: Mide un schema congelado contra gold  
`ejecutar.py`: Usa el schema final sobre documentos nuevos sin gold  
`benchmark.py`: Compara diferentes sistemas o versiones

En forma resumida:

```
entrenar
→ encontrar el mejor schema
```

```
evaluar
→ medir ese schema
```

```
ejecutar
→ usar ese schema
```

```
benchmark
→ comparar ese schema contra otros
```

# 11. Primera instalación

` python -m pip install -r requirements.txt`

Comprobar GLiNER2:

`python -c "import gliner2; print('GLiNER2 OK')"`

Comprobar DSPy:

`python -c "import dspy; print('DSPy OK')"`

Si se utiliza Ollama:

```
ollama list
```

# 12. Tests

Compilar el proyecto:

`python -m compileall -q src entrenar.py evaluar.py ejecutar.py benchmark.py test_unitarios.py`

Ejecutar tests:

`python -m unittest -v test_unitarios`

Los tests son livianos y utilizan modelos simulados.

No descargan ni ejecutan GLiNER2 real.

# 13. Smoke test

Probar únicamente dos documentos reales:

` python entrenar.py --eval-only --limit 2`

Esto sirve para comprobar:

```
dataset
↓
GLiNER2
↓
schema
↓
predicciones
↓
métricas
↓
outputs
```

sin ejecutar todavía `DSPy + GEPA.`

# 14. Baseline completo

`python entrenar.py --eval-only`

Evalúa el schema inicial sobre todo validation.

Este resultado sirve como punto de comparación antes de la optimización.

# 15. Primera optimización GEPA

Para una prueba pequeña:

`python entrenar.py --max-metric-calls 5`

Después puede aumentarse:

`python entrenar.py --max-metric-calls 10`

o:

`python entrenar.py --max-metric-calls 20`

GEPA utiliza el LLM definido en:

### llm:

y el modelo de reflexión definido en:

#### llm_reflexion:

# 16. Evaluación final

Una vez generado:

`schemas/schema_optimizado.json`

evaluar sobre test:

`python evaluar.py --schema schemas/schema_optimizado.json --split test`

El split test debe utilizarse únicamente después de finalizar la optimización.

# 17. Archivos generados

Los resultados se guardan dentro de:

`output/`

Por ejemplo:

```
smoke.csv
smoke.json
```

```
baseline.csv
baseline.json
```

```
historial.json
thresholds.json
seleccion.json
```

`evaluacion_schema_optimizado_test.csv`

El schema final queda en:

`schemas/schema_optimizado.json`

# Resumen

La responsabilidad de cada componente es:

```
YAML
↓
configuración
```

```
dataset.py
↓
prepara datos
```

```
gliner_module.py
↓
extrae con GLiNER2
```

```
metricas.py
↓
compara contra gold
```

```
schema_optimizer.py
↓
DSPy + GEPA optimizan las descripciones
```

```
entrenar.py
↓
coordina la optimización
```

```
evaluar.py
↓
mide el resultado final
```

```
ejecutar.py
↓
inferencia sobre datos nuevos
```

```
benchmark.py
↓
comparación entre sistemas
```
