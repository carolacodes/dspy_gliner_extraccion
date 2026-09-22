"""Optimización global de descripciones GLiNER con DSPy GEPA.

Arquitectura:

GEPA
    ↓
programa DSPy generador de schema
    ↓
UN schema candidato global
    ↓
GLiNER sobre documentos supervisados
    ↓
predicción consolidada a nivel documento
    ↓
métrica por persona
    ↓
score + feedback
    ↓
GEPA refina las instrucciones

Modos de entrada:

- texto_completo:
    GLiNER procesa una vez example["text"].

- fragmentos:
    GLiNER procesa cada elemento de example["fragments"].
    Luego las predicciones se consolidan a nivel documento.

El LLM nunca recibe el documento jurídico para generar
un schema específico por documento.

GEPA no modifica los pesos de GLiNER.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Mapping

from .normalizacion import (
    normalizar_cuit_cuil,
    normalizar_dni,
    normalizar_nombre,
)
from .utils import (
    KEYS,
    guardar_json,
    validar_schema,
    validar_threshold,
)


# ============================================================
# CONSTANTES
# ============================================================

MODOS_ENTRADA = {
    "texto_completo",
    "fragmentos",
}


TASK_TEXT = """
Generar exactamente tres descripciones globales de entidades para GLiNER:

- nombre_embargado
- dni
- cuit_cuil

Estas descripciones serán reutilizadas sobre documentos jurídicos
de embargos.

El objetivo es ayudar a GLiNER a identificar correctamente:

1. la persona física o razón social sobre la que recae el embargo;
2. el DNI correspondiente a esa persona;
3. el CUIT/CUIL correspondiente a esa persona.

Las descripciones deben ser generales, precisas y discriminativas.

Para nombre_embargado, deben ayudar a distinguir al embargado de otras
personas mencionadas en documentos judiciales, como jueces, secretarios,
abogados, letrados, representantes u otras autoridades.

No extraer entidades de ningún documento.
No generar respuestas particulares para un documento.
No cambiar las tres claves canónicas.

Sólo generar las descripciones globales que GLiNER utilizará como labels.
""".strip()


# ============================================================
# MODO DE ENTRADA
# ============================================================

def validar_modo_entrada(modo):
    if modo not in MODOS_ENTRADA:
        raise ValueError(
            "modo_entrada debe ser "
            "'texto_completo' o 'fragmentos'"
        )

    return modo


# ============================================================
# HELPERS PARA CONSOLIDACIÓN DE FRAGMENTOS
# ============================================================

def _split_pipe_preservando_posicion(value):
    """
    Divide strings del tipo:

    "Juan | María"
    "30123456 | "

    preservando posiciones vacías.
    """
    if value is None:
        return []

    value = str(value)

    if not value.strip():
        return []

    return [
        part.strip()
        for part in value.split("|")
    ]


def _normalizar_campo(key, value):
    if key == "nombre_embargado":
        return normalizar_nombre(value)

    if key == "dni":
        return normalizar_dni(value)

    if key == "cuit_cuil":
        return normalizar_cuit_cuil(value)

    return str(value or "").strip()


def _personas_desde_prediccion(prediction):
    """
    Convierte una predicción pipe-separated en personas.

    Ejemplo:

    nombre = Juan | María
    dni    = 123  | 456

    ->
    [
        {"nombre_embargado": "Juan", "dni": "123", ...},
        {"nombre_embargado": "María", "dni": "456", ...}
    ]
    """
    values = {
        key: _split_pipe_preservando_posicion(
            prediction.get(key, "")
        )
        for key in KEYS
    }

    count = max(
        (
            len(values[key])
            for key in KEYS
        ),
        default=0,
    )

    persons = []

    for index in range(count):
        person = {
            key: (
                values[key][index]
                if index < len(values[key])
                else ""
            )
            for key in KEYS
        }

        if any(
            str(value).strip()
            for value in person.values()
        ):
            persons.append(person)

    return persons


def _personas_compatibles(existing, candidate):
    """
    Decide si dos registros probablemente representan
    a la misma persona.

    Se consideran compatibles cuando comparten al menos
    un identificador fuerte o el mismo nombre normalizado,
    sin contradicciones explícitas.
    """
    existing_norm = {
        key: _normalizar_campo(
            key,
            existing.get(key, ""),
        )
        for key in KEYS
    }

    candidate_norm = {
        key: _normalizar_campo(
            key,
            candidate.get(key, ""),
        )
        for key in KEYS
    }

    # Si hay valores no vacíos contradictorios
    # en un mismo campo, no fusionar.
    for key in KEYS:
        a = existing_norm[key]
        b = candidate_norm[key]

        if a and b and a != b:
            # El nombre admite una pequeña excepción:
            # los identificadores fuertes pueden confirmar
            # que es la misma persona.
            if key == "nombre_embargado":
                continue

            return False

    same_dni = (
        existing_norm["dni"]
        and candidate_norm["dni"]
        and existing_norm["dni"]
        == candidate_norm["dni"]
    )

    same_cuit = (
        existing_norm["cuit_cuil"]
        and candidate_norm["cuit_cuil"]
        and existing_norm["cuit_cuil"]
        == candidate_norm["cuit_cuil"]
    )

    same_name = (
        existing_norm["nombre_embargado"]
        and candidate_norm["nombre_embargado"]
        and existing_norm["nombre_embargado"]
        == candidate_norm["nombre_embargado"]
    )

    return bool(
        same_dni
        or same_cuit
        or same_name
    )


def _fusionar_persona(existing, candidate):
    """
    Completa campos vacíos de una persona existente
    con información encontrada en otro fragmento.
    """
    merged = dict(existing)

    for key in KEYS:
        current = str(
            merged.get(key, "")
            or ""
        ).strip()

        new = str(
            candidate.get(key, "")
            or ""
        ).strip()

        if not current and new:
            merged[key] = new

    return merged


def consolidar_predicciones_fragmentos(predictions):
    """
    Consolida predicciones provenientes de varios fragmentos
    pertenecientes al mismo documento.

    Intenta evitar duplicados cuando la misma persona aparece
    repetida en distintos fragmentos.

    Los offsets de spans permanecen relativos al fragmento.
    Se agrega fragment_index para mantener trazabilidad.
    """
    consolidated_persons = []
    spans_raw = []

    for fragment_index, prediction in enumerate(predictions):
        persons = _personas_desde_prediccion(
            prediction
        )

        for candidate in persons:
            matched_index = None

            for index, existing in enumerate(
                consolidated_persons
            ):
                if _personas_compatibles(
                    existing,
                    candidate,
                ):
                    matched_index = index
                    break

            if matched_index is None:
                consolidated_persons.append(
                    dict(candidate)
                )

            else:
                consolidated_persons[
                    matched_index
                ] = _fusionar_persona(
                    consolidated_persons[
                        matched_index
                    ],
                    candidate,
                )

        for span in prediction.get(
            "spans_raw",
            [],
        ):
            span_copy = dict(span)

            span_copy[
                "fragment_index"
            ] = fragment_index

            spans_raw.append(
                span_copy
            )

    result = {}

    for key in KEYS:
        result[key] = " | ".join(
            str(
                person.get(
                    key,
                    "",
                )
                or ""
            ).strip()
            for person in consolidated_persons
        )

    result["entidades"] = [
        {
            key: person.get(
                key,
                "",
            )
            for key in KEYS
        }
        for person in consolidated_persons
    ]

    result["spans_raw"] = spans_raw

    return result


# ============================================================
# EJECUCIÓN GLINER SEGÚN MODO
# ============================================================

def predecir_ejemplo(
    extractor,
    example,
    schema,
    threshold,
    modo_entrada,
):
    """
    Ejecuta GLiNER sobre un ejemplo según el modo configurado.
    """
    modo_entrada = validar_modo_entrada(
        modo_entrada
    )

    schema = validar_schema(
        schema
    )

    threshold = validar_threshold(
        threshold
    )

    if modo_entrada == "texto_completo":
        text = str(
            example.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            raise ValueError(
                f"El ejemplo {example.get('id')} "
                "no tiene texto completo"
            )

        return extractor.predict(
            text,
            schema,
            threshold,
        )

    # --------------------------------------------------------
    # FRAGMENTOS
    # --------------------------------------------------------

    fragments = example.get(
        "fragments",
        [],
    )

    if not isinstance(
        fragments,
        list,
    ):
        raise ValueError(
            f"fragments inválido "
            f"para id {example.get('id')}"
        )

    fragments = [
        str(fragment).strip()
        for fragment in fragments
        if str(fragment or "").strip()
    ]

    if not fragments:
        raise ValueError(
            f"El ejemplo {example.get('id')} "
            "no tiene fragmentos"
        )

    predictions = []

    for fragment in fragments:
        predictions.append(
            extractor.predict(
                fragment,
                schema,
                threshold,
            )
        )

    return consolidar_predicciones_fragmentos(
        predictions
    )


# ============================================================
# EVALUACIÓN NORMAL DE UN SCHEMA
# ============================================================

def evaluar_schema(
    extractor,
    examples,
    schema,
    threshold,
    metric,
    modo_entrada="texto_completo",
):
    """
    Evalúa un schema fijo con GLiNER.

    La comparación siempre se hace a nivel documento,
    independientemente del modo de entrada.
    """
    if not examples:
        raise ValueError(
            "No se puede evaluar un split vacío"
        )

    schema = validar_schema(
        schema
    )

    threshold = validar_threshold(
        threshold
    )

    modo_entrada = validar_modo_entrada(
        modo_entrada
    )

    rows = []

    for example in examples:
        start = time.perf_counter()

        prediction = predecir_ejemplo(
            extractor,
            example,
            schema,
            threshold,
            modo_entrada,
        )

        latency = (
            time.perf_counter()
            - start
        )

        scores = metric(
            example["expected"],
            prediction,
        )

        for key in (
            *KEYS,
            "global",
        ):
            value = scores.get(
                key
            )

            if (
                not isinstance(
                    value,
                    (int, float),
                )
                or not math.isfinite(
                    value
                )
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Métrica inválida: "
                    f"{key}={value!r}"
                )

        rows.append(
            {
                "id":
                    example["id"],

                "text":
                    example.get(
                        "text",
                        "",
                    ),

                "fragments":
                    example.get(
                        "fragments",
                        [],
                    ),

                "expected":
                    example["expected"],

                "prediction":
                    prediction,

                "scores":
                    scores,

                "latencia_segundos":
                    latency,
            }
        )

    metrics = {
        key: (
            sum(
                row["scores"][key]
                for row in rows
            )
            / len(rows)
        )
        for key in (
            *KEYS,
            "global",
        )
    }

    metrics[
        "latencia_promedio"
    ] = (
        sum(
            row[
                "latencia_segundos"
            ]
            for row in rows
        )
        / len(rows)
    )

    return {
        "metricas":
            metrics,

        "resultados":
            rows,
    }


# ============================================================
# REPORTE DE ERRORES
# ============================================================

def generar_reporte_errores(
    evaluation,
    max_ejemplos=6,
    max_chars=6000,
):
    """
    Genera un diagnóstico compacto para inspección humana.
    """
    cases = []

    roles = re.compile(
        r"\b("
        r"juez|jueza|"
        r"abogado|abogada|"
        r"secretario|secretaria|"
        r"letrado|letrada"
        r")s?\b",
        re.I,
    )

    ordered = sorted(
        evaluation[
            "resultados"
        ],
        key=lambda item:
            item["scores"][
                "global"
            ],
    )

    for row in ordered:
        if len(cases) >= max_ejemplos:
            break

        if (
            row["scores"][
                "global"
            ]
            >= 1.0
        ):
            continue

        errors = row[
            "scores"
        ].get(
            "errores",
            {},
        )

        text = str(
            row.get(
                "text",
                "",
            )
            or ""
        )

        excerpt = text[:240]

        case = {
            "id":
                str(
                    row["id"]
                )[:80],

            "scores": {
                key:
                    row["scores"][
                        key
                    ]
                for key in (
                    *KEYS,
                    "global",
                )
            },

            "errores":
                errors,

            "esperado": {
                key:
                    str(
                        row[
                            "expected"
                        ].get(
                            key,
                            "",
                        )
                    )[:200]
                for key in KEYS
            },

            "predicho": {
                key:
                    str(
                        row[
                            "prediction"
                        ].get(
                            key,
                            "",
                        )
                    )[:200]
                for key in KEYS
            },

            "fragmento_texto":
                excerpt,

            "roles_cercanos_posible_confusion":
                sorted(
                    {
                        match.group(0)
                        for match
                        in roles.finditer(
                            excerpt
                        )
                    }
                ),
        }

        candidate = json.dumps(
            {
                "casos":
                    cases
                    + [case]
            },
            ensure_ascii=False,
            default=str,
        )

        if len(candidate) > max_chars:
            break

        cases.append(
            case
        )

    return json.dumps(
        {
            "casos":
                cases
        },
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# PARSING / VALIDACIÓN DE SCHEMA
# ============================================================

def parsear_schema(value):
    if isinstance(
        value,
        str,
    ):
        text = value.strip()

        if text.startswith(
            "```"
        ):
            lines = (
                text.splitlines()
            )

            if (
                len(lines) < 3
                or lines[-1].strip()
                != "```"
            ):
                raise ValueError(
                    "Bloque JSON incompleto"
                )

            text = "\n".join(
                lines[1:-1]
            )

            stripped = (
                text.lstrip()
            )

            if (
                stripped.lower()
                .startswith("json")
            ):
                text = (
                    stripped[4:]
                    .lstrip()
                )

        value = json.loads(
            text
        )

    return validar_schema(
        value
    )


# ============================================================
# CONTROL DE LEAKAGE
# ============================================================

def _separacion(
    train,
    validation,
):
    if (
        not train
        or not validation
    ):
        raise ValueError(
            "Train y validation "
            "deben ser no vacíos"
        )

    for field in (
        "id",
        "text",
        "group",
    ):

        def values(examples):
            result = set()

            for example in examples:
                value = example.get(
                    field
                )

                if value is None:
                    continue

                normalized = " ".join(
                    str(value)
                    .casefold()
                    .split()
                )

                if normalized:
                    result.add(
                        normalized
                    )

            return result

        shared = (
            values(train)
            & values(validation)
        )

        if shared:
            raise ValueError(
                "Leakage "
                "train/validation: "
                f"{field} compartido"
            )


# ============================================================
# DSPY
# ============================================================

def _importar_dspy():
    try:
        import dspy

    except ImportError as exc:
        raise RuntimeError(
            "La optimización GEPA "
            "requiere DSPy instalado"
        ) from exc

    return dspy


def _crear_lm(
    dspy,
    config: Mapping[str, Any],
    *,
    cache=True,
):
    if not isinstance(
        config,
        Mapping,
    ):
        raise ValueError(
            "Configuración LLM inválida"
        )

    provider = str(
        config.get(
            "provider",
            "",
        )
        or ""
    ).strip()

    model = str(
        config.get(
            "modelo",
            "",
        )
        or ""
    ).strip()

    if not model:
        raise ValueError(
            "Falta llm.modelo"
        )

    if (
        "/" in model
        or not provider
    ):
        qualified_model = model

    else:
        qualified_model = (
            f"{provider}/{model}"
        )

    kwargs = {}

    temperature = (
        config.get(
            "temperature"
        )
    )

    if temperature is not None:
        kwargs[
            "temperature"
        ] = temperature

    max_tokens = (
        config.get(
            "max_tokens"
        )
    )

    if max_tokens is not None:
        kwargs[
            "max_tokens"
        ] = max_tokens

    api_base = (
        config.get(
            "api_base"
        )
    )

    if api_base:
        kwargs[
            "api_base"
        ] = api_base

    api_key_env = (
        config.get(
            "api_key_env"
        )
    )

    if api_key_env:
        api_key = os.getenv(
            str(
                api_key_env
            )
        )

        if not api_key:
            raise ValueError(
                "Falta variable de entorno "
                f"{api_key_env}"
            )

        kwargs[
            "api_key"
        ] = api_key

    return dspy.LM(
        qualified_model,
        cache=cache,
        **kwargs,
    )


# ============================================================
# PROGRAMA DSPY
# ============================================================

def _crear_programa_schema(
    dspy,
    schema_inicial,
    task_lm,
):
    schema_inicial = validar_schema(
        schema_inicial
    )

    class GenerarSchemaGLiNER(
        dspy.Signature
    ):
        """
        Diseña descripciones globales de entidades
        para GLiNER.

        No extrae información de documentos.
        """

        tarea: str = (
            dspy.InputField(
                desc=(
                    "Descripción fija "
                    "del objetivo global"
                )
            )
        )

        schema_inicial: str = (
            dspy.InputField(
                desc=(
                    "Schema inicial "
                    "como referencia"
                )
            )
        )

        descripcion_nombre_embargado: str = (
            dspy.OutputField(
                desc=(
                    "Descripción global "
                    "para nombre_embargado"
                )
            )
        )

        descripcion_dni: str = (
            dspy.OutputField(
                desc=(
                    "Descripción global "
                    "para dni"
                )
            )
        )

        descripcion_cuit_cuil: str = (
            dspy.OutputField(
                desc=(
                    "Descripción global "
                    "para cuit_cuil"
                )
            )
        )

    class ProgramaSchemaGLiNER(
        dspy.Module
    ):
        def __init__(self):
            super().__init__()

            self.generador_schema = (
                dspy.Predict(
                    GenerarSchemaGLiNER
                )
            )

        def forward(
            self,
            tarea: str,
            schema_inicial: str,
        ):
            result = (
                self.generador_schema(
                    tarea=tarea,
                    schema_inicial=schema_inicial,
                )
            )

            return dspy.Prediction(
                descripcion_nombre_embargado=(
                    result
                    .descripcion_nombre_embargado
                ),

                descripcion_dni=(
                    result
                    .descripcion_dni
                ),

                descripcion_cuit_cuil=(
                    result
                    .descripcion_cuit_cuil
                ),
            )

    program = (
        ProgramaSchemaGLiNER()
    )

    program.set_lm(
        task_lm
    )

    return program


def _schema_desde_pred(
    pred,
):
    schema = {
        "nombre_embargado":
            str(
                getattr(
                    pred,
                    "descripcion_nombre_embargado",
                    "",
                )
                or ""
            ).strip(),

        "dni":
            str(
                getattr(
                    pred,
                    "descripcion_dni",
                    "",
                )
                or ""
            ).strip(),

        "cuit_cuil":
            str(
                getattr(
                    pred,
                    "descripcion_cuit_cuil",
                    "",
                )
                or ""
            ).strip(),
    }

    return validar_schema(
        schema
    )


# ============================================================
# EJEMPLOS DSPY
# ============================================================

def _a_dspy_examples(
    dspy,
    examples,
    schema_inicial,
):
    schema_json = json.dumps(
        validar_schema(
            schema_inicial
        ),
        ensure_ascii=False,
    )

    converted = []

    for example in examples:
        expected = (
            example[
                "expected"
            ]
        )

        converted.append(
            dspy.Example(
                tarea=
                    TASK_TEXT,

                schema_inicial=
                    schema_json,

                text=
                    example.get(
                        "text",
                        "",
                    ),

                fragments=
                    example.get(
                        "fragments",
                        [],
                    ),

                nombre_embargado=
                    str(
                        expected.get(
                            "nombre_embargado",
                            "",
                        )
                        or ""
                    ),

                dni=
                    str(
                        expected.get(
                            "dni",
                            "",
                        )
                        or ""
                    ),

                cuit_cuil=
                    str(
                        expected.get(
                            "cuit_cuil",
                            "",
                        )
                        or ""
                    ),

                example_id=
                    str(
                        example[
                            "id"
                        ]
                    ),
            ).with_inputs(
                "tarea",
                "schema_inicial",
            )
        )

    return converted


# ============================================================
# FEEDBACK GEPA
# ============================================================

def _feedback_gepa(
    *,
    schema,
    scores,
    expected,
    prediction,
):
    payload = {
        "score_global":
            round(
                float(
                    scores[
                        "global"
                    ]
                ),
                6,
            ),

        "score_nombre":
            round(
                float(
                    scores[
                        "nombre_embargado"
                    ]
                ),
                6,
            ),

        "score_dni":
            round(
                float(
                    scores[
                        "dni"
                    ]
                ),
                6,
            ),

        "score_cuit_cuil":
            round(
                float(
                    scores[
                        "cuit_cuil"
                    ]
                ),
                6,
            ),

        "schema_evaluado":
            schema,

        "esperado": {
            key:
                expected.get(
                    key,
                    "",
                )
            for key in KEYS
        },

        "predicho": {
            key:
                prediction.get(
                    key,
                    "",
                )
            for key in KEYS
        },

        "errores":
            scores.get(
                "errores",
                {},
            ),
    }

    text = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
    )

    if len(text) > 5000:
        text = (
            text[:5000]
            + "…"
        )

    return text


# ============================================================
# MÉTRICA GEPA
# ============================================================

def crear_metrica_gepa(
    *,
    metric,
    extractor,
    threshold,
    modo_entrada,
):
    dspy = _importar_dspy()

    threshold = validar_threshold(
        threshold
    )

    modo_entrada = validar_modo_entrada(
        modo_entrada
    )

    def gepa_metric(
        gold,
        pred,
        trace=None,
        pred_name=None,
        pred_trace=None,
        program_trace=None,
    ):
        try:
            schema = (
                _schema_desde_pred(
                    pred
                )
            )

        except Exception as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "El programa generó "
                    "un schema inválido. "
                    f"Error: {exc}"
                ),
            )

        example = {
            "id":
                getattr(
                    gold,
                    "example_id",
                    "",
                ),

            "text":
                getattr(
                    gold,
                    "text",
                    "",
                ),

            "fragments":
                list(
                    getattr(
                        gold,
                        "fragments",
                        [],
                    )
                    or []
                ),
        }

        try:
            prediction = (
                predecir_ejemplo(
                    extractor,
                    example,
                    schema,
                    threshold,
                    modo_entrada,
                )
            )

        except Exception as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "GLiNER no pudo evaluar "
                    "el schema candidato. "
                    f"Error: {exc}"
                ),
            )

        expected = {
            key:
                str(
                    getattr(
                        gold,
                        key,
                        "",
                    )
                    or ""
                )
            for key in KEYS
        }

        scores = metric(
            expected,
            prediction,
        )

        score = scores.get(
            "global"
        )

        if (
            not isinstance(
                score,
                (int, float),
            )
            or not math.isfinite(
                score
            )
            or not 0 <= score <= 1
        ):
            raise ValueError(
                "Métrica global inválida "
                f"para GEPA: {score!r}"
            )

        return dspy.Prediction(
            score=float(
                score
            ),

            feedback=(
                _feedback_gepa(
                    schema=schema,
                    scores=scores,
                    expected=expected,
                    prediction=prediction,
                )
            ),
        )

    return gepa_metric


# ============================================================
# PRESUPUESTO GEPA
# ============================================================

def _gepa_budget_kwargs(
    config,
):
    auto = config.get(
        "auto"
    )

    max_full_evals = config.get(
        "max_full_evals"
    )

    max_metric_calls = config.get(
        "max_metric_calls"
    )

    supplied = sum(
        value is not None
        for value in (
            auto,
            max_full_evals,
            max_metric_calls,
        )
    )

    if supplied != 1:
        raise ValueError(
            "GEPA requiere exactamente uno de: "
            "auto, max_full_evals o max_metric_calls"
        )

    if auto is not None:
        return {
            "auto":
                auto
        }

    if max_full_evals is not None:
        return {
            "max_full_evals":
                max_full_evals
        }

    return {
        "max_metric_calls":
            max_metric_calls
    }


# ============================================================
# RESUMEN GEPA
# ============================================================

def _resumen_gepa(
    program,
):
    details = getattr(
        program,
        "detailed_results",
        None,
    )

    if details is None:
        return {}

    candidates = getattr(
        details,
        "candidates",
        None,
    )

    return {
        "cantidad_candidatos":
            (
                len(candidates)
                if candidates
                is not None
                else None
            ),

        "best_idx":
            getattr(
                details,
                "best_idx",
                None,
            ),

        "val_aggregate_scores":
            getattr(
                details,
                "val_aggregate_scores",
                None,
            ),

        "total_metric_calls":
            getattr(
                details,
                "total_metric_calls",
                None,
            ),

        "num_full_val_evals":
            getattr(
                details,
                "num_full_val_evals",
                None,
            ),

        "seed":
            getattr(
                details,
                "seed",
                None,
            ),

        "log_dir":
            str(
                getattr(
                    details,
                    "log_dir",
                    "",
                )
                or ""
            ),
    }


# ============================================================
# SCHEMA FINAL
# ============================================================

def _generar_schema_programa(
    program,
    schema_inicial,
):
    schema_json = json.dumps(
        validar_schema(
            schema_inicial
        ),
        ensure_ascii=False,
    )

    pred = program(
        tarea=TASK_TEXT,
        schema_inicial=schema_json,
    )

    return _schema_desde_pred(
        pred
    )


# ============================================================
# OPTIMIZACIÓN GEPA
# ============================================================

def optimizar_schema(
    extractor,
    train,
    validation,
    schema,
    threshold,
    metric,
    *,
    modo_entrada,
    llm_config,
    gepa_config,
    reflection_lm_config=None,
    history_path=None,
):
    _separacion(
        train,
        validation,
    )

    schema = validar_schema(
        schema
    )

    threshold = validar_threshold(
        threshold
    )

    modo_entrada = validar_modo_entrada(
        modo_entrada
    )

    baseline_train = (
        evaluar_schema(
            extractor,
            train,
            schema,
            threshold,
            metric,
            modo_entrada,
        )
    )

    baseline_validation = (
        evaluar_schema(
            extractor,
            validation,
            schema,
            threshold,
            metric,
            modo_entrada,
        )
    )

    dspy = _importar_dspy()

    task_lm = _crear_lm(
        dspy,
        llm_config,
        cache=True,
    )

    reflection_lm = (
        _crear_lm(
            dspy,
            (
                reflection_lm_config
                or llm_config
            ),
            cache=True,
        )
    )

    student = (
        _crear_programa_schema(
            dspy,
            schema,
            task_lm,
        )
    )

    gepa_metric = (
        crear_metrica_gepa(
            metric=metric,
            extractor=extractor,
            threshold=threshold,
            modo_entrada=modo_entrada,
        )
    )

    optimizer_kwargs = {
        "metric":
            gepa_metric,

        "reflection_lm":
            reflection_lm,

        "num_threads":
            int(
                gepa_config.get(
                    "num_threads",
                    1,
                )
            ),

        "track_stats":
            True,

        "seed":
            int(
                gepa_config.get(
                    "seed",
                    0,
                )
            ),

        "candidate_selection_strategy":
            gepa_config.get(
                "candidate_selection_strategy",
                "pareto",
            ),

        "reflection_minibatch_size":
            int(
                gepa_config.get(
                    "reflection_minibatch_size",
                    3,
                )
            ),

        "skip_perfect_score":
            bool(
                gepa_config.get(
                    "skip_perfect_score",
                    True,
                )
            ),

        "add_format_failure_as_feedback":
            True,

        **_gepa_budget_kwargs(
            gepa_config
        ),
    }

    log_dir = (
        gepa_config.get(
            "log_dir"
        )
    )

    if log_dir:
        optimizer_kwargs[
            "log_dir"
        ] = str(
            log_dir
        )

    optimizer = dspy.GEPA(
        **optimizer_kwargs
    )

    trainset = (
        _a_dspy_examples(
            dspy,
            train,
            schema,
        )
    )

    valset = (
        _a_dspy_examples(
            dspy,
            validation,
            schema,
        )
    )

    optimized_program = (
        optimizer.compile(
            student,
            trainset=trainset,
            valset=valset,
        )
    )

    optimized_program.set_lm(
        task_lm
    )

    optimized_schema = (
        _generar_schema_programa(
            optimized_program,
            schema,
        )
    )

    optimized_train = (
        evaluar_schema(
            extractor,
            train,
            optimized_schema,
            threshold,
            metric,
            modo_entrada,
        )
    )

    optimized_validation = (
        evaluar_schema(
            extractor,
            validation,
            optimized_schema,
            threshold,
            metric,
            modo_entrada,
        )
    )

    accepted = (
        optimized_validation[
            "metricas"
        ][
            "global"
        ]
        >
        baseline_validation[
            "metricas"
        ][
            "global"
        ]
    )

    if accepted:
        selected_schema = (
            optimized_schema
        )

        selected_validation = (
            optimized_validation
        )

    else:
        selected_schema = schema

        selected_validation = (
            baseline_validation
        )

    gepa_summary = (
        _resumen_gepa(
            optimized_program
        )
    )

    history = {
        "metodo":
            "GEPA",

        "arquitectura":
            "schema_global",

        "modo_entrada":
            modo_entrada,

        "baseline": {
            "schema":
                schema,

            "train":
                baseline_train[
                    "metricas"
                ],

            "validation":
                baseline_validation[
                    "metricas"
                ],
        },

        "candidato_gepa": {
            "schema":
                optimized_schema,

            "train":
                optimized_train[
                    "metricas"
                ],

            "validation":
                optimized_validation[
                    "metricas"
                ],

            "accepted":
                accepted,
        },

        "gepa":
            gepa_summary,
    }

    if history_path:
        guardar_json(
            history_path,
            history,
        )

    return {
        "schema":
            selected_schema,

        "baseline":
            baseline_validation[
                "metricas"
            ],

        "metricas_validation":
            selected_validation[
                "metricas"
            ],

        "threshold":
            threshold,

        "accepted":
            accepted,

        "gepa":
            gepa_summary,

        "historial":
            history,

        "programa_optimizado":
            optimized_program,
    }


# ============================================================
# THRESHOLD
# ============================================================

def optimizar_threshold(
    extractor,
    validation,
    schema,
    initial,
    thresholds,
    metric,
    modo_entrada="texto_completo",
):
    modo_entrada = validar_modo_entrada(
        modo_entrada
    )

    best_threshold = (
        validar_threshold(
            initial
        )
    )

    best = (
        evaluar_schema(
            extractor,
            validation,
            schema,
            initial,
            metric,
            modo_entrada,
        )[
            "metricas"
        ]
    )

    trials = [
        {
            "threshold":
                initial,

            "metricas_validation":
                best,
        }
    ]

    for threshold in dict.fromkeys(
        thresholds
    ):
        threshold = (
            validar_threshold(
                threshold
            )
        )

        if threshold == initial:
            continue

        scores = (
            evaluar_schema(
                extractor,
                validation,
                schema,
                threshold,
                metric,
                modo_entrada,
            )[
                "metricas"
            ]
        )

        trials.append(
            {
                "threshold":
                    threshold,

                "metricas_validation":
                    scores,
            }
        )

        if (
            scores[
                "global"
            ]
            >
            best[
                "global"
            ]
        ):
            best_threshold = (
                threshold
            )

            best = scores

    return {
        "threshold":
            best_threshold,

        "metricas_validation":
            best,

        "historial":
            trials,
    }


# ============================================================
# CSV
# ============================================================

def resultados_csv(
    evaluation,
):
    return [
        {
            "id":
                row["id"],

            **{
                f"esperado_{key}":
                    row[
                        "expected"
                    ][
                        key
                    ]
                for key in KEYS
            },

            **{
                key:
                    row[
                        "prediction"
                    ].get(
                        key,
                        "",
                    )
                for key in KEYS
            },

            **{
                f"score_{key}":
                    row[
                        "scores"
                    ][
                        key
                    ]
                for key in (
                    *KEYS,
                    "global",
                )
            },

            "latencia_segundos":
                row[
                    "latencia_segundos"
                ],

            "spans_raw":
                row[
                    "prediction"
                ].get(
                    "spans_raw",
                    [],
                ),

            "entidades":
                row[
                    "prediction"
                ].get(
                    "entidades",
                    [],
                ),
        }

        for row
        in evaluation[
            "resultados"
        ]
    ]