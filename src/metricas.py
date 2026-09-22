"""Métricas por persona con matching bipartito simple.

Evalúa asociaciones completas:
nombre_embargado + dni + cuit_cuil

Cada persona recibe un score compuesto:
- nombre: 50%
- DNI: 25%
- CUIT/CUIL: 25%

Luego se busca la mejor asociación entre personas esperadas y predichas.
"""

from difflib import SequenceMatcher

from .utils import KEYS
from .normalizacion import (
    separar_valores,
    normalizar_nombre,
    normalizar_dni,
    normalizar_cuit_cuil,
)


PESO_NOMBRE = 0.50
PESO_DNI = 0.25
PESO_CUIT = 0.25


def _valor_en_posicion(values, index):
    """Devuelve el valor de una posición o cadena vacía si no existe."""
    return values[index] if index < len(values) else ""


def _construir_personas(data):
    """
    Convierte campos pipe-separated en personas por posición ordinal.

    Ejemplo:
        nombre_embargado = "Juan | María"
        dni = "123 | 456"

    devuelve:
        [
            {"nombre_embargado": "Juan", "dni": "123", ...},
            {"nombre_embargado": "María", "dni": "456", ...},
        ]
    """
    def separar_preservando_posiciones(value):
        if value is None or str(value).strip() == "":
            return []

        return [parte.strip() for parte in str(value).split("|")]


    nombres = separar_preservando_posiciones(
        data.get("nombre_embargado", "")
    )
    dnis = separar_preservando_posiciones(
        data.get("dni", "")
    )
    cuits = separar_preservando_posiciones(
        data.get("cuit_cuil", "")
    )

    cantidad = max(len(nombres), len(dnis), len(cuits), 0)

    personas = []

    for i in range(cantidad):
        persona = {
            "nombre_embargado": _valor_en_posicion(nombres, i),
            "dni": _valor_en_posicion(dnis, i),
            "cuit_cuil": _valor_en_posicion(cuits, i),
        }

        # Ignorar filas completamente vacías.
        if any(persona.values()):
            personas.append(persona)

    return personas


def _similitud_nombre(expected, predicted):
    expected = normalizar_nombre(expected)
    predicted = normalizar_nombre(predicted)

    if not expected and not predicted:
        return 1.0

    if not expected or not predicted:
        return 0.0

    return SequenceMatcher(None, expected, predicted).ratio()


def _similitud_exacta(expected, predicted, normalizer):
    expected = normalizer(expected)
    predicted = normalizer(predicted)

    if not expected and not predicted:
        return 1.0

    if not expected or not predicted:
        return 0.0

    return 1.0 if expected == predicted else 0.0


def _score_persona(expected, predicted):
    """
    Score total de una persona.

    Nombre admite similitud fuzzy.
    DNI y CUIT/CUIL requieren coincidencia exacta normalizada.
    """
    score_nombre = _similitud_nombre(
        expected.get("nombre_embargado", ""),
        predicted.get("nombre_embargado", ""),
    )

    score_dni = _similitud_exacta(
        expected.get("dni", ""),
        predicted.get("dni", ""),
        normalizar_dni,
    )

    score_cuit = _similitud_exacta(
        expected.get("cuit_cuil", ""),
        predicted.get("cuit_cuil", ""),
        normalizar_cuit_cuil,
    )

    global_score = (
        PESO_NOMBRE * score_nombre
        + PESO_DNI * score_dni
        + PESO_CUIT * score_cuit
    )

    return {
        "nombre_embargado": score_nombre,
        "dni": score_dni,
        "cuit_cuil": score_cuit,
        "global": global_score,
    }


def _mejor_matching(expected_personas, predicted_personas):
    """
    Matching bipartito greedy sobre score global.

    Genera todos los pares posibles y asigna primero los pares
    con mayor score, evitando reutilizar personas.

    Para el tamaño esperado del problema es suficiente y mantiene
    la implementación simple.
    """
    candidates = []

    for expected_idx, expected in enumerate(expected_personas):
        for predicted_idx, predicted in enumerate(predicted_personas):
            scores = _score_persona(expected, predicted)

            candidates.append(
                {
                    "expected_idx": expected_idx,
                    "predicted_idx": predicted_idx,
                    "scores": scores,
                }
            )

    candidates.sort(
        key=lambda item: item["scores"]["global"],
        reverse=True,
    )

    used_expected = set()
    used_predicted = set()
    matches = []

    for candidate in candidates:
        expected_idx = candidate["expected_idx"]
        predicted_idx = candidate["predicted_idx"]

        if expected_idx in used_expected:
            continue

        if predicted_idx in used_predicted:
            continue

        used_expected.add(expected_idx)
        used_predicted.add(predicted_idx)
        matches.append(candidate)

    return matches, used_expected, used_predicted


def _persona_resumida(persona):
    return {
        "nombre_embargado": persona.get("nombre_embargado", ""),
        "dni": persona.get("dni", ""),
        "cuit_cuil": persona.get("cuit_cuil", ""),
    }


def evaluar(expected, predicted):
    """
    Evalúa ground truth contra predicción.

    Retorna:
    - score nombre_embargado
    - score dni
    - score cuit_cuil
    - score global
    - errores
    - matching

    Todos los scores están entre 0 y 1.
    """
    expected_personas = _construir_personas(expected)
    predicted_personas = _construir_personas(predicted)

    # Documento sin entidades en gold ni predicción.
    if not expected_personas and not predicted_personas:
        return {
            "nombre_embargado": 1.0,
            "dni": 1.0,
            "cuit_cuil": 1.0,
            "global": 1.0,
            "errores": {
                "omitidos": [],
                "incorrectos": [],
                "asociaciones": [],
            },
            "matching": [],
        }

    matches, used_expected, used_predicted = _mejor_matching(
        expected_personas,
        predicted_personas,
    )

    # Penalización por personas faltantes o extras.
    denominator = max(len(expected_personas), len(predicted_personas), 1)

    totals = {
        "nombre_embargado": 0.0,
        "dni": 0.0,
        "cuit_cuil": 0.0,
        "global": 0.0,
    }

    matching_details = []

    for match in matches:
        scores = match["scores"]

        for key in totals:
            totals[key] += scores[key]

        matching_details.append(
            {
                "expected_idx": match["expected_idx"],
                "predicted_idx": match["predicted_idx"],
                "expected": _persona_resumida(
                    expected_personas[match["expected_idx"]]
                ),
                "predicted": _persona_resumida(
                    predicted_personas[match["predicted_idx"]]
                ),
                "scores": scores,
            }
        )

    final_scores = {
        key: totals[key] / denominator
        for key in totals
    }

    omitted = [
        _persona_resumida(persona)
        for idx, persona in enumerate(expected_personas)
        if idx not in used_expected
    ]

    incorrect = [
        _persona_resumida(persona)
        for idx, persona in enumerate(predicted_personas)
        if idx not in used_predicted
    ]

    association_errors = []

    for detail in matching_details:
        if detail["scores"]["global"] < 1.0:
            association_errors.append(
                {
                    "esperado": detail["expected"],
                    "predicho": detail["predicted"],
                    "scores": detail["scores"],
                }
            )

    return {
        "nombre_embargado": final_scores["nombre_embargado"],
        "dni": final_scores["dni"],
        "cuit_cuil": final_scores["cuit_cuil"],
        "global": final_scores["global"],
        "errores": {
            "omitidos": omitted,
            "incorrectos": incorrect,
            "asociaciones": association_errors,
        },
        "matching": matching_details,
    }