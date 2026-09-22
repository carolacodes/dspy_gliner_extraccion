"""Benchmark de GLiNER baseline, GLiNER optimizado y sistemas externos.

Compara todos los sistemas contra el mismo ground truth.

Respeta el modo de entrada utilizado en el experimento:

- texto_completo
- fragmentos
"""

from __future__ import annotations

import argparse
import math

from evaluar import preparar_evaluacion

from src.metricas import evaluar

from src.schema_optimizer import (
    evaluar_schema,
)

from src.utils import (
    KEYS,
    cargar_config,
    cargar_schema,
    crear_extractor,
    guardar_csv,
    leer_csv,
    ruta,
)


# ============================================================
# EVALUACIÓN DE SISTEMA EXTERNO
# ============================================================

def evaluar_externo(
    examples,
    rows,
    metric=evaluar,
):
    """
    Evalúa predicciones externas contra el mismo gold.

    El CSV externo debe contener exactamente
    los mismos IDs del split.
    """

    if not examples:
        raise ValueError(
            "Split vacío"
        )

    ids = [
        row.get(
            "id"
        )
        for row in rows
    ]

    expected_ids = {
        example[
            "id"
        ]
        for example in examples
    }

    received_ids = set(
        ids
    )

    if (
        len(received_ids)
        != len(ids)
    ):
        raise ValueError(
            "CSV externo contiene IDs duplicados"
        )

    if (
        received_ids
        != expected_ids
    ):
        raise ValueError(
            "CSV externo debe contener "
            "exactamente los mismos IDs "
            "del split"
        )

    mapped = {
        row[
            "id"
        ]:
            row
        for row in rows
    }

    scores = []
    latencies = []

    for example in examples:

        row = mapped[
            example[
                "id"
            ]
        ]

        # ----------------------------------------------------
        # Validar campos canónicos.
        # ----------------------------------------------------

        for key in KEYS:

            if (
                key not in row
                or row[key] is None
            ):
                raise ValueError(
                    "Faltan columnas canónicas "
                    "de predicción externa"
                )

        # ----------------------------------------------------
        # Métrica.
        # ----------------------------------------------------

        scores.append(
            metric(
                example[
                    "expected"
                ],
                row,
            )
        )

        # ----------------------------------------------------
        # Latencia opcional.
        # ----------------------------------------------------

        value = row.get(
            "latencia_segundos"
        )

        if value not in (
            None,
            "",
        ):

            latency = float(
                value
            )

            if (
                not math.isfinite(
                    latency
                )
                or latency < 0
            ):
                raise ValueError(
                    "Latencia inválida"
                )

            latencies.append(
                latency
            )

    # ========================================================
    # PROMEDIOS
    # ========================================================

    result = {
        key:
            (
                sum(
                    score[
                        key
                    ]
                    for score
                    in scores
                )
                / len(
                    scores
                )
            )
        for key in (
            *KEYS,
            "global",
        )
    }

    # Sólo reportar promedio si todos
    # los ejemplos tienen latencia.
    if (
        len(latencies)
        == len(examples)
    ):
        result[
            "latencia_promedio"
        ] = (
            sum(
                latencies
            )
            / len(
                latencies
            )
        )

    else:
        result[
            "latencia_promedio"
        ] = None

    return result


# ============================================================
# MAIN
# ============================================================

def main(argv=None):

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--config",
        default=
            "config/gliner_embargo.yaml",
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--external-csv",
        help=(
            "CSV opcional con predicciones "
            "de otro sistema."
        ),
    )

    parser.add_argument(
        "--external-name",
        default="externo",
        help=(
            "Nombre que aparecerá "
            "en el benchmark."
        ),
    )

    args = parser.parse_args(
        argv
    )

    # ========================================================
    # CONFIGURACIÓN
    # ========================================================

    config = cargar_config(
        args.config
    )

    optimized_schema_path = ruta(
        config[
            "rutas"
        ][
            "schema_optimizado"
        ]
    )

    # ========================================================
    # DATASET + SELECCIÓN
    # ========================================================

    (
        optimized_schema,
        optimized_threshold,
        splits,
        modo_entrada,
    ) = preparar_evaluacion(
        config,
        optimized_schema_path,
    )

    examples = (
        splits[
            args.split
        ]
    )

    if not examples:
        raise ValueError(
            "El split está vacío"
        )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_schema = cargar_schema(
        ruta(
            config[
                "rutas"
            ][
                "schema_inicial"
            ]
        )
    )

    baseline_threshold = (
        config[
            "gliner"
        ][
            "threshold_inicial"
        ]
    )

    # ========================================================
    # EXTRACTOR
    # ========================================================

    # Se reutiliza una única instancia de GLiNER.
    extractor = crear_extractor(
        config,
        baseline_schema,
    )

    results = []

    # ========================================================
    # GLINER BASELINE / OPTIMIZADO
    # ========================================================

    systems = (
        (
            "gliner_baseline",
            baseline_schema,
            baseline_threshold,
        ),
        (
            "gliner_optimizado",
            optimized_schema,
            optimized_threshold,
        ),
    )

    for (
        name,
        selected_schema,
        threshold,
    ) in systems:

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------
        #
        # Sólo puede hacerse directamente con texto completo.
        #
        # En fragmentos la evaluación real ya pasa por
        # predecir_ejemplo(), por lo que evitamos inventar
        # una llamada distinta al pipeline real.
        # ----------------------------------------------------

        if (
            modo_entrada
            == "texto_completo"
        ):
            extractor.predict(
                examples[
                    0
                ][
                    "text"
                ],
                selected_schema,
                threshold,
            )

        elif (
            modo_entrada
            == "fragmentos"
            and examples[
                0
            ].get(
                "fragments"
            )
        ):
            extractor.predict(
                examples[
                    0
                ][
                    "fragments"
                ][
                    0
                ],
                selected_schema,
                threshold,
            )

        # ----------------------------------------------------
        # Evaluación real
        # ----------------------------------------------------

        evaluation = evaluar_schema(
            extractor=
                extractor,

            examples=
                examples,

            schema=
                selected_schema,

            threshold=
                threshold,

            metric=
                evaluar,

            modo_entrada=
                modo_entrada,
        )

        results.append(
            {
                "sistema":
                    name,

                "modo_entrada":
                    modo_entrada,

                **evaluation[
                    "metricas"
                ],
            }
        )

    # ========================================================
    # SISTEMA EXTERNO
    # ========================================================

    if args.external_csv:

        external_rows = leer_csv(
            ruta(
                args.external_csv
            )
        )

        external_result = (
            evaluar_externo(
                examples,
                external_rows,
            )
        )

        results.append(
            {
                "sistema":
                    args.external_name,

                "modo_entrada":
                    "externo",

                **external_result,
            }
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = (
        ruta(
            config[
                "rutas"
            ][
                "output"
            ]
        )
        / (
            f"benchmark_"
            f"{args.split}.csv"
        )
    )

    guardar_csv(
        output,
        results,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "Split:",
        args.split,
    )

    print(
        "Modo GLiNER:",
        modo_entrada,
    )

    print()

    for row in results:
        print(
            row
        )

    print()

    print(
        "CSV:",
        output,
    )


if __name__ == "__main__":
    main()