"""Evaluación de un schema congelado sobre validation o test.

Respeta el mismo modo de entrada utilizado durante la optimización:

- texto_completo
- fragmentos

No utiliza DSPy, GEPA ni LLM.
"""

from __future__ import annotations

import argparse

from src.dataset import (
    cargar_splits,
    manifiesto,
)

from src.metricas import evaluar

from src.schema_optimizer import (
    evaluar_schema,
    resultados_csv,
)

from src.utils import (
    cargar_config,
    cargar_seleccion,
    crear_extractor,
    guardar_csv,
    guardar_json,
    leer_json,
    ruta,
)


# ============================================================
# PREPARACIÓN DE EVALUACIÓN
# ============================================================

def preparar_evaluacion(
    config,
    schema_path,
    threshold=None,
):
    """
    Prepara una evaluación reproducible.

    Comprueba:

    - schema + threshold seleccionados;
    - mismos splits que durante optimización;
    - mismo modo de entrada.
    """

    schema, threshold = (
        cargar_seleccion(
            config,
            schema_path,
            threshold,
        )
    )

    splits = cargar_splits(
        config
    )

    current_manifest = manifiesto(
        splits
    )

    modo_entrada = (
        config[
            "datos"
        ][
            "modo_entrada"
        ]
    )

    selection_path = (
        ruta(
            config[
                "rutas"
            ][
                "output"
            ]
        )
        / "seleccion.json"
    )

    # --------------------------------------------------------
    # Si existe una selección optimizada,
    # comprobar reproducibilidad.
    # --------------------------------------------------------

    if selection_path.exists():

        selection = leer_json(
            selection_path
        )

        saved_manifest = (
            selection.get(
                "splits"
            )
        )

        if (
            saved_manifest
            is not None
            and saved_manifest
            != current_manifest
        ):
            raise ValueError(
                "Dataset/splits cambiaron "
                "desde la optimización; "
                "la evaluación ya no sería comparable"
            )

        saved_mode = (
            selection.get(
                "modo_entrada"
            )
        )

        if (
            saved_mode is not None
            and saved_mode
            != modo_entrada
        ):
            raise ValueError(
                "El modo de entrada cambió "
                "desde la optimización: "
                f"optimizado={saved_mode}, "
                f"actual={modo_entrada}"
            )

    return (
        schema,
        threshold,
        splits,
        modo_entrada,
    )


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
        help=(
            "Ruta al archivo YAML "
            "de configuración."
        ),
    )

    parser.add_argument(
        "--schema",
        default=None,
        help=(
            "Schema JSON a evaluar. "
            "Por defecto se usa "
            "schema_optimizado.json."
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
        help=(
            "Split sobre el que se "
            "realizará la evaluación."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Threshold explícito. "
            "Si no se indica, se utiliza "
            "el seleccionado previamente."
        ),
    )

    parser.add_argument(
        "--output",
        help=(
            "Ruta opcional para el CSV "
            "de resultados."
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

    schema_path = ruta(
        args.schema
        or config[
            "rutas"
        ][
            "schema_optimizado"
        ]
    )

    # ========================================================
    # DATASET + SELECCIÓN
    # ========================================================

    (
        schema,
        threshold,
        splits,
        modo_entrada,
    ) = preparar_evaluacion(
        config,
        schema_path,
        args.threshold,
    )

    examples = (
        splits[
            args.split
        ]
    )

    # ========================================================
    # GLINER
    # ========================================================

    extractor = crear_extractor(
        config,
        schema,
    )

    # ========================================================
    # EVALUACIÓN
    # ========================================================

    result = evaluar_schema(
        extractor=
            extractor,

        examples=
            examples,

        schema=
            schema,

        threshold=
            threshold,

        metric=
            evaluar,

        modo_entrada=
            modo_entrada,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    if args.output:

        output = ruta(
            args.output
        )

    else:

        output = (
            ruta(
                config[
                    "rutas"
                ][
                    "output"
                ]
            )
            / (
                f"evaluacion_"
                f"{schema_path.stem}_"
                f"{args.split}.csv"
            )
        )

    guardar_csv(
        output,
        resultados_csv(
            result
        ),
    )

    guardar_json(
        output.with_suffix(
            ".json"
        ),
        {
            "split":
                args.split,

            "modo_entrada":
                modo_entrada,

            "schema":
                schema,

            "threshold":
                threshold,

            "metricas":
                result[
                    "metricas"
                ],

            "splits":
                manifiesto(
                    splits
                ),
        },
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "Split:",
        args.split,
    )

    print(
        "Modo de entrada:",
        modo_entrada,
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "Métricas:",
        result[
            "metricas"
        ],
    )

    print(
        "CSV:",
        output,
    )


if __name__ == "__main__":
    main()