"""Inferencia final con GLiNER2 sobre documentos sin ground truth.

Este script NO utiliza:

- DSPy
- GEPA
- LLM
- gold supervisado

Utiliza únicamente:

- GLiNER2
- schema seleccionado
- threshold seleccionado

El texto de entrada depende de:

datos.modo_entrada

Valores soportados:

- texto_completo
- fragmentos
"""

from __future__ import annotations

import argparse

from src.utils import (
    cargar_config,
    cargar_seleccion,
    crear_extractor,
    guardar_csv,
    leer_csv,
    ruta,
)


# ============================================================
# RESOLVER COLUMNA DE TEXTO
# ============================================================

def resolver_columna_texto(
    config,
    columna_cli=None,
):
    """
    Determina qué columna del CSV debe enviarse a GLiNER2.

    Prioridad:

    1. --text-column si fue indicada manualmente.
    2. configuración YAML según datos.modo_entrada.
    """

    if columna_cli:
        return columna_cli

    datos = config[
        "datos"
    ]

    modo = datos[
        "modo_entrada"
    ]

    columnas = (
        datos[
            "extraccion"
        ][
            "columnas"
        ]
    )

    if modo == "texto_completo":
        return columnas[
            "texto_completo"
        ]

    if modo == "fragmentos":
        return columnas[
            "fragmento"
        ]

    raise ValueError(
        "Modo de entrada desconocido: "
        f"{modo}"
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
            "Ruta al YAML de configuración."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV con documentos o fragmentos "
            "sobre los que ejecutar GLiNER2."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "CSV donde se guardarán "
            "las predicciones."
        ),
    )

    parser.add_argument(
        "--schema",
        help=(
            "Schema JSON a utilizar. "
            "Si no se indica se utiliza "
            "schema_optimizado.json."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Threshold global explícito. "
            "Si no se indica se utiliza "
            "el seleccionado durante "
            "la optimización."
        ),
    )

    parser.add_argument(
        "--text-column",
        help=(
            "Nombre explícito de la columna "
            "que contiene el texto. "
            "Si no se indica se determina "
            "a partir de datos.modo_entrada."
        ),
    )

    args = parser.parse_args(
        argv
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = cargar_config(
        args.config
    )

    modo_entrada = (
        config[
            "datos"
        ][
            "modo_entrada"
        ]
    )

    # ========================================================
    # SCHEMA
    # ========================================================

    schema_path = ruta(
        args.schema
        or config[
            "rutas"
        ][
            "schema_optimizado"
        ]
    )

    schema, threshold = (
        cargar_seleccion(
            config,
            schema_path,
            args.threshold,
        )
    )

    # ========================================================
    # INPUT
    # ========================================================

    input_path = ruta(
        args.input
    )

    rows = leer_csv(
        input_path
    )

    if not rows:
        raise ValueError(
            "El CSV de entrada está vacío"
        )

    text_column = (
        resolver_columna_texto(
            config,
            args.text_column,
        )
    )

    # ========================================================
    # VALIDACIÓN DEL CSV
    # ========================================================

    for index, row in enumerate(
        rows
    ):

        if text_column not in row:
            raise ValueError(
                f"Falta la columna "
                f"'{text_column}' "
                f"en la fila {index}"
            )

        text = row.get(
            text_column
        )

        if (
            text is None
            or not str(
                text
            ).strip()
        ):
            raise ValueError(
                f"Texto vacío en fila "
                f"{index} para columna "
                f"'{text_column}'"
            )

    # ========================================================
    # GLINER2
    # ========================================================

    extractor = crear_extractor(
        config,
        schema,
    )

    # ========================================================
    # INFERENCIA
    # ========================================================

    output_rows = []

    for row in rows:

        text = str(
            row[
                text_column
            ]
        ).strip()

        prediction = (
            extractor.predict(
                text,
                schema,
                threshold,
            )
        )

        output_rows.append(
            {
                **row,

                "pred_nombre_embargado":
                    prediction.get(
                        "nombre_embargado",
                        "",
                    ),

                "pred_dni":
                    prediction.get(
                        "dni",
                        "",
                    ),

                "pred_cuit_cuil":
                    prediction.get(
                        "cuit_cuil",
                        "",
                    ),

                "pred_entidades":
                    prediction.get(
                        "entidades",
                        [],
                    ),

                "pred_spans_raw":
                    prediction.get(
                        "spans_raw",
                        [],
                    ),
            }
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_path = ruta(
        args.output
    )

    guardar_csv(
        output_path,
        output_rows,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "Modo de entrada:",
        modo_entrada,
    )

    print(
        "Columna de texto:",
        text_column,
    )

    print(
        "Documentos/registros procesados:",
        len(
            output_rows
        ),
    )

    print(
        "Schema:",
        schema_path,
    )

    print(
        "Threshold global:",
        threshold,
    )

    print(
        "CSV:",
        output_path,
    )


if __name__ == "__main__":
    main()