"""Entrenamiento/optimización de descripciones GLiNER con DSPy + GEPA.

TRAIN:
    se utiliza durante la optimización/reflexión de GEPA.

VALIDATION:
    selecciona el programa/schema y el threshold.

TEST:
    nunca participa en este comando.

Modos de entrada:

- texto_completo
- fragmentos

La inferencia final sigue siendo únicamente GLiNER.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

from src.dataset import (
    cargar_splits,
    manifiesto,
)
from src.metricas import evaluar
from src.schema_optimizer import (
    evaluar_schema,
    optimizar_schema,
    optimizar_threshold,
    resultados_csv,
)
from src.utils import (
    cargar_config,
    cargar_schema,
    crear_extractor,
    fingerprint,
    guardar_csv,
    guardar_json,
    ruta,
    threshold_grid,
)


# ============================================================
# VERSIONES
# ============================================================

def _version_paquete(package: str):
    """
    Devuelve la versión instalada de un paquete
    sin hacer fallar la ejecución si no existe.
    """

    candidates = [
        package,
    ]

    if package == "dspy":
        candidates.append(
            "dspy-ai"
        )

    for candidate in candidates:
        try:
            return version(
                candidate
            )

        except PackageNotFoundError:
            continue

    return None


# ============================================================
# PRESUPUESTO GEPA
# ============================================================

def _configurar_presupuesto_gepa(
    config,
    args,
):
    """
    Permite sobrescribir desde CLI el presupuesto
    definido en gliner_embargo.yaml.

    GEPA debe utilizar exactamente uno de:

    - auto
    - max_full_evals
    - max_metric_calls
    """

    gepa = dict(
        config["gepa"]
    )

    overrides = sum(
        value is not None
        for value in (
            args.gepa_auto,
            args.max_full_evals,
            args.max_metric_calls,
        )
    )

    if overrides > 1:
        raise ValueError(
            "Usar sólo uno de: "
            "--gepa-auto, "
            "--max-full-evals o "
            "--max-metric-calls"
        )

    if args.gepa_auto is not None:
        gepa["auto"] = (
            args.gepa_auto
        )

        gepa["max_full_evals"] = None
        gepa["max_metric_calls"] = None

    elif args.max_full_evals is not None:
        if args.max_full_evals < 1:
            raise ValueError(
                "--max-full-evals debe ser >= 1"
            )

        gepa["auto"] = None

        gepa["max_full_evals"] = (
            args.max_full_evals
        )

        gepa["max_metric_calls"] = None

    elif args.max_metric_calls is not None:
        if args.max_metric_calls < 1:
            raise ValueError(
                "--max-metric-calls debe ser >= 1"
            )

        gepa["auto"] = None
        gepa["max_full_evals"] = None

        gepa["max_metric_calls"] = (
            args.max_metric_calls
        )

    return gepa


# ============================================================
# GUARDAR BASELINE / SMOKE
# ============================================================

def _guardar_baseline(
    *,
    out,
    prefix,
    evaluation,
    schema,
    threshold,
    limit,
    manifest,
    modo_entrada,
):
    """
    Guarda resultados de baseline o smoke test.
    """

    guardar_json(
        out
        / f"{prefix}.json",
        {
            "metricas_validation":
                evaluation[
                    "metricas"
                ],

            "schema":
                schema,

            "threshold":
                threshold,

            "modo_entrada":
                modo_entrada,

            "limit":
                limit,

            "splits":
                manifest,
        },
    )

    guardar_csv(
        out
        / f"{prefix}.csv",
        resultados_csv(
            evaluation
        ),
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
            "Ruta al YAML "
            "de configuración."
        ),
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Evalúa el schema inicial "
            "sobre validation sin usar "
            "DSPy, GEPA ni LLM."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Limita validation después "
            "del split. "
            "Sólo puede utilizarse "
            "junto con --eval-only."
        ),
    )

    parser.add_argument(
        "--gepa-auto",
        choices=(
            "light",
            "medium",
            "heavy",
        ),
        help=(
            "Sobrescribe gepa.auto "
            "para esta corrida."
        ),
    )

    parser.add_argument(
        "--max-full-evals",
        type=int,
        help=(
            "Sobrescribe el presupuesto "
            "GEPA por cantidad de "
            "evaluaciones completas."
        ),
    )

    parser.add_argument(
        "--max-metric-calls",
        type=int,
        help=(
            "Sobrescribe el presupuesto "
            "GEPA por cantidad máxima "
            "de llamadas a la métrica."
        ),
    )

    args = parser.parse_args(
        argv
    )

    # --------------------------------------------------------
    # VALIDAR CLI
    # --------------------------------------------------------

    if args.limit is not None:
        if (
            args.limit < 1
            or not args.eval_only
        ):
            parser.error(
                "--limit debe ser positivo "
                "y sólo puede utilizarse "
                "con --eval-only"
            )

    # ========================================================
    # CONFIGURACIÓN
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
    # DATASET
    # ========================================================

    splits = cargar_splits(
        config
    )

    split_manifest = manifiesto(
        splits
    )

    train = (
        splits[
            "train"
        ]
    )

    validation = (
        splits[
            "validation"
        ]
    )

    # TEST se mantiene completamente
    # fuera de este script de optimización.

    # ========================================================
    # SCHEMA INICIAL
    # ========================================================

    schema_inicial_path = ruta(
        config[
            "rutas"
        ][
            "schema_inicial"
        ]
    )

    schema_inicial = cargar_schema(
        schema_inicial_path
    )

    initial_threshold = (
        config[
            "gliner"
        ][
            "threshold_inicial"
        ]
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = ruta(
        config[
            "rutas"
        ][
            "output"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RESUMEN DATASET
    # ========================================================

    print(
        "Modo de entrada:",
        modo_entrada,
    )

    print(
        "Train:",
        len(train),
        "documentos",
    )

    print(
        "Validation:",
        len(validation),
        "documentos",
    )

    print(
        "Test:",
        len(
            splits[
                "test"
            ]
        ),
        "documentos reservados",
    )

    # ========================================================
    # CREAR GLINER
    # ========================================================

    extractor = crear_extractor(
        config,
        schema_inicial,
    )

    # ========================================================
    # BASELINE / SMOKE TEST
    # ========================================================

    if args.eval_only:
        validation_eval = (
            validation
        )

        if (
            args.limit
            is not None
        ):
            validation_eval = (
                validation[
                    :args.limit
                ]
            )

        result = evaluar_schema(
            extractor=
                extractor,

            examples=
                validation_eval,

            schema=
                schema_inicial,

            threshold=
                initial_threshold,

            metric=
                evaluar,

            modo_entrada=
                modo_entrada,
        )

        prefix = (
            "smoke"
            if args.limit
            is not None
            else "baseline"
        )

        _guardar_baseline(
            out=
                output_dir,

            prefix=
                prefix,

            evaluation=
                result,

            schema=
                schema_inicial,

            threshold=
                initial_threshold,

            limit=
                args.limit,

            manifest=
                split_manifest,

            modo_entrada=
                modo_entrada,
        )

        print()

        print(
            "Baseline validation:",
            result[
                "metricas"
            ],
        )

        print(
            "Threshold:",
            initial_threshold,
        )

        print(
            "Modo:",
            modo_entrada,
        )

        print(
            "Schema:",
            schema_inicial,
        )

        print()

        print(
            "Archivos:",
            output_dir
            / f"{prefix}.json",
            "y",
            output_dir
            / f"{prefix}.csv",
        )

        return

    # ========================================================
    # PRESUPUESTO GEPA
    # ========================================================

    gepa_config = (
        _configurar_presupuesto_gepa(
            config,
            args,
        )
    )

    print()

    print(
        "Iniciando optimización GEPA..."
    )

    if (
        gepa_config.get(
            "max_metric_calls"
        )
        is not None
    ):
        print(
            "Presupuesto GEPA:",
            gepa_config[
                "max_metric_calls"
            ],
            "metric calls",
        )

    elif (
        gepa_config.get(
            "max_full_evals"
        )
        is not None
    ):
        print(
            "Presupuesto GEPA:",
            gepa_config[
                "max_full_evals"
            ],
            "full evals",
        )

    else:
        print(
            "Presupuesto GEPA auto:",
            gepa_config[
                "auto"
            ],
        )

    # ========================================================
    # OPTIMIZACIÓN GEPA
    # ========================================================

    result = optimizar_schema(
        extractor=
            extractor,

        train=
            train,

        validation=
            validation,

        schema=
            schema_inicial,

        threshold=
            initial_threshold,

        metric=
            evaluar,

        modo_entrada=
            modo_entrada,

        llm_config=
            config[
                "llm"
            ],

        reflection_lm_config=
            (
                config.get(
                    "llm_reflexion"
                )
                or config[
                    "llm"
                ]
            ),

        gepa_config=
            gepa_config,

        history_path=
            output_dir
            / "historial.json",
    )

    selected_schema = (
        result[
            "schema"
        ]
    )

    # ========================================================
    # OPTIMIZACIÓN DEL THRESHOLD
    # ========================================================

    threshold_result = {
        "threshold":
            initial_threshold,

        "metricas_validation":
            result[
                "metricas_validation"
            ],

        "historial":
            [],
    }

    optimization_config = (
        config[
            "optimizacion"
        ]
    )

    if (
        optimization_config[
            "optimizar_threshold"
        ]
    ):
        threshold_result = (
            optimizar_threshold(
                extractor=
                    extractor,

                validation=
                    validation,

                schema=
                    selected_schema,

                initial=
                    initial_threshold,

                thresholds=
                    threshold_grid(
                        optimization_config[
                            "threshold_min"
                        ],
                        optimization_config[
                            "threshold_max"
                        ],
                        optimization_config[
                            "threshold_paso"
                        ],
                    ),

                metric=
                    evaluar,

                modo_entrada=
                    modo_entrada,
            )
        )

    final_threshold = (
        threshold_result[
            "threshold"
        ]
    )

    # ========================================================
    # VERSIONES
    # ========================================================

    versions = {
        "gliner":
            _version_paquete(
                "gliner"
            ),

        "dspy":
            _version_paquete(
                "dspy"
            ),

        "PyYAML":
            _version_paquete(
                "PyYAML"
            ),
    }

    # ========================================================
    # SELECCIÓN FINAL
    # ========================================================

    selection = {
        "metodo_optimizacion":
            "GEPA",

        "arquitectura":
            "schema_global",

        "modo_entrada":
            modo_entrada,

        "schema_hash":
            fingerprint(
                selected_schema
            ),

        "schema":
            selected_schema,

        "threshold_inicial":
            initial_threshold,

        "threshold_final":
            final_threshold,

        "baseline_validation":
            result[
                "baseline"
            ],

        "mejor_validation":
            threshold_result[
                "metricas_validation"
            ],

        "schema_gepa_aceptado":
            result.get(
                "accepted",
                False,
            ),

        "gepa":
            result.get(
                "gepa",
                {},
            ),

        "gepa_config":
            gepa_config,

        "gliner":
            config[
                "gliner"
            ],

        "llm": {
            key:
                value
            for (
                key,
                value,
            ) in config[
                "llm"
            ].items()
            if key
            not in {
                "api_key",
            }
        },

        "llm_reflexion": {
            key:
                value
            for (
                key,
                value,
            ) in (
                config.get(
                    "llm_reflexion"
                )
                or config[
                    "llm"
                ]
            ).items()
            if key
            not in {
                "api_key",
            }
        },

        "versiones":
            versions,

        "splits":
            split_manifest,
    }

    # ========================================================
    # GUARDAR SCHEMA
    # ========================================================

    optimized_schema_path = ruta(
        config[
            "rutas"
        ][
            "schema_optimizado"
        ]
    )

    guardar_json(
        optimized_schema_path,
        selected_schema,
    )

    # ========================================================
    # GUARDAR THRESHOLDS
    # ========================================================

    guardar_json(
        output_dir
        / "thresholds.json",

        threshold_result[
            "historial"
        ],
    )

    # ========================================================
    # GUARDAR SELECCIÓN
    # ========================================================

    guardar_json(
        output_dir
        / "seleccion.json",

        selection,
    )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    print()

    print(
        "Baseline validation:",
        f"{result['baseline']['global']:.4f}",
    )

    print(
        "Schema GEPA aceptado:",
        result.get(
            "accepted",
            False,
        ),
    )

    print(
        "Mejor validation:",
        (
            f"{threshold_result['metricas_validation']['global']:.4f}"
        ),
    )

    print(
        "Threshold:",
        initial_threshold,
        "->",
        final_threshold,
    )

    print(
        "Modo de entrada:",
        modo_entrada,
    )

    print()

    print(
        "Schema seleccionado:"
    )

    for (
        key,
        description,
    ) in selected_schema.items():
        print(
            f"  {key}: "
            f"{description}"
        )

    gepa_summary = (
        result.get(
            "gepa",
            {},
        )
    )

    if gepa_summary:
        print()

        print(
            "GEPA candidatos:",
            gepa_summary.get(
                "cantidad_candidatos"
            ),
        )

        print(
            "GEPA llamadas a métrica:",
            gepa_summary.get(
                "total_metric_calls"
            ),
        )

    print()

    print(
        "Schema guardado en:",
        optimized_schema_path,
    )

    print(
        "Resultados en:",
        output_dir,
    )


if __name__ == "__main__":
    main()