"""Tests livianos para DSPy + GEPA + GLiNER2.

Estos tests NO:

- descargan modelos reales;
- ejecutan GLiNER2 real;
- arrancan Ollama;
- llaman APIs;
- utilizan GPU.

Las dependencias pesadas se reemplazan por stubs/mocks.
"""

from __future__ import annotations

import builtins
import copy
import csv
import importlib
import json
import sys
import tempfile
import types
import unittest

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from benchmark import evaluar_externo

from src.dataset import (
    cargar_extraccion,
    cargar_gold,
    construir_dataset,
    dividir_dataset,
    manifiesto,
)

from src.gliner_module import (
    GLiNERModule,
    STRUCTURE_NAME,
)

from src.metricas import evaluar

from src.normalizacion import (
    normalizar_cuit_cuil,
    normalizar_dni,
    normalizar_nombre,
    separar_valores,
)

from src.schema_optimizer import (
    consolidar_predicciones_fragmentos,
    crear_metrica_gepa,
    evaluar_schema,
    generar_reporte_errores,
    optimizar_schema,
    optimizar_threshold,
    parsear_schema,
    predecir_ejemplo,
    validar_modo_entrada,
)

from src.utils import (
    ROOT,
    cargar_config,
    cargar_schema,
    leer_json,
    threshold_grid,
    validar_schema,
)


# ============================================================
# SCHEMA BASE
# ============================================================

SCHEMA = cargar_schema(
    ROOT
    / "schemas"
    / "schema_inicial.json"
)


# ============================================================
# HELPERS GENERALES
# ============================================================

def example(
    identifier,
    *,
    name="Juan Pérez",
    dni="30123456",
    cuit="20301234567",
    group=None,
):
    """
    Ejemplo equivalente al contrato interno
    generado por dataset.py.
    """

    return {
        "id":
            str(identifier),

        "group":
            group,

        "text":
            (
                f"Documento {identifier}. "
                f"Se ordena embargo de {name}, "
                f"DNI {dni}, CUIT {cuit}."
            ),

        "fragments": [
            f"Embargo de {name}.",
            f"DNI {dni}.",
            f"CUIT {cuit}.",
        ],

        "expected": {
            "nombre_embargado":
                name,

            "dni":
                dni,

            "cuit_cuil":
                cuit,
        },

        "cantidad_personas":
            1,
    }


def escribir_csv_punto_y_coma(
    path,
    rows,
):
    """
    Escribe CSV compatible con los loaders
    reales de dataset.py.
    """

    rows = list(
        rows
    )

    if not rows:
        raise ValueError(
            "No hay filas"
        )

    fields = list(
        rows[0].keys()
    )

    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# EXTRACTOR SIMULADO PARA GEPA
# ============================================================

class FakeExtractor:
    """
    Simula el contrato final del extractor.

    Schema inicial:
        sólo encuentra nombre.

    Si schema["dni"] == "mejor":
        encuentra nombre + DNI + CUIT.
    """

    def __init__(
        self,
    ):
        self.calls = []

    def predict(
        self,
        text,
        schema,
        threshold,
    ):
        self.calls.append(
            {
                "text":
                    text,

                "schema":
                    dict(
                        schema
                    ),

                "threshold":
                    threshold,
            }
        )

        improved = (
            schema["dni"]
            == "mejor"
        )

        return {
            "nombre_embargado":
                "Juan Pérez",

            "dni":
                (
                    "30123456"
                    if improved
                    else ""
                ),

            "cuit_cuil":
                (
                    "20301234567"
                    if (
                        improved
                        and threshold <= 0.5
                    )
                    else ""
                ),

            "spans_raw":
                [],

            "entidades":
                [],
        }


# ============================================================
# EXTRACTOR SIMULADO PARA FRAGMENTOS
# ============================================================

class FragmentExtractor:
    """
    Simula información distribuida
    entre varios fragmentos.
    """

    def __init__(
        self,
    ):
        self.calls = []

    def predict(
        self,
        text,
        schema,
        threshold,
    ):
        self.calls.append(
            text
        )

        if "Embargo" in text:

            return {
                "nombre_embargado":
                    "Juan Pérez",

                "dni":
                    "",

                "cuit_cuil":
                    "",

                "spans_raw":
                    [],

                "entidades":
                    [],
            }

        if "DNI" in text:

            return {
                "nombre_embargado":
                    "Juan Pérez",

                "dni":
                    "30123456",

                "cuit_cuil":
                    "",

                "spans_raw":
                    [],

                "entidades":
                    [],
            }

        if "CUIT" in text:

            return {
                "nombre_embargado":
                    "Juan Pérez",

                "dni":
                    "",

                "cuit_cuil":
                    "20301234567",

                "spans_raw":
                    [],

                "entidades":
                    [],
            }

        return {
            "nombre_embargado":
                "",

            "dni":
                "",

            "cuit_cuil":
                "",

            "spans_raw":
                [],

            "entidades":
                [],
        }


# ============================================================
# GLINER2 STRUCTURED SIMULADO
# ============================================================

class FakeStructuredSchema:
    """
    Simula el builder:

        create_schema()
        .structure(...)
        .field(...)
    """

    def __init__(
        self,
    ):
        self.structure_name = None

        self.fields = []

    def structure(
        self,
        name,
    ):
        self.structure_name = (
            name
        )

        return self

    def field(
        self,
        name,
        **kwargs,
    ):
        self.fields.append(
            {
                "name":
                    name,

                **kwargs,
            }
        )

        return self


class FakeGLiNER2Model:
    """
    Simula AutoExtractor / GLiNER2.

    Guarda:

    - schema construido;
    - texto enviado;
    - threshold global;
    - kwargs de extract().
    """

    def __init__(
        self,
    ):
        self.created_schemas = []

        self.extract_calls = []

    def create_schema(
        self,
    ):
        schema = (
            FakeStructuredSchema()
        )

        self.created_schemas.append(
            schema
        )

        return schema

    def extract(
        self,
        text,
        schema,
        **kwargs,
    ):
        self.extract_calls.append(
            {
                "text":
                    text,

                "schema":
                    schema,

                "kwargs":
                    dict(
                        kwargs
                    ),
            }
        )

        return {
            "persona_embargada": [
                {
                    "nombre_embargado": {
                        "value":
                            "Juan Pérez",

                        "confidence":
                            0.91,

                        "start":
                            10,

                        "end":
                            20,
                    },

                    "dni_embargado": {
                        "value":
                            "30.123.456",

                        "confidence":
                            0.88,

                        "start":
                            25,

                        "end":
                            35,
                    },

                    "cuit_cuil_embargado": {
                        "value":
                            "20-30123456-7",

                        "confidence":
                            0.86,

                        "start":
                            40,

                        "end":
                            53,
                    },
                }
            ]
        }


# ============================================================
# DSPY SIMULADO
# ============================================================

def fake_dspy_module():
    """
    Implementación mínima de DSPy
    utilizada exclusivamente por tests.
    """

    module = types.ModuleType(
        "dspy"
    )

    # --------------------------------------------------------
    # SIGNATURE
    # --------------------------------------------------------

    class Signature:
        pass

    def InputField(
        **kwargs,
    ):
        return None

    def OutputField(
        **kwargs,
    ):
        return None

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    class Prediction:

        def __init__(
            self,
            **kwargs,
        ):
            for (
                key,
                value,
            ) in kwargs.items():

                setattr(
                    self,
                    key,
                    value,
                )

    # --------------------------------------------------------
    # EXAMPLE
    # --------------------------------------------------------

    class Example:

        def __init__(
            self,
            **kwargs,
        ):
            self._inputs = ()

            for (
                key,
                value,
            ) in kwargs.items():

                setattr(
                    self,
                    key,
                    value,
                )

        def with_inputs(
            self,
            *names,
        ):
            self._inputs = (
                names
            )

            return self

    # --------------------------------------------------------
    # MODULE
    # --------------------------------------------------------

    class Module:

        def __init__(
            self,
        ):
            self._lm = None

        def __call__(
            self,
            *args,
            **kwargs,
        ):
            return self.forward(
                *args,
                **kwargs,
            )

        def forward(
            self,
            *args,
            **kwargs,
        ):
            raise NotImplementedError(
                "Las subclases deben "
                "implementar forward()."
            )

        def set_lm(
            self,
            lm,
        ):
            self._lm = lm

            return self

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    class Predict:

        def __init__(
            self,
            signature,
        ):
            self.signature = (
                signature
            )

            self.improved = False

        def __call__(
            self,
            **kwargs,
        ):
            if self.improved:

                return Prediction(
                    descripcion_nombre_embargado=(
                        "Persona física o razón social "
                        "sobre la que recae directamente "
                        "la medida judicial"
                    ),

                    descripcion_dni=
                        "mejor",

                    descripcion_cuit_cuil=(
                        "CUIT o CUIL perteneciente "
                        "al sujeto embargado"
                    ),
                )

            return Prediction(
                descripcion_nombre_embargado=(
                    SCHEMA[
                        "nombre_embargado"
                    ]
                ),

                descripcion_dni=(
                    SCHEMA[
                        "dni"
                    ]
                ),

                descripcion_cuit_cuil=(
                    SCHEMA[
                        "cuit_cuil"
                    ]
                ),
            )

    # --------------------------------------------------------
    # LM
    # --------------------------------------------------------

    class LM:

        def __init__(
            self,
            model,
            **kwargs,
        ):
            self.model = model

            self.kwargs = (
                kwargs
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    class DetailedResults:

        def __init__(
            self,
        ):
            self.candidates = [
                "baseline",
                "optimized",
            ]

            self.best_idx = 1

            self.val_aggregate_scores = [
                0.5,
                1.0,
            ]

            self.total_metric_calls = (
                4
            )

            self.num_full_val_evals = (
                1
            )

            self.seed = 42

            self.log_dir = (
                "fake_logs"
            )

    # --------------------------------------------------------
    # GEPA
    # --------------------------------------------------------

    class GEPA:

        def __init__(
            self,
            **kwargs,
        ):
            self.kwargs = (
                kwargs
            )

            self.metric = (
                kwargs[
                    "metric"
                ]
            )

        def compile(
            self,
            student,
            trainset=None,
            valset=None,
            **kwargs,
        ):
            if not hasattr(
                student,
                "generador_schema",
            ):
                raise AssertionError(
                    "Student sin "
                    "generador_schema"
                )

            student.generador_schema.improved = (
                True
            )

            if valset:

                gold = (
                    valset[0]
                )

                pred = student(
                    tarea=
                        gold.tarea,

                    schema_inicial=
                        gold.schema_inicial,
                )

                result = (
                    self.metric(
                        gold,
                        pred,
                    )
                )

                if not hasattr(
                    result,
                    "score",
                ):
                    raise AssertionError(
                        "GEPA metric "
                        "sin score"
                    )

                if not hasattr(
                    result,
                    "feedback",
                ):
                    raise AssertionError(
                        "GEPA metric "
                        "sin feedback"
                    )

            student.detailed_results = (
                DetailedResults()
            )

            return student

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    def context(
        **kwargs,
    ):
        return nullcontext()

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    setattr(
        module,
        "Signature",
        Signature,
    )

    setattr(
        module,
        "InputField",
        InputField,
    )

    setattr(
        module,
        "OutputField",
        OutputField,
    )

    setattr(
        module,
        "Prediction",
        Prediction,
    )

    setattr(
        module,
        "Example",
        Example,
    )

    setattr(
        module,
        "Module",
        Module,
    )

    setattr(
        module,
        "Predict",
        Predict,
    )

    setattr(
        module,
        "LM",
        LM,
    )

    setattr(
        module,
        "GEPA",
        GEPA,
    )

    setattr(
        module,
        "context",
        context,
    )

    return module


# ============================================================
# TESTS
# ============================================================

class Tests(
    unittest.TestCase
):

    # ========================================================
    # IMPORTS
    # ========================================================

    def test_imports_sin_dependencias_pesadas(
        self,
    ):
        original = (
            builtins.__import__
        )

        def guarded(
            name,
            *args,
            **kwargs,
        ):
            if (
                name.split(".")[0]
                in {
                    "dspy",
                    "gliner",
                    "gliner2",
                    "torch",
                    "pandas",
                }
            ):
                raise AssertionError(
                    f"Import pesado: {name}"
                )

            return original(
                name,
                *args,
                **kwargs,
            )

        with patch(
            "builtins.__import__",
            side_effect=guarded,
        ):

            for name in (
                "src.gliner_module",
                "src.schema_optimizer",
                "entrenar",
                "evaluar",
                "ejecutar",
                "benchmark",
            ):

                importlib.reload(
                    importlib.import_module(
                        name
                    )
                )

    # ========================================================
    # CONFIG
    # ========================================================

    def test_config_nueva(
        self,
    ):
        config = (
            cargar_config()
        )

        self.assertEqual(
            config[
                "datos"
            ][
                "modo_entrada"
            ],
            "texto_completo",
        )

        self.assertEqual(
            config[
                "gliner"
            ][
                "modelo"
            ],
            "fastino/gliner2-multi-v1",
        )

        self.assertEqual(
            config[
                "gliner"
            ][
                "threshold_inicial"
            ],
            0.55,
        )

        self.assertEqual(
            config[
                "gliner"
            ][
                "field_thresholds"
            ],
            {
                "nombre_embargado":
                    0.60,

                "dni":
                    0.50,

                "cuit_cuil":
                    0.50,
            },
        )

        self.assertIn(
            "gold",
            config[
                "datos"
            ],
        )

        self.assertIn(
            "extraccion",
            config[
                "datos"
            ],
        )

        self.assertEqual(
            config[
                "datos"
            ][
                "split"
            ],
            {
                "train":
                    0.60,

                "validation":
                    0.20,

                "test":
                    0.20,
            },
        )

    def test_gepa_budget_config(
        self,
    ):
        config = (
            cargar_config()
        )

        self.assertEqual(
            config[
                "gepa"
            ][
                "max_metric_calls"
            ],
            20,
        )

        self.assertIsNone(
            config[
                "gepa"
            ][
                "auto"
            ]
        )

        self.assertIsNone(
            config[
                "gepa"
            ][
                "max_full_evals"
            ]
        )

    def test_modo_invalido(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):

            validar_modo_entrada(
                "otro"
            )

    # ========================================================
    # SCHEMA
    # ========================================================

    def test_schema_valido(
        self,
    ):
        self.assertEqual(
            validar_schema(
                SCHEMA
            ),
            SCHEMA,
        )

    def test_schema_invalido(
        self,
    ):
        invalid = {
            **SCHEMA,

            "dni":
                "",
        }

        with self.assertRaises(
            ValueError
        ):

            validar_schema(
                invalid
            )

    def test_parse_schema(
        self,
    ):
        value = (
            "```json\n"
            + json.dumps(
                SCHEMA
            )
            + "\n```"
        )

        self.assertEqual(
            parsear_schema(
                value
            ),
            SCHEMA,
        )

    # ========================================================
    # NORMALIZACIÓN
    # ========================================================

    def test_normalizacion(
        self,
    ):
        self.assertEqual(
            normalizar_nombre(
                "  JUÁN   Pérez "
            ),
            "juan perez",
        )

        self.assertEqual(
            normalizar_dni(
                "30.123.456"
            ),
            "30123456",
        )

        self.assertEqual(
            normalizar_cuit_cuil(
                "20-30123456-7"
            ),
            "20301234567",
        )

        self.assertEqual(
            separar_valores(
                "Juan | | María"
            ),
            [
                "Juan",
                "María",
            ],
        )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    def test_metric_personas_correctas(
        self,
    ):
        truth = {
            "nombre_embargado":
                "Juan Pérez | María López",

            "dni":
                "30123456 | 28987654",

            "cuit_cuil":
                "20301234567 | 27289876543",
        }

        result = evaluar(
            truth,
            dict(
                truth
            ),
        )

        self.assertEqual(
            result[
                "global"
            ],
            1.0,
        )

    def test_metric_detecta_dni_intercambiado(
        self,
    ):
        truth = {
            "nombre_embargado":
                "Juan Pérez | María López",

            "dni":
                "30123456 | 28987654",

            "cuit_cuil":
                "20301234567 | 27289876543",
        }

        prediction = {
            "nombre_embargado":
                "Juan Pérez | María López",

            "dni":
                "28987654 | 30123456",

            "cuit_cuil":
                "20301234567 | 27289876543",
        }

        result = evaluar(
            truth,
            prediction,
        )

        self.assertEqual(
            result[
                "dni"
            ],
            0.0,
        )

        self.assertLess(
            result[
                "global"
            ],
            1.0,
        )

    # ========================================================
    # GOLD
    # ========================================================

    def test_gold_consolida_multiples_personas(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "gold.csv"
            )

            escribir_csv_punto_y_coma(
                path,
                [
                    {
                        "id":
                            "100",

                        "numero_archivo":
                            "1",

                        "embargado":
                            "Juan Pérez",

                        "dni":
                            "30123456",

                        "cuit_cuil":
                            "",
                    },

                    {
                        "id":
                            "100",

                        "numero_archivo":
                            "1",

                        "embargado":
                            "María López",

                        "dni":
                            "28987654",

                        "cuit_cuil":
                            "27289876543",
                    },
                ],
            )

            gold = cargar_gold(
                path
            )

            self.assertEqual(
                len(gold),
                1,
            )

            self.assertEqual(
                gold[
                    "100"
                ][
                    "expected"
                ][
                    "nombre_embargado"
                ],
                "Juan Pérez | María López",
            )

            self.assertEqual(
                gold[
                    "100"
                ][
                    "expected"
                ][
                    "dni"
                ],
                "30123456 | 28987654",
            )

            self.assertEqual(
                gold[
                    "100"
                ][
                    "expected"
                ][
                    "cuit_cuil"
                ],
                " | 27289876543",
            )

    # ========================================================
    # EXTRACCIÓN DATASET
    # ========================================================

    def test_extraccion_consolida_fragmentos(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "fragmentos.csv"
            )

            escribir_csv_punto_y_coma(
                path,
                [
                    {
                        "id":
                            "100",

                        "numero_archivo":
                            "1",

                        "fragmento":
                            "fragmento uno",

                        "texto_completo":
                            "documento completo",
                    },

                    {
                        "id":
                            "100",

                        "numero_archivo":
                            "1",

                        "fragmento":
                            "fragmento dos",

                        "texto_completo":
                            "documento completo",
                    },
                ],
            )

            data = cargar_extraccion(
                path
            )

            self.assertEqual(
                data[
                    "100"
                ][
                    "text"
                ],
                "documento completo",
            )

            self.assertEqual(
                data[
                    "100"
                ][
                    "fragments"
                ],
                [
                    "fragmento uno",
                    "fragmento dos",
                ],
            )

    # ========================================================
    # MERGE
    # ========================================================

    def test_construir_dataset_excluye_sin_gold(
        self,
    ):
        gold = {
            "100": {
                "id":
                    "100",

                "numero_archivo":
                    "1",

                "expected": {
                    "nombre_embargado":
                        "Juan",

                    "dni":
                        "123",

                    "cuit_cuil":
                        "",
                },

                "cantidad_personas":
                    1,
            }
        }

        extraction = {
            "100": {
                "id":
                    "100",

                "numero_archivo":
                    "1",

                "text":
                    "texto 100",

                "fragments":
                    [
                        "fragmento"
                    ],
            },

            "999": {
                "id":
                    "999",

                "numero_archivo":
                    "99",

                "text":
                    "sin gold",

                "fragments":
                    [
                        "fragmento"
                    ],
            },
        }

        dataset = construir_dataset(
            gold,
            extraction,
        )

        self.assertEqual(
            len(dataset),
            1,
        )

        self.assertEqual(
            dataset[
                0
            ][
                "id"
            ],
            "100",
        )

    # ========================================================
    # SPLITS
    # ========================================================

    def test_splits_reproducibles(
        self,
    ):
        examples = [
            example(
                i,
                group=str(i),
            )
            for i
            in range(30)
        ]

        first = dividir_dataset(
            examples,
            seed=42,
        )

        second = dividir_dataset(
            examples,
            seed=42,
        )

        self.assertEqual(
            manifiesto(
                first
            ),
            manifiesto(
                second
            ),
        )

    def test_seed_distinto_cambia_split(
        self,
    ):
        examples = [
            example(
                i,
                group=str(i),
            )
            for i
            in range(30)
        ]

        first = dividir_dataset(
            examples,
            seed=42,
        )

        second = dividir_dataset(
            examples,
            seed=99,
        )

        self.assertNotEqual(
            manifiesto(
                first
            ),
            manifiesto(
                second
            ),
        )

    def test_same_group_no_se_separa(
        self,
    ):
        examples = [
            example(
                i,
                group=str(i),
            )
            for i
            in range(20)
        ]

        examples[
            0
        ][
            "group"
        ] = "MISMO"

        examples[
            1
        ][
            "group"
        ] = "MISMO"

        splits = dividir_dataset(
            examples,
            seed=42,
        )

        locations = {}

        for (
            split_name,
            rows,
        ) in splits.items():

            for row in rows:

                locations[
                    row[
                        "id"
                    ]
                ] = (
                    split_name
                )

        self.assertEqual(
            locations["0"],
            locations["1"],
        )

    # ========================================================
    # GLINER2 STRUCTURED
    # ========================================================

    def test_gliner2_construye_schema_structured(
        self,
    ):
        model = (
            FakeGLiNER2Model()
        )

        module = GLiNERModule(
            schema=
                SCHEMA,

            threshold=
                0.55,

            model=
                model,

            field_thresholds={
                "nombre_embargado":
                    0.60,

                "dni":
                    0.50,

                "cuit_cuil":
                    0.50,
            },

            use_validators=
                False,
        )

        result = module.predict(
            "Se ordena embargo de Juan Pérez."
        )

        # ----------------------------------------------------
        # Se creó un schema.
        # ----------------------------------------------------

        self.assertEqual(
            len(
                model.created_schemas
            ),
            1,
        )

        native_schema = (
            model.created_schemas[
                0
            ]
        )

        # ----------------------------------------------------
        # Structure correcto.
        # ----------------------------------------------------

        self.assertEqual(
            native_schema.structure_name,
            STRUCTURE_NAME,
        )

        self.assertEqual(
            native_schema.structure_name,
            "persona_embargada",
        )

        # ----------------------------------------------------
        # Fields nativos.
        # ----------------------------------------------------

        fields_by_name = {
            field[
                "name"
            ]:
                field
            for field
            in native_schema.fields
        }

        self.assertEqual(
            set(
                fields_by_name
            ),
            {
                "nombre_embargado",
                "dni_embargado",
                "cuit_cuil_embargado",
            },
        )

        # ----------------------------------------------------
        # Thresholds por field.
        # ----------------------------------------------------

        self.assertEqual(
            fields_by_name[
                "nombre_embargado"
            ][
                "threshold"
            ],
            0.60,
        )

        self.assertEqual(
            fields_by_name[
                "dni_embargado"
            ][
                "threshold"
            ],
            0.50,
        )

        self.assertEqual(
            fields_by_name[
                "cuit_cuil_embargado"
            ][
                "threshold"
            ],
            0.50,
        )

        # ----------------------------------------------------
        # Descripciones correctas.
        # ----------------------------------------------------

        self.assertEqual(
            fields_by_name[
                "nombre_embargado"
            ][
                "description"
            ],
            SCHEMA[
                "nombre_embargado"
            ],
        )

        self.assertEqual(
            fields_by_name[
                "dni_embargado"
            ][
                "description"
            ],
            SCHEMA[
                "dni"
            ],
        )

        self.assertEqual(
            fields_by_name[
                "cuit_cuil_embargado"
            ][
                "description"
            ],
            SCHEMA[
                "cuit_cuil"
            ],
        )

        # ----------------------------------------------------
        # Threshold global.
        # ----------------------------------------------------

        self.assertEqual(
            len(
                model.extract_calls
            ),
            1,
        )

        extract_call = (
            model.extract_calls[
                0
            ]
        )

        self.assertEqual(
            extract_call[
                "kwargs"
            ][
                "threshold"
            ],
            0.55,
        )

        self.assertTrue(
            extract_call[
                "kwargs"
            ][
                "include_confidence"
            ]
        )

        self.assertTrue(
            extract_call[
                "kwargs"
            ][
                "include_spans"
            ]
        )

        # ----------------------------------------------------
        # Normalización al contrato canónico.
        # ----------------------------------------------------

        self.assertEqual(
            result[
                "nombre_embargado"
            ],
            "Juan Pérez",
        )

        self.assertEqual(
            result[
                "dni"
            ],
            "30.123.456",
        )

        self.assertEqual(
            result[
                "cuit_cuil"
            ],
            "20-30123456-7",
        )

        self.assertEqual(
            len(
                result[
                    "entidades"
                ]
            ),
            1,
        )

        self.assertEqual(
            len(
                result[
                    "spans_raw"
                ]
            ),
            3,
        )

    # ========================================================
    # CACHE DEL SCHEMA NATIVO
    # ========================================================

    def test_gliner2_reutiliza_schema_nativo(
        self,
    ):
        model = (
            FakeGLiNER2Model()
        )

        module = GLiNERModule(
            schema=
                SCHEMA,

            threshold=
                0.55,

            model=
                model,

            field_thresholds={
                "nombre_embargado":
                    0.60,

                "dni":
                    0.50,

                "cuit_cuil":
                    0.50,
            },

            use_validators=
                False,
        )

        module.predict(
            "documento uno"
        )

        module.predict(
            "documento dos"
        )

        # El schema nativo debe construirse
        # una sola vez.
        self.assertEqual(
            len(
                model.created_schemas
            ),
            1,
        )

        # Pero se realizan dos inferencias.
        self.assertEqual(
            len(
                model.extract_calls
            ),
            2,
        )

    # ========================================================
    # ACTUALIZACIÓN FIELD THRESHOLDS
    # ========================================================

    def test_gliner2_actualiza_field_thresholds(
        self,
    ):
        model = (
            FakeGLiNER2Model()
        )

        module = GLiNERModule(
            schema=
                SCHEMA,

            model=
                model,

            field_thresholds={
                "nombre_embargado":
                    0.60,

                "dni":
                    0.50,

                "cuit_cuil":
                    0.50,
            },

            use_validators=
                False,
        )

        module.predict(
            "primero"
        )

        module.actualizar_field_thresholds(
            {
                "nombre_embargado":
                    0.55
            }
        )

        module.predict(
            "segundo"
        )

        # Cambiar field threshold invalida
        # el schema nativo cacheado.
        self.assertEqual(
            len(
                model.created_schemas
            ),
            2,
        )

        second_schema = (
            model.created_schemas[
                1
            ]
        )

        fields = {
            field[
                "name"
            ]:
                field
            for field
            in second_schema.fields
        }

        self.assertEqual(
            fields[
                "nombre_embargado"
            ][
                "threshold"
            ],
            0.55,
        )

    # ========================================================
    # PREDECIR TEXTO COMPLETO
    # ========================================================

    def test_predecir_texto_completo(
        self,
    ):
        extractor = (
            FakeExtractor()
        )

        ex = example(
            1
        )

        result = (
            predecir_ejemplo(
                extractor,
                ex,
                SCHEMA,
                0.45,
                "texto_completo",
            )
        )

        self.assertEqual(
            len(
                extractor.calls
            ),
            1,
        )

        self.assertEqual(
            extractor.calls[
                0
            ][
                "text"
            ],
            ex[
                "text"
            ],
        )

        self.assertEqual(
            result[
                "nombre_embargado"
            ],
            "Juan Pérez",
        )

    # ========================================================
    # PREDECIR FRAGMENTOS
    # ========================================================

    def test_predecir_fragmentos(
        self,
    ):
        extractor = (
            FragmentExtractor()
        )

        ex = example(
            1
        )

        result = (
            predecir_ejemplo(
                extractor,
                ex,
                SCHEMA,
                0.45,
                "fragmentos",
            )
        )

        self.assertEqual(
            len(
                extractor.calls
            ),
            3,
        )

        self.assertEqual(
            result[
                "nombre_embargado"
            ],
            "Juan Pérez",
        )

        self.assertEqual(
            result[
                "dni"
            ],
            "30123456",
        )

        self.assertEqual(
            result[
                "cuit_cuil"
            ],
            "20301234567",
        )

    # ========================================================
    # CONSOLIDACIÓN DE FRAGMENTOS
    # ========================================================

    def test_consolidar_predicciones_fragmentos(
        self,
    ):
        predictions = [
            {
                "nombre_embargado":
                    "Juan Pérez",

                "dni":
                    "30123456",

                "cuit_cuil":
                    "",

                "spans_raw":
                    [],
            },

            {
                "nombre_embargado":
                    "Juan Pérez",

                "dni":
                    "",

                "cuit_cuil":
                    "20301234567",

                "spans_raw":
                    [],
            },
        ]

        result = (
            consolidar_predicciones_fragmentos(
                predictions
            )
        )

        self.assertEqual(
            result[
                "nombre_embargado"
            ],
            "Juan Pérez",
        )

        self.assertEqual(
            result[
                "dni"
            ],
            "30123456",
        )

        self.assertEqual(
            result[
                "cuit_cuil"
            ],
            "20301234567",
        )

    # ========================================================
    # EVALUAR SCHEMA
    # ========================================================

    def test_evaluar_schema_texto_completo(
        self,
    ):
        result = (
            evaluar_schema(
                FakeExtractor(),
                [
                    example(
                        1
                    )
                ],
                SCHEMA,
                0.45,
                evaluar,
                "texto_completo",
            )
        )

        self.assertAlmostEqual(
            result[
                "metricas"
            ][
                "global"
            ],
            0.5,
        )

    def test_evaluar_schema_fragmentos(
        self,
    ):
        result = (
            evaluar_schema(
                FragmentExtractor(),
                [
                    example(
                        1
                    )
                ],
                SCHEMA,
                0.45,
                evaluar,
                "fragmentos",
            )
        )

        self.assertEqual(
            result[
                "metricas"
            ][
                "global"
            ],
            1.0,
        )

    # ========================================================
    # GEPA METRIC
    # ========================================================

    def test_metric_gepa_score_feedback(
        self,
    ):
        fake_dspy = (
            fake_dspy_module()
        )

        with patch.dict(
            sys.modules,
            {
                "dspy":
                    fake_dspy
            },
        ):

            metric = (
                crear_metrica_gepa(
                    metric=
                        evaluar,

                    extractor=
                        FakeExtractor(),

                    threshold=
                        0.45,

                    modo_entrada=
                        "texto_completo",
                )
            )

            ex = example(
                1
            )

            gold = (
                fake_dspy.Example(
                    text=
                        ex[
                            "text"
                        ],

                    fragments=
                        ex[
                            "fragments"
                        ],

                    example_id=
                        ex[
                            "id"
                        ],

                    nombre_embargado=
                        ex[
                            "expected"
                        ][
                            "nombre_embargado"
                        ],

                    dni=
                        ex[
                            "expected"
                        ][
                            "dni"
                        ],

                    cuit_cuil=
                        ex[
                            "expected"
                        ][
                            "cuit_cuil"
                        ],
                )
            )

            pred = (
                fake_dspy.Prediction(
                    descripcion_nombre_embargado=
                        "Persona embargada",

                    descripcion_dni=
                        "mejor",

                    descripcion_cuit_cuil=
                        "CUIT embargado",
                )
            )

            result = metric(
                gold,
                pred,
            )

        self.assertEqual(
            result.score,
            1.0,
        )

        self.assertIn(
            "score_global",
            result.feedback,
        )

    # ========================================================
    # GEPA GLOBAL
    # ========================================================

    def test_gepa_schema_global(
        self,
    ):
        fake_dspy = (
            fake_dspy_module()
        )

        config = (
            cargar_config()
        )

        train = [
            example(
                "TRAIN",
                group="A",
            )
        ]

        validation = [
            example(
                "VALIDATION",
                group="B",
            )
        ]

        with tempfile.TemporaryDirectory() as directory:

            history_path = (
                Path(directory)
                / "history.json"
            )

            with patch.dict(
                sys.modules,
                {
                    "dspy":
                        fake_dspy
                },
            ):

                result = (
                    optimizar_schema(
                        extractor=
                            FakeExtractor(),

                        train=
                            train,

                        validation=
                            validation,

                        schema=
                            SCHEMA,

                        threshold=
                            0.45,

                        metric=
                            evaluar,

                        modo_entrada=
                            "texto_completo",

                        llm_config=
                            config[
                                "llm"
                            ],

                        reflection_lm_config=
                            config[
                                "llm_reflexion"
                            ],

                        gepa_config=
                            config[
                                "gepa"
                            ],

                        history_path=
                            history_path,
                    )
                )

            self.assertTrue(
                result[
                    "accepted"
                ]
            )

            self.assertEqual(
                result[
                    "schema"
                ][
                    "dni"
                ],
                "mejor",
            )

            history = leer_json(
                history_path
            )

            self.assertEqual(
                history[
                    "modo_entrada"
                ],
                "texto_completo",
            )

    # ========================================================
    # GEPA FRAGMENTOS
    # ========================================================

    def test_gepa_metric_fragmentos(
        self,
    ):
        fake_dspy = (
            fake_dspy_module()
        )

        ex = example(
            1
        )

        with patch.dict(
            sys.modules,
            {
                "dspy":
                    fake_dspy
            },
        ):

            metric = (
                crear_metrica_gepa(
                    metric=
                        evaluar,

                    extractor=
                        FragmentExtractor(),

                    threshold=
                        0.45,

                    modo_entrada=
                        "fragmentos",
                )
            )

            gold = (
                fake_dspy.Example(
                    text=
                        ex[
                            "text"
                        ],

                    fragments=
                        ex[
                            "fragments"
                        ],

                    example_id=
                        ex[
                            "id"
                        ],

                    nombre_embargado=
                        ex[
                            "expected"
                        ][
                            "nombre_embargado"
                        ],

                    dni=
                        ex[
                            "expected"
                        ][
                            "dni"
                        ],

                    cuit_cuil=
                        ex[
                            "expected"
                        ][
                            "cuit_cuil"
                        ],
                )
            )

            pred = (
                fake_dspy.Prediction(
                    descripcion_nombre_embargado=
                        "Persona embargada",

                    descripcion_dni=
                        "mejor",

                    descripcion_cuit_cuil=
                        "CUIT embargado",
                )
            )

            result = metric(
                gold,
                pred,
            )

        self.assertEqual(
            result.score,
            1.0,
        )

    # ========================================================
    # LEAKAGE
    # ========================================================

    def test_gepa_rechaza_leakage(
        self,
    ):
        config = (
            cargar_config()
        )

        same = example(
            1,
            group="MISMO",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Leakage",
        ):

            optimizar_schema(
                extractor=
                    FakeExtractor(),

                train=[
                    same
                ],

                validation=[
                    copy.deepcopy(
                        same
                    )
                ],

                schema=
                    SCHEMA,

                threshold=
                    0.45,

                metric=
                    evaluar,

                modo_entrada=
                    "texto_completo",

                llm_config=
                    config[
                        "llm"
                    ],

                reflection_lm_config=
                    config[
                        "llm_reflexion"
                    ],

                gepa_config=
                    config[
                        "gepa"
                    ],
            )

    # ========================================================
    # THRESHOLD
    # ========================================================

    def test_threshold_search(
        self,
    ):
        optimized_schema = {
            **SCHEMA,

            "dni":
                "mejor",
        }

        result = (
            optimizar_threshold(
                extractor=
                    FakeExtractor(),

                validation=[
                    example(
                        "VAL"
                    )
                ],

                schema=
                    optimized_schema,

                initial=
                    0.60,

                thresholds=
                    threshold_grid(
                        0.30,
                        0.70,
                        0.05,
                    ),

                metric=
                    evaluar,

                modo_entrada=
                    "texto_completo",
            )
        )

        self.assertEqual(
            result[
                "threshold"
            ],
            0.30,
        )

        self.assertEqual(
            result[
                "metricas_validation"
            ][
                "global"
            ],
            1.0,
        )

    # ========================================================
    # REPORTE ERRORES
    # ========================================================

    def test_reporte_errores_limitado(
        self,
    ):
        ex = example(
            1
        )

        ex[
            "text"
        ] = (
            "Juez "
            + "texto " * 30000
        )

        evaluation = (
            evaluar_schema(
                FakeExtractor(),
                [
                    ex
                ],
                SCHEMA,
                0.45,
                evaluar,
                "texto_completo",
            )
        )

        report = (
            generar_reporte_errores(
                evaluation,
                max_chars=2000,
            )
        )

        self.assertLessEqual(
            len(
                report
            ),
            2000,
        )

        self.assertIn(
            "Juez",
            report,
        )

    # ========================================================
    # BENCHMARK EXTERNO
    # ========================================================

    def test_external_alignment(
        self,
    ):
        examples = [
            example(
                1
            ),
            example(
                2
            ),
        ]

        rows = [
            {
                "id":
                    ex[
                        "id"
                    ],

                **ex[
                    "expected"
                ],

                "latencia_segundos":
                    0.1,
            }

            for ex
            in reversed(
                examples
            )
        ]

        result = (
            evaluar_externo(
                examples,
                rows,
            )
        )

        self.assertEqual(
            result[
                "global"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            result[
                "latencia_promedio"
            ],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()