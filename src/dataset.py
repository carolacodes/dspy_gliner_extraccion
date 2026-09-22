"""Carga y preparación del dataset supervisado para DSPy + GLiNER.

Este proyecto trabaja con dos CSV separados:

1. GOLD supervisado:
   contiene las respuestas correctas por documento.

2. BASE DE EXTRACCIÓN:
   contiene fragmentos y texto_completo por documento.

Ambos archivos se unen por `id`.

El loader consolida múltiples embargados del mismo documento usando " | ".
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

from .utils import KEYS, ruta


# ============================================================
# LECTURA CSV
# ============================================================

def _leer_csv_semicolon(path):
    """
    Lee CSV separado por punto y coma.

    Se usa newline="" para soportar correctamente
    campos multilínea como texto_completo.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream,
            delimiter=";",
        )

        if not reader.fieldnames:
            raise ValueError(
                f"CSV sin cabecera: {path}"
            )

        return list(reader)


# ============================================================
# HELPERS
# ============================================================

def _clean(value):
    if value is None:
        return ""

    return str(value).strip()


def _normalizar_texto_grupo(texto):
    """
    Se usa sólo para detectar documentos idénticos
    entre splits.

    No modifica el texto enviado a GLiNER.
    """
    return re.sub(
        r"\s+",
        " ",
        _clean(texto).casefold(),
    )


def _hash_texto(texto):
    return hashlib.sha256(
        str(texto).encode("utf-8")
    ).hexdigest()


def _join_pipe(values):
    """
    Une valores manteniendo posiciones.

    Ejemplo:

    ["Juan", "María"]
    ->
    "Juan | María"

    ["", "2030..."]
    ->
    " | 2030..."
    """
    return " | ".join(
        _clean(value)
        for value in values
    )


# ============================================================
# GOLD
# ============================================================

def cargar_gold(
    path,
    *,
    id_column="id",
    numero_archivo_column="numero_archivo",
    nombre_column="embargado",
    dni_column="dni",
    cuit_column="cuit_cuil",
):
    """
    Carga y consolida la base supervisada.

    Puede haber varias filas para el mismo id
    cuando un documento tiene varios embargados.

    Ejemplo:

    id 123:
      Juan | DNI 1
      María | DNI 2

    se convierte en:

    expected:
      nombre_embargado = "Juan | María"
      dni = "1 | 2"
    """

    rows = _leer_csv_semicolon(
        path
    )

    if not rows:
        raise ValueError(
            "La base gold está vacía"
        )

    required = {
        id_column,
        numero_archivo_column,
        nombre_column,
        dni_column,
        cuit_column,
    }

    missing = (
        required
        - set(rows[0].keys())
    )

    if missing:
        raise ValueError(
            "Faltan columnas en gold: "
            + ", ".join(
                sorted(missing)
            )
        )

    grouped = defaultdict(list)

    for index, row in enumerate(rows):
        identifier = _clean(
            row.get(id_column)
        )

        if not identifier:
            raise ValueError(
                f"Gold con id vacío en fila {index}"
            )

        grouped[
            identifier
        ].append(row)

    consolidated = {}

    for identifier, group_rows in grouped.items():

        numero_archivo_values = {
            _clean(
                row.get(
                    numero_archivo_column
                )
            )
            for row in group_rows
            if _clean(
                row.get(
                    numero_archivo_column
                )
            )
        }

        if len(
            numero_archivo_values
        ) > 1:
            raise ValueError(
                f"El id {identifier} tiene "
                "más de un numero_archivo en gold"
            )

        numero_archivo = (
            next(
                iter(
                    numero_archivo_values
                ),
                None,
            )
        )

        nombres = []
        dnis = []
        cuits = []

        for row in group_rows:
            nombres.append(
                _clean(
                    row.get(
                        nombre_column
                    )
                )
            )

            dnis.append(
                _clean(
                    row.get(
                        dni_column
                    )
                )
            )

            cuits.append(
                _clean(
                    row.get(
                        cuit_column
                    )
                )
            )

        consolidated[
            identifier
        ] = {
            "id":
                identifier,

            "numero_archivo":
                numero_archivo,

            "expected": {
                "nombre_embargado":
                    _join_pipe(
                        nombres
                    ),

                "dni":
                    _join_pipe(
                        dnis
                    ),

                "cuit_cuil":
                    _join_pipe(
                        cuits
                    ),
            },

            "cantidad_personas":
                len(group_rows),
        }

    return consolidated


# ============================================================
# BASE DE EXTRACCIÓN
# ============================================================

def cargar_extraccion(
    path,
    *,
    id_column="id",
    numero_archivo_column="numero_archivo",
    fragmento_column="fragmento",
    texto_completo_column="texto_completo",
):
    """
    Carga fragmentos.csv y consolida por id.

    Devuelve para cada documento:

    {
        id,
        numero_archivo,
        text,
        fragments
    }

    `text` contiene texto_completo.
    `fragments` contiene todos los fragmentos del id.
    """

    rows = _leer_csv_semicolon(
        path
    )

    if not rows:
        raise ValueError(
            "La base de extracción está vacía"
        )

    required = {
        id_column,
        numero_archivo_column,
        fragmento_column,
        texto_completo_column,
    }

    missing = (
        required
        - set(rows[0].keys())
    )

    if missing:
        raise ValueError(
            "Faltan columnas en extracción: "
            + ", ".join(
                sorted(missing)
            )
        )

    grouped = defaultdict(list)

    for index, row in enumerate(rows):
        identifier = _clean(
            row.get(
                id_column
            )
        )

        if not identifier:
            raise ValueError(
                f"Extracción con id vacío "
                f"en fila {index}"
            )

        grouped[
            identifier
        ].append(row)

    consolidated = {}

    for identifier, group_rows in grouped.items():

        numero_archivo_values = {
            _clean(
                row.get(
                    numero_archivo_column
                )
            )
            for row in group_rows
            if _clean(
                row.get(
                    numero_archivo_column
                )
            )
        }

        if len(
            numero_archivo_values
        ) > 1:
            raise ValueError(
                f"El id {identifier} tiene "
                "más de un numero_archivo "
                "en la base de extracción"
            )

        numero_archivo = (
            next(
                iter(
                    numero_archivo_values
                ),
                None,
            )
        )

        textos_completos = {
            _clean(
                row.get(
                    texto_completo_column
                )
            )
            for row in group_rows
            if _clean(
                row.get(
                    texto_completo_column
                )
            )
        }

        if len(
            textos_completos
        ) > 1:
            raise ValueError(
                f"El id {identifier} tiene "
                "más de un texto_completo diferente"
            )

        texto_completo = (
            next(
                iter(
                    textos_completos
                ),
                "",
            )
        )

        if not texto_completo:
            raise ValueError(
                f"El id {identifier} "
                "no tiene texto_completo"
            )

        fragments = []

        for row in group_rows:
            fragment = _clean(
                row.get(
                    fragmento_column
                )
            )

            if fragment:
                fragments.append(
                    fragment
                )

        consolidated[
            identifier
        ] = {
            "id":
                identifier,

            "numero_archivo":
                numero_archivo,

            "text":
                texto_completo,

            "fragments":
                fragments,
        }

    return consolidated


# ============================================================
# MERGE GOLD + EXTRACCIÓN
# ============================================================

def construir_dataset(
    gold,
    extraccion,
):
    """
    Une gold y extracción por id.

    Sólo se conservan documentos con ground truth.

    Los ids presentes en extracción pero ausentes
    en gold se excluyen porque no pueden evaluarse.
    """

    examples = []

    ids_gold = set(
        gold.keys()
    )

    ids_extraccion = set(
        extraccion.keys()
    )

    common_ids = (
        ids_gold
        & ids_extraccion
    )

    if not common_ids:
        raise ValueError(
            "No existen IDs coincidentes "
            "entre gold y extracción"
        )

    missing_extraction = (
        ids_gold
        - ids_extraccion
    )

    if missing_extraction:
        raise ValueError(
            "Existen documentos gold "
            "sin texto de extracción: "
            + ", ".join(
                sorted(
                    missing_extraction
                )
            )
        )

    # Los que están en extracción pero no en gold
    # se excluyen intencionalmente.
    for identifier in sorted(
        common_ids
    ):
        gold_item = gold[
            identifier
        ]

        extraction_item = (
            extraccion[
                identifier
            ]
        )

        gold_numero = (
            gold_item.get(
                "numero_archivo"
            )
        )

        extraction_numero = (
            extraction_item.get(
                "numero_archivo"
            )
        )

        if (
            gold_numero
            and extraction_numero
            and gold_numero
            != extraction_numero
        ):
            raise ValueError(
                f"numero_archivo distinto "
                f"para id {identifier}: "
                f"gold={gold_numero}, "
                f"extraccion={extraction_numero}"
            )

        group = (
            gold_numero
            or extraction_numero
        )

        examples.append(
            {
                "id":
                    identifier,

                "group":
                    group,

                "text":
                    extraction_item[
                        "text"
                    ],

                "fragments":
                    list(
                        extraction_item[
                            "fragments"
                        ]
                    ),

                "expected":
                    dict(
                        gold_item[
                            "expected"
                        ]
                    ),

                "cantidad_personas":
                    gold_item[
                        "cantidad_personas"
                    ],
            }
        )

    return examples


# ============================================================
# SPLITS
# ============================================================

class _UnionFind:

    def __init__(
        self,
        size,
    ):
        self.parent = list(
            range(size)
        )

    def find(
        self,
        value,
    ):
        while (
            self.parent[value]
            != value
        ):
            self.parent[value] = (
                self.parent[
                    self.parent[
                        value
                    ]
                ]
            )

            value = (
                self.parent[
                    value
                ]
            )

        return value

    def union(
        self,
        a,
        b,
    ):
        root_a = self.find(
            a
        )

        root_b = self.find(
            b
        )

        if root_a != root_b:
            self.parent[
                root_b
            ] = root_a


def _construir_grupos(
    examples,
):
    """
    Evita leakage entre splits.

    Se mantienen juntos documentos que:

    - comparten group / numero_archivo;
    - tienen texto completo idéntico.
    """

    uf = _UnionFind(
        len(examples)
    )

    by_group = {}
    by_text = {}

    for index, example in enumerate(
        examples
    ):

        group = example.get(
            "group"
        )

        if group:
            normalized_group = (
                _clean(
                    group
                ).casefold()
            )

            if (
                normalized_group
                in by_group
            ):
                uf.union(
                    index,
                    by_group[
                        normalized_group
                    ],
                )
            else:
                by_group[
                    normalized_group
                ] = index

        normalized_text = (
            _normalizar_texto_grupo(
                example[
                    "text"
                ]
            )
        )

        if (
            normalized_text
            in by_text
        ):
            uf.union(
                index,
                by_text[
                    normalized_text
                ],
            )

        else:
            by_text[
                normalized_text
            ] = index

    groups = defaultdict(
        list
    )

    for index, example in enumerate(
        examples
    ):
        groups[
            uf.find(
                index
            )
        ].append(
            example
        )

    return list(
        groups.values()
    )


def dividir_dataset(
    examples,
    *,
    seed=42,
    split=None,
):
    """
    Divide documentos en:

    - train
    - validation
    - test

    La división se realiza por grupos
    para evitar leakage.
    """

    if not examples:
        raise ValueError(
            "No hay ejemplos para dividir"
        )

    if split is None:
        split = {
            "train":
                0.60,

            "validation":
                0.20,

            "test":
                0.20,
        }

    if (
        not isinstance(
            split,
            dict,
        )
        or set(
            split
        )
        != {
            "train",
            "validation",
            "test",
        }
    ):
        raise ValueError(
            "split debe contener exactamente "
            "train, validation y test"
        )

    train_ratio = float(
        split[
            "train"
        ]
    )

    validation_ratio = float(
        split[
            "validation"
        ]
    )

    test_ratio = float(
        split[
            "test"
        ]
    )

    if (
        train_ratio <= 0
        or validation_ratio <= 0
        or test_ratio <= 0
    ):
        raise ValueError(
            "Las proporciones deben ser positivas"
        )

    if abs(
        (
            train_ratio
            + validation_ratio
            + test_ratio
        )
        - 1.0
    ) > 1e-9:
        raise ValueError(
            "Las proporciones deben sumar 1"
        )

    groups = _construir_grupos(
        examples
    )

    if len(
        groups
    ) < 3:
        raise ValueError(
            "Se requieren al menos "
            "tres grupos independientes"
        )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        groups
    )

    total_groups = len(
        groups
    )

    train_count = max(
        1,
        round(
            total_groups
            * train_ratio
        ),
    )

    validation_count = max(
        1,
        round(
            total_groups
            * validation_ratio
        ),
    )

    # Garantizar al menos un grupo para test.
    if (
        train_count
        + validation_count
        >= total_groups
    ):
        excess = (
            train_count
            + validation_count
            - total_groups
            + 1
        )

        if (
            train_count
            >= validation_count
            and train_count
            - excess
            >= 1
        ):
            train_count -= (
                excess
            )

        elif (
            validation_count
            - excess
            >= 1
        ):
            validation_count -= (
                excess
            )

        else:
            raise ValueError(
                "No es posible crear "
                "train/validation/test "
                "no vacíos"
            )

    train_groups = groups[
        :train_count
    ]

    validation_groups = groups[
        train_count:
        train_count
        + validation_count
    ]

    test_groups = groups[
        train_count
        + validation_count:
    ]

    def flatten(
        group_list,
    ):
        return [
            example
            for group
            in group_list
            for example
            in group
        ]

    result = {
        "train":
            flatten(
                train_groups
            ),

        "validation":
            flatten(
                validation_groups
            ),

        "test":
            flatten(
                test_groups
            ),
    }

    if any(
        not values
        for values
        in result.values()
    ):
        raise ValueError(
            "Alguno de los splits "
            "quedó vacío"
        )

    return result


# ============================================================
# MANIFIESTO
# ============================================================

def manifiesto(
    splits,
):
    """
    Metadata reproducible de los splits.
    """

    result = {}

    for (
        split_name,
        examples,
    ) in splits.items():

        result[
            split_name
        ] = [
            {
                "id":
                    example[
                        "id"
                    ],

                "group":
                    example.get(
                        "group"
                    ),

                "text_hash":
                    _hash_texto(
                        example[
                            "text"
                        ]
                    ),

                "expected":
                    example[
                        "expected"
                    ],
            }

            for example
            in examples
        ]

    return result


# ============================================================
# CARGA COMPLETA
# ============================================================

def cargar_splits(
    config,
):
    """
    Punto principal utilizado por entrenar.py,
    evaluar.py y benchmark.py.

    Lee:

    - gold supervisado;
    - base de fragmentos / texto completo;

    los une por id y genera los splits.
    """

    datos = config[
        "datos"
    ]

    gold_config = datos[
        "gold"
    ]

    extraction_config = datos[
        "extraccion"
    ]

    gold = cargar_gold(
        ruta(
            gold_config[
                "ruta"
            ]
        ),
        id_column=gold_config[
            "columnas"
        ][
            "id"
        ],
        numero_archivo_column=(
            gold_config[
                "columnas"
            ][
                "numero_archivo"
            ]
        ),
        nombre_column=(
            gold_config[
                "columnas"
            ][
                "nombre_embargado"
            ]
        ),
        dni_column=(
            gold_config[
                "columnas"
            ][
                "dni"
            ]
        ),
        cuit_column=(
            gold_config[
                "columnas"
            ][
                "cuit_cuil"
            ]
        ),
    )

    extraccion = cargar_extraccion(
        ruta(
            extraction_config[
                "ruta"
            ]
        ),
        id_column=(
            extraction_config[
                "columnas"
            ][
                "id"
            ]
        ),
        numero_archivo_column=(
            extraction_config[
                "columnas"
            ][
                "numero_archivo"
            ]
        ),
        fragmento_column=(
            extraction_config[
                "columnas"
            ][
                "fragmento"
            ]
        ),
        texto_completo_column=(
            extraction_config[
                "columnas"
            ][
                "texto_completo"
            ]
        ),
    )

    examples = construir_dataset(
        gold,
        extraccion,
    )

    return dividir_dataset(
        examples,
        seed=datos[
            "seed"
        ],
        split=datos[
            "split"
        ],
    )


# ============================================================
# DSPY
# ============================================================

def como_dspy(
    examples,
):
    """
    Conversión opcional a dspy.Example.

    El texto jurídico no se usa como input
    del generador de schema GEPA.

    Esta función se mantiene como helper general.
    """

    try:
        import dspy

    except ImportError as exc:
        raise RuntimeError(
            "como_dspy requiere "
            "DSPy instalado"
        ) from exc

    converted = []

    for example in examples:

        expected = (
            example[
                "expected"
            ]
        )

        converted.append(
            dspy.Example(
                text=
                    example[
                        "text"
                    ],

                fragments=
                    example.get(
                        "fragments",
                        [],
                    ),

                nombre_embargado=
                    expected[
                        "nombre_embargado"
                    ],

                dni=
                    expected[
                        "dni"
                    ],

                cuit_cuil=
                    expected[
                        "cuit_cuil"
                    ],

                example_id=
                    example[
                        "id"
                    ],
            )
        )

    return converted