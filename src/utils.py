"""Utilidades compartidas, validación y persistencia.

Este módulo no importa GLiNER ni DSPy de forma pesada al cargarse.
Centraliza:

- contratos;
- validación de schema;
- configuración YAML;
- GEPA;
- LLM;
- dataset gold + extracción;
- CSV / JSON;
- thresholds;
- paths;
- creación del extractor;
- selección persistida.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Protocol


# ============================================================
# CONSTANTES
# ============================================================

KEYS = (
    "nombre_embargado",
    "dni",
    "cuit_cuil",
)

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# CONTRATOS
# ============================================================

class Extractor(Protocol):
    def predict(
        self,
        text: str,
        schema: Mapping[str, str],
        threshold: float,
    ) -> dict:
        ...


class Metric(Protocol):
    def __call__(
        self,
        expected: Mapping[str, str],
        predicted: Mapping[str, str],
    ) -> dict:
        """
        Debe devolver como mínimo:

        {
            "nombre_embargado": float,
            "dni": float,
            "cuit_cuil": float,
            "global": float
        }

        Todos los scores deben estar entre 0 y 1.
        """
        ...


# ============================================================
# SCHEMA
# ============================================================

def validar_schema(schema):
    """
    Valida que el schema contenga exactamente
    las tres entidades canónicas.
    """
    if (
        not isinstance(schema, dict)
        or set(schema) != set(KEYS)
    ):
        raise ValueError(
            f"El schema debe contener exactamente {KEYS}"
        )

    result = {}

    for key in KEYS:
        value = schema[key]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"La descripción de {key} "
                "debe ser un string no vacío"
            )

        value = value.strip()

        if len(value) > 2000:
            raise ValueError(
                f"La descripción de {key} "
                "supera 2000 caracteres"
            )

        result[key] = value

    descriptions = {
        value.casefold()
        for value in result.values()
    }

    if len(descriptions) != len(KEYS):
        raise ValueError(
            "Las tres descripciones del schema "
            "deben ser distintas"
        )

    return result


# ============================================================
# JSON
# ============================================================

def leer_json(path):
    path = Path(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def guardar_json(path, data):
    """
    Escritura atómica simple.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temp.replace(path)


def cargar_schema(path):
    return validar_schema(
        leer_json(path)
    )


# ============================================================
# HASH / REPRODUCIBILIDAD
# ============================================================

def fingerprint(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


# ============================================================
# THRESHOLD
# ============================================================

def validar_threshold(value):
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (float, int),
        )
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            "Threshold debe ser un número entre 0 y 1"
        )

    return float(value)


def threshold_grid(
    start,
    stop,
    step,
):
    """
    Genera thresholds evitando errores acumulativos
    de punto flotante.
    """
    from decimal import Decimal

    validar_threshold(start)
    validar_threshold(stop)

    if (
        isinstance(step, bool)
        or not isinstance(
            step,
            (float, int),
        )
        or not math.isfinite(step)
        or step <= 0
        or stop < start
    ):
        raise ValueError(
            "Rango de thresholds inválido"
        )

    current = Decimal(
        str(start)
    )

    end = Decimal(
        str(stop)
    )

    delta = Decimal(
        str(step)
    )

    result = []

    while current <= end:
        result.append(
            float(current)
        )

        if len(result) > 10000:
            raise ValueError(
                "Demasiados thresholds"
            )

        current += delta

    return result


# ============================================================
# VALIDACIÓN LLM
# ============================================================

def _validar_llm(
    config,
    section_name,
):
    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"La sección {section_name} "
            "debe ser un objeto"
        )

    model = config.get(
        "modelo"
    )

    if (
        not isinstance(model, str)
        or not model.strip()
    ):
        raise ValueError(
            f"{section_name}.modelo es obligatorio"
        )

    provider = config.get(
        "provider"
    )

    if (
        provider is not None
        and (
            not isinstance(provider, str)
            or not provider.strip()
        )
    ):
        raise ValueError(
            f"{section_name}.provider inválido"
        )

    api_base = config.get(
        "api_base"
    )

    if (
        api_base is not None
        and (
            not isinstance(api_base, str)
            or not api_base.strip()
        )
    ):
        raise ValueError(
            f"{section_name}.api_base inválido"
        )

    temperature = config.get(
        "temperature"
    )

    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(
                temperature,
                (float, int),
            )
            or not math.isfinite(
                temperature
            )
            or temperature < 0
        ):
            raise ValueError(
                f"{section_name}.temperature inválido"
            )

    max_tokens = config.get(
        "max_tokens"
    )

    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(
                max_tokens,
                int,
            )
            or max_tokens < 1
        ):
            raise ValueError(
                f"{section_name}.max_tokens inválido"
            )

    api_key_env = config.get(
        "api_key_env"
    )

    if (
        api_key_env is not None
        and (
            not isinstance(
                api_key_env,
                str,
            )
            or not api_key_env.strip()
        )
    ):
        raise ValueError(
            f"{section_name}.api_key_env inválido"
        )


# ============================================================
# VALIDACIÓN GEPA
# ============================================================

def _validar_gepa(config):
    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "La sección gepa debe ser un objeto"
        )

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
        if auto not in {
            "light",
            "medium",
            "heavy",
        }:
            raise ValueError(
                "gepa.auto debe ser "
                "light, medium o heavy"
            )

    if max_full_evals is not None:
        if (
            isinstance(
                max_full_evals,
                bool,
            )
            or not isinstance(
                max_full_evals,
                int,
            )
            or max_full_evals < 1
        ):
            raise ValueError(
                "gepa.max_full_evals "
                "debe ser un entero >= 1"
            )

    if max_metric_calls is not None:
        if (
            isinstance(
                max_metric_calls,
                bool,
            )
            or not isinstance(
                max_metric_calls,
                int,
            )
            or max_metric_calls < 1
        ):
            raise ValueError(
                "gepa.max_metric_calls "
                "debe ser un entero >= 1"
            )

    num_threads = config.get(
        "num_threads",
        1,
    )

    if (
        isinstance(num_threads, bool)
        or not isinstance(
            num_threads,
            int,
        )
        or num_threads < 1
    ):
        raise ValueError(
            "gepa.num_threads debe ser >= 1"
        )

    seed = config.get(
        "seed",
        0,
    )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError(
            "gepa.seed debe ser un entero"
        )

    minibatch = config.get(
        "reflection_minibatch_size",
        3,
    )

    if (
        isinstance(minibatch, bool)
        or not isinstance(
            minibatch,
            int,
        )
        or minibatch < 1
    ):
        raise ValueError(
            "gepa.reflection_minibatch_size "
            "debe ser >= 1"
        )

    strategy = config.get(
        "candidate_selection_strategy",
        "pareto",
    )

    if (
        not isinstance(strategy, str)
        or not strategy.strip()
    ):
        raise ValueError(
            "gepa.candidate_selection_strategy inválido"
        )

    skip = config.get(
        "skip_perfect_score",
        True,
    )

    if not isinstance(
        skip,
        bool,
    ):
        raise ValueError(
            "gepa.skip_perfect_score "
            "debe ser booleano"
        )

    log_dir = config.get(
        "log_dir"
    )

    if (
        log_dir is not None
        and (
            not isinstance(log_dir, str)
            or not log_dir.strip()
        )
    ):
        raise ValueError(
            "gepa.log_dir inválido"
        )


# ============================================================
# VALIDACIÓN SPLITS
# ============================================================

def _validar_splits(datos):
    split = datos.get(
        "split"
    )

    if not isinstance(
        split,
        dict,
    ):
        raise ValueError(
            "datos.split debe ser un objeto"
        )

    required = {
        "train",
        "validation",
        "test",
    }

    if set(split) != required:
        raise ValueError(
            "datos.split debe contener exactamente "
            "train, validation y test"
        )

    values = [
        split["train"],
        split["validation"],
        split["test"],
    ]

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (float, int),
            )
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "Las proporciones del split "
                "deben ser números positivos"
            )

    if not math.isclose(
        sum(values),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Las proporciones train/validation/test "
            "deben sumar 1"
        )

    seed = datos.get(
        "seed"
    )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError(
            "datos.seed debe ser un entero"
        )


# ============================================================
# VALIDACIÓN FUENTES DE DATOS
# ============================================================

def _validar_gold(datos):
    gold = datos.get(
        "gold"
    )

    if not isinstance(
        gold,
        dict,
    ):
        raise ValueError(
            "datos.gold debe ser un objeto"
        )

    path = gold.get(
        "ruta"
    )

    if (
        not isinstance(path, str)
        or not path.strip()
    ):
        raise ValueError(
            "datos.gold.ruta es obligatoria"
        )

    columns = gold.get(
        "columnas"
    )

    if not isinstance(
        columns,
        dict,
    ):
        raise ValueError(
            "datos.gold.columnas debe ser un objeto"
        )

    required = (
        "id",
        "numero_archivo",
        "nombre_embargado",
        "dni",
        "cuit_cuil",
    )

    for key in required:
        value = columns.get(
            key
        )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"Falta datos.gold.columnas.{key}"
            )

    column_values = [
        columns[key].strip()
        for key in required
    ]

    if len(
        set(column_values)
    ) != len(
        column_values
    ):
        raise ValueError(
            "Las columnas configuradas en gold "
            "deben ser distintas"
        )


def _validar_extraccion(datos):
    extraction = datos.get(
        "extraccion"
    )

    if not isinstance(
        extraction,
        dict,
    ):
        raise ValueError(
            "datos.extraccion debe ser un objeto"
        )

    path = extraction.get(
        "ruta"
    )

    if (
        not isinstance(path, str)
        or not path.strip()
    ):
        raise ValueError(
            "datos.extraccion.ruta es obligatoria"
        )

    columns = extraction.get(
        "columnas"
    )

    if not isinstance(
        columns,
        dict,
    ):
        raise ValueError(
            "datos.extraccion.columnas "
            "debe ser un objeto"
        )

    required = (
        "id",
        "numero_archivo",
        "fragmento",
        "texto_completo",
    )

    for key in required:
        value = columns.get(
            key
        )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"Falta datos.extraccion.columnas.{key}"
            )

    column_values = [
        columns[key].strip()
        for key in required
    ]

    if len(
        set(column_values)
    ) != len(
        column_values
    ):
        raise ValueError(
            "Las columnas configuradas en extracción "
            "deben ser distintas"
        )


def _validar_modo_entrada(datos):
    mode = datos.get(
        "modo_entrada"
    )

    allowed = {
        "texto_completo",
        "fragmentos",
    }

    if mode not in allowed:
        raise ValueError(
            "datos.modo_entrada debe ser "
            "'texto_completo' o 'fragmentos'"
        )


def _validar_datos(datos):
    if not isinstance(
        datos,
        dict,
    ):
        raise ValueError(
            "La sección datos debe ser un objeto"
        )

    _validar_modo_entrada(
        datos
    )

    _validar_gold(
        datos
    )

    _validar_extraccion(
        datos
    )

    _validar_splits(
        datos
    )


# ============================================================
# VALIDACIÓN GLINER
# ============================================================

def _validar_gliner(config):
    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "La sección gliner debe ser un objeto"
        )

    model = config.get(
        "modelo"
    )

    if (
        not isinstance(model, str)
        or not model.strip()
    ):
        raise ValueError(
            "gliner.modelo es obligatorio"
        )

    device = config.get(
        "device"
    )

    if (
        not isinstance(device, str)
        or not device.strip()
    ):
        raise ValueError(
            "gliner.device inválido"
        )

    validar_threshold(
        config.get(
            "threshold_inicial"
        )
    )

    for key in (
        "flat_ner",
        "multi_label",
        "local_files_only",
    ):
        if not isinstance(
            config.get(key),
            bool,
        ):
            raise ValueError(
                f"gliner.{key} debe ser booleano"
            )

    chunk_words = config.get(
        "chunk_words"
    )

    overlap_words = config.get(
        "overlap_words"
    )

    if (
        isinstance(chunk_words, bool)
        or not isinstance(
            chunk_words,
            int,
        )
        or chunk_words < 1
        or isinstance(
            overlap_words,
            bool,
        )
        or not isinstance(
            overlap_words,
            int,
        )
        or not 0 <= overlap_words < chunk_words
    ):
        raise ValueError(
            "Ventanas GLiNER inválidas"
        )


# ============================================================
# VALIDACIÓN OPTIMIZACIÓN
# ============================================================

def _validar_optimizacion(config):
    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "La sección optimizacion "
            "debe ser un objeto"
        )

    optimize = config.get(
        "optimizar_threshold"
    )

    if not isinstance(
        optimize,
        bool,
    ):
        raise ValueError(
            "optimizacion.optimizar_threshold "
            "debe ser booleano"
        )

    threshold_grid(
        config.get(
            "threshold_min"
        ),
        config.get(
            "threshold_max"
        ),
        config.get(
            "threshold_paso"
        ),
    )


# ============================================================
# VALIDACIÓN RUTAS
# ============================================================

def _validar_rutas(config):
    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "La sección rutas debe ser un objeto"
        )

    for key in (
        "schema_inicial",
        "schema_optimizado",
        "output",
    ):
        value = config.get(
            key
        )

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"rutas.{key} inválida"
            )


# ============================================================
# CONFIG PRINCIPAL
# ============================================================

def cargar_config(
    path=ROOT / "config/gliner_embargo.yaml",
):
    """
    Carga y valida toda la configuración.
    """
    import yaml

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe config: {path}"
        )

    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "La configuración YAML es inválida"
        )

    required_sections = (
        "gliner",
        "llm",
        "gepa",
        "datos",
        "optimizacion",
        "rutas",
    )

    for section in required_sections:
        if not isinstance(
            config.get(section),
            dict,
        ):
            raise ValueError(
                f"Falta sección {section}"
            )

    _validar_gliner(
        config["gliner"]
    )

    _validar_llm(
        config["llm"],
        "llm",
    )

    if config.get(
        "llm_reflexion"
    ) is not None:
        _validar_llm(
            config["llm_reflexion"],
            "llm_reflexion",
        )

    _validar_gepa(
        config["gepa"]
    )

    _validar_datos(
        config["datos"]
    )

    _validar_optimizacion(
        config["optimizacion"]
    )

    _validar_rutas(
        config["rutas"]
    )

    return config


# ============================================================
# PATHS
# ============================================================

def ruta(value):
    """
    Las rutas relativas se resuelven desde
    la raíz del proyecto.
    """
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


# ============================================================
# CSV GENÉRICO
# ============================================================

def leer_csv(path):
    """
    CSV genérico separado por coma.

    Se conserva principalmente para:
    - benchmark;
    - inferencia;
    - outputs.

    El dataset supervisado real utiliza
    su propio loader en dataset.py.
    """
    path = Path(path)

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream
        )

        if (
            not reader.fieldnames
            or len(
                set(
                    reader.fieldnames
                )
            )
            != len(
                reader.fieldnames
            )
        ):
            raise ValueError(
                "CSV sin cabecera "
                "o con columnas repetidas"
            )

        return list(
            reader
        )


def guardar_csv(
    path,
    rows,
):
    rows = list(
        rows
    )

    if not rows:
        raise ValueError(
            "No hay resultados para escribir"
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            value,
                            ensure_ascii=False,
                        )
                        if isinstance(
                            value,
                            (dict, list),
                        )
                        else value
                    )
                    for key, value
                    in row.items()
                }
            )


# ============================================================
# CREACIÓN EXTRACTOR
# ============================================================

def crear_extractor(
    config,
    schema,
):
    """
    Import diferido para evitar cargar GLiNER
    durante tests livianos.
    """
    from .gliner_module import GLiNERModule

    cfg = config[
        "gliner"
    ]

    return GLiNERModule(
        schema=schema,

        threshold=cfg[
            "threshold_inicial"
        ],

        model_name=cfg[
            "modelo"
        ],

        device=cfg[
            "device"
        ],

        chunk_words=cfg[
            "chunk_words"
        ],

        overlap_words=cfg[
            "overlap_words"
        ],

        flat_ner=cfg[
            "flat_ner"
        ],

        multi_label=cfg[
            "multi_label"
        ],

        local_files_only=cfg[
            "local_files_only"
        ],
    )


# ============================================================
# SELECCIÓN OPTIMIZADA
# ============================================================

def cargar_seleccion(
    config,
    schema_path,
    threshold=None,
):
    """
    Carga schema + threshold.

    Si se utiliza el schema optimizado,
    valida también seleccion.json.
    """
    schema = cargar_schema(
        schema_path
    )

    if threshold is not None:
        return (
            schema,
            validar_threshold(
                threshold
            ),
        )

    optimized_path = ruta(
        config[
            "rutas"
        ][
            "schema_optimizado"
        ]
    ).resolve()

    current_path = Path(
        schema_path
    ).resolve()

    if current_path == optimized_path:
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

        if not selection_path.exists():
            raise ValueError(
                "No existe seleccion.json. "
                "El schema optimizado todavía "
                "no fue generado experimentalmente."
            )

        selection = leer_json(
            selection_path
        )

        if (
            selection.get(
                "schema_hash"
            )
            != fingerprint(
                schema
            )
        ):
            raise ValueError(
                "Schema y seleccion.json "
                "no coinciden"
            )

        if (
            selection.get(
                "gliner"
            )
            != config[
                "gliner"
            ]
        ):
            raise ValueError(
                "La configuración GLiNER "
                "es distinta de la utilizada "
                "durante la optimización"
            )

        return (
            schema,
            validar_threshold(
                selection[
                    "threshold_final"
                ]
            ),
        )

    return (
        schema,
        config[
            "gliner"
        ][
            "threshold_inicial"
        ],
    )