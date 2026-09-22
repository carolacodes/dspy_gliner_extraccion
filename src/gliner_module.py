"""Adaptador GLiNER2 para extracción structured.

Arquitectura:

AutoExtractor.from_pretrained(...)
        ↓
model.create_schema()
        ↓
.structure("persona_embargada")
        ↓
.field(...)
        ↓
model.extract(...)

Se mantiene un contrato canónico para el resto del proyecto:

- nombre_embargado
- dni
- cuit_cuil

Internamente, el schema GLiNER2 utiliza los nombres del V7:

- nombre_embargado
- dni_embargado
- cuit_cuil_embargado
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .utils import (
    KEYS,
    validar_schema,
    validar_threshold,
)


# ============================================================
# CONSTANTES
# ============================================================

STRUCTURE_NAME = "persona_embargada"


FIELD_MAP = {
    "nombre_embargado":
        "nombre_embargado",

    "dni":
        "dni_embargado",

    "cuit_cuil":
        "cuit_cuil_embargado",
}


REVERSE_FIELD_MAP = {
    native:
        canonical
    for canonical, native
    in FIELD_MAP.items()
}


DNI_REGEX = (
    r"^\s*"
    r"(?:(?:D\.?\s*N\.?\s*I\.?|"
    r"DOCUMENTO(?:\s+NACIONAL\s+DE\s+IDENTIDAD)?)"
    r"\s*(?:(?:NRO|N[º°o])\.?\s*)?"
    r"[:#-]?\s*)?"
    r"\d{1,2}(?:[.\s-]?\d{3}){2}"
    r"\s*$"
)


CUIT_CUIL_REGEX = (
    r"^\s*"
    r"(?:(?:C\.?\s*U\.?\s*I\.?\s*L\.?|"
    r"C\.?\s*U\.?\s*I\.?\s*T\.?|"
    r"CUIL\s*/\s*CUIT|"
    r"CUIT\s*/\s*CUIL)"
    r"\s*(?:(?:NRO|N[º°o])\.?\s*)?"
    r"[:#-]?\s*)?"
    r"\d{2}(?:[-.\s]?\d){8}[-.\s]?\d"
    r"\s*$"
)


# ============================================================
# HELPERS
# ============================================================

def _valor_simple(
    value: Any,
) -> str:
    """
    Obtiene el texto de un field GLiNER2.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return value.strip()

    if isinstance(
        value,
        (int, float),
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "value",
            "text",
            "span",
        ):
            candidate = (
                value.get(key)
            )

            if candidate is not None:
                return str(
                    candidate
                ).strip()

        return ""

    return str(
        value
    ).strip()


def _valor_campo(
    value: Any,
) -> str:
    """
    Convierte un field GLiNER2 en texto.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        list,
    ):
        values = [
            _valor_simple(
                item
            )
            for item in value
        ]

        values = [
            item
            for item in values
            if item
        ]

        return " | ".join(
            values
        )

    return _valor_simple(
        value
    )


def _extraer_spans_campo(
    value: Any,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    """
    Conserva confidence y offsets si GLiNER2
    los devuelve.
    """

    values = (
        value
        if isinstance(
            value,
            list,
        )
        else [value]
    )

    result: list[
        dict[str, Any]
    ] = []

    for item in values:

        if not isinstance(
            item,
            dict,
        ):
            continue

        text = _valor_simple(
            item
        )

        span: dict[
            str,
            Any,
        ] = {
            "label":
                field_name,

            "text":
                text,

            "score":
                item.get(
                    "confidence",
                    item.get(
                        "score"
                    ),
                ),

            "start":
                item.get(
                    "start"
                ),

            "end":
                item.get(
                    "end"
                ),
        }

        span = {
            key: value
            for key, value
            in span.items()
            if value is not None
        }

        if (
            text
            or "start" in span
            or "end" in span
        ):
            result.append(
                span
            )

    return result


# ============================================================
# GLINER MODULE
# ============================================================

class GLiNERModule:
    """
    Wrapper reutilizable para GLiNER2 structured.

    El modelo se carga una sola vez.

    GEPA puede enviar distintos schemas candidatos,
    pero todos se transforman al mismo structure
    persona_embargada.
    """

    def __init__(
        self,
        schema,
        threshold=0.55,
        *,
        model_name=(
            "fastino/"
            "gliner2-privacy-filter-PII-multi"
        ),
        model=None,
        device="cpu",
        field_thresholds: Mapping[
            str,
            float,
        ] | None = None,

        # Compatibilidad temporal con configuración anterior.
        chunk_words=160,
        overlap_words=40,
        flat_ner=True,
        multi_label=False,
        local_files_only=False,

        use_validators=True,
    ):
        self.schema = validar_schema(
            schema
        )

        self.threshold = (
            validar_threshold(
                threshold
            )
        )

        self.model_name = (
            model_name
        )

        self.device = device

        self.use_validators = bool(
            use_validators
        )

        # ----------------------------------------------------
        # THRESHOLDS POR FIELD
        # ----------------------------------------------------

        defaults = {
            "nombre_embargado":
                0.60,

            "dni":
                0.50,

            "cuit_cuil":
                0.50,
        }

        if field_thresholds:
            defaults.update(
                field_thresholds
            )

        self.field_thresholds = {
            key:
                validar_threshold(
                    defaults[key]
                )
            for key in KEYS
        }

        # ----------------------------------------------------
        # Compatibilidad
        # ----------------------------------------------------

        self.chunk_words = (
            chunk_words
        )

        self.overlap_words = (
            overlap_words
        )

        self.flat_ner = (
            flat_ner
        )

        self.multi_label = (
            multi_label
        )

        self.local_files_only = (
            local_files_only
        )

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        if model is None:
            model = (
                self._cargar_modelo(
                    model_name=
                        model_name,

                    device=
                        device,
                )
            )

        self.model = model

        # Cache de schema nativo.
        self._native_schema = None

        self._native_schema_source = (
            None
        )


    # ========================================================
    # MODELO
    # ========================================================

    @staticmethod
    def _cargar_modelo(
        *,
        model_name,
        device,
    ):
        """
        Misma estrategia que identificacion_roles.
        """

        try:
            from gliner2 import (
                AutoExtractor,
            )

        except ImportError as exc:
            raise RuntimeError(
                "No se pudo importar gliner2. "
                "Instalá GLiNER2 antes de "
                "ejecutar inferencia real."
            ) from exc

        try:
            return (
                AutoExtractor
                .from_pretrained(
                    model_name,
                    map_location=device,
                )
            )

        except Exception as auto_error:

            try:
                from gliner2 import (
                    GLiNER2,
                )

                return (
                    GLiNER2
                    .from_pretrained(
                        model_name,
                        map_location=device,
                    )
                )

            except Exception as fallback_error:
                raise RuntimeError(
                    "No se pudo cargar "
                    f"GLiNER2 '{model_name}'. "
                    f"AutoExtractor: "
                    f"{auto_error}. "
                    "GLiNER2 fallback: "
                    f"{fallback_error}"
                ) from fallback_error


    # ========================================================
    # UPDATE
    # ========================================================

    def actualizar_schema(
        self,
        schema,
    ):
        self.schema = validar_schema(
            schema
        )

        self._invalidar_cache()


    def actualizar_threshold(
        self,
        threshold,
    ):
        self.threshold = (
            validar_threshold(
                threshold
            )
        )


    def actualizar_field_thresholds(
        self,
        thresholds,
    ):
        if not isinstance(
            thresholds,
            Mapping,
        ):
            raise ValueError(
                "field_thresholds "
                "debe ser un mapping"
            )

        updated = dict(
            self.field_thresholds
        )

        for key, value in (
            thresholds.items()
        ):
            if key not in KEYS:
                raise ValueError(
                    "Field threshold "
                    f"desconocido: {key}"
                )

            updated[key] = (
                validar_threshold(
                    value
                )
            )

        self.field_thresholds = (
            updated
        )

        self._invalidar_cache()


    def _invalidar_cache(
        self,
    ):
        self._native_schema = None

        self._native_schema_source = (
            None
        )


    # ========================================================
    # VALIDATORS
    # ========================================================

    def _crear_validadores(
        self,
        canonical_field,
    ):
        if not self.use_validators:
            return []

        if canonical_field not in {
            "dni",
            "cuit_cuil",
        }:
            return []

        try:
            from gliner2 import (
                RegexValidator,
            )

        except ImportError as exc:
            raise RuntimeError(
                "No se pudo importar "
                "RegexValidator de gliner2"
            ) from exc

        if canonical_field == "dni":

            return [
                RegexValidator(
                    DNI_REGEX,
                    mode="full",
                    exclude=False,
                    flags=re.IGNORECASE,
                )
            ]

        return [
            RegexValidator(
                CUIT_CUIL_REGEX,
                mode="full",
                exclude=False,
                flags=re.IGNORECASE,
            )
        ]


    # ========================================================
    # SCHEMA NATIVO
    # ========================================================

    def _construir_schema_nativo(
        self,
        schema,
    ):
        """
        Convierte:

        nombre_embargado
        dni
        cuit_cuil

        a fields GLiNER2:

        nombre_embargado
        dni_embargado
        cuit_cuil_embargado
        """

        schema = validar_schema(
            schema
        )

        if not hasattr(
            self.model,
            "create_schema",
        ):
            raise RuntimeError(
                "El modelo cargado no expone "
                "create_schema(). "
                "No parece ser un GLiNER2 "
                "structured compatible."
            )

        native = (
            self.model
            .create_schema()
        )

        structure = (
            native.structure(
                STRUCTURE_NAME
            )
        )

        # ----------------------------------------------------
        # NOMBRE EMBARGADO
        # ----------------------------------------------------

        structure = structure.field(
            FIELD_MAP[
                "nombre_embargado"
            ],

            dtype="str",

            description=
                schema[
                    "nombre_embargado"
                ],

            threshold=
                self.field_thresholds[
                    "nombre_embargado"
                ],
        )

        # ----------------------------------------------------
        # DNI EMBARGADO
        # ----------------------------------------------------

        dni_kwargs: dict[
            str,
            Any,
        ] = {
            "dtype":
                "str",

            "description":
                schema[
                    "dni"
                ],

            "threshold":
                self.field_thresholds[
                    "dni"
                ],
        }

        validators = (
            self._crear_validadores(
                "dni"
            )
        )

        if validators:
            dni_kwargs[
                "validators"
            ] = validators

        structure = (
            structure.field(
                FIELD_MAP[
                    "dni"
                ],
                **dni_kwargs,
            )
        )

        # ----------------------------------------------------
        # CUIT / CUIL EMBARGADO
        # ----------------------------------------------------

        cuit_kwargs: dict[
            str,
            Any,
        ] = {
            "dtype":
                "str",

            "description":
                schema[
                    "cuit_cuil"
                ],

            "threshold":
                self.field_thresholds[
                    "cuit_cuil"
                ],
        }

        validators = (
            self._crear_validadores(
                "cuit_cuil"
            )
        )

        if validators:
            cuit_kwargs[
                "validators"
            ] = validators

        structure = (
            structure.field(
                FIELD_MAP[
                    "cuit_cuil"
                ],
                **cuit_kwargs,
            )
        )

        return structure


    def _obtener_schema_nativo(
        self,
        schema,
    ):
        """
        Cache del schema GLiNER2.

        El threshold global NO forma parte
        del schema structured.

        Los thresholds por field sí.
        """

        schema = validar_schema(
            schema
        )

        source = (
            tuple(
                (
                    key,
                    schema[key],
                )
                for key in KEYS
            ),

            tuple(
                (
                    key,
                    self.field_thresholds[
                        key
                    ],
                )
                for key in KEYS
            ),
        )

        if (
            self._native_schema
            is not None
            and self._native_schema_source
            == source
        ):
            return (
                self._native_schema
            )

        native_schema = (
            self._construir_schema_nativo(
                schema
            )
        )

        self._native_schema = (
            native_schema
        )

        self._native_schema_source = (
            source
        )

        return native_schema


    # ========================================================
    # NORMALIZAR RESPUESTA
    # ========================================================

    def _normalizar_respuesta(
        self,
        raw_response,
    ):
        empty: dict[
            str,
            Any,
        ] = {
            "nombre_embargado":
                "",

            "dni":
                "",

            "cuit_cuil":
                "",

            "entidades":
                [],

            "spans_raw":
                [],
        }

        if raw_response is None:
            return empty

        if not isinstance(
            raw_response,
            dict,
        ):
            raise ValueError(
                "GLiNER2 devolvió una "
                "respuesta structured "
                "que no es dict"
            )

        structures = (
            raw_response.get(
                STRUCTURE_NAME
            )
        )

        if structures is None:
            return empty

        if not isinstance(
            structures,
            list,
        ):
            structures = [
                structures
            ]

        entities: list[
            dict[str, str]
        ] = []

        spans_raw: list[
            dict[str, Any]
        ] = []

        for structure_index, item in enumerate(
            structures
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            entity: dict[
                str,
                str,
            ] = {}

            # ------------------------------------------------
            # Convertir fields nativos → contrato canónico.
            # ------------------------------------------------

            for (
                canonical,
                native,
            ) in FIELD_MAP.items():

                entity[
                    canonical
                ] = _valor_campo(
                    item.get(
                        native
                    )
                )

            if not any(
                entity.values()
            ):
                continue

            entities.append(
                entity
            )

            # ------------------------------------------------
            # Spans
            # ------------------------------------------------

            for (
                canonical,
                native,
            ) in FIELD_MAP.items():

                field_spans = (
                    _extraer_spans_campo(
                        item.get(
                            native
                        ),

                        field_name=
                            canonical,
                    )
                )

                for span in field_spans:

                    span[
                        "structure_index"
                    ] = (
                        structure_index
                    )

                    span[
                        "native_field"
                    ] = native

                    spans_raw.append(
                        span
                    )

        if not entities:
            return empty

        result: dict[
            str,
            Any,
        ] = {
            key:
                " | ".join(
                    str(
                        entity.get(
                            key,
                            "",
                        )
                        or ""
                    ).strip()
                    for entity
                    in entities
                )

            for key in KEYS
        }

        result[
            "entidades"
        ] = entities

        result[
            "spans_raw"
        ] = spans_raw

        return result


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        text,
        schema=None,
        threshold=None,
    ):
        """
        Ejecuta GLiNER2.

        threshold:
            threshold global de model.extract()

        field_thresholds:
            thresholds definidos dentro
            del schema structured.
        """

        schema = (
            self.schema
            if schema is None
            else validar_schema(
                schema
            )
        )

        threshold = (
            self.threshold
            if threshold is None
            else validar_threshold(
                threshold
            )
        )

        text = str(
            text
            or ""
        ).strip()

        if not text:
            return {
                "nombre_embargado":
                    "",

                "dni":
                    "",

                "cuit_cuil":
                    "",

                "entidades":
                    [],

                "spans_raw":
                    [],
            }

        native_schema = (
            self._obtener_schema_nativo(
                schema
            )
        )

        raw_response = (
            self.model.extract(
                text,
                native_schema,

                include_confidence=True,

                include_spans=True,

                threshold=threshold,
            )
        )

        return (
            self._normalizar_respuesta(
                raw_response
            )
        )