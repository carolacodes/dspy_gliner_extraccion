"""Normaliza valores extraídos, nunca modifica el documento ni sus offsets."""
import re
import unicodedata


def normalizar_nombre(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def normalizar_dni(value):
    return re.sub(r"[\s.\-]", "", str(value).strip())


def normalizar_cuit_cuil(value):
    return normalizar_dni(value)


def separar_valores(value):
    return [p.strip() for p in str(value or "").split("|") if p.strip()]


def unir_valores(values):
    return " | ".join(values)


def normalizar_valor(key, value):
    functions = {"nombre_embargado": normalizar_nombre, "dni": normalizar_dni, "cuit_cuil": normalizar_cuit_cuil}
    return functions[key](value)


def valores_normalizados(key, value):
    return {v for part in separar_valores(value) if (v := normalizar_valor(key, part))}
