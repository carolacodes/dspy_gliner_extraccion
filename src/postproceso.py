"""Agrupación ordinal de spans, con normalizador inyectable y sin DSPy."""
from .utils import KEYS


def postprocesar(spans, normalizer=None):
    if normalizer is None:
        from .normalizacion import normalizar_valor
        normalizer = normalizar_valor
    grouped = {k: [] for k in KEYS}
    seen = {k: set() for k in KEYS}
    raw = [dict(s) for s in spans]
    for span in sorted(raw, key=lambda s: (s["start"], s["end"])):
        key = span["label"]
        if key not in grouped:
            continue
        value = normalizer(key, span["text"])
        if not value or value in seen[key]:
            continue
        seen[key].add(value)
        grouped[key].append({**span, "value": span["text"].strip() if key == "nombre_embargado" else value})
    count = max(map(len, grouped.values()), default=0)
    entities = [{k: grouped[k][i] if i < len(grouped[k]) else None for k in KEYS} for i in range(count)]
    result = {k: " | ".join(row[k]["value"] if row[k] else "" for row in entities) for k in KEYS}
    return {**result, "entidades": entities, "spans_raw": raw}
