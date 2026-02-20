import json

def extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith(""):
        t = t[len(""):].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return json.loads(t)

def validate(schema_cls, text: str):
    data = extract_json(text)
    return schema_cls.model_validate(data)