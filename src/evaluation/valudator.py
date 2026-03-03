import json
import json_repair
def extract_json(text: str) -> dict:
    t = text.strip()
    return json_repair.loads(t)

def validate(schema_cls, text: str):
    data = extract_json(text)
    return schema_cls.model_validate(data)