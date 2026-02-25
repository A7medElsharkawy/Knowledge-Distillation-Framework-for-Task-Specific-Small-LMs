import json_repair
def parse_json(text):
    try:
        return json_repair.loads(text)
    except:
        return None