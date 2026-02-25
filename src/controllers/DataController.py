from .BaseController import BaseController
import os
import json
import random

class DataController(BaseController):
    def __init__(self):
        super().__init__()

    
    def load_example_story(self) -> str:
        example_path = os.path.join(self.raw_data_dir , "example.txt")
        with open(example_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def load_raw_data(self) -> list[str]:
        raw_data_path = os.path.join(self.raw_data_dir, "news-sample.jsonl")

        raw_data = []
        
        for line in open(raw_data_path, "r", encoding="utf-8"):
            if line.strip() == "":
                continue
            raw_data.append(json.loads(line.strip()))

        random.Random(42).shuffle(raw_data)
        return raw_data
        