from .BaseController import BaseController
import os


class DataController(BaseController):
    def __init__(self):
        super().__init__()

    
    def load_example_story(self) -> str:
        example_path = os.path.join(self.raw_data_dir , "example.txt")
        with open(example_path, "r", encoding="utf-8") as f:
            return f.read().strip()

