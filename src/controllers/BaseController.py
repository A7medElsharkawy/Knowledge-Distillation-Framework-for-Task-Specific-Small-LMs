import os
from helper import get_settings

class BaseController:
    def __init__(self):
        self.settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname((os.path.dirname(__file__))))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')



