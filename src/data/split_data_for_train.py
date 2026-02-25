import os
from controllers import BaseController
import json

def split_data(data):
    data_lenght = len(data)
    train_value = int(data_lenght * .8)
    train_ds = data[:train_value]
    eval_ds = data[train_value:]
    data_dir = BaseController().processed_data_dir

    with open(os.path.join(data_dir,"train.json"), "w") as dest:
        json.dump(train_ds, dest, ensure_ascii=False, default=str)

    with open(os.path.join(data_dir,  "val.json"), "w", encoding="utf8") as dest:
        json.dump(eval_ds, dest, ensure_ascii=False, default=str)
