from controllers import DataController
from controllers import BaseController
from tqdm.auto import tqdm
from models.enums import ModelEnum
from utils.prompt_template import create_details_extraction_prompt
from utils.prompt_template import translation_messagges_prompt
from models.shcemes import ExtractNewsDetails
from models.shcemes import TranslationStory
from evaluation import run_task
from utils import parse_json
import json

def prepare_rawdata(target_lang,runner,build_messages_fn,schema_cls,raw_data,output_file):


    id = 0
    for story in tqdm(raw_data):
 
        raw_output, _ = run_task(
            runner=runner,
            build_messages_fn=build_messages_fn,
            schema_cls=schema_cls,
            text=story['content'].strip(),
            target_lang=target_lang,
        )
        llm_resp_dict = parse_json(raw_output)

        if not llm_resp_dict:
            continue
        
        with open(output_file, "a", encoding="utf8") as dest:
            dest.write(json.dumps({
                "id": id,
                "story": story['content'].strip(),
                "task": "Extrat the story details into a JSON.",
                "output_scheme": json.dumps( schema_cls.model_json_schema(), ensure_ascii=False ),
                "response": llm_resp_dict,
            }, ensure_ascii=False, default=str)  + "\n" ) 
        
        id += 1

        if(id % 3) == 0:
            print(f"Processed {id} stories")
        



