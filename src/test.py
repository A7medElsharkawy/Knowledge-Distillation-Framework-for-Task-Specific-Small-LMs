from utils.prompt_template import create_details_extraction_prompt
from utils.prompt_template import translation_messagges_prompt
from models.shcemes import ExtractNewsDetails
from models.shcemes import TranslationStory
from models.enums import  ModelEnum
from controllers import ModelController
from controllers import DataController
from evaluation import run_task,LocalRunner,OpenAIRunner
from data.prepare_data import prepare_rawdata
import os
from data import format_data_for_finetuning
from data import split_data

if __name__ == "__main__":
    # model_name = ModelEnum.BASE_MODEL_QWEN.value
    # target_lang = ModelEnum.TARGET_LANG.value
    # text = DataController().load_example_story()
    # messages_details_task = create_details_extraction_prompt(ExtractNewsDetails,text)
    # messages_traslate_task = translation_messagges_prompt(TranslationStory,text=text,target_lang=target_lang)
    # # output= EvaluateModel.eval_base_model(model_name=model_name,messages=messages_traslate_task)
    # # print(output)

    # test opwn ai model
    # openai_model = ModelEnum.OPENAI_MODEL.value
    # runners = OpenAIRunner(openai_model)
    # raw_output, validated = run_task(
    #     runner=runners,
    #     build_messages_fn=translation_messagges_prompt,
    #     schema_cls=TranslationStory,
    #     text=text,
    #     target_lang=target_lang,
    # )

    # openai_model = ModelEnum.BASE_MODEL_QWEN.value
    # runners = LocalRunner(openai_model)
    # target_lang = ModelEnum.TARGET_LANG.value
    # raw_data = DataController().load_raw_data()
    output_file = os.path.join(DataController().processed_data_dir, "prepared_data.jsonl")

    # prepare_rawdata(
    #                 target_lang=target_lang,
    #                 runner=runners,
    #                 build_messages_fn=translation_messagges_prompt,
    #                 schema_cls=TranslationStory,
    #                 raw_data=raw_data,
    #                 output_file=output_file)

    #


    data = format_data_for_finetuning(output_file)
    split_data(data)

    