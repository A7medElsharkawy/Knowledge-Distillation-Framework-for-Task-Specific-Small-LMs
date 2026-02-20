from utils.prompt_template import create_details_extraction_prompt
from utils.prompt_template import translation_messagges_prompt
from models.shcemes import ExtractNewsDetails
from models.shcemes import TranslationStory
from models.enums import  ModelEnum
from controllers import ModelController
from controllers import DataController
from evaluation import run_task,LocalRunner,OpenAIRunner

if __name__ == "__main__":
    model_name = ModelEnum.BASE_MODEL_QWEN.value
    target_lang = ModelEnum.TARGET_LANG.value
    text = DataController().load_example_story()
    messages_details_task = create_details_extraction_prompt(ExtractNewsDetails,text)
    messages_traslate_task = translation_messagges_prompt(TranslationStory,text=text,target_lang=target_lang)
    # output= EvaluateModel.eval_base_model(model_name=model_name,messages=messages_traslate_task)
    # print(output)

    openai_model = ModelEnum.OPENAI_MODEL.value
    runners = OpenAIRunner(openai_model)
    raw_output, validated = run_task(
        runner=runners,
        build_messages_fn=translation_messagges_prompt,
        schema_cls=TranslationStory,
        text=text,
        target_lang=target_lang,
    )
    print(raw_output)