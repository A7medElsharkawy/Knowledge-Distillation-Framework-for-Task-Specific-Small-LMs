from urllib3 import response
from utils.prompt_template import create_details_extraction_prompt
from models.shcemes import NewsDetails
from models.enums import  ModelEnum
from controllers import ModelController

def eval_base_model():
    modelcontroller = ModelController()
    model, tokenizer = modelcontroller.load_model_and_tokenizer(ModelEnum.BASE_MODEL_QWEN.value)

    messages = create_details_extraction_prompt(NewsDetails)

    text = modelcontroller.apply_chat_templete(messages = messages,tokenizer = tokenizer)
    response = modelcontroller.model_output(text,tokenizer,model)

    return response

