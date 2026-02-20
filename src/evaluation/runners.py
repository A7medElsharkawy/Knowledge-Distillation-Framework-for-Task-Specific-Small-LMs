from controllers.ModelController import ModelController
from openai import OpenAI
from helper import get_settings
class LocalRunner:
    def __init__(self, model_name: str):

        self.mc = ModelController()
        self.model, self.tokenizer = self.mc.load_model_and_tokenizer(model_name)

    def generate(self, messages) -> str:
        prompt = self.mc.apply_chat_templete(messages, self.tokenizer)
        return self.mc.model_output(prompt, self.tokenizer, self.model)


class OpenAIRunner:
    def __init__(self, model_name: str):
        self.setting = get_settings()
        self.api_key = self.setting.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def generate(self, messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content