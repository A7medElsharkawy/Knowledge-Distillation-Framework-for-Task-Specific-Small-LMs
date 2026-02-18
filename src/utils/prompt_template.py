from models.shcemes import NewsDetails
from pathlib import Path
from controllers import DataController
import json



def create_details_extraction_prompt(news_details: NewsDetails):

    example_story = DataController().load_example_story()
    details_extraction_prompt = [
        {
            "role": "system",
                 "content": "\n".join([
                "You are an NLP data parser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the output in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## story",
                example_story,
                "",

                "## Pydantic Details:",
                json.dumps(news_details.model_json_schema(), ensure_ascii=False),
                "",
                "## Story Details:",
                '```json'
            ])
        }
    ]
    return details_extraction_prompt