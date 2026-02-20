import json



def create_details_extraction_prompt(ExtractNewsDetails,text) -> list:


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
                text.strip(),
                "",

                "## Pydantic Details:",
                json.dumps(ExtractNewsDetails.model_json_schema(), ensure_ascii=False),
                "",
                "## Story Details:",
                '```json'
            ])
        }
    ]
    return details_extraction_prompt


def translation_messagges_prompt(TranslationStory,text,target_lang)-> str:
    
    translation_message = [
        {
            "role":"system",
            "content":'\n'.join([
                    "You are a professional translator.",
                    "You will be provided by an Arabic text.",
                    "You have to translate the text into the `Targeted Language`.",
                    "Follow the provided Scheme to generate a JSON",
                    "Do not generate any introduction or conclusion."])
        },
        {
            "role":"user",
            "content":'\n'.join([
                "## story",
                text.strip(),
                "",
                "## Pydantic Details:",
                json.dumps(TranslationStory.model_json_schema(), ensure_ascii=False),
                "",
                "## Targeted Langauge:",
                target_lang,
                "",
                "## Translated Story:",
                "```json"

            ])

        }
    ]
    return translation_message

