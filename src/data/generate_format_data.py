import json
import random

def format_data_for_finetuning(data_path) -> list[dict]:

    llm_finetunning_data = []

    system_message = '\n'.join([
        "you are a professional NLP data parser",
        "Follow the provided `Task` by the user and `Outpt Shema` to generate the `Output JSOn`.",
        "Do not generate any introduction or conclusion."
    ])

    for line in open(data_path):
        if line.strip() == "":
            continue

        rec = json.loads(line.strip())
        llm_finetunning_data.append(
            {
                "system" : system_message,
                "instruction" : "\n".join([
                    "# Story:",
                    rec['story'],
                    "# Task:",
                    rec['task'],
                    "# Output Schema:",
                    rec['output_scheme'],
                    "",
                    "# Output JSON:",
                    "```json"

                ]),
                "input" : "",
                "output": "\n".join([
            "```json",
            json.dumps(rec["response"], ensure_ascii=False, default=str),
            "```"
        ]),
        "history": []
            }
        )
    random.Random(101).shuffle(llm_finetunning_data)   
     
    return llm_finetunning_data