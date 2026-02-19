from .BaseController import BaseController
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelController(BaseController):
        def __init__(self):
            super().__init__()
        
        def load_model_and_tokenizer(self,model_name: str):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype= None)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        
        def apply_chat_templete(self,messages,tokenizer:AutoTokenizer):

                text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt = True)

                return text

        def model_output(self,text : str,tokenizer:AutoTokenizer,model:AutoModelForCausalLM):
            model_inputs = tokenizer([text], return_tensors="pt")
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=False, top_k=None, temperature=None, top_p=None
            )
            generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
