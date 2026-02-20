from controllers import ModelController


def eval_base_model(model_name,messages):

    modelcontroller = ModelController()
    model, tokenizer = modelcontroller.load_model_and_tokenizer(model_name)
    output = modelcontroller.apply_chat_templete(messages = messages,tokenizer = tokenizer)
    response = modelcontroller.model_output(output,tokenizer,model)

    return response

