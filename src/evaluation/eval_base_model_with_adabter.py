from controllers import ModelController,BaseController


def eval_base_model_with_adapter(model_name,messages):

    modelcontroller = ModelController()
    bc = BaseController()
    model, tokenizer = modelcontroller.load_model_and_tokenizer(model_name)
    adapter = bc.adapter_model
    model.load_adapter(adapter)
    output = modelcontroller.apply_chat_templete(messages = messages,tokenizer = tokenizer)
    response = modelcontroller.model_output(output,tokenizer,model)


    return response
