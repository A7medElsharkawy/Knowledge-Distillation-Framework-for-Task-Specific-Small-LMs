from .valudator import validate
import inspect
def run_task(runner, build_messages_fn, schema_cls, **kwargs):
    """
        kwargs can include:
            runner: model runner
            build_messages_fn: fn for create a prompt
            schema_cls: pydantic schema
            text: text story
            target_lang: langauge 
            ...
        Passed dynamically to build_messages_fn.
    """  

    sig = inspect.signature(build_messages_fn)
    supported_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters}
    messages = build_messages_fn(schema_cls,**supported_args)  # your prompt_template style
    output = runner.generate(messages)
    validated = validate(schema_cls, output)
    return output, validated