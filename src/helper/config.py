from pydantic_settings import BaseSettings

class Setting(BaseSettings):

    HUGGINGFACE_TOKEN: str
    WANDB_API_KEY: str
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"

def get_settings() -> Setting:
    return Setting()
