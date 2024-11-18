from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        # try:
        #     logger.info("Loading settings from the ZenML secret store.")
        #
        #     settings_secrets = Client().get_secret("settings")
        #     settings = Settings(**settings_secrets.secret_values)
        # except (RuntimeError, KeyError):
        #     logger.warning(
        #         "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
        #     )
        #     settings = Settings()
        #
        # return settings
        logger.info("Loading settings")
        return Settings()


# Global singleton
settings = Settings.load_settings()
