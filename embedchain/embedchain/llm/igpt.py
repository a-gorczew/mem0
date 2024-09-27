import logging
from collections.abc import Iterable
from typing import Optional, Union

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from mfd_amber_rag.core.llm.iGPT_langchain_chat import IGPTChatEndpoint

from embedchain.config import BaseLlmConfig
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.llm.base import BaseLlm

logger = logging.getLogger(__name__)


@register_deserializable
class IGPTLlm(BaseLlm):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config=config)
        if self.config.model is None:
            self.config.model = "gpt-4o"  # https://wiki.ith.intel.com/display/GenAI/Model+Information
        """IGPT URL"""
        self.config.base_url = "https://apis-internal.intel.com/generativeaiinference/v1"
        self.config.api_key = None

        """IGPT Infer Params"""
        self.config.temperature = 0.6
        self.config.top_p = 0.7
        stop_sequences: list[str] | None = None
        frequency_penalty: float = 0
        presence_penalty: float = 0
        self.config.max_tokens = 4096

        """Enable stream chat mode."""
        self.config.stream = False

        """Key/value arguments to pass to the model. Reserved for future use"""
        self.config. model_kwargs = None

        timeout: int | None = 5000

        # client = Client(host=config.base_url)
        # local_models = client.list()["models"]
        # if not any(model.get("name") == self.config.model for model in local_models):
        #     logger.info(f"Pulling {self.config.model} from Ollama!")
        #     client.pull(self.config.model)

    # @root_validator(pre=True)
    # def validate_environment(cls: "IGPTChatEndpoint", values: dict[str, Any]) -> dict:
    #     """
    #     Validate that api key and python package exists in environment.
    #
    #     :param values: Values to validate
    #     :return: Validated values
    #     """
    #     values["base_url"] = get_from_dict_or_env(values, "base_url", "API_URL")
    #
    #     return values

    def get_llm_model_answer(self, prompt):
        return self._get_answer(prompt=prompt, config=self.config)

    @staticmethod
    def _get_answer(prompt: str, config: BaseLlmConfig) -> Union[str, Iterable]:
        if config.stream:
            callbacks = config.callbacks if config.callbacks else [StreamingStdOutCallbackHandler()]
        else:
            callbacks = [StdOutCallbackHandler()]

        llm = IGPTChatEndpoint(
            model=config.model,
            system=config.system_prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            callback_manager=CallbackManager(callbacks),
            api_url=config.base_url,
        )

        return llm.invoke(prompt)
