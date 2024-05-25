"""
Reasoning module based on HF models
"""
from typing import Any, Literal
import logging

from reasoning_strategies.models.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class Reasoner(ModelWrapper):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_init_kwargs: dict[str, Any],
        tokenizer_path: str,
        tokenizer_init_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(model_name, model_path, model_init_kwargs, tokenizer_path, tokenizer_init_kwargs)

    def inference(
        self,
        encoded_input: dict[str, Any],
        inference_kwargs: dict[str, Any],
        reasoning_strategy: Literal["naive", "cot", "cot_sc", "tot"] = "cot",
    ) -> tuple[Any, Any]:
        """
        Perform inference using the specified reasoning strategy.

        Args:
            encoded_input (dict[str, Any]): The encoded input for the inference.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference.
            reasoning_strategy (Literal["naive", "cot", "cot_sc", "tot"], optional): The reasoning strategy to use. Defaults to "cot".

        Returns:
            Tuple[Any, Any]: A tuple containing the decoded input and the decoded output.
        """
        if reasoning_strategy in ["naive", "cot"]:
            decoded_input, decoded_output = self.forward(encoded_input=encoded_input, inference_kwargs=inference_kwargs)
        elif reasoning_strategy in ["cot_sc", "tot"]:
            error_msg = f"Reasoning strategy '{reasoning_strategy}' is not implemented yet. Future work will consider this setup."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        else:
            error_message = f"Invalid reasoning strategy: {reasoning_strategy}"
            logging.error(error_message)
            raise ValueError(error_message)

        return decoded_input, decoded_output
