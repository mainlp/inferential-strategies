"""
Utility functions
"""
import os
from typing import Any, Dict, Callable
import json
import yaml
import logging
from tqdm import tqdm
import random
import numpy as np
import torch

from reasoning_strategies.models.model_args import ModelArgs, PromptArgs

# Get the existing logger
logger = logging.getLogger(__name__)


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its content.

    Args:
    file_path (str): The path to the YAML file.

    Returns:
    Dict[str, Any]: A dictionary containing the content of the YAML file.

    Raises:
    FileNotFoundError: If the YAML file does not exist.
    yaml.YAMLError: If the YAML file cannot be parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                logger.error(f"Error parsing YAML file at {file_path}: {exc}")
                raise
    except FileNotFoundError:
        logger.error(f"YAML file not found at {file_path}.")
        raise FileNotFoundError(f"YAML file not found at {file_path}.")

def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file and returns it as a string.

    Args:
    file_path (str): The path to the text file.

    Returns:
    str: The content of the file.

    Raises:
    FileNotFoundError: If the text file does not exist.
    IOError: If the file could not be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"The file {file_path} was not found.")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except IOError as e:
        logger.error(f"Could not read file: {file_path}. Error: {e}")
        raise IOError(f"Could not read the file {file_path}.")


def write_to_text_file(content: str, file_path: str) -> None:
    """
    Writes a given string to a text file, creating the file if it doesn't exist.
    If necessary, directories in the path will also be created.

    Args:
    content (str): The string to be written to the file.
    file_path (str): The path (or file-like object) where the content should be written.

    Raises:
    IOError: If the file cannot be written.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
    except IOError as e:
        logger.error(f"Failed to write to file: {file_path}. Error: {e}")
        raise IOError(f"Failed to write to file {file_path}: {e}")


def save_dict_to_json(data: dict, file_path: str) -> None:
    """
    Saves a given dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to be saved.
    - file_path (str): The path (including file name) where the JSON file will be saved.

    Raises:
    - Exception: Propagates any exceptions that occur during file creation or JSON serialization.

    This function will create the directory path if it does not exist.
    It handles exceptions related to file writing and JSON serialization by raising them.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save dictionary to JSON: {file_path}. Error: {e}")
        raise e


def load_data_from_json(file_path: str) -> dict[str, Any]:
    """
    Loads data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file to be read.

    Returns:
    - dict[str, Any]: The data loaded from the JSON file.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - JSONDecodeError: If the file is not a valid JSON.

    This function handles file existence checks and ensures that the file is a valid JSON.
    If the file is not found or not a valid JSON, the function raises the appropriate exception.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {file_path}. Error: {e}")
        raise

def get_upper_dir(path: str, level: int = 1) -> str:
    """
    Get the directory 'level' levels up from the given path.

    Args:
    path (str): The initial file or directory path.
    level (int): The number of levels up to traverse.

    Returns:
    str: The path of the upper directory.
    """
    for _ in range(level):
        path = os.path.dirname(path)
    return path


def load_model_args(model_args_dict: dict[str, Any]) -> ModelArgs:
    """
    Load the model arguments from the given dictionary.

    Args:
        model_args_dict (dict[str, Any]): The dictionary containing the model arguments.

    Returns:
        ModelArgs: The model arguments object.
    """

    if not "init" in model_args_dict:
        logging.warning("No 'init' argument provided in model-config.yaml file. Using no additional parameters...")
        init_kwargs = {}
    else:
        init_kwargs = model_args_dict["init"]

        # account for torch_dtype
        if "torch_dtype" in init_kwargs:
            mapping = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            init_kwargs["torch_dtype"] = mapping[init_kwargs["torch_dtype"]]

    if not "inference" in model_args_dict:
        logging.warning("No 'inference' argument provided in model-config.yaml file. Using no additional parameters...")
        inference_kwargs = {}
    else:
        inference_kwargs = model_args_dict["inference"]

    return ModelArgs(init_kwargs=init_kwargs, inference_kwargs=inference_kwargs)


def load_prompt_args(prompt_args_dict: dict[str, Any]) -> PromptArgs:
    """
    Create a PromptArgs object from the given prompt_args_dict.

    Args:
        prompt_args_dict (dict[str, Any]): A dictionary containing prompt arguments.

    Returns:
        PromptArgs: The PromptArgs object created from the prompt_args_dict.
    """
    return PromptArgs(**prompt_args_dict)


def load_model_config(yaml_file_path: str) -> tuple[ModelArgs, dict[str, Any], PromptArgs]:
    """
    Loads the model configuration from a YAML file.

    Args:
        yaml_file_path (str): The path to the YAML file.

    Returns:
        tuple[ModelArgs, dict[str, Any], dict[str, Any]]: A tuple containing the model, tokenizer and prompt arguments.
    """
    if not yaml_file_path.endswith(".yaml"):
        error_message = f"Provided file path must point to YAML file: {yaml_file_path}"
        logging.error(error_message)
        raise ValueError(error_message)

    model_config_dict = read_yaml_file(yaml_file_path)

    # model kwargs
    model_config = model_config_dict["model"] if "model" in model_config_dict else {}
    model_args = load_model_args(model_config)

    # tokenizer kwargs
    tokenizer_kwargs = model_config_dict["tokenizer"] if "tokenizer" in model_config_dict else {}

    # prompt kwargs
    promt_config = model_config_dict["prompt"] if "prompt" in model_config_dict else {}
    prompt_args = load_prompt_args(promt_config)

    return model_args, tokenizer_kwargs, prompt_args


def inference_pipeline(
    encoded_input_dict: dict[str, Any],
    inference_function: Callable[..., tuple[list[str], list[str]]],
    function_kwargs: dict,
    batch_size: int,
) -> dict[str, list[str]]:
    """
    Runs an inference pipeline using the provided encoded input dictionary, inference function, function kwargs, and batch size.

    Args:
        encoded_input_dict (dict[str, Any]): A dictionary containing the encoded input data.
        inference_function (Callable[..., tuple[list[str], list[str]]]): The function used for inference.
        function_kwargs (dict): Additional keyword arguments to be passed to the inference function.
        batch_size (int): The batch size used for processing the input data.

    Returns:
        dict[str, list[str]]: A dictionary containing the generated output, with keys "input" and "generated_output".
    """
    generated_output: dict[str, list[str]] = {"input": [], "generated_output": []}
    num_batches = len(encoded_input_dict["input_ids"]) // batch_size + int(
        len(encoded_input_dict["input_ids"]) % batch_size > 0
    )

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        input_batch = {key: value[start_idx:end_idx] for key, value in encoded_input_dict.items()}

        # Perform inference
        decoded_input, decoded_output = inference_function(encoded_input=input_batch, **function_kwargs)
        generated_output["input"].extend(decoded_input)
        generated_output["generated_output"].extend(decoded_output)

    return generated_output


def set_seed(seed: int = 0) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation. Defaults to 0.

    Returns:
        None
    """
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
