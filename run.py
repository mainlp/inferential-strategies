"""
Main script to run experiments on the inferential strategies LLMs employ in problems of propositional logic.
"""
import os
import logging
import argparse
from typing import Any
from tqdm import tqdm
from datasets import load_dataset

from reasoning_strategies.utils.utils import read_yaml_file, load_model_config, inference_pipeline, set_seed
from reasoning_strategies.contexts.context import Context
from reasoning_strategies.dataprep.preprocessing import inject_context
from reasoning_strategies.prompts.prompt_manager import PromptManager
from reasoning_strategies.models.reasoner import Reasoner
from reasoning_strategies.models.model_args import ModelArgs, PromptArgs

CONTEXT_PATH = os.path.join("reasoning_strategies", "contexts")
PROMPT_PATH = os.path.join("reasoning_strategies", "prompts")
MODEL_ARG_PATH = os.path.join("reasoning_strategies", "models", "model_config")
OUTPUT_PATH = os.path.join("experimental_results")
SAVE_PATH = os.path.join("hf_models")

def setup_logging(verbose: int = 1) -> None:
    """
    Sets up the logging level based on the verbosity level.

    Args:
        verbose (int): Verbosity level provided by the user.
                       0 for WARNING, 1 for INFO, and 2 or higher for DEBUG.
    """
    if verbose == 0:
        logging_level = logging.WARNING
    elif verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser(
        "Comparing Inferential Strategies of Humans and Large Language Models in Deductive Reasoning Tasks"
    )

    # General configs
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[-1, 0, 1], help="Verbose mode (-1: DEBUG, 0: WARNING, 1: INFO)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
    parser.add_argument("--answer-only", action="store_true", help="Whether to record only the answer")

    # Configs about context
    parser.add_argument(
        "--context",
        type=str,
        default="marbles",
        choices=["marbles", "zoo", "letters"],
        help="Context for premises and question",
    )

    # Configs about experiment
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2, 3, 4], help="Experiment number")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of experiment iterations")

    # Configs about model
    parser.add_argument("--model", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Large Language Model to use")
    parser.add_argument(
        "--strategy", type=str, default="cot", choices=["naive", "cot", "cot_sc", "tot"], help="LLM reasoning strategy"
    )

    return parser.parse_args()


def prepare_dataset(args: argparse.Namespace, dataset_name: str, dataset_config: str, split: str = "test") -> Any:
    """
    Prepares the dataset of propositional logic. When a context is specified, letters in the
    problem statements are replaced respectively.

    Args:
        args (argparse.Namespace): Command line arguments.
        dataset_name (str): Name of dataset. Has to be a valid HF dataset repository.
        dataset_config (str): The configuration of the dataset.
        split (str, optional): Specifies which split to use. Defaults to 'test'.

    Returns:
        Any: The prepared dataset.
    """
    # Load and preprocess dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if args.context != "letters":
        context_file = os.path.join(CONTEXT_PATH, f"{args.context}.yaml")
        context_dict = read_yaml_file(context_file)
        context = Context(**context_dict)

        dataset = dataset.map(inject_context, fn_kwargs={"context": context}, batched=False)

    return dataset


def load_args(args: argparse.Namespace) -> tuple[ModelArgs, dict[str, Any], PromptArgs]:
    """
    Load the arguments for model, tokenizer and prompts.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple[ModelArgs, dict[str, Any], PromptArgs]: ModelArgs, TokenizerArgs and PromptArgs.
    """
    model_config_file = os.path.join(MODEL_ARG_PATH, f"{args.model.replace('/', '_')}.yaml")
    return load_model_config(model_config_file)


def initialize_reasoner(args: argparse.Namespace) -> Reasoner:
    """
    Initializes the model wrapper based on the provided arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        Reasoner: The initialized model wrapper.
    """
    # model & tokenizer
    model_args, tokenizer_kwargs, _ = load_args(args)

    model_path = os.path.join(SAVE_PATH, "model", args.model)
    tokenizer_path = os.path.join(SAVE_PATH, "tokenizer", args.model)

    reasoner = Reasoner(
        model_name=args.model,
        model_path=model_path,
        model_init_kwargs=model_args.init_kwargs,
        tokenizer_path=tokenizer_path,
        tokenizer_init_kwargs=tokenizer_kwargs,
    )

    return reasoner


def run_inference(args: argparse.Namespace, dataset: Any, reasoner: Reasoner) -> None:
    """
    Runs the inference pipeline on the dataset using the reasoner.

    Args:
        args (argparse.Namespace): Command line arguments.
        dataset (Any): The prepared dataset.
        reasoner (Reasoner): The initialized model wrapper.
    """
    model_args, tokenizer_kwargs, prompt_args = load_args(args)
    model_args.inference_kwargs.update(
        {"eos_token_id": reasoner.tokenizer.eos_token_id, "pad_token_id": reasoner.tokenizer.pad_token_id}
    )

    # get task prompts
    task_prompt_path = os.path.join(PROMPT_PATH, "task_prompts", f"experiment{args.experiment}")
    sys_message_file = os.path.join(task_prompt_path, "system_message.txt")
    prompt_file = os.path.join(task_prompt_path, "prompt.txt")
    cot_prompt_file = (
        os.path.join(PROMPT_PATH, "reasoning_prompts", args.strategy, f"{args.strategy}_prompt.txt")
        if args.strategy in ["cot", "cot_sc"]
        else None
    )
    cautious_sys_message_file = (
        os.path.join(task_prompt_path, "cautious_system_instruction.txt") if prompt_args.cautious_mode else None
    )

    prompt_manager = PromptManager()
    sys_message, prompt = prompt_manager.get_input_prompts(
        sys_message_file_path=sys_message_file,
        prompt_file_path=prompt_file,
        prompt_suffix_file_path=cot_prompt_file,
        cautious_sys_message_file_path=cautious_sys_message_file,
    )

    # convert prompts and encode
    dataset = dataset.map(
        prompt_manager.create_task_prompt,
        fn_kwargs={
            "prompt": prompt,
            "tokenizer": reasoner.tokenizer,
            "allow_system_message": prompt_args.system_message,
            "system_message": sys_message,
        },
        batched=True,
        batch_size=len(dataset),
        load_from_cache_file=False,
    )
    encoded_input = reasoner.tokenizer(dataset["task_prompt"], padding=True, return_tensors="pt").to(args.device)

    # inference
    for run_nr in tqdm(range(args.num_samples), desc="Processing Samples"):
        generated_output = inference_pipeline(
            encoded_input_dict=encoded_input,
            inference_function=reasoner.inference,
            function_kwargs={"inference_kwargs": model_args.inference_kwargs, "reasoning_strategy": args.strategy},
            batch_size=args.batch_size,
        )

        # write results to files
        model_path_name = f"{args.model}_cautious" if prompt_args.cautious_mode else args.model
        result_path = os.path.join(
            OUTPUT_PATH,
            f"experiment{args.experiment}",
            "inference_results",
            model_path_name,
            args.strategy,
            f"sample_{run_nr}",
        )
        metadata = {
            "Correct answers": dataset["answer"],
            "model": args.model,
            "strategy": args.strategy,
            "context": args.context,
            "experiment": args.experiment,
            "sample id": run_nr,
            "batch size": args.batch_size,
            "model_init_kwargs": dict((k, v) for k, v in model_args.init_kwargs.items() if k != "torch_dtype"),
            "model_inference_kwargs": model_args.inference_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
        }

        reasoner.save_results_to_json(
            result_dict=generated_output, file_path=os.path.join(result_path, f"results.json"), metadata=metadata
        )
        reasoner.save_results_to_txt(
            folder_path=result_path,
            decoded_output=generated_output["generated_output"],
            decoded_input=generated_output["input"],
        )


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    dataset = prepare_dataset(args, "mainlp/henst_prop_logic", f"experiment{args.experiment}")
    reasoner = initialize_reasoner(args)

    run_inference(args, dataset, reasoner)


if __name__ == "__main__":
    main()
