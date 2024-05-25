"""
Interface to handle different LLMs
"""
from typing import Any, Literal
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from reasoning_strategies.utils.utils import read_text_file


class PromptManager:
    def __init__(self):
        pass

    @staticmethod
    def load_prompts(
        sys_message_file_path: str,
        prompt_file_path: str,
        prompt_prefix_file_path: str | None,
        prompt_suffix_file_path: str | None,
    ) -> tuple[str, str | None, str, str | None]:
        """Loads the system message, prompt, prompt prefix, and prompt suffix from the specified file paths.

        Args:
            sys_message_file_path (str): The file path of the system message file.
            prompt_file_path (str): The file path of the prompt file.
            prompt_prefix_file_path (str | None): The file path of the prompt prefix file.
            prompt_suffix_file_path (str | None): The file path of the prompt suffix file (optional).
        Returns:
            tuple[str, str | None, str, str | None]: A tuple containing the system message, prompt prefix, prompt, and prompt suffix.
        """
        sys_message = read_text_file(sys_message_file_path)
        prompt = read_text_file(prompt_file_path)
        prompt_prefix = read_text_file(prompt_prefix_file_path) if prompt_prefix_file_path else None
        prompt_suffix = read_text_file(prompt_suffix_file_path) if prompt_suffix_file_path else None

        return sys_message, prompt_prefix, prompt, prompt_suffix

    @staticmethod
    def assemble_prompt(
        prompt: str,
        prompt_prefix: str | None = None,
        prompt_suffix: str | None = None,
        prompt_mapping: dict[str, str] = {},
    ) -> str:
        """
        Assembles the prompt by adding a prefix and suffix to the given prompt.

        Args:
            prompt (str): The original prompt.
            prompt_prefix (str | None, optional): The prefix to add to the prompt. Defaults to None.
            prompt_suffix (str | None, optional): The suffix to add to the prompt. Defaults to None.

        Returns:
            str: The assembled prompt.
        """
        # add examples for few-shot settings
        if prompt_prefix is not None:
            prompt = f"{prompt_prefix}\n\n{prompt}"

        # add specifics for prompt type
        if prompt_suffix is not None:
            prompt += f"\n{prompt_suffix}"

        # replace any special tokens with prompt mapping
        for special_token, replacement in prompt_mapping.items():
            prompt = prompt.replace(special_token, replacement)

        return prompt

    def get_input_prompts(
        self,
        sys_message_file_path: str,
        prompt_file_path: str,
        prompt_prefix_file_path: str | None = None,
        prompt_suffix_file_path: str | None = None,
        cautious_sys_message_file_path: str | None = None,
        prompt_mapping: dict[str, Any] = {},
    ) -> tuple[str, str]:
        """
        Get input prompts from the specified file paths and return the system message and prompt.

        Args:
            sys_message_file_path (str): The file path for system message.
            prompt_file_path (str): The file path for prompt.
            prompt_prefix_file_path (str | None): The file path for prompt prefix. Defaults to None.
            prompt_suffix_file_path (str | None): The file path for prompt suffix. Defaults to None.
            cautious_sys_message_file_path (str | None): The file path for cautious system message. Defaults to None.
            prompt_mapping (dict[str, Any]): The mapping of prompts to their corresponding values. Defaults to an empty dictionary.

        Returns:
            tuple[str, str]: A tuple containing the system message and the prompt.
        """
        sys_message, prompt_prefix, prompt, prompt_suffix = self.load_prompts(
            sys_message_file_path=sys_message_file_path,
            prompt_file_path=prompt_file_path,
            prompt_prefix_file_path=prompt_prefix_file_path,
            prompt_suffix_file_path=prompt_suffix_file_path,
        )

        if cautious_sys_message_file_path:
            prompt_prefix = sys_message
            sys_message = read_text_file(cautious_sys_message_file_path)

        prompt = self.assemble_prompt(
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            prompt_mapping=prompt_mapping,
        )

        return sys_message, prompt

    @staticmethod
    def get_chat_prompt(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        allow_system_message: bool,
        system_message: str,
        prompt: str,
        tokenize: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ):
        """
        Generates a chat prompt for a conversation using the given tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use for tokenizing the chat prompt.
            allow_system_message (bool): A flag indicating whether to allow a system message in the chat prompt.
            system_message (str): The system message to include in the chat prompt.
            prompt (str): The user's prompt for the conversation.
            tokenize (bool, optional): A flag indicating whether to tokenize the chat prompt. Defaults to True.
            return_tensors (Literal["pt"], optional): The type of tensors to return. Defaults to "pt".

        Returns:
            chat: The generated chat prompt.

        Examples:
            >>> tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
            >>> allow_system_message = True
            >>> system_message = "Welcome to the chatbot!"
            >>> prompt = "Hello, how can I help you?"
            >>> chat = get_chat_prompt(tokenizer, allow_system_message, system_message, prompt)
        """
        if allow_system_message:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": f"{system_message}\n\n{prompt}",
                },
            ]
        chat = tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True, return_tensors=return_tensors
        )
        return chat

    @staticmethod
    def create_task_prompt(
        examples: dict[str, Any],
        prompt: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        allow_system_message: bool,
        system_message: str,
        tokenize: bool = False,
    ) -> dict[str, list]:
        """
        Generates a task prompt for creating a task.

        Args:
            examples (dict[str, Any]): A dictionary containing examples for the task.
            prompt (str): The prompt for the task.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use for tokenization.
            allow_system_message (bool): Flag indicating whether to allow system messages.
            system_message (str): The system message to include in the task prompt.
            tokenize (bool, optional): Flag indicating whether to tokenize the task prompt. Defaults to False.

        Returns:
            dict[str, list]: A dictionary containing the generated task prompts.

        """
        task_prompts: list[str | int] = []

        for premises, conclusion in zip(examples["premises"], examples["question"]):
            formatted_premises = "\n".join(premises)
            task_prompt = prompt.replace("<premises>", f"Statements:\n{formatted_premises}")
            task_prompt = task_prompt.replace("<conclusion>", f"Conclusion: {conclusion}")

            chat = PromptManager.get_chat_prompt(
                tokenizer=tokenizer,
                allow_system_message=allow_system_message,
                system_message=system_message,
                prompt=task_prompt,
                tokenize=tokenize,
            )
            task_prompts.append(chat)

        return {"task_prompt": task_prompts}

    @staticmethod
    def create_evaluation_prompt(
        examples: dict[str, Any],
        prompt: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        allow_system_message: bool,
        system_message: str,
        tokenize: bool = False,
    ) -> dict[str, list]:
        """
        Generates a task prompt for creating a task.

        Args:
            examples (dict[str, Any]): A dictionary containing examples for the task.
            prompt (str): The prompt for the task.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use for tokenization.
            allow_system_message (bool): Flag indicating whether to allow system messages.
            system_message (str): The system message to include in the task prompt.
            tokenize (bool, optional): Flag indicating whether to tokenize the task prompt. Defaults to False.

        Returns:
            dict[str, list]: A dictionary containing the generated task prompts.

        """
        evaluation_prompts: list[str | int] = []

        for generated_output in examples["generated_output"]:
            evaluation_prompt = prompt.replace("<reasoning-path>", generated_output)

            chat = PromptManager.get_chat_prompt(
                tokenizer=tokenizer,
                allow_system_message=allow_system_message,
                system_message=system_message,
                prompt=evaluation_prompt,
                tokenize=tokenize,
            )
            evaluation_prompts.append(chat)

        return {"evaluation_prompt": evaluation_prompts}

    @staticmethod
    def create_train_prompt(
        examples: dict[str, Any],
        input_column_name: str,
        gt_column_name: str,
    ) -> dict[str, list]:
        """
        Generates a task prompt for creating a task.

        Args:
            examples (dict[str, Any]): A dictionary containing examples for the task.
            prompt (str): The prompt for the task.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use for tokenization.
            allow_system_message (bool): Flag indicating whether to allow system messages.
            system_message (str): The system message to include in the task prompt.
            tokenize (bool, optional): Flag indicating whether to tokenize the task prompt. Defaults to False.

        Returns:
            dict[str, list]: A dictionary containing the generated task prompts.

        """
        train_prompts: list[str] = [
            f"{input_text} {gt_text}"
            for input_text, gt_text in zip(examples[input_column_name], examples[gt_column_name])
        ]
        return {"train_prompt": train_prompts}
