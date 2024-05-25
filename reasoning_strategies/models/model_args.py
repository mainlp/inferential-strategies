"""
Data class to handle model arguments
"""
from typing import Any
from dataclasses import dataclass


@dataclass
class ModelArgs:
    init_kwargs: dict[str, Any]
    inference_kwargs: dict[str, Any]


@dataclass
class PromptArgs:
    system_message: bool = False
    cautious_mode: bool = False
