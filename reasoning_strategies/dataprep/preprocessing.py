"""
Functions and modules to pre-process data
"""
import random
from typing import Any, Protocol
import re

class ContextForm(Protocol):
    adjectives: list[str]
    objects: list[str]
    places: list[str]

def replace_words(original_string: str, replacement_dict: dict) -> str:
    """
    Replace whole words in a string based on a dictionary mapping.

    Args:
    original_string (str): The string where replacements need to be made.
    replacement_dict (dict): A dictionary where each key is a word in the original string 
                             that needs to be replaced with its corresponding value.

    Returns:
    str: A new string with all replacements performed.

    Example:
    >>> replace_words("This is A and this is B in Antarctica.", {"A": "hello", "B": "bye"})
    'This is hello and this is bye in Antarctica.'
    """
    # Pattern to match whole words only
    pattern = r'\b(' + '|'.join(re.escape(key) for key in replacement_dict.keys()) + r')\b'
    
    # Replacement function
    def replacer(match):
        return replacement_dict[match.group(0)]

    return re.sub(pattern, replacer, original_string)

def starts_with_vowel_sound(word: str) -> bool:
    """
    Check if a given word starts with a vowel sound.

    Parameters:
        word (str): The word to check.

    Returns:
        bool: True if the word starts with a vowel sound, False otherwise.
    """
    vowel_letters = "aeiou"
    return word[0].lower() in vowel_letters

def draw_adjective(adjectives: list[str]) -> str:
    """
    Draws an adjective from the given list of adjectives without replacement.

    Parameters:
        adjectives (list[str]): A list of adjectives from which to draw an adjective.

    Returns:
        str: The drawn adjective.

    Raises:
        ValueError: If the list of adjectives is empty.
    """
    if not adjectives:
        raise ValueError("No more adjectives to draw.")
    random.shuffle(adjectives)
    return adjectives.pop()

def create_context(adjectives: list[str], objects: list[str], place: str) -> str:
    """
    Generate a context by combining an adjective, an object, and a place.

    Args:
        adjectives (list[str]): A list of adjectives to choose from.
        objects (list[str]): A list of objects to choose from.
        place (str): The place to include in the context.

    Returns:
        str: The generated context consisting of an article, an adjective, an object, and a place.
    """
    adjective = draw_adjective(adjectives)
    object = random.choice(objects)
    article = "an" if starts_with_vowel_sound(adjective) else "a"
    return " ".join([article, adjective, object, place])

def create_context_mapping(context: ContextForm) -> dict[str, str]:
    """
    Generate a context mapping dictionary based on the provided context form.

    Args:
        context (ContextForm): The context form containing the necessary information.

    Returns:
        dict[str, str]: A dictionary mapping letters to context strings.

    """
    adjectives = context.adjectives.copy()
    place = random.choice(context.places)
    letters = ["A", "B", "C", "D", "E"]
    context_mapping = {letter: create_context(adjectives, context.objects, place) for letter in letters}

    return context_mapping

def inject_context(example: dict[str, Any], context: ContextForm) -> dict[str, Any]:
    """
    Injects the given context into the example dictionary.

    Parameters:
        example (dict[str, Any]): The example dictionary to inject the context into.
        context (ContextForm): The context to inject into the example dictionary.

    Returns:
        dict[str, Any]: The example dictionary with the injected context.
    """
    context_mapping = create_context_mapping(context)

    # premises
    example["premises"] = [replace_words(sample, context_mapping) for sample in example["premises"]]
    
    # question
    example["question"] = replace_words(example["question"], context_mapping)

    return example

