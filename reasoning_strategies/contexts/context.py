"""
Dataclass holding attributes for the context of the problem statements.
"""

from dataclasses import dataclass

@dataclass
class Context:
    adjectives: list[str]
    objects: list[str]
    places: list[str]