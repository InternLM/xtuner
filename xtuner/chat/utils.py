from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig():

    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.1)
    top_k: int = field(default=40)
    top_p: float = field(default=0.75)
    repetition_penalty: float = field(default=1.0)
    stop_words: list = field(default_factory=lambda: [])
    seed: Optional[int] = field(default=None)
