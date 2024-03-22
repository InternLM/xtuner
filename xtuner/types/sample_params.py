from typing import Optional

from pydantic import BaseModel


class SampleParams(BaseModel):

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.75
    repetition_penalty: float = 1.0
    stop_words: list = []
    seed: Optional[int] = None
