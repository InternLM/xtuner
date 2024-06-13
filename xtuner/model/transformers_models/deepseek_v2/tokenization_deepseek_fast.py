from typing import List, Optional, Union

from transformers.models.llama import LlamaTokenizerFast


class DeepseekTokenizerFast(LlamaTokenizerFast):

    def convert_ids_to_tokens(
            self,
            ids: Union[int, List[int]],
            skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """Converts a single index or a sequence of indices in a token or a
        sequence of tokens, using the vocabulary and added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            token = self._tokenizer.id_to_token(index)
            tokens.append(token if token is not None else '')
        return tokens

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        token = self._tokenizer.id_to_token(int(index))
        return token if token is not None else ''
