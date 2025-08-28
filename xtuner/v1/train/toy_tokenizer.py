class UTF8ByteTokenizer:
    """字节级 UTF-8 tokenizer：

    - 普通字节的 token id 范围为 [0, 255]
    - 可选特殊符号会占用 256 以上的 id
    """

    def __init__(
        self,
        bos_token_id: int | None = 256,
        eos_token_id: int | None = 257,
        pad_token_id: int | None = 258,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.special_ids = {x for x in [bos_token_id, eos_token_id, pad_token_id] if x is not None}
        self.vocab_size = 256 + len(self.special_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 严格 UTF-8 编码，遇到非法字符会抛错；如需忽略可用 errors='ignore'
        b = text.encode("utf-8", errors="strict")
        ids = list(b)  # 每个字节 0..255

        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        bytes_list = []
        for t in ids:
            if t < 0:
                raise ValueError(f"invalid token id: {t}")
            if t < 256:
                bytes_list.append(t)
            else:
                if not skip_special_tokens and t in self.special_ids:
                    # 特殊符号默认不还原为字符；如有需要可自定义占位符
                    pass
                # 其余 >=256 的非特殊 id 直接忽略
        return bytes(bytes_list).decode("utf-8", errors="strict")
