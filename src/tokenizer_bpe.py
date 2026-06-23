from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def encode_large(self, text: str, chunk_chars: int = 2_000_000) -> list[int]:
        # tokenizer.encode() on the whole string at once builds one giant Rust
        # Encoding object -- on a 15GB-RAM box that OOMs for a 150MB+ input.
        # Chunking bounds peak memory to one chunk's worth at a time.
        ids: list[int] = []
        for i in range(0, len(text), chunk_chars):
            ids.extend(self.tokenizer.encode(text[i : i + chunk_chars]).ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))
