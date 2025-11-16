import json
from pathlib import Path

from tqdm import tqdm
from wasabi import msg


class SimpleTokenizer:
    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, corpus: str, vocab_size: int) -> None:
        tokens = corpus.encode("utf-8")
        tokens = list(map(int, tokens))
        self.merges = {}
        for i in tqdm(range(vocab_size - 256), desc="Merging byte pairs"):
            combinations = self._count_combinations(tokens)
            top_pair = max(combinations, key=combinations.get)
            idx = 256 + i
            tokens = self._merge_bytepairs(tokens, top_pair, idx)
            self.merges[top_pair] = idx
        self.vocab = self._build_vocab()

    def encode(self, input: str) -> list[int]:
        tokens = list(input.encode("utf-8"))
        for (token_1, token_2), value in self.merges.items():
            tokens = self._merge_bytepairs(tokens, (token_1, token_2), value)
        return tokens

    def decode(self, tokenized_input) -> str:
        tokens = b"".join(self.vocab[idx] for idx in tokenized_input)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _build_vocab(self) -> dict:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")
        return vocab

    def add_special_token(self, special_str: str):
        if len(self.merges) == 0:
            msg.fail("Special token addition must be done after training has finished")
            return
        if self.special_tokens.get(special_str, None) is not None:
            msg.fail(f"Skipping: The `{special_str}` special token already exists")
        else:
            self.special_tokens[special_str] = (
                256 + len(self.merges) + len(self.special_tokens)
            )

    def save(self, folder_name):
        save_path = Path(folder_name)
        if not save_path.exists():
            save_path.mkdir()

        merges_path = save_path / "tokenizer_merges.vm"

        with open(merges_path, "w") as merges_file:
            for idx1, idx2 in self.merges.keys():
                merges_file.write(f"{idx1} {idx2}\n")
        merges_file.close()

        special_tokens_path = save_path / "tokenizer_special.vm"

        with open(special_tokens_path, "w") as special_tokens_file:
            for special_token in self.special_tokens.keys():
                special_tokens_file.write(f"{special_token}\n")
        special_tokens_file.close()

    def load(self, folder_name):
        save_path = Path(folder_name)
        if not save_path.exists():
            raise ValueError("The folder that you provided does not exist")

        merges_path = save_path / "tokenizer_merges.vm"
        with open(merges_path, "r", encoding="utf-8") as merges_file:
            merges = merges_file.read()
            merges = merges.split("\n")[:-1]
            for i, row in enumerate(merges):
                idx1, idx2 = row.split(" ")
                idx1 = int(idx1)
                idx2 = int(idx2)
                self.merges[(idx1, idx2)] = i + 256
        merges_length = len(self.merges)

        special_tokens_path = save_path / "tokenizer_special.vm"

        with open(special_tokens_path, "r", encoding="utf-8") as special_tokens_file:
            special_tokens = special_tokens_file.read()
            special_tokens = special_tokens.split("\n")[:-1]
            for i, special_token in enumerate(special_tokens):
                self.special_tokens[special_token] = i + 256 + merges_length

        self.vocab = self._build_vocab()

    @staticmethod
    def _count_combinations(tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @staticmethod
    def _merge_bytepairs(tokens, pair, index):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(index)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
