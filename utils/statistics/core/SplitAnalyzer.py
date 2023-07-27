import sys

sys.path.append("../../../jetnn/data_processing/tree_code_representation")
sys.path.append("../../")
sys.path.append("../../../")

import json
from shared_utils.utils import tokenize, remove_java_comments
from transformers import RobertaTokenizerFast
from string import punctuation, whitespace
from my_code_tree import MyCodeTree
import colorama
from colorama import Fore


class SplitAnalyzer:
    def __init__(self, path_to_data, path_to_log=None, language="java"):
        self._path_to_data = path_to_data
        self._path_to_log = path_to_log
        self._language = language
        with open(path_to_data, "r") as f:
            self._data = f.readlines()[:100]
        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            "microsoft/codebert-base"
        )
        self._max_tokens = 2048
        self._code_tree = MyCodeTree("/home/tim/JetNN/jetnn/vendor/tree-sitter-java")
        self._max_subsequence_size = 10
        self._colors = [
            Fore.RED,
            Fore.YELLOW,
            Fore.GREEN,
            Fore.CYAN,
            Fore.BLUE,
            Fore.MAGENTA,
        ]

    def process_data(self, log=True):
        self._tokens_distribution = []
        for elem in self._data:
            symbols = json.loads(elem)["code"]
            cleaned_code = remove_java_comments(symbols)
            code = "".join(
                [
                    (ch if ch not in (punctuation + whitespace) else " ")
                    for ch in cleaned_code
                ]
            )
            code = " ".join(code.split())
            tokenized_code = tokenize(self._tokenizer, code, self._max_tokens)
            tokens = list(
                filter(
                    lambda x: x != "<pad>",
                    [self._tokenizer.decode(token) for token in tokenized_code],
                )
            )[1:-1]
            self._code_tree.process_code(
                cleaned_code, tokens, self._max_subsequence_size
            )
            tokens_split = self._code_tree.get_split_positions()
            for i in range(len(tokens_split) - 1):
                tokens_split[i][1] = max(tokens_split[i][1], tokens_split[i + 1][0] - 1)
            tokens_split[len(tokens_split) - 1][1] = len(cleaned_code)
            # for i, split in enumerate(tokens_split):
            #     print(self._colors[i % len(self._colors)] + cleaned_code[split[0]: split[1] + 1], end='')
            # print(len(cleaned_code), tokens_split[-1][1])
            data = {"code": cleaned_code, "split": tokens_split}
            with open("/home/tim/splits/split_10.jsonl", "a") as f:
                json.dump(data, f)
                f.write("\n")
        if log:
            pass

    def get_path_to_log(self):
        return self._path_to_log


if __name__ == "__main__":
    sa = SplitAnalyzer("/home/tim/JetNN/datasets/gradle_new_2048/test.jsonl")
    sa.process_data()
