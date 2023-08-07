import argparse
import json

import numpy as np
from transformers import RobertaTokenizerFast
from core.my_code_tree import MyCodeTree
from string import punctuation, whitespace
from core.gradle_format_converter import run_gradle_reformat
from draw import run_draw


def remove_comments(code):
    def non_comment(line):
        if len(line) == 0:
            return False
        return not (line[0][0] == "*" or line[0] == "/**")

    return "\n".join(
        list(
            filter(
                lambda s: non_comment(
                    list(filter(lambda ss: len(ss) > 0, s.split(" ")))
                ),
                code.splitlines(),
            )
        )
    )


def tokenize(tokenizer, text: str, max_parts: int):
    return tokenizer.encode(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_parts,
        truncation="longest_first",
    )


def run(mode='split'):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('log_file')
    args = parser.parse_args()

    my_code_tree = MyCodeTree()
    max_subsequence_size = 32
    max_code_parts = 2048
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    with open(args.data_file, 'r') as data_file, open(args.log_file, 'w') as log_file:
        for raw_sample in data_file.readlines():
            try:
                sample = json.loads(raw_sample)
            except:
                continue
            cleaned_code = remove_comments(sample["code"])
            code = "".join(
                [
                    (ch if ch not in (punctuation + whitespace) else " ")
                    for ch in cleaned_code
                ]
            )
            code = " ".join(code.split())
            tokenized_code = tokenize(tokenizer, code, max_code_parts)
            tokens = list(
                filter(
                    lambda x: x != "<pad>",
                    [tokenizer.decode(token) for token in tokenized_code],
                )
            )[1:-1]
            tokens_split = my_code_tree.process_code(
                cleaned_code, tokens, max_subsequence_size
            )
            if mode == 'split':
                for sz in tokens_split:
                    log_file.write(str(sz) + ", ")
            elif mode == 'length':
                log_file.write(str(len(tokens_split)) + ": " + str(sum(tokens_split)) + "\n")
            else:
                log_file.write(str(len(tokens)) + ", ")


"""
    Dataset Metrics:
        Median Len
        Average Len
        0.95 quantile
        + image of distribution
    Tokens split:
        Median Len
        Average Len
        0.95 quantile
        + image distribution
"""


if __name__ == '__main__':
    # run_gradle_reformat()
    run_draw("/Users/timofeyvasilevskij/JetBrains/JetNN/tree_code_structure/log.txt")
    # run('split')
    # result = list()
    # with open("/Users/timofeyvasilevskij/JetBrains/JetNN/tree_code_structure/length_log.txt", 'r') as f:
    #     for l in f.readlines():
    #         result.append(int((l.split(": ")[0])))
    # print(np.mean(result))
