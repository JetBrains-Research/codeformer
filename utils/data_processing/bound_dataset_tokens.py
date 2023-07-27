import json
import os
import sys
import argparse

sys.path.append("/home/tim/JetNN/utils")
from shared_utils.utils import tokenize, remove_java_comments
from transformers import RobertaTokenizerFast


class DatasetProcessor:
    def __init__(self, src_path, dst_path, tokens_number):
        self._src_path = src_path
        self._dst_path = dst_path
        self._tokens_number = tokens_number
        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            "microsoft/codebert-base"
        )
        self._tokens_ranges = [0, 512, 1024, 2048, int(1e6)]
        self._test_files = list()

    def _find_bucket(self, tokens_number):
        idx = 0
        while self._tokens_ranges[idx] < tokens_number:
            idx += 1
        return idx - 1

    def _create_test_files(self):
        for token_range in self._tokens_ranges:
            test_file_path = os.path.join(
                self._src_path + "_" + str(tokens_number),
                "test" + str(token_range) + ".jsonl",
            )
            test_file = open(test_file_path, "w")
            self._test_files.append(test_file)

    def _process_shared(self, src_path, dst_path, mode="train"):
        with open(src_path, "r") as src_file:
            for raw_sample in src_file.readlines():
                if len(raw_sample) <= 1:
                    continue
                try:
                    sample = json.loads(raw_sample)
                except:
                    continue
                source_code = sample["code"]
                source_code = remove_java_comments(source_code)
                tokens = tokenize(self._tokenizer, source_code, self._tokens_number)
                if mode == "train":
                    if tokens[self._tokens_number - 1] == 2:
                        continue
                    with open(dst_path, "a") as dst_file:
                        dst_file.write(raw_sample)
                elif mode == "test":
                    tokens_number = tokens.index(2) + 1
                    test_file = self._test_files[self._find_bucket(tokens_number)]
                    test_file.write(raw_sample)

    def _process_train_val_test(self):
        src_train = os.path.join(self._src_path, "train.jsonl")
        dst_train = os.path.join(self._dst_path, "train.jsonl")
        src_val = os.path.join(self._src_path, "val.jsonl")
        dst_val = os.path.join(self._dst_path, "val.jsonl")
        src_test = os.path.join(self._src_path, "test.jsonl")

        self._create_test_files()
        # self._process_shared(src_train, dst_train, 'train')
        # self._process_shared(src_val, dst_val, 'train')
        self._process_shared(src_test, None, "test")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tokens_number", default=2048)
    args = parser.parse_args()
    tokens_number = args.tokens_number
    src_path = "/home/tim/JetNN/datasets/gradle_new"
    dst_path = src_path + "_" + str(tokens_number)
    os.makedirs(dst_path, exist_ok=True)
    dataset_processor = DatasetProcessor(src_path, dst_path, tokens_number)
    dataset_processor._process_train_val_test()
