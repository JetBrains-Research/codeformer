import json
from shared_utils.utils import tokenize, remove_java_comments
from transformers import RobertaTokenizerFast


class DatasetStatistics:
    def __init__(self, path_to_data, path_to_log=None, language="java"):
        self._path_to_data = path_to_data
        self._path_to_log = path_to_log
        self._language = language
        with open(path_to_data, "r") as f:
            self._data = f.readlines()
        self._tokens_distribution = list()
        self._symbols_distribution = list()
        self._tokenizer = RobertaTokenizerFast.from_pretrained(
            "microsoft/codebert-base"
        )
        self._max_tokens = 8096

    @staticmethod
    def write_list_to_file(data, path_to_file):
        with open(path_to_file, "w") as f:
            for elem in data:
                f.write(str(elem) + ",")

    def calc_tokens_distribution(self, log=True):
        self._tokens_distribution = []
        for elem in self._data:
            try:
                symbols = json.loads(elem)["code"]
            except:
                continue
            symbols = remove_java_comments(symbols)
            symbols = symbols.replace(" ", "").replace("\\n", "").replace("\n", "")
            tokenized_code = tokenize(self._tokenizer, symbols, self._max_tokens)
            tokenized_code = list(filter(lambda x: x > 2, tokenized_code))
            self._tokens_distribution.append(len(tokenized_code))
        if log:
            DatasetStatistics.write_list_to_file(
                self._tokens_distribution, self._path_to_log
            )

    def get_tokens_distribution(self):
        return self._tokens_distribution

    def calc_symbols_distribution(self, log=True):
        self._symbols_distribution = []
        for elem in self._data:
            try:
                symbols = json.loads(elem)["code"]
            except:
                continue
            symbols = remove_java_comments(symbols)
            symbols = symbols.replace(" ", "").replace("\\n", "").replace("\n", "")
            self._symbols_distribution.append(len(symbols))
        if log:
            DatasetStatistics.write_list_to_file(
                self._symbols_distribution, self._path_to_log
            )

    def get_symbols_distribution(self):
        return self._symbols_distribution

    def get_path_to_log(self):
        return self._path_to_log
