import json
import numpy as np
from tree_sitter import Language, Parser
import matplotlib.pyplot as plt


class TreeStat:
    def __init__(self, filename, tokenizer=None):
        Language.build_library("build/my-languages.so", ["tree-sitter-java"])
        java_language = Language("build/my-languages.so", "java")
        self._parser = Parser()
        self._parser.set_language(java_language)
        self._filename = filename
        self._accumulated_node_info = dict()
        self._cnt_node = 0
        self._cnt_predict_node = 0
        self._batch_value = 20

    def calculate_number_of_specific_nodes(
        self, node_predicate=lambda x: x.endswith("statement")
    ):
        with open(self._filename, "r") as f:
            for jsonl in f.readlines():
                java_code = json.loads(jsonl)["code"]
                tree = self._parser.parse(bytes(java_code, "utf8"))
                cursor = tree.walk()
                self._cnt_node = 0
                self._cnt_predict_node = 0
                self._calculate_for_one_tree(cursor, node_predicate)
                if (
                    self._cnt_node // self._batch_value
                    not in self._accumulated_node_info
                ):
                    self._accumulated_node_info[
                        self._cnt_node // self._batch_value
                    ] = list()
                self._accumulated_node_info[self._cnt_node // self._batch_value].append(
                    self._cnt_predict_node
                )
        for key in self._accumulated_node_info:
            print(key, np.mean(np.array(self._accumulated_node_info[key])))
        self._draw_plots()

    def _calculate_for_one_tree(self, node, node_predicate):
        self._cnt_node += 1
        if node_predicate(node.node.type):
            self._cnt_predict_node += 1
        is_leaf = not node.goto_first_child()
        if not is_leaf:
            self._calculate_for_one_tree(node, node_predicate)
            while node.goto_next_sibling():
                self._calculate_for_one_tree(node, node_predicate)
            node.goto_parent()

    def _draw_plots(self):
        fig, ax = plt.subplots()
        ax.scatter(
            np.array([key for key in self._accumulated_node_info.keys()])
            * self._batch_value,
            [
                np.mean(np.array(value))
                for value in self._accumulated_node_info.values()
            ],
        )

        ax.set(
            xlabel="number of nodes in tree",
            ylabel="number of expression nodes",
            title="Expression nodes in the tree",
        )
        ax.grid()

        fig.savefig("test.png")
        plt.show()
