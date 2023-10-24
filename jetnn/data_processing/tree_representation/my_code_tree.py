from jetnn.data_processing.tree_representation.my_tree_sitter import MyTreeSitter
import random
import sys
from jetnn.data_processing.tree_representation.tree_utils import MyTokens, MyNode, split_big_leaves, merge_left


class MyCodeTree:
    def __init__(
            self,
            programming_language: str="java",
            path_to_tree_sitter: str="../vendor/tree-sitter-java",
    ):
        self._tree_sitter = MyTreeSitter(programming_language, path_to_tree_sitter)
        self._tokens = None
        self._root = None
        random.seed(10)
        sys.setrecursionlimit(10000)

    def remove_comments(self, code: str, method_location: tuple[int, int]=(0, 0)) -> (str, str, tuple[int, int]):
        return self._tree_sitter.remove_comments_from_code(code, method_location)

    @staticmethod
    def _post_process_sequence_split(sequence_split: list[int], max_subtree_size: int) -> list[int]:
        sequence_split = split_big_leaves(sequence_split, max_subtree_size)
        sequence_split = merge_left(sequence_split, max_subtree_size)
        return sequence_split

    def process_code_random(self, tokens: list[str], max_subtree_size: int) -> list[int]:
        result = list()
        cnt = 0
        while cnt < len(tokens):
            random_split = random.randint(1, max_subtree_size)
            result.append(min(random_split, len(tokens) - cnt))
            cnt += random_split
        return result

    def process_code(
            self,
            java_code: str,
            tokens: list[str],
            max_subtree_size: int=16,
    ) -> list[int]:
        self._tree_sitter.process_code(java_code)
        self._tokens = MyTokens(java_code, tokens)
        self._root = self._init_traverse_tree_sitter()
        self._tree_sitter.reset()
        sequence_split = self.get_sequence_split(self._root, max_subtree_size)
        return MyCodeTree._post_process_sequence_split(
            [node.get_num_tokens() for node in sequence_split], max_subtree_size
        )

    def _init_traverse_tree_sitter(self) -> MyNode:
        start_pos, end_pos = self._tree_sitter.get_start_end_bytes()
        node = MyNode(start_pos, end_pos)
        is_leaf = not self._tree_sitter.goto_first_child()
        if is_leaf:
            while node.contains_token(self._tokens.get_current_token()):
                node.add_token(self._tokens.get_current_token_info())
                self._tokens.increase_index()
        else:
            child_node = self._init_traverse_tree_sitter()
            node.add_children(child_node)
            while self._tree_sitter.goto_next_child():
                child_node = self._init_traverse_tree_sitter()
                node.add_children(child_node)
            self._tree_sitter.goto_parent()
        return node

    def get_sequence_split(self, start_node: MyNode, max_subtree_size: int) -> list[int]:
        subtree_split: list[int] = list()
        if start_node.get_num_tokens() < max_subtree_size:
            subtree_split.append(start_node)
        else:
            for chile_node in start_node.get_children():
                subtree_split.extend(
                    self.get_sequence_split(chile_node, max_subtree_size)
                )
            if len(subtree_split) == 0 and len(start_node.get_children()) == 0:
                subtree_split.append(start_node)
        return subtree_split
