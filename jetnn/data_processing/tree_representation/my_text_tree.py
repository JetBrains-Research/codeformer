import spacy
from tree_utils import MyTokens, MyNode, split_big_leaves, merge_left
import random


class MyTextTree:

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")
        self._tree_doc = None
        self._tokens = None
        self._spacy_root = None
        self._root = None
        random.seed(10)

    def _find_root_node(self):
        for token in self._tree_doc:
            if token.dep_ == 'ROOT':
                return token
        raise RuntimeError("Root not found!")

    @staticmethod
    def _post_process_sequence_split(sequence_split, max_subtree_size):
        sequence_split = split_big_leaves(sequence_split, max_subtree_size)
        sequence_split = merge_left(sequence_split, max_subtree_size)
        return sequence_split

    def process_text(self, text, tokens, max_subtree_size=16, tokens_process_function=lambda x: x):
        self._tree_doc = self._nlp(text)
        self._spacy_root = self._find_root_node()
        self._tokens = MyTokens(text, tokens, tokens_process_function)
        self._root = self._init_traverse_tree(self._spacy_root)
        sequence_split = self._get_sequence_split(self._root, max_subtree_size)
        return MyTextTree._post_process_sequence_split(
            [node.get_num_tokens() for node in sequence_split], max_subtree_size
        )

    def _init_traverse_tree(self, spacy_node):
        start_pos = spacy_node.idx
        end_pos = start_pos + len(spacy_node.text)
        node = MyNode(start_pos, end_pos)
        is_leaf = True

        for chile_spacy_node in spacy_node.children:
            is_leaf = False
            child_node = self._init_traverse_tree(chile_spacy_node)
            node.add_children(child_node)

        if is_leaf:
            while node.contains_token(self._tokens.get_current_token()):
                node.add_token(self._tokens.get_current_token_info())
                self._tokens.increase_index()

        return node

    def _get_sequence_split(self, start_node, max_subtree_size):
        subtree_split = list()
        if start_node.get_num_tokens() < max_subtree_size:
            subtree_split.append(start_node)
        else:
            for chile_node in start_node.get_children():
                subtree_split.extend(
                    self._get_sequence_split(chile_node, max_subtree_size)
                )
        return subtree_split


