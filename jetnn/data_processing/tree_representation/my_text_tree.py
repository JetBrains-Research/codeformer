import spacy
from jetnn.data_processing.tree_representation.tree_utils import MyTokens, MyNode, split_big_leaves, merge_left
import random


class MyTextTree:

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")
        self._tree_doc = None
        self._tokens = None
        self._spacy_roots = None
        self._root = None
        random.seed(10)

    def _find_root_nodes(self):
        root_nodes = list()
        for token in self._tree_doc:
            if token.dep_ == 'ROOT':
                root_nodes.append(token)
        return root_nodes

    @staticmethod
    def _post_process_sequence_split(sequence_split, max_subtree_size):
        sequence_split = list(filter(lambda x: x != 0, sequence_split))
        sequence_split = split_big_leaves(sequence_split, max_subtree_size)
        sequence_split = merge_left(sequence_split, max_subtree_size)
        return sequence_split

    def process_text(self, text, tokens, max_subtree_size=16, tokens_process_function=lambda x: x):
        self._tree_doc = self._nlp(text)
        self._spacy_roots = self._find_root_nodes()
        self._tokens = MyTokens(text, tokens, tokens_process_function)

        self._root = MyNode(0, len(text))
        for spacy_root in self._spacy_roots:
            self._root.add_children(self._init_traverse_tree(spacy_root))

        sequence_split = self._get_sequence_split(self._root, max_subtree_size)
        result = MyTextTree._post_process_sequence_split(
            sequence_split, max_subtree_size
        )
        return result

    def _init_traverse_tree(self, spacy_node):
        start_pos = spacy_node.idx
        end_pos = start_pos + len(spacy_node.text)
        node = MyNode(start_pos, end_pos)
        left_children = list()
        right_children = list()

        for child_space_node in spacy_node.children:
            if child_space_node.idx < spacy_node.idx:
                left_children.append(child_space_node)
            else:
                right_children.append(child_space_node)    

        for child_spacy_node in left_children:
            child_node = self._init_traverse_tree(child_spacy_node)
            node.add_children(child_node)

        while node.contains_token(self._tokens.get_current_token()):
            node.add_token(self._tokens.get_current_token_info())
            self._tokens.increase_index()

        for child_spacy_node in right_children:
            child_node = self._init_traverse_tree(child_spacy_node)
            node.add_children(child_node)

        return node

    def _get_sequence_split(self, start_node, max_subtree_size):
        subtree_split = list()
        if start_node.get_num_tokens() < max_subtree_size:
            subtree_split.append(start_node.get_num_tokens())
        else:
            left_subtree_split = list()
            right_subtree_split = list()
            for child_node in start_node.get_children():
                if child_node.get_current_node_start_end()[0] < start_node.get_current_node_start_end()[0]:
                    left_subtree_split.extend(
                        self._get_sequence_split(child_node, max_subtree_size)
                    )
                else:
                    right_subtree_split.extend(
                        self._get_sequence_split(child_node, max_subtree_size)
                    )
            subtrees_sum_size = sum(left_subtree_split) + sum(right_subtree_split)
            subtree_split.extend(left_subtree_split)
            subtree_split.append(start_node.get_num_tokens() - subtrees_sum_size)
            subtree_split.extend(right_subtree_split)
        return subtree_split


