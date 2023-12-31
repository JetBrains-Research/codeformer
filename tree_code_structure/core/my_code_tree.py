from tree_code_structure.core.my_tree_sitter import MyTreeSitter


class MyTokens:
    def __init__(self, java_code, tokens, tokens_process_function):
        self._tokens = [tokens_process_function(token) for token in tokens]
        self._tokens_positions = list()
        self._tokens_positions.append(0)
        current_position = 0
        for token in self._tokens:
            pos = java_code.find(token, current_position)
            if pos != -1:
                current_position = pos + len(token)
            # else:
            # raise Exception("Token not found")
            self._tokens_positions.append(current_position)
        self._tokens_positions.append(self._tokens_positions[-1])
        self._index = 0

    def get_current_token_info(self):
        return self.get_current_index(), self.get_current_token()

    def get_current_token(self):
        if self._index >= len(self._tokens_positions):
            return None
        return self._tokens_positions[self._index]

    def get_current_index(self):
        return self._index

    def increase_index(self):
        self._index += 1


class MyNode:
    def __init__(self, tree_sitter: MyTreeSitter):
        self._start_pos, self._end_pos = tree_sitter.get_current_node_start_end()
        self._children = []
        self._start_token_index = int(1e9)
        self._num_tokens = 0
        self._tokens = []

    def add_children(self, my_node):
        # print("add_children")
        if my_node.get_num_tokens() == 0:
            return
        self._start_token_index = min(
            self._start_token_index, my_node.get_start_token_index()
        )
        self._children.append(my_node)
        self._num_tokens += my_node.get_num_tokens()

    def get_start_token_index(self):
        return self._start_token_index

    def get_end_token_index(self):
        return self._start_token_index + self._num_tokens

    def get_num_tokens(self):
        return self._num_tokens

    def get_children(self):
        return self._children

    def is_leaf(self):
        return self.get_children() == 0

    def add_token(self, token_info):
        self._start_token_index = min(self._start_token_index, token_info[0])
        self._tokens.append(token_info)
        self._num_tokens += 1

    def contains_token(self, token):
        if token is None:
            return False
        return self._end_pos >= token


class MyCodeTree:
    def __init__(self):
        self._tree_sitter = MyTreeSitter()
        self._tokens = None
        self._root = None

    @staticmethod
    def _accumulate_ones(sequence_split, max_subtree_size):
        result = list()
        for split in sequence_split:
            if split == 1 and len(result) > 0 and result[-1] < max_subtree_size:
                result[-1] += 1
            else:
                result.append(split)
        return result

    @staticmethod
    def _split_big_leaves(sequence_split, max_subtree_size):
        result = list()
        for split in sequence_split:
            if split <= max_subtree_size:
                result.append(split)
            else:
                while split > max_subtree_size:
                    result.append(max_subtree_size)
                    split -= max_subtree_size
                result.append(split)
        return result

    @staticmethod
    def _post_process_sequence_split(sequence_split, max_subtree_size):
        sequence_split = MyCodeTree._accumulate_ones(sequence_split, max_subtree_size)
        sequence_split = MyCodeTree._split_big_leaves(sequence_split, max_subtree_size)
        return sequence_split

    def process_code(
        self,
        java_code,
        tokens,
        max_subtree_size=16,
        tokens_process_function=lambda x: x[1:] if len(x) > 1 and x[0] == " " else x,
    ):
        self._tree_sitter.process_code(java_code)
        self._tokens = MyTokens(java_code, tokens, tokens_process_function)
        self._root = self._init_traverse_tree_sitter()
        self._tree_sitter.reset()
        sequence_split = self.get_sequence_split(None, max_subtree_size)
        return MyCodeTree._post_process_sequence_split(
            [node.get_num_tokens() for node in sequence_split], max_subtree_size
        )

    def _init_traverse_tree_sitter(self):
        node = MyNode(self._tree_sitter)
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

    def get_sequence_split(self, start_node, max_subtree_size):
        if start_node is None:
            start_node = self._root
        # print("num tokens in subtree", start_node.get_num_tokens(), len(start_node.get_children()), start_node.is_leaf())
        subtree_split = list()
        if start_node.get_num_tokens() < max_subtree_size:
            subtree_split.append(start_node)
        else:
            for chile_node in start_node.get_children():
                subtree_split.extend(
                    self.get_sequence_split(chile_node, max_subtree_size)
                )
        return subtree_split
