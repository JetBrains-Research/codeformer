class MyNode:
    def __init__(self, start_pos, end_pos):
        self._start_pos = start_pos
        self._end_pos = end_pos
        self._children = []
        self._num_tokens = 0
        self._tokens = []

    def get_current_node_start_end(self):
        return self._start_pos, self._end_pos

    def add_children(self, my_node):
        if my_node.get_num_tokens() == 0:
            return
        my_node_start_pos, my_node_end_pos = my_node.get_current_node_start_end()
        self._start_pos = min(self._start_pos, my_node_start_pos)
        self._end_pos = max(self._end_pos, my_node_end_pos)
        self._children.append(my_node)
        self._num_tokens += my_node.get_num_tokens()

    def get_num_tokens(self):
        return self._num_tokens

    def get_children(self):
        return self._children

    def is_leaf(self):
        return self.get_children() == 0

    def add_token(self, token_info):
        self._tokens.append(token_info)
        self._num_tokens += 1

    def contains_token(self, token):
        if token is None:
            return False
        return self._end_pos >= token


class MyTokens:
    def __init__(self, text, tokens):
        self._tokens = tokens
        self._tokens_positions = list()
        current_position = 0
        for token in self._tokens:
            pos = text.find(token, current_position)
            if pos != -1:
                current_position = pos + len(token)
            self._tokens_positions.append(current_position)
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


def merge_left(sequence_split, max_subtree_size):
    result = list()
    for split in sequence_split:
        if len(result) > 0 and result[-1] + split <= max_subtree_size:
            result[-1] += split
        else:
            result.append(split)
    return result


def split_big_leaves(sequence_split, max_subtree_size):
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
