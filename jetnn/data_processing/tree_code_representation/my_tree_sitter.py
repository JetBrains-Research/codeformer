from tree_sitter import Language, Parser


class MyTreeSitter:
    def __init__(self, programming_language, path_to_tree_sitter):
        self._init = False
        self._parser = None
        self._p_sum = list()
        self._tree = None
        self._current_node = None
        path_to_build = 'tree_sitter_build_' + programming_language + '/my-languages.so'
        Language.build_library(
            path_to_build,
            [path_to_tree_sitter],
        )
        language = Language(path_to_build, programming_language)
        self._parser = Parser()
        self._parser.set_language(language)

    def process_code(self, code):
        self._tree = self._parser.parse(bytes(code, "utf8"))
        self._current_node = self._tree.walk()
        code_lines = code.split("\n")
        self._p_sum = [0 for _ in range(len(code_lines) + 1)]
        for i in range(1, len(code_lines) + 1):
            self._p_sum[i] = self._p_sum[i - 1] + len(code_lines[i - 1]) + 1

    def remove_comments_from_code(self, code):
        tree = self._parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        comment_nodes = []

        def walk(node):
            if 'comment' in node.type.lower(): 
                comment_nodes.append(node)
            for child in node.children:
                walk(child)
        walk(root_node)

        comment_positions = [(node.start_byte, node.end_byte) for node in comment_nodes]
        comment_positions.reverse()
        clean_code = code
        for start, end in comment_positions:
            clean_code = clean_code[:start] + clean_code[end:]
        return clean_code

    def get_current_node_start_end(self):
        start_pos = self._p_sum[int(self._current_node.node.start_point[0])] + int(
            self._current_node.node.start_point[1]
        )
        end_pos = self._p_sum[int(self._current_node.node.end_point[0])] + int(
            self._current_node.node.end_point[1]
        )
        return start_pos, end_pos

    def get_current_node(self):
        return self._current_node

    def get_start_end_bytes(self):
        return self._current_node.node.start_byte, self._current_node.node.end_byte

    def goto_first_child(self):
        return self._current_node.goto_first_child()

    def goto_next_child(self):
        return self._current_node.goto_next_sibling()

    def goto_parent(self):
        return self._current_node.goto_parent()

    def reset(self):
        self._current_node = self._tree.walk()
