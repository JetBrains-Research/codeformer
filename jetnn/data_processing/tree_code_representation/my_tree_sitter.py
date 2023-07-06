from tree_sitter import Language, Parser


class MyTreeSitter:
    def __init__(self):
        self._init = False
        self._parser = None
        self._p_sum = list()
        self._tree = None
        self._current_node = None

    def init(self):
        self._init = True
        Language.build_library(
            "tree_sitter_build/my-languages.so",
            ["../vendor/tree-sitter-java"],
        )
        java_language = Language("tree_sitter_build/my-languages.so", "java")
        self._parser = Parser()
        self._parser.set_language(java_language)

    def process_code(self, java_code):
        if not self._init:
            self.init()
        code_lines = java_code.split("\n")
        self._p_sum = [0 for _ in range(len(code_lines) + 1)]
        for i in range(1, len(code_lines) + 1):
            self._p_sum[i] = self._p_sum[i - 1] + len(code_lines[i - 1]) + 1
        self._tree = self._parser.parse(bytes(java_code, "utf8"))
        self._current_node = self._tree.walk()

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

    def goto_first_child(self):
        return self._current_node.goto_first_child()

    def goto_next_child(self):
        return self._current_node.goto_next_sibling()

    def goto_parent(self):
        return self._current_node.goto_parent()

    def reset(self):
        self._current_node = self._tree.walk()
