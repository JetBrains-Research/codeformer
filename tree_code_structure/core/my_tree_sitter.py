from tree_sitter import Language, Parser


class MethodIdentifier:

    def __init__(self, identifier, start_pos, end_pos):
        self._identifier = identifier
        self._location = (start_pos, end_pos)

    def get_identifier(self):
        return self._identifier
    
    def get_location(self):
        return self._location


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
            ["/home/tim/JetNN/tree_code_structure/core/tree-sitter-java"],
        )
        java_language = Language("tree_sitter_build/my-languages.so", "java")
        self._parser = Parser()
        self._parser.set_language(java_language)

    def process_code(self, java_code):
        if not self._init:
            self.init()
        self._code = java_code
        self._identifier_to_locations = dict()
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
    
    def find_method_names(self):
        if self.get_current_node().node.type == 'identifier':
            start_pos, end_pos = self.get_current_node_start_end()
            method_name = self._code[start_pos: end_pos].lower()
            if method_name not in self._identifier_to_locations:
                self._identifier_to_locations[method_name] = list()
            self._identifier_to_locations[method_name].append((start_pos, end_pos))
            
        is_leaf = not self.goto_first_child()
        if is_leaf:
            pass
        else:
            self.find_method_names()
            while self.goto_next_child():
                self.find_method_names()
            self.goto_parent()

    def get_identifiers_locations(self):
        return self._identifier_to_locations

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
