from tree_sitter import Language, Parser


class MyTreeSitter:
    def __init__(self, programming_language, path_to_tree_sitter):
        self._init = False
        # self._parser = None
        self._pl = programming_language
        self._p_sum = list()
        self._tree = None
        self._current_node = None
        self._path_to_build = "tree_sitter_build_" + programming_language + "/my-languages.so"
        Language.build_library(
            self._path_to_build,
            [path_to_tree_sitter],
        )
        

    def process_code(self, code):
        parser = Parser()
        parser.set_language(Language(self._path_to_build, self._pl))
        self._tree = parser.parse(bytes(code, "utf8"))
        self._current_node = self._tree.walk()

    def remove_comments_from_code(self, code, method_location):
        parser = Parser()
        parser.set_language(Language(self._path_to_build, self._pl))
        code_bytes = bytes(code, 'utf8')
        bytes_method_location = [len(bytes(code[:method_location[i]], 'utf8')) for i in range(2)]
        tree = parser.parse(code_bytes)
        root_node = tree.root_node
        shift = 0
        shrink = 0
        comment_nodes = []

        def walk(node):
            if "comment" in node.type.lower():
                comment_nodes.append(node)
            for child in node.children:
                walk(child)

        walk(root_node)

        comment_positions = [(node.start_byte, node.end_byte) for node in comment_nodes]
        comment_text = "\n".join([str(code_bytes[start_byte: end_byte], 'utf8') for start_byte, end_byte in comment_positions])
        
        for start_byte, end_byte in comment_positions:
            if start_byte < bytes_method_location[0]:
                assert end_byte <= bytes_method_location[0]
                shift -= (end_byte - start_byte)
            elif start_byte >= bytes_method_location[0] and start_byte < bytes_method_location[1]:
                assert end_byte <= bytes_method_location[1]
                shrink -= (end_byte - start_byte)
                
        new_bytes_method_location = (bytes_method_location[0] + shift, bytes_method_location[1] + shift + shrink)
        new_method_location = [len(str(code_bytes[:new_bytes_method_location[i]], 'utf8')) for i in range(2)]
        comment_positions.reverse()
        clean_bytes = code_bytes
        for start, end in comment_positions:
            clean_bytes = clean_bytes[:start] + clean_bytes[end:]
        clean_code = str(clean_bytes, 'utf8')
        return clean_code, comment_text, new_method_location

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
