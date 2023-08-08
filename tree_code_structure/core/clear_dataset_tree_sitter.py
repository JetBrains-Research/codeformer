import json
import os
import sys
from jetnn.data_processing.tree_representation.my_code_tree import MyCodeTree


def main():
    code_tree = MyCodeTree(
        'python', '/Users/Timofei.Vasilevskii/codeformer/jetnn/vendor/tree-sitter-python'
    )
    core_src = "/home/tim/codeformer/datasets/python_starcoder"
    core_dst = "/home/tim/codeformer/datasets/python_starcoder_processed"
    for data_file in os.listdir(core_src):
        src_file = os.path.join(core_src, data_file)
        dst_file = os.path.join(core_dst, data_file)

        with open(src_file, 'r') as f_read, open(dst_file, 'w') as f_write:
            for raw_sample in f_read.readlines():
                try:
                    sample = json.loads(raw_sample)
                    cleaned_code = code_tree.remove_comments(sample["code"])
                    if len(cleaned_code) < 5:
                        continue
                    f_write.write(raw_sample + '\n')
                except:
                    pass


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
