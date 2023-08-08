import json
import os
import sys
from my_tree_sitter import MyTreeSitter

class MethodNameNotFound(Exception):
    pass


def _process_label(label: str) -> str:
    s = list()
    for c in label:
        if c.isupper():
            s.append("|")
            s.append(c.lower())
        else:
            s.append(c)
    result = "".join(s)
    return result


def _get_method_name_lower_from_label(label):
    return "".join(label.split("|"))


def _process_code(mask_method_name, code):
    tree_sitter = MyTreeSitter()
    tree_sitter.process_code(code)
    # print(code)
    tree_sitter.find_method_names()

    identifiers_locations = tree_sitter.get_identifiers_locations()
    if mask_method_name not in identifiers_locations:
        raise MethodNameNotFound
    
    method_name_locations = identifiers_locations[mask_method_name]
    return _replace_method_names(code, method_name_locations)


def _replace_method_names(code, method_name_locations):
    replace_symbol = 'METHOD_NAME'
    new_code = str()
    if len(method_name_locations) == 0:
        raise MethodNameNotFound
    
    last_idx = 0
    for loc in method_name_locations:
        new_code += (code[last_idx: loc[0]] + replace_symbol)
        last_idx = loc[1]
    new_code += code[last_idx: len(code)]
    # print(new_code)
    return new_code


def main():
    core_src = "/home/tim/JetNN/gradle_data/val"
    core_dst = "/home/tim/JetNN/raw_data/val"
    cnt_all = 0
    cnt_invalid = 0
    for src_dir_name in os.listdir(core_src):
        src_dir = os.path.join(core_src, src_dir_name)
        src_file = os.path.join(src_dir, 'result.jsonl')
        dst_dir = os.path.join(core_dst, src_dir_name)
        os.mkdir(dst_dir)
        dst_file = os.path.join(dst_dir, 'result.jsonl')

        if not os.path.exists(src_file):
            continue 

        with open(src_file, 'r') as f_read, open(dst_file, 'w') as f_write:
            for raw_sample in f_read.readlines():
                cnt_all += 1
                sample = json.loads(raw_sample)
                first_label = sample["label"].split(',')[0]
                processed_label = _process_label(first_label)
                sample["label"] = processed_label
                try:
                    method_name = _get_method_name_lower_from_label(first_label)
                    processed_code = _process_code(method_name, sample["code"])
                    sample["code"] = processed_code
                    json.dump(sample, f_write)
                    f_write.write('\n')
                except MethodNameNotFound:
                    cnt_invalid += 1
    print("percentage of invalid samples", cnt_invalid / cnt_all)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
