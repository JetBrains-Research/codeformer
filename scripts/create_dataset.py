import os
import random

src_path = "/home/tim/JetNN/raw_data"
dst_path = "/home/tim/JetNN/datasets/gradle_new"

for dataset in os.listdir(src_path):
    path = os.path.join(src_path, dataset)
    projects_list = os.listdir(path)
    random.shuffle(projects_list)
    path_to_dst = os.path.join(dst_path, dataset + ".jsonl")
    with open(os.path.join(path_to_dst), "w") as out_file:
        for project in projects_list:
            path_to_project = os.path.join(path, project)
            path_to_file = os.path.join(path_to_project, "result.jsonl")
            with open(path_to_file) as in_file:
                data = in_file.read()
                out_file.write(data + "\n")
