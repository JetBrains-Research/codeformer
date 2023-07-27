import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def remove_java_comments(code):
    def non_comment(line):
        if len(line) == 0:
            return False
        return not (
            line[0][0] == "*"
            or line[0] == "/**"
            or (len(line) > 1 and (line[0][0:2] == "/*" or line[0][0:2] == "//"))
        )

    return "\n".join(
        list(
            filter(
                lambda s: non_comment(
                    list(filter(lambda ss: len(ss) > 0, s.split(" ")))
                ),
                code.splitlines(),
            )
        )
    )


def tokenize(tokenizer, text: str, max_parts: int):
    return tokenizer.encode(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_parts,
        truncation="longest_first",
    )


def read_list_data(data_file):
    with open(data_file, "r") as f:
        result = [int(x) for x in list(f.readline().split(","))[1:-1]]
    print(np.mean(result))
    return np.array(sorted(result))


def draw_hist(data_file):
    data = read_list_data(data_file)
    # x_data, y_data = list(), list()
    # for i in range(1, 32):
    #     x_data.append(i)
    #     y_data.append(np.count_nonzero(data == i))
    data = data.clip(0, 2e4)
    plt.hist(data, bins=10, ls="dotted", alpha=0.5)  # density=False would make counts
    # plt.scatter(x_data, y_data)
    plt.title("Distribution of number of symbols in the file dataset")
    plt.ylabel("Frequency")
    plt.xlabel("Number of symbols")
    plt.show()
