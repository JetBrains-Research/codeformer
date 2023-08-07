import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('MacOSX')


def read_list_data(data_file):
    with open(data_file, 'r') as f:
        result = [int(x) for x in list(f.readline().split(", "))[1:-1]]
    print(np.mean(result))
    return np.array(sorted(result))


def draw_hist(data_file):
    data = read_list_data(data_file)
    x_data, y_data = list(), list()
    for i in range(1, 32):
        x_data.append(i)
        y_data.append(np.count_nonzero(data == i))
    # plt.hist(data, bins=32, ls='dotted', alpha=0.5)  # density=False would make counts
    plt.scatter(x_data, y_data)
    plt.title('Distribution of the split sizes in the method dataset')
    plt.ylabel('Frequency')
    plt.xlabel('Split size (number of tokens)')
    plt.show()


def run_draw(data_file):
    draw_hist(data_file)
