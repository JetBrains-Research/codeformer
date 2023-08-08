from core.DatasetStatistics import DatasetStatistics
from utils.utils import draw_hist

path_to_data = "/Users/timofeyvasilevskij/JetBrains/JetNN/datasets/gradle/train.jsonl"
path_to_log = "/Users/timofeyvasilevskij/JetBrains/JetNN/logs/gradle_symbols.txt"
ds_stat = DatasetStatistics(path_to_data, path_to_log)

ds_stat.calc_symbols_distribution()
draw_hist(ds_stat.get_path_to_log())
