import argparse
import jsonlines, yaml

from lm_eval import evaluator, tasks


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/data/shared-data/codeformer/models/rand_init_codeformer.yaml", help="Config file location")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = yaml.safe_load(open(args.config))

    results = evaluator.evaluate(adaptor, tasks.get_task_dict(["scrolls_contractnli",
                                                               "scrolls_govreport",
                                                               "scrolls_narrativeqa",
                                                               "scrolls_qasper",
                                                               "scrolls_qmsum",
                                                               "scrolls_quality",
                                                               "scrolls_summscreenfd",]), False, 0, None)
    with jsonlines.open('scrolls.jsonl', mode='w') as writer:
        for item in results:
            writer.write(item)
    print(results)