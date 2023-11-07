import argparse
import jsonlines, yaml
from lm.utils import get_model_from_config, get_tokenizer_from_config
from lm.lm_eval_adapter import *
from lm_eval import evaluator, tasks


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/aspr/PycharmProjects/codeformer/artifacts/models/rand_init_codeformer.yaml", help="Config file location")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    #params = yaml.safe_load(open(args.config))
    model = get_model_from_config(args.config)
    tokenizer = get_tokenizer_from_config(args.config)
    adapter = EvalHarnessAdapter(model, tokenizer)
    results = evaluator.evaluate(adapter, tasks.get_task_dict(["scrolls_contractnli"]), False, 0, None)
                                                               # "scrolls_govreport",
                                                               # "scrolls_narrativeqa",
                                                               # "scrolls_qasper",
                                                               # "scrolls_qmsum",
                                                               # "scrolls_quality",
                                                               # "scrolls_summscreenfd",]), False, 0, None)
    with jsonlines.open('scrolls.jsonl', mode='w') as writer:
        for item in results:
            writer.write(item)
    print(results)

