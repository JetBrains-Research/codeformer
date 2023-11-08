# simplest sanity check for whether lm_eval works at all
import jsonlines
from lm_eval import evaluator, tasks


if __name__ == "__main__":
    results = evaluator.simple_evaluate(model="hf-causal",
        model_args="pretrained=EleutherAI/gpt-j-6B", tasks = tasks.get_task_dict(["scrolls_contractnli"]), device="cpu")
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

