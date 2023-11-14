import numpy as np
import torch
from tqdm.auto import tqdm
import random
from datasets import load_dataset
import numpy as np
from transformers import TrainingArguments, Trainer
import evaluate

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.nn import functional


def tokenize_function(tokenizer, examples, part, size):
    return (tokenizer(examples[part][0:size]["input"], padding="max_length", truncation=True, return_tensors='pt'),
            tokenizer(examples[part][0:size]["output"], padding="max_length", truncation=True, return_tensors='pt'))


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_dataset(dataset_path):
    raw_dataset = load_dataset(dataset_path)
    dataset_train = tokenize_function(tokenizer, raw_dataset, 'train', len(raw_dataset['train']))
    dataset_val = tokenize_function(tokenizer, raw_dataset, 'val', len(raw_dataset['val']))
    dataset_test = tokenize_function(tokenizer, raw_dataset, 'test', len(raw_dataset['test']))
    return dataset_train, dataset_val, dataset_test


class FineTunedModel:
    def __init__(self, model_path, tokenizer_path, num_labels):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def compute_loss(self, model, dataset, tokenizer, use_grad=False):
        with torch.set_grad_enabled(use_grad):
            model_logits = model(input_ids=dataset[0]['input_ids'], attention_mask=dataset[0]['attention_mask'],
                                 decoder_input_ids=dataset[1]['input_ids'],
                                 decoder_attention_mask=dataset[1]['attention_mask']).logits
            cross_entropy_tensor = functional.cross_entropy(model_logits[:, :-1].reshape(-1, model_logits.shape[-1]),
                                                            dataset[1]['input_ids'][:, 1:].reshape(-1),
                                                            reduction="none",
                                                            ignore_index=tokenizer.pad_token_id).reshape(
                model_logits.shape[0], model_logits.shape[1] - 1)
        return None


if __name__ == "__main__":
    metric = evaluate.load("accuracy")
    bart_model = FineTunedModel("facebook/bart-large", "facebook/bart-large", num_labels=3)
    dataset_train, dataset_val, dataset_test = load_dataset("tau/scrolls", 'quality')
    training_args = TrainingArguments(output_dir="test_training", evaluation_strategy="steps")
    trainer = Trainer(
        model=bart_model.model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
