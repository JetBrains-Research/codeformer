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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(tokenizer, examples, part, size):
    return (tokenizer(examples[part][0:size]["input"], padding="max_length", truncation=True, return_tensors='pt'),
            tokenizer(examples[part][0:size]["output"], padding="max_length", truncation=True, return_tensors='pt'))

def bart_inference(dataset):
    raw_dataset = load_dataset(dataset)
    model = AutoModelForCausalLM.from_pretrained("facebook/bart-large", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    dataset_train = tokenize_function(tokenizer, raw_dataset, 'train', len(raw_dataset['train']))
    dataset_val = tokenize_function(tokenizer, raw_dataset, 'val', len(raw_dataset['val']))
    dataset_test = tokenize_function(tokenizer, raw_dataset, 'test', len(raw_dataset['test']))


if __name__ == "__main__":
    metric = evaluate.load("accuracy")
    raw_dataset = load_dataset("tau/scrolls", 'quality')
    print(raw_dataset["train"][0])
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    dataset_train = tokenize_function(tokenizer, raw_dataset, 'train')
    dataset_val = tokenize_function(tokenizer, raw_dataset, 'val')

    training_args = TrainingArguments(output_dir="test_training", evaluation_strategy="steps")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()