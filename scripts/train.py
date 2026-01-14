"""
This script fine-tuned a sentiment analysis model on product reviews
and saves the trained model in a local directory (/sentiment_model).
"""

import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    logging as hf_logging
)
hf_logging.set_verbosity_error()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# this function computes accuracy and f1 score during evaluation
def compute_metrics(eval_pred):
    """
    Compute accuracy and f1 score from evaluation predictions.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # calculate accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # calculate f1 score
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# main training function
def train_model():
    """
    The function loads a dataset from a URL, converts ratings to sentiment labels,
    fine-tunes the model, and saves the trained model locally.
    """
    # load dataset via URL from huggingface
    data_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl"

    dataset_stream = load_dataset("json", data_files= data_url, split="train", streaming=True)

    # create empty lists for each sentiment category, set a target number for each category
    negatives, neutrals, positives = [], [], []
    target = 3000

    # process data to suit our calassification needs
    for example in dataset_stream:
        rating = example['rating']
        if rating <= 2.0 and len(negatives) < target:
            example['label'] = 0
            negatives.append(example)
        elif rating == 3 and len(neutrals) < target:
            example['label'] = 1
            neutrals.append(example)
        elif rating >= 4 and len(positives) < target:
            example['label'] = 2
            positives.append(example)
        
        if len(negatives) == target and len(neutrals) == target and len(positives) == target:
            break

    # create dataset object and shuffle the data
    df = pd.DataFrame(negatives + neutrals + positives)
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)

    # tokenization and model setup
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map((preprocess_function), batched=True)
    
    # split dataset into train and test sets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    # load pre-trained model and set up training arguments
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
        )
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # initialize trainer, include compute_metrics for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics = compute_metrics
    )

    # start training
    trainer.train()
    
    # save the trained model
    trainer.save_model("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

if __name__ == "__main__":
    train_model()
