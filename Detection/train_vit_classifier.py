import os
from PIL import Image
import torch
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def main():
    # === Step 1: Load dataset ===
    data_path = "./datasets/imagenet"  # Make sure train/val/test folders are under this
    dataset = load_dataset("imagefolder", data_dir=data_path)
    print(dataset)

    # === Step 2: Load processor (image transforms) ===
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    transform = Compose([
        Resize(processor.size["height"]),
        CenterCrop(processor.size["height"]),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    def transform_example(example):
        image = example["image"].convert("RGB")
        example["pixel_values"] = transform(image)
        return example

    dataset = dataset.map(transform_example, remove_columns=["image"])

    # === Step 3: Setup binary labels ===
    label2id = {"adm": 0, "real": 1}
    id2label = {0: "adm", 1: "real"}

    def encode_labels(example):
        label = example["label"]
        if isinstance(label, str):  # If label is still string, map it
            example["label"] = label2id[label]
        return example

    dataset = dataset.map(encode_labels)

    # === Step 4: Load model ===
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # === Step 5: Training arguments ===
    training_args = TrainingArguments(
        output_dir=r"D:\VS Projects\ECE-580\Proj\Detection\vit_model",
        overwrite_output_dir=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = np.mean(preds == labels)
        return {"accuracy": acc}

    # === Step 6: Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # === Step 7: Train and evaluate ===
    trainer.train()
    trainer.evaluate(dataset["test"])

    # === Step 8: Save model & processor ===
    save_path = r"D:\VS Projects\ECE-580\Proj\Detection"
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)

if __name__ == "__main__":
    main()
