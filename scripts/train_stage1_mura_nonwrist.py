"""
Stage 1: Pretraining on MURA non-wrist regions.

Trains (1) a DeiT classifier and (2) a DenseNet201 classifier on the
non-wrist splits saved as NumPy arrays.

Data layout expected:
    data_root/
        train/images.npy
        train/labels.npy
        val/images.npy
        val/labels.npy
"""

import os
import math
import argparse
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AdamW, get_cosine_schedule_with_warmup

import data_utils as m          # has SingleImageDataset, _train_transforms, _val_transforms
import hybrid_vision as hv      # has densenet201_model, collate_fn_xc, compute_metrics


def load_splits(data_root: str):
    y_train = np.load(os.path.join(data_root, "train", "labels.npy"))
    y_val   = np.load(os.path.join(data_root, "val",   "labels.npy"))

    x_train = np.load(os.path.join(data_root, "train", "images.npy"))
    x_val   = np.load(os.path.join(data_root, "val",   "images.npy"))

    train_dataset = m.SingleImageDataset(x_train, y_train, transform=m._train_transforms)
    val_dataset   = m.SingleImageDataset(x_val,   y_val,   transform=m._val_transforms)
    return train_dataset, val_dataset


def train_deit_stage1(train_dataset, val_dataset, out_dir: str):
    learning_rate = 2e-5
    weight_decay = 0.01
    train_bs = 256
    epochs = 20

    model = m.DeiTClassifier()

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / train_bs)
    total_steps = num_update_steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "stage1_deit"),
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="recall",
        greater_is_better=True,
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=m.collate_fn,
        compute_metrics=m.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


def train_densenet_stage1(train_dataset, val_dataset, out_dir: str):
    learning_rate = 2e-5
    weight_decay = 0.03
    train_bs = 256
    epochs = 50

    model = hv.densenet201_model    # from hybrid_vision.py
    model.requires_grad_(True)      # train 100%

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / train_bs)
    total_steps = num_update_steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=learning_rate, weight_decay=weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "stage1_densenet"),
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=hv.collate_fn_xc,
        compute_metrics=hv.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to MURA non-wrist npy data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--train_deit", action="store_true",
                        help="Train DeiT on Stage 1")
    parser.add_argument("--train_densenet", action="store_true",
                        help="Train DenseNet201 on Stage 1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds = load_splits(args.data_root)

    if args.train_deit:
        train_deit_stage1(train_ds, val_ds, args.output_dir)

    if args.train_densenet:
        train_densenet_stage1(train_ds, val_ds, args.output_dir)
