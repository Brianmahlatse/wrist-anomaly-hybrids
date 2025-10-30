"""
Stage 3: Hybrid CNN–ViT training on the MURA wrist dataset.

This script shows TWO concrete examples:
1. DenseNet201 (wrist) + ViT-B16 (wrist) → SequentialHybridClassifier
   - CNN trainable fraction: 0.2
   - ViT trainable fraction: 0.2
2. Xception (wrist) + DeiT-B (wrist) → ParallelHybridClassifier
   - CNN trainable fraction: 0.4
   - ViT trainable fraction: 0.2

Both use:
- AdamW
- cosine schedule with 10% step-based warm-up
- early stopping (patience = 5)
- batch size 64
- 50 epochs

This matches Stage 3 in the manuscript.
"""

import os
import math
import argparse
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AdamW, get_cosine_schedule_with_warmup

import data_utils as m   # SingleImageDataset, transforms, collate_fn, compute_metrics
import hybrid_vision as hv  # CNNExtractor, ViTFeatureLayer, SequentialHybridClassifier, ParallelHybridClassifier


def load_wrist_splits(data_root: str):
    y_train = np.load(os.path.join(data_root, "train", "labels.npy"))
    y_val   = np.load(os.path.join(data_root, "val",   "labels.npy"))
    y_test  = np.load(os.path.join(data_root, "test",  "labels.npy"))

    x_train = np.load(os.path.join(data_root, "train", "images.npy"))
    x_val   = np.load(os.path.join(data_root, "val",   "images.npy"))
    x_test  = np.load(os.path.join(data_root, "test",  "images.npy"))

    train_dataset = m.SingleImageDataset(x_train, y_train, transform=m._train_transforms)
    val_dataset   = m.SingleImageDataset(x_val,   y_val,   transform=m._val_transforms)
    test_dataset  = m.SingleImageDataset(x_test,  y_test,  transform=m._val_transforms)
    return train_dataset, val_dataset, test_dataset


def make_scheduler(optimizer, dataset_len, batch_size, epochs):
    num_update_steps_per_epoch = math.ceil(dataset_len / batch_size)
    total_steps = num_update_steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


def train_sequential_dn_vit(train_ds, val_ds, args):
    # 1) load CNN (DenseNet201 wrist) and ViT (ViT-B16 wrist)
    dn_state = torch.load(args.densenet_ckpt, map_location="cpu")
    vit_state = torch.load(args.vit_ckpt, map_location="cpu")

    dn_model = hv.densenet201_model
    dn_model.load_state_dict(dn_state)

    vit_model = hv.vit_base_model
    vit_model.load_state_dict(vit_state)

    # 2) wrap into extractors
    cnn_extractor = hv.CNNExtractor(dn_model)
    vit_extractor = hv.ViTFeatureLayer(vit_model, dropout_p=0.1)

    # freeze 80%, train last 20% (as in your code)
    m.set_trainable_fraction(cnn_extractor, kind="cnn", fraction=0.2, freeze_embeddings=True)
    m.set_trainable_fraction(vit_extractor, kind="vit", fraction=0.2, freeze_embeddings=True)

    # 3) build hybrid
    hybrid = hv.SequentialHybridClassifier(
        cnn_extractor=cnn_extractor,
        vit_extractor=vit_extractor,
        num_labels=1,
    )

    # 4) optimizer/scheduler
    lr = 2e-4
    wd = 0.03
    bs = 64
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, hybrid.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)

    use_fp16 = torch.cuda.is_available()

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "stage3_sequential_dn_vit"),
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=hybrid,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=m.collate_fn,
        compute_metrics=m.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()


def train_parallel_xc_deit(train_ds, val_ds, args):
    # 1) load CNN (Xception wrist) and DeiT (wrist)
    xc_state = torch.load(args.xception_ckpt, map_location="cpu")
    deit_state = torch.load(args.deit_ckpt, map_location="cpu")

    xc_model = m.XceptionClassifier()
    xc_model.load_state_dict(xc_state)

    deit_model = m.DeiTClassifier()
    deit_model.load_state_dict(deit_state)

    # 2) wrap into extractors
    cnn_extractor = hv.CNNExtractor(xc_model)
    vit_extractor = hv.ViTFeatureLayer(deit_model, dropout_p=0.1)

    # Stage 3 fractions from your code: CNN 0.4, ViT 0.2
    m.set_trainable_fraction(cnn_extractor, kind="cnn", fraction=0.4, freeze_embeddings=True)
    m.set_trainable_fraction(vit_extractor, kind="vit", fraction=0.2, freeze_embeddings=True)

    # 3) build hybrid
    hybrid = hv.ParallelHybridClassifier(
        cnn_extractor=cnn_extractor,
        vit_extractor=vit_extractor,
        num_labels=1,
    )

    # 4) optimizer/scheduler
    lr = 2e-4
    wd = 0.03
    bs = 64
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, hybrid.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)

    use_fp16 = torch.cuda.is_available()

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "stage3_parallel_xc_deit"),
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=hybrid,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=m.collate_fn,
        compute_metrics=m.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to MURA wrist npy data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs_stage3",
                        help="Where to save Stage 3 checkpoints/logs")
    # sequential (DenseNet + ViT)
    parser.add_argument("--densenet_ckpt", type=str, default=None,
                        help="Path to Stage 2 DenseNet201 wrist checkpoint")
    parser.add_argument("--vit_ckpt", type=str, default=None,
                        help="Path to Stage 2 ViT-B16 wrist checkpoint")
    # parallel (Xception + DeiT)
    parser.add_argument("--xception_ckpt", type=str, default=None,
                        help="Path to Stage 2 Xception wrist checkpoint")
    parser.add_argument("--deit_ckpt", type=str, default=None,
                        help="Path to Stage 2 DeiT-B wrist checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds, val_ds, _ = load_wrist_splits(args.data_root)

    # run whichever pairs the user provided
    if args.densenet_ckpt is not None and args.vit_ckpt is not None:
        train_sequential_dn_vit(train_ds, val_ds, args)

    if args.xception_ckpt is not None and args.deit_ckpt is not None:
        train_parallel_xc_deit(train_ds, val_ds, args)
