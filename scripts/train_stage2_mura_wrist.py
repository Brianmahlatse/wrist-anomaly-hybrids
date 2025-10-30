
"""
Stage 2: Fine-tuning on the MURA wrist subset.

This script shows two examples:
1. Fine-tune Xception from a Stage 1 checkpoint (freeze ~30%, train last 70%).
2. Fine-tune DeiT-B from a Stage 1 checkpoint (freeze embeddings, train last 60%).

It expects the wrist data already saved as NumPy arrays:
    data_root/
        train/images.npy
        train/labels.npy
        val/images.npy
        val/labels.npy
        test/images.npy
        test/labels.npy
and Stage 1 checkpoints saved as .pth files.
"""

import os
import math
import argparse
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AdamW, get_cosine_schedule_with_warmup

import data_utils as m    
import hybrid_vision as hv  


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


def finetune_xception(train_ds, val_ds, ckpt_path: str, out_dir: str):
    # 1) load Stage 1 checkpoint
    model = m.XceptionClassifier()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    # 2) unfreeze last 70% of CNN
    m.set_trainable_fraction(model, kind="cnn", fraction=0.7)

    # 3) optimizer/scheduler
    lr = 2e-5
    wd = 0.03
    bs = 64
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "stage2_xception"),
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=m.collate_fn_xc,
        compute_metrics=m.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()


def finetune_deit(train_ds, val_ds, ckpt_path: str, out_dir: str):
    # 1) load Stage 1 checkpoint
    deit_model = m.DeiTClassifier()
    state = torch.load(ckpt_path, map_location="cpu")
    deit_model.load_state_dict(state)

    # 2) unfreeze ~60% transformer layers, keep embeddings frozen
    m.set_trainable_fraction(deit_model, kind="vit", fraction=0.6, freeze_embeddings=True)

    lr = 2e-4        
    wd = 0.03
    bs = 64
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, deit_model.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "stage2_deit"),
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
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=deit_model,
        args=args,
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
    parser.add_argument("--output_dir", type=str, default="./outputs_stage2",
                        help="Where to save Stage 2 checkpoints/logs")
    parser.add_argument("--xception_ckpt", type=str, default=None,
                        help="Path to Stage 1 Xception checkpoint (.pth)")
    parser.add_argument("--deit_ckpt", type=str, default=None,
                        help="Path to Stage 1 DeiT checkpoint (.pth)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds, val_ds, _ = load_wrist_splits(args.data_root)

    if args.xception_ckpt is not None:
        finetune_xception(train_ds, val_ds, args.xception_ckpt, args.output_dir)

    if args.deit_ckpt is not None:
        finetune_deit(train_ds, val_ds, args.deit_ckpt, args.output_dir)
