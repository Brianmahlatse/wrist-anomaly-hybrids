"""
Stage 4: External fine-tuning on Al-Huda using model_loader.
"""

import os
import math
import argparse
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AdamW, get_cosine_schedule_with_warmup

import data_utils as m
import model_loader as ml
import hybrid_vision as hv   # for set_trainable_fraction if you put it there, else from data_utils


def load_alhuda(data_root: str):
    ft_x = np.load(os.path.join(data_root, "finetune_images.npy"))
    ft_y = np.load(os.path.join(data_root, "finetune_labels.npy"))
    va_x = np.load(os.path.join(data_root, "val_images.npy"))
    va_y = np.load(os.path.join(data_root, "val_labels.npy"))
    te_x = np.load(os.path.join(data_root, "test_images.npy"))
    te_y = np.load(os.path.join(data_root, "test_labels.npy"))

    train_ds = m.SingleImageDataset(ft_x, ft_y, transform=m._train_transforms)
    val_ds   = m.SingleImageDataset(va_x, va_y, transform=m._val_transforms)
    test_ds  = m.SingleImageDataset(te_x, te_y, transform=m._val_transforms)
    return train_ds, val_ds, test_ds


def make_scheduler(optimizer, dataset_len, batch_size, epochs):
    steps_per_epoch = math.ceil(dataset_len / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


def unfreeze_last_5pct(hybrid_model):
    # your hybrids have .cnn_extractor and .vit_extractor inside
    hv.set_trainable_fraction(hybrid_model.cnn_extractor, kind="cnn", fraction=0.05, freeze_embeddings=True)
    hv.set_trainable_fraction(hybrid_model.vit_extractor, kind="vit", fraction=0.05, freeze_embeddings=True)


def finetune(hybrid_model, train_ds, val_ds, out_dir, lr, wd):
    bs = 32
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, hybrid_model.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)
    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=out_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_dir=os.path.join(out_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=hybrid_model,
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
    parser.add_argument("--data_root", required=True,
                        help="Path to Al-Huda .npy files")
    parser.add_argument("--output_dir", default="./outputs_stage4")
    # stage 3 ckpts
    parser.add_argument("--dn_vit_stage3", type=str, help="Stage 3 DenseNet--ViT hybrid ckpt")
    parser.add_argument("--xc_deit_stage3", type=str, help="Stage 3 Xception--DeiT hybrid ckpt")
    # optional standalone ckpts (if Stage 3 was trained with them)
    parser.add_argument("--dn_ckpt", type=str)
    parser.add_argument("--vit_ckpt", type=str)
    parser.add_argument("--xc_ckpt", type=str)
    parser.add_argument("--deit_ckpt", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds, val_ds, _ = load_alhuda(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DN--ViT external
    if args.dn_vit_stage3:
        dn_vit = ml.load_hybrid_from_checkpoint(
            fusion_type="parallel",      # or "sequential" 
            pair_name="dn_vit",
            hybrid_ckpt_path=args.dn_vit_stage3,
            cnn_ckpt_path=args.dn_ckpt,
            vit_ckpt_path=args.vit_ckpt,
            map_location="cpu",
        ).to(device)
        unfreeze_last_5pct(dn_vit)
        finetune(dn_vit, train_ds, val_ds,
                 os.path.join(args.output_dir, "stage4_dn_vit"),
                 lr=1e-3, wd=0.01)

    # 2) XC--DeiT external
    if args.xc_deit_stage3:
        xc_deit = ml.load_hybrid_from_checkpoint(
            fusion_type="parallel",
            pair_name="xc_deit",
            hybrid_ckpt_path=args.xc_deit_stage3,
            cnn_ckpt_path=args.xc_ckpt,
            vit_ckpt_path=args.deit_ckpt,
            map_location="cpu",
        ).to(device)
        unfreeze_last_5pct(xc_deit)
        finetune(xc_deit, train_ds, val_ds,
                 os.path.join(args.output_dir, "stage4_xc_deit"),
                 lr=1e-4, wd=0.02)  
