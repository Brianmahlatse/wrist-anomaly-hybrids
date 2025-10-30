"""
Stage 3: Hybrid training on MURA wrist using model_loader services.
"""

import os
import math
import argparse
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AdamW, get_cosine_schedule_with_warmup

import data_utils as m
import model_loader as ml   # <--- use this instead of hand-building
# ml exposes: build_backbones_pair, build_hybrid, load_hybrid_from_checkpoint


def load_wrist(data_root: str):
    y_tr = np.load(os.path.join(data_root, "train", "labels.npy"))
    y_va = np.load(os.path.join(data_root, "val",   "labels.npy"))
    y_te = np.load(os.path.join(data_root, "test",  "labels.npy"))

    x_tr = np.load(os.path.join(data_root, "train", "images.npy"))
    x_va = np.load(os.path.join(data_root, "val",   "images.npy"))
    x_te = np.load(os.path.join(data_root, "test",  "images.npy"))

    train_ds = m.SingleImageDataset(x_tr, y_tr, transform=m._train_transforms)
    val_ds   = m.SingleImageDataset(x_va, y_va, transform=m._val_transforms)
    test_ds  = m.SingleImageDataset(x_te, y_te, transform=m._val_transforms)
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


def train_one_hybrid(train_ds, val_ds, model, out_dir: str):
    lr = 2e-4
    wd = 0.03
    bs = 64
    epochs = 50

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, len(train_ds), bs, epochs)
    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=out_dir,
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
        data_collator=m.collate_fn,
        compute_metrics=m.compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="MURA wrist npy directory")
    parser.add_argument("--output_dir", default="./outputs_stage3")
    # Stage 2 checkpoints for building the backbones
    parser.add_argument("--xc_ckpt", type=str, help="Stage 2 Xception wrist checkpoint")
    parser.add_argument("--deit_ckpt", type=str, help="Stage 2 DeiT wrist checkpoint")
    parser.add_argument("--dn_ckpt", type=str, help="Stage 2 DenseNet201 wrist checkpoint")
    parser.add_argument("--vit_ckpt", type=str, help="Stage 2 ViT-B16 wrist checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds, val_ds, _ = load_wrist(args.data_root)

    # 1) DenseNet + ViT (sequential)
    if args.dn_ckpt and args.vit_ckpt:
        dn_vit_model = ml.build_hybrid(
            fusion_type="sequential",
            cnn_model=ml.load_backbone_weights(ml.build_backbones_pair("dn_vit")[0],
                                               ml.build_backbones_pair("dn_vit")[1],
                                               cnn_ckpt_path=args.dn_ckpt,
                                               vit_ckpt_path=args.vit_ckpt)[0],
            vit_model=ml.load_backbone_weights(ml.build_backbones_pair("dn_vit")[0],
                                               ml.build_backbones_pair("dn_vit")[1],
                                               cnn_ckpt_path=args.dn_ckpt,
                                               vit_ckpt_path=args.vit_ckpt)[1],
        )
        train_one_hybrid(train_ds, val_ds, dn_vit_model,
                         os.path.join(args.output_dir, "stage3_seq_dn_vit"))

    # 2) Xception + DeiT (parallel)
    if args.xc_ckpt and args.deit_ckpt:
        # simpler path: build pair, load weights, then build hybrid
        xc_model, deit_model = ml.build_backbones_pair("xc_deit")
        xc_model, deit_model = ml.load_backbone_weights(
            xc_model, deit_model,
            cnn_ckpt_path=args.xc_ckpt,
            vit_ckpt_path=args.deit_ckpt,
        )
        xc_deit_model = ml.build_hybrid(
            fusion_type="parallel",
            cnn_model=xc_model,
            vit_model=deit_model,
            dropout_p=0.1,
            num_labels=1,
        )
        train_one_hybrid(train_ds, val_ds, xc_deit_model,
                         os.path.join(args.output_dir, "stage3_par_xc_deit"))
