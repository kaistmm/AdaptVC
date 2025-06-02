import lightning as L
from lightning.pytorch import callbacks
from lightning.pytorch import loggers
import torch
import os
import argparse
import yaml
from munch import Munch

torch.set_float32_matmul_precision("high")  # medium
from models.adaptvc import AdaptVC
from dataset import AdaptVCDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=48000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_epoch_interval", type=int, default=50)
    args = parser.parse_args()

    config = Munch.fromDict(yaml.safe_load(open(args.config)))
    model = AdaptVC(**config)

    train_ds = AdaptVCDataset(
        metadata="assets/metadata/train.csv",
        max_len=args.max_len,
        max_len_ref=48000,
        data_root="data/LibriTTS",
        evalmode=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_ds.collate_fn,
    )

    val_ds = AdaptVCDataset(
        metadata="assets/metadata/val.csv",
        max_len=-1,
        max_len_ref=48000,
        data_root="data/LibriTTS",
        evalmode=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_ds.collate_fn,
    )

    experiment_root = os.path.join("./logs", args.exp_name)
    os.makedirs(experiment_root, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="gpu",
        precision=args.precision,
        log_every_n_steps=args.log_interval,
        enable_model_summary=False,
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
            callbacks.RichProgressBar(),
            callbacks.LearningRateMonitor(logging_interval="epoch"),
            callbacks.ModelCheckpoint(
                dirpath=experiment_root,
                save_top_k=-1,
                every_n_epochs=5,
                save_last=True,
            ),
        ],
        check_val_every_n_epoch=args.eval_epoch_interval,
        logger=loggers.WandbLogger(
            project="ICASSP-VC", save_dir=experiment_root, name=args.exp_name
        ),
    )
    trainer.logger.log_hyperparams(config)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
