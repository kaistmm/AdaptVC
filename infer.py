import lightning as L
import torch
from models.adaptvc import AdaptVC
from dataset import AdaptVCDataset
import argparse
from munch import Munch
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()

    test_ds = AdaptVCDataset(
        metadata="assets/metadata/test.csv",
        max_len=-1,
        max_len_ref=72000,
        data_root="data/LibriTTS",
        evalmode=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=4,
        drop_last=False,
    )

    trainer = L.Trainer(devices=1, logger=False)
    model = AdaptVC.load_from_checkpoint(args.ckpt_path, map_location="cpu", config="configs/config.yaml")

    model.infer_save_root = args.ckpt_path.replace(".ckpt", "")
    trainer.test(model, test_loader)
