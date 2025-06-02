import lightning as L
import torch
import os
from abc import ABC

from utils import plot_specs, plot_weight


class LightningClass(L.LightningModule, ABC):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss_prior",
        }

    def get_loss(self, batch):
        model_output = self(**batch)
        loss_dict = {}
        for key, value in model_output.items():
            if key.startswith("loss_"):
                loss_dict[key] = value * self.loss_weights[key]

        return loss_dict

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            plot_content = plot_weight(
                self.adapter_content.fc.weight.softmax(1).detach().cpu().numpy()
            )
            plot_speaker = plot_weight(
                self.adapter_speaker.fc.weight.softmax(1).detach().cpu().numpy()
            )
            self.logger.log_image(
                "weight/content", [plot_content], step=self.current_epoch
            )
            self.logger.log_image(
                "weight/speaker", [plot_speaker], step=self.current_epoch
            )
        loss_dict = self.get_loss(batch)
        total_loss = sum(loss_dict.values())
        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch["audio_src"].size(0),
            )
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["audio_src"].size(0),
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            model_output = self(**batch)
            mel_len = torch.sum(model_output["mel_mask"], dim=-1).int()

            for i in range(2):
                spec = plot_specs(
                    [
                        model_output["mel"][i, :, : mel_len[i]].detach().cpu().numpy(),
                        model_output["mel_pred"][i, :, : mel_len[i]]
                        .detach()
                        .cpu()
                        .numpy(),
                    ],
                    ["mel", "mel_pred"],
                )
                self.logger.experiment.log(
                    {
                        f"mel_{i}": spec,

                    }
                )

        loss_dict = self.get_loss(batch)
        total_loss = sum(loss_dict.values())
        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch["audio_src"].size(0),
            )
        self.log(
            "val/loss",
            total_loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["audio_src"].size(0),
        )
        return {"val_loss": total_loss, "log": loss_dict}

    def on_test_start(self) -> None:
        assert self.infer_save_root is not None
        if not os.path.exists(self.infer_save_root):
            os.makedirs(self.infer_save_root)
        self.rtf_audio_duration = 0
        self.rtf_inference_time = 0
        import sys

        sys.path.append("hifigan")
        from hifigan.generator import HifiganGenerator

        self.vocoder = HifiganGenerator().to(self.device)
        # ckpt = torch.load("hifigan/logs/libritts3/model-best.pt", map_location="cpu")
        ckpt = torch.load(
            "assets/hifigan.pt",
            map_location="cpu",
        )
        model_ckpt = ckpt["generator"]["model"]
        for k in list(model_ckpt.keys()):
            if k.startswith("module."):
                model_ckpt[k[7:]] = model_ckpt.pop(k)
        self.vocoder.load_state_dict(model_ckpt)
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.vocoder.eval()

    def test_step(self, batch, batch_idx):
        import os
        import soundfile as sf
        import time

        rtf_start = time.time()
        model_output = self(**batch)
        audio_pred = self.vocoder(model_output["mel_pred"])
        rtf_end = time.time()
        self.rtf_inference_time += rtf_end - rtf_start
        self.rtf_audio_duration += batch["audio_src_len"].sum().item()

        loss = 0

        if len(audio_pred.shape) == 3:
            audio_pred = audio_pred.squeeze(1)

        for idx in range(len(model_output["mel_pred"])):
            save_idx = batch_idx * batch["audio_src"].size(0) + idx
            audio = audio_pred[idx, : batch["audio_src_len"][idx]].detach().cpu()
            audio = audio / audio.abs().max()
            sf.write(
                os.path.join(self.infer_save_root, f"{save_idx:04d}_pred.wav"),
                audio,
                samplerate=16000,
            )
            torch.save(
                model_output["mel_pred"][idx].detach().cpu(),
                os.path.join(self.infer_save_root, f"{save_idx:04d}.pt"),
            )
            audio_src = batch["audio_src"][idx, : batch["audio_src_len"][idx]]
            sf.write(
                os.path.join(self.infer_save_root, f"{save_idx:04d}_src.wav"),
                audio_src.detach().cpu(),
                samplerate=16000,
            )
            audio_ref = batch["audio_ref"][idx, : batch["audio_ref_len"][idx]]
            sf.write(
                os.path.join(self.infer_save_root, f"{save_idx:04d}_ref.wav"),
                audio_ref.detach().cpu(),
                samplerate=16000,
            )

        return loss

    def on_test_end(self) -> None:
        self.rtf_audio_duration = float(self.rtf_audio_duration) / 16000
        print(f"RTF: {self.rtf_inference_time / self.rtf_audio_duration:.4f}")

    def state_dict(self):
        drop_list = ["vocoder", "ssl_model"]

        state_dict = super().state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if not any([x in k for x in drop_list]):
                new_state_dict[k] = v
        return new_state_dict

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, config=None):
        # Use torch.load instead of pl_load
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = checkpoint["state_dict"]

        # Drop unwanted keys
        drop_keys = ["ssl_model", "vocoder"]
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not any(d in k for d in drop_keys)
        }
        
        # Instantiate the model with the config
        import yaml
        from munch import Munch
        config = Munch.fromDict(yaml.safe_load(open(config)))
        model = cls(**config)

