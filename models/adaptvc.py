import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
from transformers.utils import logging
import math

logging.set_verbosity_error()
import vector_quantize_pytorch

from models.flow_matching import CFM
from models.lightningclass import LightningClass
from munch import Munch


class Adapter(nn.Module):
    def __init__(self, in_dim=12, out_dim=1):
        super(Adapter, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc.weight.data.uniform_(1 / in_dim, 1 / in_dim)

    def forward(self, x):
        return torch.matmul(
            x.permute(0, 2, 3, 1),
            F.softmax(self.fc.weight, dim=1).T,
        ).squeeze(-1)


class LogMelSpectrogram(torch.nn.Module):

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1280,
        window_length=1280,
        hop_length=320,
        center=False,
        normalized=False,
        power=1.0,
        n_mels=80,
        clamp_min=1e-5,
        eps=1e-9,
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.center = center
        self.power = power
        self.normalized = normalized
        self.clamp_min = clamp_min
        self.eps = eps

        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.window_length,
            hop_length=self.hop_length,
            center=self.center,
            power=self.power,
            n_mels=self.n_mels,
            normalized=self.normalized,
        )

    def forward(self, wav, wav_len):
        wav = F.pad(
            wav,
            (
                (self.window_length - self.hop_length) // 2,
                (self.window_length - self.hop_length) // 2,
            ),
            "reflect",
        )
        mel = self.melspctrogram(wav)
        mel_len = wav_len // self.hop_length
        logmel = torch.log(torch.clamp(mel, min=self.clamp_min) + self.eps)
        return logmel, mel_len


class AdaptVC(LightningClass):
    def __init__(
        self,
        mel_params,
        attention_params,
        hidden_dim,
        codebook_dim,
        adapter_layers,
        ssl_model,
        cache_dir,
        cfm_params,
        decoder_params,
        optimizer_params,
        scheduler_params,
        loss_weights,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.loss_weights = loss_weights
        self.mel_params = mel_params
        self.mel_transform = LogMelSpectrogram(**self.mel_params)

        self.adapter_content = Adapter(in_dim=adapter_layers, out_dim=1)
        self.vq = vector_quantize_pytorch.VectorQuantize(
            hidden_dim,
            codebook_size=codebook_dim,
        )
        self.adapter_speaker = Adapter(in_dim=adapter_layers, out_dim=1)

        self.speaker_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, **attention_params
        )

        self.proj_mel = nn.Linear(hidden_dim, self.mel_params["n_mels"])

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ssl_model, cache_dir=cache_dir)
        self.ssl_model = AutoModel.from_pretrained(ssl_model, cache_dir=cache_dir)
        print(f'Using SSL model: {ssl_model}')
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        cfm_params = Munch(cfm_params)
        decoder_params = Munch(decoder_params)
        self.decoder = CFM(
            in_channels=self.mel_params["n_mels"] * 2,
            out_channel=self.mel_params["n_mels"],
            cfm_params=cfm_params,
            decoder_params=decoder_params,
        )

        self.recon_loss = nn.L1Loss()

        self.infer_save_root = None
        self.strict_loading = False

    def forward(
        self,
        audio_src,
        audio_src_len,
        audio_ref,
        audio_ref_len,
        n_timesteps=10,
        **kwargs,
    ):
        ssl_src = self.extract_ssl_features(audio_src, audio_src_len)
        feat_content = self.adapter_content(ssl_src)
        quantized, ids, loss_vq = self.vq(feat_content.transpose(1, 2))
        if len(loss_vq.shape) == 1:
            loss_vq = loss_vq.squeeze()
        feat_content = quantized.transpose(1, 2)

        ssl_ref = self.extract_ssl_features(audio_ref, audio_ref_len)
        feat_speaker = self.adapter_speaker(ssl_ref)

        feat_fused, _ = self.speaker_attention(
            feat_content.transpose(1, 2),
            feat_speaker.transpose(1, 2),
            feat_speaker.transpose(1, 2),
        )
        feat_fused = feat_fused.transpose(1, 2)


        mel, mel_mask = self.get_mel(audio_src, audio_src_len)
        max_len = min(feat_fused.shape[-1], mel.shape[-1])
        if max_len % 2 != 0:
            max_len -= 1
        mel = mel[:, :, :max_len]
        mel_mask = mel_mask[:, :, :max_len]
        feat_fused = feat_fused[:, :, :max_len]
        feat_len = torch.sum(mel_mask, -1).int()

        assert (
            feat_len.max() == feat_fused.shape[-1]
        ), f"{feat_len.max()} != {feat_fused.shape[-1]}"

        mu_mel = self.proj_mel(feat_fused.transpose(1, 2)).transpose(1, 2)

        if self.training:
            loss_diff, mel_pred = self.decoder.compute_loss(
                x1=mel, mask=mel_mask, mu=mu_mel, spks=feat_speaker.transpose(1, 2)
            )
        else:
            loss_diff = torch.tensor(0.0).to(mel.device)
            mel_pred = self.decoder(
                mu_mel,
                mel_mask,
                n_timesteps=n_timesteps,
                spks=feat_speaker.transpose(1, 2),
            )
        loss_prior = torch.sum(
            0.5 * ((mel - mu_mel) ** 2 + math.log(2 * math.pi)) * mel_mask
        )
        loss_prior = loss_prior / (torch.sum(mel_mask) * self.mel_params["n_mels"])

        weight_content = self.adapter_content.fc.weight.softmax(1).unsqueeze(1)
        weight_speaker = self.adapter_speaker.fc.weight.softmax(1).unsqueeze(1)

        return {
            "mel": mel,
            "mel_pred": mel_pred,
            "mel_mask": mel_mask,
            "loss_prior": loss_prior,
            "loss_diff": loss_diff,
            "loss_vq": loss_vq,
            "weight_content": weight_content,
            "weight_speaker": weight_speaker,
        }

    @staticmethod
    def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(
            max_length, device=lengths.device, dtype=lengths.dtype
        ).expand(batch_size, max_length) < lengths.unsqueeze(1)
        return padding_mask.unsqueeze(1)

    @torch.no_grad()
    def extract_ssl_features(self, audio, audio_len):
        self.ssl_model.eval()
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        audio = F.pad(audio, (40, 40), "reflect")
        audio_len = audio_len + 80
        # audio.shape = (batch_size, audio_len)
        audio_mask = self._lengths_to_padding_mask(audio_len).squeeze(1)
        # audio_mask.shape = (batch_size, audio_len)
        audio = (
            self.feature_extractor(audio, return_tensors="pt")
            .input_values.squeeze(0)
            .to(audio.device)
        )
        ssl_output = self.ssl_model(
            audio, attention_mask=audio_mask, output_hidden_states=True
        )
        ssl_features = torch.stack(ssl_output.hidden_states[1:], dim=1).transpose(
            -1, -2
        )

        return ssl_features

    def get_mel(self, audio, audio_len):
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        mel, mel_len = self.mel_transform(audio, audio_len)
        mel_mask = self._lengths_to_padding_mask(mel_len)

        return mel, mel_mask


if __name__ == "__main__":
    mel_transform = LogMelSpectrogram()

    audio_src = torch.randn(2, 48000)
    import pdb; pdb.set_trace()
    mel = mel_transform(audio_src)
    print(f"Mel shape: {mel.shape}")
