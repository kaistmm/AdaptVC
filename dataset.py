import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import random
import numpy as np
import os
from munch import Munch
import soundfile as sf


class AdaptVCDataset(Dataset):
    def __init__(
        self, metadata, max_len, max_len_ref, data_root, evalmode=False, **kwargs
    ):
        self.max_len = max_len
        self.max_len_ref = max_len_ref
        self.evalmode = evalmode

        self.metadata = pd.read_csv(metadata)
        if not evalmode:
            self.metadata = self.metadata[self.metadata.duration > 3]
        self.data_root = data_root
        self.resample = torchaudio.transforms.Resample(24000, 16000)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if self.evalmode:
            row_ref = row
            row = pd.Series(
                {"path": row_ref.source_path, "speaker": row_ref.source_speaker}
            )
        else:
            row_ref = row

        audio_ref, sr_ref = sf.read(os.path.join(self.data_root, row_ref.path))
        assert sr_ref == 24000
        audio_ref = torch.FloatTensor(audio_ref).unsqueeze(0)
        audio_ref = self.resample(audio_ref)
        audio_ref_len = audio_ref.size(1)
        if not self.evalmode and audio_ref_len > self.max_len_ref:
            audio_ref = audio_ref[:, : self.max_len_ref]
            audio_ref_len = self.max_len_ref

        audio_src, sr_src = sf.read(os.path.join(self.data_root, row.path))
        assert sr_src == 24000
        audio_src = torch.FloatTensor(audio_src).unsqueeze(0)
        audio_src = self.resample(audio_src)
        audio_src_len = audio_src.size(1)

        if not self.evalmode and self.max_len > 0:
            audio_src, audio_src_len = self.slice_audio(audio_src, audio_src_len)

        return (
            audio_src,
            audio_src_len,
            audio_ref,
            audio_ref_len,
            row.path,
            row_ref.path,
        )

    def get_random_sample(self, speaker, intra=True):
        if intra:
            candidates = self.metadata[self.metadata.speaker == speaker]
        else:
            candidates = self.metadata[self.metadata.speaker != speaker]

        return candidates.sample().iloc[0]

    def slice_audio(self, audio, audio_len):
        if audio_len > self.max_len:
            start = random.randint(0, audio_len - self.max_len)
            audio = audio[:, start : start + self.max_len]
            audio_len = self.max_len

        return audio, audio_len

    def collate_fn(self, batch):
        (
            audio_src,
            audio_src_len,
            audio_ref,
            audio_ref_len,
            path_src,
            path_ref,
        ) = zip(*batch)
        B = len(audio_src)
        max_audio_src_len = max(audio_src_len)
        audio_src_pad = torch.zeros(B, 1, max_audio_src_len)
        max_audio_ref_len = max(audio_ref_len)
        audio_ref_pad = torch.zeros(B, 1, max_audio_ref_len)

        for i in range(B):
            audio_src_pad[i, :, : audio_src_len[i]] = audio_src[i]
            audio_ref_pad[i, :, : audio_ref_len[i]] = audio_ref[i]

        audio_src_pad = audio_src_pad.squeeze(1)
        audio_ref_pad = audio_ref_pad.squeeze(1)

        audio_src_len = torch.LongTensor(audio_src_len)
        audio_ref_len = torch.LongTensor(audio_ref_len)

        return {
            "audio_src": audio_src_pad,
            "audio_ref": audio_ref_pad,
            "audio_src_len": audio_src_len,
            "audio_ref_len": audio_ref_len,
            "path_src": path_src,
            "path_ref": path_ref,
        }


if __name__ == "__main__":
    dataset = AdaVCDataset(
        metadata="assets/metadata/vc_train2.csv",
        max_len=-1,
        max_len_ref=48000,
        data_root="data/LibriTTS",
        evalmode=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))
    import pdb; pdb.set_trace()
    print(batch)
