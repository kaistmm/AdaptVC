# AdaptVC: High Quality Voice Conversion with Adaptive Learning

[![arXiv](https://img.shields.io/badge/arXiv-2501.01347-b31b1b.svg)](https://arxiv.org/abs/2501.01347)
[![Demo](https://img.shields.io/badge/Demo-Online-blue)](https://mm.kaist.ac.kr/projects/AdaptVC/)

![](./images/main.png)

This repository implements a high-quality zero-shot voice conversion system that uses adaptive learning to disentangle linguistic content from voice style through self-supervised speech features and conditional flow matching.

# Abstract

The goal of voice conversion is to transform the
speech of a source speaker to sound like that of a reference
speaker while preserving the original content. A key challenge
is to extract disentangled linguistic content from the source and
voice style from the reference. While existing approaches leverage
various methods to isolate the two, a generalization still requires
further attention, especially for robustness in zero-shot scenarios.
In this paper, we achieve successful disentanglement of content
and speaker features by tuning self-supervised speech features
with adapters. The adapters are trained to dynamically encode
nuanced features from rich self-supervised features, and the
decoder fuses them to produce speech that accurately resembles
the reference with minimal loss of content. Moreover, we leverage
a conditional flow matching decoder with cross-attention speaker
conditioning to further boost the synthesis quality and efficiency.
Subjective and objective evaluations in a zero-shot scenario
demonstrate that the proposed method outperforms existing
models in speech quality and similarity to the reference speech.

## Demo

Check out our [online demo](https://mm.kaist.ac.kr/projects/AdaptVC/) to hear AdaptVC voice conversion examples.

# Checkpoints

| Model | Link |
|-------|------|
| AdaptVC model | [Download](https://drive.google.com/file/d/1NiSGdQPcRiuyHN9qFotDh2FWuhLTe7hk/view?usp=drive_link) |
| Vocoder | [Download](https://drive.google.com/file/d/1c09z1y2BhBegNbh2PSF7eT93iBPSOhqq/view?usp=drive_link) |


# Quick Start

Navigate to AdaptVC root and install dependencies:

### Tested on Python 3.10 with PyTorch 2.7.0

## 1. Create and activate a conda environment

```
conda env create -f environment.yaml python=3.10
conda activate adaptvc
```

If you need to update the environment later

```
conda env update -f environment.yaml
```



## 2. Train AdaptVC (supports multi-GPU)

Update data paths for LibriTTS datasets in ```train.py``` and ```infer.py```

```
python train.py ${experiment_name} --devices 0 1 2 3
```

The code will automatically create `logs/${experiment_name}` directory and save checkpoint. Multi-GPU is enabled when more than one device ids are provided in `--devices` argument

## 3. Inference
```
python infer.py ${checkpoint_path}
```

The inference code will save mel-spectrograms and audio files in `${checkpoint_path}` directory



# Citations

If you find AdaptVC useful, please cite:

    @inproceedings{kim2025adaptvc,
        title={AdaptVC: High Quality Voice Conversion with Adaptive Learning},
        author={Kim, Jaehun and Kim, Ji-Hoon and Choi, Yeunju and Nguyen, Tan Dat and Mun, Seongkyu and Chung, Joon Son},
        booktitle={International Conference on Acoustics, Speech and Signal Processing},
        year={2025}
    }