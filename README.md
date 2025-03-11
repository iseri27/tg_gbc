# Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning

![](figs/gbc.png)

## Introduction

We propose a zero-shot, retraining-free pruning method that accelerates 3D object detection models via key pruning.

## Get Started

You need `pytorch` to use tgGBC, and prepare any .

```bash
# 1. install your pytorch.
pip3 install torch torchvision torchaudio

# 2. clone this repo.
git clone https://github.com/iseri27/tg_gbc
cd tg_gbc

# 3. install tggbc
python setup.py develop
```