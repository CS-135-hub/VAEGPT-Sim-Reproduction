# VAEGPT-Sim Reproduction / 论文复现

## Overview / 项目简介
This repository reproduces a VAE+GPT framework for sentence representation learning and conditional generation on STS-B.  
本项目复现 VAE+GPT 句向量学习框架，并在 STS-B 上进行评估与生成实验。

## Features / 主要内容
- VAE encoder (BERT) + GPT-2 decoder with multi-token latent prefix  
  VAE 编码器（BERT）+ GPT-2 解码器，多 token 前缀承载潜变量
- Semantic supervision head (μ → score)  
  语义监督头（μ → 相似度分数）
- KL warmup + free-bits + decoder token dropout  
  KL 退火 + free-bits + 解码端随机遮蔽
- Evaluation on STS-B dev (Spearman/Pearson)  
  STS-B dev 集评估（Spearman/Pearson）

## Paper Source / 论文来源
Proceedings of the 2024 Findings of the Association for Computational Linguistics (ACL 2024 Findings)  
DOI: 10.18653/v1/2024.findings-acl.513

## Requirements / 环境
- Python 3.10  
- Conda env: `vaegptsim`  

Conda (recommended):  
```
conda env create -f env_vaegptsim.yml
conda activate vaegptsim
```

Pip (optional):  
```
pip install -r requirements.txt
```

## Train / 训练
Default training (2000 steps):  
```
python train_sim.py
```

Resume training:  
```
python train_sim.py --resume_from checkpoints/vaegpt_train.pt
```

## Evaluate / 评估
```
python eval_stsb.py checkpoints/vaegpt_best.pt
```

Or use the training script for eval only:  
```
python train_sim.py --eval_only --eval_ckpt checkpoints/vaegpt_best.pt
```

## Generate / 生成
```
python generate.py --ckpt checkpoints/vaegpt_best.pt --src "A boy is playing a guitar."
```

## Results / 指标
- Best (dev): step 1000 | Spearman 0.4286 | Pearson 0.3407 | n=1500

## Docs / 文档
- Experiment report: `REPORT.md`  
- Talking points & tuning guide: `REPRO_TALKING_POINTS.md`

