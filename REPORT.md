# VAEGPT-Sim 实验报告（2026-2-11）

## 论文来源
本文复现与验证的模型框架来源于：

Proceedings of the 2024 Findings of the Association for Computational Linguistics (ACL 2024 Findings)  
DOI: 10.18653/v1/2024.findings-acl.513

## 实验目的
复现/验证 VAE+GPT 框架在 STS-B 上的句向量质量；加入相似度监督头（μ→score）。

## 环境
- OS: Windows
- Conda env: vaegptsim
- Python: 3.10
- 关键包版本：见 `runs/<date>/requirements.txt`、`env_vaegptsim.yml`

## 数据
- 训练：GLUE STS-B (train)
- 评估：GLUE STS-B (dev，n=1500)
- 默认训练仅使用高相似样本（label >= 4.0，可通过参数关闭）

## 实验复现 Pipeline

### 1. 环境准备
- 方案 A（conda）：
  - `conda env create -f env_vaegptsim.yml`
  - `conda activate vaegptsim`
- 方案 B（pip）：
  - `pip install -r requirements.txt`

### 2. 数据准备
训练/验证默认使用 HuggingFace Datasets 的 GLUE STS-B：
- 代码内部会自动 `load_dataset("glue", "stsb")`
- 如需离线/本地 TSV：
  - 传参：`--local_stsb_train_tsv <path>`、`--local_stsb_dev_tsv <path>`
  - 离线模式：`--offline`（必要时配合 `--bert_dir`、`--gpt2_dir`）
  - 默认会过滤低相似样本：`--min_label 4.0`（要全量请设为 `0`）

`prepare_dataset.py` 提供 `SentencePairDataset`（编码 `sentence1 -> encoder`，`sentence2 -> decoder`，并保留 decoder attention mask）。

### 3. 训练
```
python train_sim.py
```
特点：
- KL warmup + free-bits、sim head 监督
- decoder token dropout 强化 z 的作用
- 默认不解冻 GPT-2（可自行开启分阶段解冻）
- 支持中断保存/续训与单独评估
- 输出：
  - `checkpoints/vaegpt.pt`（最新）
  - `checkpoints/vaegpt_best.pt`（dev Spearman 最佳）
  - `checkpoints/vaegpt_train.pt`（含 sim_head 与 step，用于继续训练）

### 4. 评估（STS-B dev）
```
python eval_stsb.py checkpoints/vaegpt.pt
```
输出 Spearman / Pearson。
也可用训练脚本直接评估：
```
python train_sim.py --eval_only --eval_ckpt checkpoints/vaegpt_best.pt
```

### 5. 生成（可选）
```
python generate.py --ckpt checkpoints/vaegpt_best.pt --src "A boy is playing a guitar."
```
可调参数（采样、top-p/top-k、temperature 等）见脚本 CLI。

### 6. 复现实验建议
- 固定随机种子：脚本内 `SEED = 42`
- 记录关键超参：batch size / latent size / KL warmup / sim head 权重
- 保存 `requirements.txt` 与运行日志（建议放到 `runs/<date>/`）

## 模型与训练
- Encoder: BERT base (μ/logσ²)
- Decoder: GPT-2 (z→prefix embed)
- Loss: CE (reconstruction) + β·KL + λ·MSE(μ→score)
- 关键超参：
  - batch_size=16, latent_size=32, num_steps=2000
  - β 退火步数=1500, β_max=0.5, free_bits=0.05
  - λ=3.0, decoder_drop_p=0.3
  - 默认冻结 GPT-2（如需解冻可调整参数）

## 指标
- 最近一次 best（dev）：step 1000 | STS-B dev | Spearman: 0.4286 | Pearson: 0.3407 | n=1500

## 现象与结论
- 生成文本仍有重复倾向，说明语义承载与解码策略仍需加强。
- STS-B 相关性仍偏低但已能稳定跑通，作为基线可接受。

## 下一步计划
1. 继续优化训练目标（仅高相似样本 / 更合适的复述数据集）
2. 调整 KL 约束与解码策略，降低重复与通用句式
3. 做系统化 ablation（λ、β_max、dropout、解冻策略）
