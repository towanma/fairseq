# 藏语 HuBERT 训练指南

本文档是藏语 HuBERT 模型训练的完整指南，基于自动化 Pipeline 工具实现端到端训练。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [配置文件](#3-配置文件)
4. [启动训练](#4-启动训练)
5. [训练监控](#5-训练监控)
6. [故障排除](#6-故障排除)

---

## 1. 环境准备

### 1.1 安装 Fairseq

```bash
# 从 fairseq 根目录
pip install -e .[dev]

# 安装额外依赖
pip install soundfile torchaudio matplotlib pyyaml
```

### 1.2 验证安装

```bash
python -c "import fairseq; import soundfile; import torchaudio; print('OK')"
```

---

## 2. 数据准备

### 2.1 音频格式要求

| 项目 | 要求 |
|------|------|
| **格式** | WAV |
| **采样率** | 16000 Hz (16kHz) |
| **声道** | 单声道 (mono) |
| **时长** | 2-15 秒（推荐） |

### 2.2 音频转换（如需）

```bash
# 使用 ffmpeg 批量转换
find /原始音频目录 -name "*.wav" | while read file; do
    ffmpeg -i "$file" -ar 16000 -ac 1 "/目标目录/$(basename $file)"
done

# 或使用 sox
sox input.wav -r 16000 -c 1 output.wav
```

### 2.3 目录结构

请将音频文件组织为以下结构：

```
/data/tibetan_audio/          ← 音频根目录
├── train/                    ← 训练集（大部分数据）
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── valid/                    ← 验证集（可选，约1%数据）
│   └── ...
└── test/                     ← 测试集（可选）
    └── ...
```

> **注意**：如果只有一个目录，Pipeline 会自动按比例划分训练集/验证集。

### 2.4 生成 Manifest 文件

Manifest 是 TSV 格式的索引文件，记录每个音频的路径和帧数。

```bash
# 从 fairseq 根目录运行
python examples/wav2vec/wav2vec_manifest.py \
    /data/tibetan_audio/train \
    --dest /data/tibetan_manifest \
    --ext wav \
    --valid-percent 0.01   # 1% 作为验证集
```

生成文件：
- `/data/tibetan_manifest/train.tsv` - 训练集索引
- `/data/tibetan_manifest/valid.tsv` - 验证集索引

### 2.5 数据量建议

| 规模 | 音频时长 | 说明 |
|------|----------|------|
| 最小可行 | 100+ 小时 | 可以训练，效果有限 |
| 推荐 | 500+ 小时 | 较好的效果 |
| 理想 | 1000+ 小时 | 最佳效果 |

---

## 3. 配置文件

### 3.1 复制并编辑配置模板

```bash
cp examples/hubert/tib_hubert/configs/tibetan_hubert_config.yaml my_config.yaml
```

### 3.2 配置文件说明

```yaml
# my_config.yaml

# ==================== 数据配置 ====================
data:
  # Manifest 文件目录（包含 train.tsv, valid.tsv）
  manifest_dir: /data/tibetan_manifest
  
  # 工作目录（保存中间文件、模型检查点等）
  work_dir: /data/tibetan_hubert_work
  
  # 音频采样率
  sample_rate: 16000
  
  # 是否有独立的测试集
  has_test_split: false
  
  # 数据验证配置
  validation:
    num_workers: 8           # 并行验证进程数
    min_duration: 2.0        # 最小音频时长（秒）
    max_duration: 15.625     # 最大音频时长（秒）
    skip_mfcc_check: false   # 是否跳过 MFCC 检查

# ==================== 训练配置 ====================
training:
  distributed_world_size: 1  # GPU 数量
  nproc_per_node: 1          # 每节点进程数（通常等于 GPU 数）
  master_port: 29501         # 分布式训练端口

# ==================== 阶段配置 ====================
stages:
  # 第一阶段：基于 MFCC
  stage1:
    nshard: 100              # 并行分片数
    n_clusters: 100          # K-means 聚类数
    percent: 0.1             # K-means 采样比例
    
    train_overrides:
      optimization.max_update: 100000   # 最大更新步数
      dataset.max_tokens: 1400000       # 每批次 token 数（显存不足时减小）
      common.fp16: true                 # 使用 FP16
      optimization.clip_norm: 10.0      # 梯度裁剪

  # 第二阶段：基于 HuBERT L6 特征
  stage2:
    nshard: 100
    n_clusters: 500          # 通常增加到 500
    percent: 0.1
    layer: 6                 # 提取第 6 层特征
    
    train_overrides:
      optimization.max_update: 100000
      dataset.max_tokens: 1400000
      common.fp16: true
      optimization.clip_norm: 10.0
```

### 3.3 显存调整

如果遇到 CUDA OOM（显存不足）：

```yaml
train_overrides:
  dataset.max_tokens: 700000        # 减半
  optimization.update_freq: [2]     # 梯度累积
```

---

## 4. 启动训练

### 4.1 完整训练（推荐）

```bash
# 从 fairseq 根目录运行
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage all
```

这会依次执行：
1. **数据验证** - 过滤问题音频
2. **Stage 1** - MFCC 特征 → K-means → 训练
3. **Stage 2** - HuBERT L6 特征 → K-means → 训练

### 4.2 分阶段运行

```bash
# 只运行第一阶段（包含数据验证）
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage stage1

# 只运行第二阶段
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage stage2

# 只运行数据验证
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage validate
```

### 4.3 中断恢复

Pipeline 会自动保存状态，中断后可恢复：

```bash
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage all \
    --resume
```

### 4.4 重新开始

```bash
python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
    --config my_config.yaml \
    --stage all \
    --reset-state
```

### 4.5 命令行参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--stage` | 运行阶段：`all`, `validate`, `stage1`, `stage2` |
| `--resume` | 恢复运行，跳过已完成阶段 |
| `--reset-state` | 清除状态，重新开始 |
| `--skip-validation` | 跳过数据验证（不推荐） |

---

## 5. 训练监控

### 5.1 实时监控

在另一个终端运行：

```bash
python examples/hubert/tib_hubert/scripts/monitor_training.py \
    --log-dir /data/tibetan_hubert_work/stage1/checkpoints \
    --mode monitor \
    --alert-on-nan
```

### 5.2 训练后分析

```bash
python examples/hubert/tib_hubert/scripts/monitor_training.py \
    --log-dir /data/tibetan_hubert_work/stage1/checkpoints \
    --mode analyze \
    --output /data/analysis
```

生成：
- `training_metrics.png` - 训练曲线
- `training_report.html` - 完整报告

---

## 6. 故障排除

### 6.1 出现 NaN 梯度

1. **定位问题数据**：
   ```bash
   python examples/hubert/tib_hubert/scripts/dump_bad_batch.py \
       --run-dir /data/tibetan_hubert_work/stage1/checkpoints \
       --target-update 36213 \
       --epoch 568
   ```

2. **检查并移除问题文件**，重新运行数据验证

3. **调整超参数**：
   - 减小 `dataset.max_tokens`
   - 增大 `optimization.clip_norm`

### 6.2 数据验证过滤太多

检查过滤原因：
```bash
head -20 /data/tibetan_hubert_work/invalid_files_train.txt
```

常见问题：
- **采样率不对** → 重新转换音频为 16kHz
- **时长不符** → 调整 `min_duration` / `max_duration`
- **静音** → 检查音频内容

### 6.3 显存不足

```yaml
train_overrides:
  dataset.max_tokens: 700000      # 减小
  optimization.update_freq: [2]   # 梯度累积
```

### 6.4 训练速度慢

```bash
# 检查 GPU 利用率
nvidia-smi

# 增加数据加载进程
# 在 config.yaml 中：
dataset:
  num_workers: 8
```

---

## 📁 目录结构总结

```
/data/
├── tibetan_audio/           # 原始音频
│   └── train/
├── tibetan_manifest/        # Manifest 索引文件
│   ├── train.tsv
│   └── valid.tsv
└── tibetan_hubert_work/     # 工作目录
    ├── pipeline_state.json  # Pipeline 状态
    ├── stage1/
    │   ├── mfcc_feat/       # MFCC 特征
    │   ├── labels/          # 聚类标签
    │   └── checkpoints/     # 模型检查点
    └── stage2/
        ├── features/        # HuBERT 特征
        ├── labels/
        └── checkpoints/     # 最终模型
```

---

## 🎯 训练完成后

最终模型保存在：
```
/data/tibetan_hubert_work/stage2/checkpoints/checkpoint_best.pt
```

可用于：
- 下游任务微调（ASR、说话人识别等）
- 提取音频特征
- 继续训练第三阶段

---

## ⏱️ 预计时间

| 阶段 | 单机 4×GPU | 说明 |
|------|-----------|------|
| 数据验证 | 10-30 分钟 | 取决于数据量 |
| Stage 1 训练 | 1-3 天 | 100k updates |
| Stage 2 训练 | 1-3 天 | 100k updates |

---

祝训练顺利！🎉
