# 藏语 HuBERT 继续预训练详解

本文档记录在本仓库上使用藏语语料继续预训练 HuBERT 模型的完整流程，包括数据整理、伪标签生成、多阶段训练以及单卡/多卡启动方式。所有命令均假设你已在仓库根目录执行，并完成 `pip install -e .[dev]` 等环境准备。

---

## 1. 数据整理与清单构建
1. **音频准备**  
   - 收集全部藏语语音，统一为 16 kHz、单声道 WAV。  
   - 可通过 `sox input.wav -r 16000 -c 1 output.wav` 或 `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav` 批处理转换。  
   - 按数据类型（训练/验证/测试）分目录，例如：
     ```
     /data/tibetan_audio/
       train/
       valid/
       test/
     ```

2. **生成 TSV manifest**  
   - 训练+验证（按比例切分）：
     ```bash
     python examples/wav2vec/wav2vec_manifest.py \
       /data/tibetan_audio/train \
       --dest /data/tibetan_manifest \
       --ext wav \
       --valid-percent 0.01
     ```
     - `/data/tibetan_audio/train`：包含训练全集的根目录；
     - `--valid-percent 0.01`：1% 音频随机落入 `valid.tsv`（若已有固定验证集，可设 0 并手动覆盖）。

   - 测试集（若存在独立目录）：
     ```bash
     python examples/wav2vec/wav2vec_manifest.py \
       /data/tibetan_audio/test \
       --dest /data/tibetan_manifest_test \
       --ext wav \
       --valid-percent 0.0
     mv /data/tibetan_manifest_test/train.tsv /data/tibetan_manifest/test.tsv
     ```
   - 最终你应拥有 `/data/tibetan_manifest/{train.tsv,valid.tsv,test.tsv}`。

3. **检查与补充**
   - 确认 TSV 第一行均为音频根目录，后续每行形如 `relative/path.wav<TAB>num_frames`。  
   - 若需数据增强（速度扰动、说话人平衡等），应在生成 TSV 前完成，并确保增强后的音频亦在清单中。

---

## 2. 第一阶段：基于 MFCC 的伪标签
为方便并行，大量脚本支持分片（`nshard`）和当前分片号（`rank`）。以下示例设 `nshard=100`，可根据硬件调整。

### 2.1 提取 MFCC 特征
```bash
export MANIFEST_DIR=/data/tibetan_manifest
export MFCC_FEAT=/data/hubert_stage1/mfcc_feat
export NSHARD=100

mkdir -p ${MFCC_FEAT}
for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_mfcc_feature.py \
      ${MANIFEST_DIR} ${split} ${NSHARD} ${rank} ${MFCC_FEAT}
  done
done
```
- `dump_mfcc_feature.py` 会在 `${MFCC_FEAT}` 下生成 `${split}_${rank}_${NSHARD}.{npy,len}`，用于后续聚类。

### 2.2 训练 K-means
```bash
python examples/hubert/simple_kmeans/learn_kmeans.py \
  ${MFCC_FEAT} train ${NSHARD} /data/hubert_stage1/mfcc_km500.bin 500 --percent 0.1
```
- `/data/hubert_stage1/mfcc_km500.bin`：保存聚类模型；  
- `500`：簇数（可改为 100 获得更平滑标签）；  
- `--percent 0.1`：仅抽样 10% 特征拟合，若数据量不大可设为 `-1` 使用全部。

### 2.3 应用聚类并生成标签
```bash
export LABEL_STAGE1=/data/hubert_stage1/labels
mkdir -p ${LABEL_STAGE1}

for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_km_label.py \
      ${MFCC_FEAT} ${split} /data/hubert_stage1/mfcc_km500.bin ${NSHARD} ${rank} ${LABEL_STAGE1}
  done

  cat ${LABEL_STAGE1}/${split}_*_${NSHARD}.km > ${LABEL_STAGE1}/${split}.km
  rm ${LABEL_STAGE1}/${split}_*_${NSHARD}.km
done

for i in $(seq 0 499); do echo "$i 1"; done > ${LABEL_STAGE1}/dict.km.txt
```
- 通过 `cat` 合并全部分片，生成 `train.km`、`valid.km`、`test.km`。  
- 若聚类簇数不是 500，请将 `seq 0 499` 替换为 `seq 0 $((NUM_CLUSTERS-1))`。

### 2.4 第一阶段训练
**单 GPU / 单进程：**
```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=${MANIFEST_DIR} \
  task.label_dir=${LABEL_STAGE1} \
  task.labels='["km"]' \
  model.label_rate=100 \
  checkpoint.save_dir=/data/hubert_stage1/exp \
  distributed_training.distributed_world_size=1 \
  distributed_training.distributed_backend=nccl
```
- `task.data`：指向 TSV 的目录；  
- `task.label_dir`：包含 `train.km` 等文件；  
- `model.label_rate`：与聚类帧率一致（MFCC 默认 100Hz）；  
- `distributed_training.distributed_world_size=1`：避免初始化分布式；若在 CPU 上试跑，可将 backend 改为 `gloo`。

**多 GPU（示例：单机 4 卡）：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 --master_port=29501 \
  fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=${MANIFEST_DIR} \
  task.label_dir=${LABEL_STAGE1} \
  task.labels='["km"]' \
  model.label_rate=100 \
  checkpoint.save_dir=/data/hubert_stage1/exp \
  distributed_training.distributed_world_size=4
```
- `torchrun` 会自动初始化进程组；  
- `CUDA_VISIBLE_DEVICES` 控制使用 GPU；  
- `--master_port` 需与其他任务错开避免冲突；  
- `distributed_world_size` 与 `--nproc_per_node` 保持一致。

训练完成后，将 `checkpoint_best.pt` 或 `checkpoint_last.pt` 用于下一阶段特征导出。

---

## 3. 第二阶段：L6 表征聚类
第二阶段使用第一阶段模型第 6 层隐藏向量作为聚类特征，一般提升簇数至 500。

### 3.1 提取 HuBERT 特征（第 6 层）
```bash
export STAGE1_CKPT=/data/hubert_stage1/exp/checkpoint_best.pt
export HUBERT_FEAT2=/data/hubert_stage2/features
mkdir -p ${HUBERT_FEAT2}

for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_hubert_feature.py \
      ${MANIFEST_DIR} ${split} ${STAGE1_CKPT} 6 ${NSHARD} ${rank} ${HUBERT_FEAT2}
  done
done
```
- 第 4 个参数 `6` 表示抽取第 6 层表示。

### 3.2 聚类并生成标签
```bash
python examples/hubert/simple_kmeans/learn_kmeans.py \
  ${HUBERT_FEAT2} train ${NSHARD} /data/hubert_stage2/hubert_L6_km500.bin 500 --percent 0.1

export LABEL_STAGE2=/data/hubert_stage2/labels
mkdir -p ${LABEL_STAGE2}
for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_km_label.py \
      ${HUBERT_FEAT2} ${split} /data/hubert_stage2/hubert_L6_km500.bin ${NSHARD} ${rank} ${LABEL_STAGE2}
  done
  cat ${LABEL_STAGE2}/${split}_*_${NSHARD}.km > ${LABEL_STAGE2}/${split}.km
  rm ${LABEL_STAGE2}/${split}_*_${NSHARD}.km
done

for i in $(seq 0 499); do echo "$i 1"; done > ${LABEL_STAGE2}/dict.km.txt
```

### 3.3 第二阶段训练
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 --master_port=29502 \
  fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=${MANIFEST_DIR} \
  task.label_dir=${LABEL_STAGE2} \
  task.labels='["km"]' \
  model.label_rate=100 \
  checkpoint.save_dir=/data/hubert_stage2/exp \
  checkpoint.finetune_from_model=${STAGE1_CKPT} \
  checkpoint.reset_optimizer=true \
  checkpoint.reset_lr_scheduler=true \
  distributed_training.distributed_world_size=2
```
- `checkpoint.finetune_from_model`：从阶段 1 权重热启动；  
- `reset_optimizer/ reset_lr_scheduler`：确保新阶段重新调度学习率；  
- 其他参数同上，可根据资源调节。

---

## 4. 第三阶段：L9 表征聚类（可选）
第三阶段参考官方 pipeline，从第二阶段模型第 9 层提特征并重新聚类。

```bash
export STAGE2_CKPT=/data/hubert_stage2/exp/checkpoint_best.pt
export HUBERT_FEAT3=/data/hubert_stage3/features
mkdir -p ${HUBERT_FEAT3}

for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_hubert_feature.py \
      ${MANIFEST_DIR} ${split} ${STAGE2_CKPT} 9 ${NSHARD} ${rank} ${HUBERT_FEAT3}
  done
done

python examples/hubert/simple_kmeans/learn_kmeans.py \
  ${HUBERT_FEAT3} train ${NSHARD} /data/hubert_stage3/hubert_L9_km500.bin 500 --percent 0.1

export LABEL_STAGE3=/data/hubert_stage3/labels
mkdir -p ${LABEL_STAGE3}
for split in train valid test; do
  for rank in $(seq 0 $((NSHARD-1))); do
    python examples/hubert/simple_kmeans/dump_km_label.py \
      ${HUBERT_FEAT3} ${split} /data/hubert_stage3/hubert_L9_km500.bin ${NSHARD} ${rank} ${LABEL_STAGE3}
  done
  cat ${LABEL_STAGE3}/${split}_*_${NSHARD}.km > ${LABEL_STAGE3}/${split}.km
  rm ${LABEL_STAGE3}/${split}_*_${NSHARD}.km
done

for i in $(seq 0 499); do echo "$i 1"; done > ${LABEL_STAGE3}/dict.km.txt
```

训练命令同第二阶段，替换 `label_dir`、`save_dir` 与 `checkpoint.finetune_from_model=${STAGE2_CKPT}` 即可。

---

## 5. 训练监控与后续使用
- **日志**：`common.tensorboard_logdir` 可写入 `--config-name` 中默认路径，或在命令后追加 `common.tensorboard_logdir=/tmp/tb_stageX` 使用 TensorBoard 观察。  
- **继续训练/恢复**：若中断，可保留 `checkpoint_last.pt` 并添加 `checkpoint.restore_file=checkpoint_last.pt` 自动续跑。  
- **GPU 资源不足**：降低 `dataset.max_tokens` 或增大 `optimization.update_freq`（梯度累计）。  
- **下游任务**：通常使用阶段 3 模型；调用 `fairseq/hubert/hubert_model.HubertModel.extract_features` 获得隐藏层特征，或继续走 HuBERT CTC 微调流程。

---

## 6. 常见问题与提示
- **缺少 `train.km`/`valid.km`**：需确保执行 `cat ... > split.km` 合并分片，并删除 `_rank_nshard` 临时文件；训练只读取无分片后缀的文件。  
- **簇数不一致**：`dict.km.txt` 行数必须与聚类簇数相同；若阶段间修改簇数，应同步更新标签、字典与训练命令。  
- **label_rate 设置**：MFCC 生成的标签默认 100Hz，若在 HuBERT 特征提取时修改 `--layer` 或 `--max_chunk` 影响帧率，请相应调整 `model.label_rate`。  
- **多机多卡**：可使用 `torchrun --nnodes=N --node_rank=i --master_addr=...` 等参数；确保 `distributed_world_size = N * nproc_per_node`。  
- **磁盘清理**：分片特征和中间 KM 标签体积大，建议阶段完成后压缩或删除旧版文件，避免混淆。
- **定位异常 batch**：若 loss 突然飙升，可运行脚本恢复对应样本：
  ```bash
  python scripts/dump_bad_batch.py \
    --run-dir /root/fairseq/outputs/2025-10-21/19-30-00 \
    --target-update 36213 \
    --epoch 568
  ```
  该脚本读取 Hydra 保存的 `.hydra/config.yaml`，根据日志中的 `num_updates` 与 `epoch` 输出同批次音频的完整路径，可选 `--split` 指定数据集（默认 `train`），便于人工排查异常样本。

遵循以上流程即可完成藏语 HuBERT 的多阶段继续预训练，并为后续的 ASR、关键词检索等任务提供高质量音频表征。祝实验顺利！
