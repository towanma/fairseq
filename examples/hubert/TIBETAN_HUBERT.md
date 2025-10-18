# 藏语 HuBERT 继续预训练流程

本指南汇总了在本仓库上以藏语语料继续预训练 HuBERT 的推荐步骤，涵盖数据准备、伪标签生成以及多阶段训练。假设你已经完成开发环境搭建 (`pip install -e .[dev]`) 并熟悉基本的命令行操作。

## 1. 数据准备
- 汇总全部藏语原始音频，采样率统一为 16 kHz、单声道（和 HuBERT 默认一致）；若原始采样率不同，先使用 `sox` 或 `ffmpeg` 重采样。
- 使用 `examples/wav2vec/wav2vec_manifest.py` 生成 `train.tsv`、`valid.tsv`（可再额外划分 `test.tsv`）：
  ```bash
  python examples/wav2vec/wav2vec_manifest.py \
    /path/to/tibetan_audio \
    --dest /path/to/tibetan_manifest \
    --ext wav --valid-percent 0.01
  ```
- 检查 TSV 是否覆盖全部音频、时长分布正常，并根据需要补充速度扰动、说话人平衡等数据增强。

## 2. 第一阶段：MFCC 聚类与预训练
1. **特征抽取（MFCC）**  
   ```bash
   python examples/hubert/simple_kmeans/dump_mfcc_feature.py \
     /path/to/tibetan_manifest train 100 0 /path/to/mfcc_feat
   ```
   - 参数含义：`100` 为总分片数，`0` 为当前分片编号；使用 GNU Parallel 或调度系统遍历全部 `rank`。
2. **训练 K-means (`km100` 或 `km500`)**
   ```bash
   python examples/hubert/simple_kmeans/learn_kmeans.py \
     /path/to/mfcc_feat train 100 /path/to/kmeans/mfcc_km500.bin 500 --percent 0.1
   ```
3. **生成伪标签与字典**
   ```bash
   python examples/hubert/simple_kmeans/dump_km_label.py \
     /path/to/mfcc_feat train /path/to/kmeans/mfcc_km500.bin 100 0 /path/to/labels/stage1
   ```
   合并全部分片后，创建 `dict.km.txt`：
   ```bash
   for i in $(seq 0 499); do echo "$i 1"; done > /path/to/labels/stage1/dict.km.txt
   ```
4. **运行第一阶段预训练**
   ```bash
   python fairseq_cli/hydra_train.py \
     --config-dir examples/hubert/config/pretrain \
     --config-name hubert_base_librispeech \
     task.data=/path/to/tibetan_manifest \
     task.label_dir=/path/to/labels/stage1 \
     task.labels='["km"]' \
     model.label_rate=100 \
     checkpoint.save_dir=/path/to/exp/stage1_tibetan
   ```
   常用调整：`distributed_training.distributed_world_size`、`dataset.max_tokens`、`optimization.max_update`。

## 3. 第二阶段：L6 表征聚类
1. 使用第一阶段模型（`checkpoint_best.pt` 或 `checkpoint_last.pt`）抽取第 6 层表征：
   ```bash
   python examples/hubert/simple_kmeans/dump_hubert_feature.py \
     /path/to/tibetan_manifest train \
     /path/to/exp/stage1_tibetan/checkpoint_best.pt \
     6 100 0 /path/to/features/stage2
   ```
2. 重复 K-means、标签生成与字典构建，得到 `stage2` 标签（通常命名为 `L6_km500`）。
3. 继续预训练：
   ```bash
   python fairseq_cli/hydra_train.py \
     --config-dir examples/hubert/config/pretrain \
     --config-name hubert_base_librispeech \
     task.data=/path/to/tibetan_manifest \
     task.label_dir=/path/to/labels/stage2 \
     task.labels='["km"]' \
     model.label_rate=100 \
     checkpoint.save_dir=/path/to/exp/stage2_tibetan \
     checkpoint.finetune_from_model=/path/to/exp/stage1_tibetan/checkpoint_best.pt \
     checkpoint.reset_optimizer=true \
     checkpoint.reset_lr_scheduler=true
   ```

## 4. 第三阶段：L9 表征聚类（可选）
1. 用第二阶段模型抽取第 9 层隐藏向量：
   ```bash
   python examples/hubert/simple_kmeans/dump_hubert_feature.py \
     /path/to/tibetan_manifest train \
     /path/to/exp/stage2_tibetan/checkpoint_best.pt \
     9 100 0 /path/to/features/stage3
   ```
2. 重新聚类（仍推荐 500 簇）并生成标签与 `dict.km.txt`。
3. 启动第三阶段训练，将 `checkpoint.finetune_from_model` 指向第二阶段权重，保存到新的输出目录。

## 5. 训练监控与扩展
- 各阶段训练日志默认保存在 `checkpoint.save_dir` 内，结合 `tensorboard_logdir` 与验证集损失追踪收敛。
- 若 GPU 资源有限，可调低 `dataset.max_tokens`、增大 `optimization.update_freq` 实现梯度累计。
- 继续预训练不同规模模型（Large/X-Large）时，改用对应 YAML 配置并同步放宽资源配置。
- 完成全部阶段后，使用最终模型（通常 Stage 3）作为下游任务的特征提取器或初始化，进行 CTC 微调、关键词检索等实验。

## 6. 常见注意事项
- 保持各阶段 `dict.km.txt` 尺寸与 K-means 簇数一致，否则会与模型输出维度冲突。
- 多阶段之间务必在生成新标签前冻结上一阶段模型；禁止在训练过程中动态替换标签。
- `model.label_rate` 需匹配聚类使用的帧率（MFCC 默认 100Hz，HuBERT 特征 50Hz 时要同步调整）。
- 在大规模集群上运行 `dump_*` 和 `learn_kmeans.py` 时，善用 `--nshard` 与作业调度保持数据对齐。
