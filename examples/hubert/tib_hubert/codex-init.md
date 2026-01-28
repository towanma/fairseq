# Codex Init - examples/hubert/tib_hubert

## 这份目录在做什么

`examples/hubert/tib_hubert` 是一套“藏语 HuBERT 继续预训练”的材料与脚本，目标是把官方 HuBERT 的多阶段流程（特征提取 -> K-means 聚类 -> 伪标签 -> 训练）做成更易复现的端到端 Pipeline，并补充数据校验、训练监控与问题 batch 定位工具。

核心思路：
- 输入：`manifest_dir` 里的 `train.tsv/valid.tsv(/test.tsv)`（由 `examples/wav2vec/wav2vec_manifest.py` 生成）。
- Stage 0：过滤可能导致训练不稳定（NaN/Inf、静音、采样率不符等）的音频。
- Stage 1：MFCC 特征 -> K-means -> `*.km` 标签 -> 用 `hubert_base_librispeech` 配置训练。
- Stage 2：用 Stage 1 模型第 6 层特征 -> K-means -> 新标签 -> 从 Stage 1 checkpoint 热启动继续训练。
- Stage 3（可选）：用 Stage 2 模型更高层（默认第 9 层）特征 -> K-means -> 新标签 -> 从 Stage 2 checkpoint 热启动继续训练。

## 文件/模块速览

- `README.md`
  - 面向使用者的“操作手册”，解释环境、数据准备、配置、启动 pipeline、监控与排障。
- `TIBETAN_HUBERT.md`
  - 更“原始/手动”的流程记录（逐条命令），包含可选 Stage 3（L9）思路，并提示如何用 Pipeline 启用 stage3。
- `configs/tibetan_hubert_config.yaml`
  - Pipeline 的 YAML 配置模板：数据路径、分布式参数、各 stage 的 nshard/聚类数/训练 overrides。
- `scripts/tibetan_hubert_pipeline.py`
  - 端到端执行器：串联 validate、MFCC、K-means、labels、训练、二/三阶段特征/聚类/训练；用 `work_dir/pipeline_state.json` 记录已完成阶段。
- `scripts/audio_validator.py`
  - TSV manifest 级别的数据验证与过滤；输出 filtered manifest + report + invalid list。
- `scripts/monitor_training.py`
  - 训练日志解析/实时监控/报告生成；支持传入 run dir 或直接传 `train.log`，并会自动选择最新的 `train.log`。
- `scripts/dump_bad_batch.py`
  - 给定 Hydra `run-dir` 和 `target-update/epoch`，复现 dataloader 顺序并打印该 update 对应的音频路径；支持 DDP 场景下通过 `--num-shards/--shard-id` 匹配某个 rank 的数据分片。
- `scripts/quickstart_tibetan_hubert.sh`
  - 交互式“快速开始”：检查依赖、生成 manifest、跑校验、写一份 config，然后启动 pipeline。

## Pipeline 行为细节（scripts/tibetan_hubert_pipeline.py）

### 状态与目录

- `work_dir/`
  - `pipeline_state.json`：记录 `completed_stages`，用于“断点续跑/跳过已完成阶段”。
  - `validation_report_{split}.json`、`invalid_files_{split}.txt`：数据校验产物。
  - `stage1/`：`mfcc_feat/`、`labels/`、`checkpoints/`、`mfcc_km*.bin`
  - `stage2/`：`features/`、`labels/`、`checkpoints/`、`hubert_L6_km*.bin`
  - `stage3/`（可选）：`features/`、`labels/`、`checkpoints/`、`hubert_L9_km*.bin`

### Stage 0: validate_data

对 `train/valid(/test)` 的 manifest 逐条检查音频文件：
- 采样率必须等于 `sample_rate`（默认 16k）。
- 时长范围 `[min_duration, max_duration]`（默认 2s - 15.625s）。
- 过滤 NaN/Inf、近静音、极端幅度；可选做一次 MFCC 计算以发现数值问题。

实现上会：
- 生成 `${split}_filtered.tsv`
- 备份原始 `${split}.tsv` 为 `${split}_original.tsv`（若不存在）
- 用 filtered 版本覆盖原 `${split}.tsv`

### Stage 1: MFCC -> K-means -> labels -> train

- MFCC：调用 `examples/hubert/simple_kmeans/dump_mfcc_feature.py`
  - 输出：`work_dir/stage1/mfcc_feat/{split}_{rank}_{nshard}.npy/.len`
- K-means：`learn_kmeans.py`
  - 输出：`work_dir/stage1/mfcc_km{n_clusters}.bin`
- labels：`dump_km_label.py` + 合并分片
  - 输出：`work_dir/stage1/labels/{train,valid,test}.km`、`dict.km.txt`
- train：`fairseq_cli/hydra_train.py`（config 固定为 `hubert_base_librispeech`）
  - 关键参数：
    - `task.data=<manifest_dir>`
    - `task.label_dir=<work_dir>/stage1/labels`
    - `task.labels=["km"]`
    - `model.label_rate=100`（与 MFCC 帧率一致）
    - `checkpoint.save_dir=<work_dir>/stage1/checkpoints`
    - `distributed_training.distributed_world_size=<world_size>`
    - `hydra.run.dir=<work_dir>/stage1/checkpoints`（让 `train.log` 与 `.hydra/*` 和 checkpoint 同目录，便于监控与排障）
  - 训练可通过 `train_overrides` 注入覆盖项（max_update/max_tokens/fp16/clip_norm 等）。

### Stage 2: HuBERT(L6) -> K-means -> labels -> train

- features：`dump_hubert_feature.py`
  - 输入 checkpoint：优先 `stage1/checkpoints/checkpoint_best.pt`，否则 `checkpoint_last.pt`
  - 输出：`work_dir/stage2/features/...`
- K-means/labels：与 Stage 1 同结构
- train：同 `hubert_base_librispeech`，但会额外：
  - `checkpoint.finetune_from_model=<stage1_ckpt>`
  - `checkpoint.reset_optimizer=true`
  - `checkpoint.reset_lr_scheduler=true`
  - `hydra.run.dir=<work_dir>/stage2/checkpoints`

### Stage 3: HuBERT(L9 默认) -> K-means -> labels -> train（可选）

- features：`dump_hubert_feature.py`
  - 输入 checkpoint：优先 `stage2/checkpoints/checkpoint_best.pt`，否则 `checkpoint_last.pt`
  - 输出：`work_dir/stage3/features/...`
- K-means/labels：与 Stage 2 同结构（文件名为 `hubert_L{layer}_km{n_clusters}.bin`）
- train：同 `hubert_base_librispeech`，但会额外：
  - `checkpoint.finetune_from_model=<stage2_ckpt>`
  - `checkpoint.reset_optimizer=true`
  - `checkpoint.reset_lr_scheduler=true`
  - `hydra.run.dir=<work_dir>/stage3/checkpoints`

## 配置文件关键项（configs/tibetan_hubert_config.yaml）

- `data.manifest_dir`：必须包含 `train.tsv`、`valid.tsv`（可选 `test.tsv`）。
- `data.work_dir`：中间文件、聚类模型、标签、checkpoints 的根目录。
- `data.has_test_split`：若为 `true` 且存在 `test.tsv`，Pipeline 会处理 `test` split；若缺失 `test.tsv` 会 warning 并跳过。
- `data.validation.*`：校验并行度与阈值；`skip_mfcc_check` 可提速但降低排雷力度。
- `training.*`：`distributed_world_size/nproc_per_node/master_port` 用于 `torchrun` 启动。
- `stages.stage{1,2,3}.*`：
  - `nshard`：把全量特征/标签分片生成的分片数（会循环 rank=0..nshard-1）。
  - `n_clusters/percent/layer`：聚类簇数/抽样比例/提取层数。
  - `train_overrides`：透传到 Hydra 的覆盖参数（`key=value`）。
  - `stages.stage3.enabled`：是否启用 stage3（默认关闭）。

## 已完成的改进（对应原“风险点”）

1. `scripts/audio_validator.py`：修复并行验证的结果顺序
   - 现在直接在 `as_completed()` 中用 `idx = futures[future]` 把 `future.result()` 写回到对应位置，保证 filtered manifest 与原 TSV 顺序严格对齐。

2. `scripts/tibetan_hubert_pipeline.py`：训练失败不再写入 completed，并会中断 pipeline
   - `stage{1,2,3}_train` 现在仅在 `returncode==0` 时 `_mark_stage_complete()`；否则抛出异常停止后续阶段，避免误跳过。

3. Pipeline split 处理更稳健
   - 现在 `train.tsv/valid.tsv` 会被显式校验为必需文件；`has_test_split=true` 但缺少 `test.tsv` 时会 warning 并跳过 test split，避免后续 stage 直接报错。

4. Stage 3 已在 pipeline 中实现
   - `tibetan_hubert_pipeline.py` 新增 `stage3_features/kmeans/labels/train`，并提供 `--stage stage3`；通过 `stages.stage3.enabled=true` 启用。

5. 训练日志路径与选择逻辑已增强
   - `monitor_training.py` 支持传入目录或文件，并会自动选择“最新”的 `train.log`。
   - pipeline 训练命令显式设置 `hydra.run.dir=<stageX/checkpoints>`，让 `train.log`/`.hydra/*` 与 checkpoint 同目录，默认用法更一致。

6. `dump_bad_batch.py` 支持 DDP 分片与 `update_freq` schedule
   - 新增 `--num-shards/--shard-id`；`update_freq` 会按 epoch 选择（长度不足则用最后一个值），更贴近真实训练迭代逻辑。

## 建议的使用姿势（最小踩坑版）

- 没有 test 集：把 `has_test_split` 设为 `false`。
- 先跑 `--stage validate`，确认过滤比例与 `invalid_files_*.txt` 里的原因是否合理。
- `nshard` 初次建议用较小值（例如 1/10），先确认整条链路跑通，再按资源调整。
- 训练监控：优先使用 `<work_dir>/stageX/checkpoints`（pipeline 已放置 `train.log` 在这里）；并确保日志为 JSON lines（否则脚本解析不到指标）。
