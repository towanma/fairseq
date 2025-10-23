#!/usr/bin/env python3
"""
Utility to inspect the dataset indices that formed a specific optimizer update
during HuBERT (or other Hydra-powered fairseq) training.

Example
-------
python scripts/dump_bad_batch.py \
  --run-dir /root/fairseq/outputs/2025-10-21/19-30-00 \
  --target-update 36213 \
  --epoch 568
"""

import argparse
import os
from typing import List

from omegaconf import OmegaConf

from fairseq import tasks


def load_manifest_paths(manifest_path: str) -> List[str]:
    with open(manifest_path, "r") as f:
        root = f.readline().strip()
        rel_paths = [line.strip().split("\t")[0] for line in f]
    return [os.path.join(root, relp) for relp in rel_paths]


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    OmegaConf.resolve(cfg)

    if isinstance(cfg.task.label_rate, int):
        cfg.task.label_rate = float(cfg.task.label_rate)
    elif isinstance(cfg.task.label_rate, str):
        cfg.task.label_rate = float(cfg.task.label_rate)
    task = tasks.setup_task(cfg.task)

    split = args.split
    task.load_dataset(split)

    dataset = task.dataset(split)
    manifest_path = os.path.join(cfg.task.data, f"{split}.tsv")
    sample_paths = load_manifest_paths(manifest_path)

    update_freq = cfg.optimization.update_freq or [1]
    update_period = update_freq[0]

    target_update = args.target_update
    global_update = 0

    for epoch in range(1, args.epoch + 1):
        epoch_itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.max_sentences,
            max_positions=task.max_positions(),
            seed=cfg.common.seed,
            epoch=epoch,
            shuffle=True,
            num_shards=1,
            shard_id=0,
        ).next_epoch_itr(shuffle=True)

        micro_batches: List[List[int]] = []
        for sample in epoch_itr:
            sample_ids = sample["id"].cpu().tolist()
            micro_batches.append(sample_ids)

            if len(micro_batches) == update_period:
                global_update += 1
                if global_update == target_update:
                    print(
                        f"Found optimizer update {target_update} "
                        f"during epoch {epoch} (micro-batches {len(micro_batches)})"
                    )
                    ids_to_report = [idx for mb in micro_batches for idx in mb]
                    for idx in ids_to_report:
                        print(sample_paths[idx])
                    return
                micro_batches = []

        if global_update >= target_update:
            break

    raise ValueError(
        f"Could not locate update {target_update}. "
        "Check that --epoch is at least as large as the epoch containing the target update."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate data for a specific update.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Hydra run directory containing .hydra/config.yaml and train.log.",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        required=True,
        help="Optimizer update number reported in train_inner/train logs.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Epoch to iterate over (from train logs).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect (default: train).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
