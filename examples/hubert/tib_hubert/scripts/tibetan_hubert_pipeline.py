#!/usr/bin/env python3
"""
End-to-end pipeline for Tibetan HuBERT training.

This script automates the entire training process described in TIBETAN_HUBERT.md:
1. Data validation and filtering
2. MFCC feature extraction
3. K-means clustering
4. Label generation
5. Multi-stage HuBERT training

Usage:
    # Run from fairseq root directory
    python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
        --config examples/hubert/tib_hubert/configs/tibetan_hubert_config.yaml \
        --stage all
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("tibetan_hubert_pipeline")


class HubertPipeline:
    """Manages the complete HuBERT training pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config["data"]
        self.training_config = config["training"]
        self.stages_config = config.get("stages", {})

        # Set up directories
        self.manifest_dir = Path(self.data_config["manifest_dir"])
        self.work_dir = Path(self.data_config["work_dir"])
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state file
        self.state_file = self.work_dir / "pipeline_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"completed_stages": []}

    def _save_state(self):
        """Save pipeline state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _mark_stage_complete(self, stage_name: str):
        """Mark a stage as completed."""
        if stage_name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage_name)
            self._save_state()
            logger.info(f"✓ Stage '{stage_name}' completed")

    def _is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage is already completed."""
        return stage_name in self.state["completed_stages"]

    def _run_command(self, cmd: List[str], description: str, check: bool = True):
        """Run a shell command with logging."""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(cmd, check=check)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ Completed in {elapsed:.1f}s")
        else:
            logger.error(f"✗ Failed with return code {result.returncode}")
            if check:
                sys.exit(1)

        return result

    def stage_0_validate_data(self):
        """Stage 0: Validate and filter audio data."""
        stage_name = "validate_data"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 0: Data Validation and Filtering")
        logger.info(f"{'='*60}")

        validation_config = self.data_config.get("validation", {})
        splits = ["train", "valid"]

        if self.data_config.get("has_test_split", False):
            splits.append("test")

        for split in splits:
            manifest_path = self.manifest_dir / f"{split}.tsv"
            output_path = self.manifest_dir / f"{split}_filtered.tsv"
            report_path = self.work_dir / f"validation_report_{split}.json"
            invalid_list_path = self.work_dir / f"invalid_files_{split}.txt"

            if not manifest_path.exists():
                logger.warning(f"Manifest not found: {manifest_path}, skipping")
                continue

            cmd = [
                "python", "examples/hubert/tib_hubert/scripts/audio_validator.py",
                "--manifest", str(manifest_path),
                "--output", str(output_path),
                "--report", str(report_path),
                "--invalid-list", str(invalid_list_path),
                "--num-workers", str(validation_config.get("num_workers", 8)),
                "--min-duration", str(validation_config.get("min_duration", 2.0)),
                "--max-duration", str(validation_config.get("max_duration", 15.625)),
                "--sample-rate", str(self.data_config.get("sample_rate", 16000)),
            ]

            if validation_config.get("skip_mfcc_check", False):
                cmd.append("--skip-mfcc-check")

            self._run_command(cmd, f"Validating {split} split")

            # Backup original manifest and use filtered version
            if output_path.exists():
                backup_path = self.manifest_dir / f"{split}_original.tsv"
                if not backup_path.exists():
                    shutil.copy(manifest_path, backup_path)
                shutil.copy(output_path, manifest_path)
                logger.info(f"Replaced {split}.tsv with filtered version")

        self._mark_stage_complete(stage_name)

    def stage_1_mfcc_features(self):
        """Stage 1: Extract MFCC features."""
        stage_name = "stage1_mfcc"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1: MFCC Feature Extraction")
        logger.info(f"{'='*60}")

        stage1_config = self.stages_config.get("stage1", {})
        nshard = stage1_config.get("nshard", 100)
        feat_dir = self.work_dir / "stage1" / "mfcc_feat"
        feat_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "valid"]
        if self.data_config.get("has_test_split", False):
            splits.append("test")

        for split in splits:
            logger.info(f"Extracting MFCC for {split} split...")
            for rank in range(nshard):
                cmd = [
                    "python", "examples/hubert/simple_kmeans/dump_mfcc_feature.py",
                    str(self.manifest_dir), split, str(nshard), str(rank), str(feat_dir),
                ]
                self._run_command(cmd, f"MFCC extraction {split} shard {rank}/{nshard}")

        self._mark_stage_complete(stage_name)

    def stage_1_kmeans(self):
        """Stage 1: K-means clustering on MFCC."""
        stage_name = "stage1_kmeans"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1: K-means Clustering (MFCC)")
        logger.info(f"{'='*60}")

        stage1_config = self.stages_config.get("stage1", {})
        nshard = stage1_config.get("nshard", 100)
        n_clusters = stage1_config.get("n_clusters", 100)
        percent = stage1_config.get("percent", 0.1)

        feat_dir = self.work_dir / "stage1" / "mfcc_feat"
        km_model_path = self.work_dir / "stage1" / f"mfcc_km{n_clusters}.bin"

        cmd = [
            "python", "examples/hubert/simple_kmeans/learn_kmeans.py",
            str(feat_dir), "train", str(nshard), str(km_model_path),
            str(n_clusters), "--percent", str(percent),
        ]

        self._run_command(cmd, f"Training K-means (k={n_clusters})")
        self._mark_stage_complete(stage_name)

    def stage_1_labels(self):
        """Stage 1: Generate K-means labels."""
        stage_name = "stage1_labels"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1: Generate K-means Labels")
        logger.info(f"{'='*60}")

        stage1_config = self.stages_config.get("stage1", {})
        nshard = stage1_config.get("nshard", 100)
        n_clusters = stage1_config.get("n_clusters", 100)

        feat_dir = self.work_dir / "stage1" / "mfcc_feat"
        km_model_path = self.work_dir / "stage1" / f"mfcc_km{n_clusters}.bin"
        label_dir = self.work_dir / "stage1" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "valid"]
        if self.data_config.get("has_test_split", False):
            splits.append("test")

        for split in splits:
            logger.info(f"Generating labels for {split} split...")

            # Dump labels for each shard
            for rank in range(nshard):
                cmd = [
                    "python", "examples/hubert/simple_kmeans/dump_km_label.py",
                    str(feat_dir), split, str(km_model_path),
                    str(nshard), str(rank), str(label_dir),
                ]
                self._run_command(cmd, f"Label generation {split} shard {rank}/{nshard}")

            # Merge shards
            logger.info(f"Merging label shards for {split}...")
            output_file = label_dir / f"{split}.km"
            shard_files = [label_dir / f"{split}_{rank}_{nshard}.km" for rank in range(nshard)]

            with open(output_file, "w") as outf:
                for shard_file in shard_files:
                    if shard_file.exists():
                        with open(shard_file, "r") as inf:
                            outf.write(inf.read())
                        shard_file.unlink()  # Remove shard file

            logger.info(f"Created {output_file}")

        # Create dictionary
        dict_file = label_dir / "dict.km.txt"
        with open(dict_file, "w") as f:
            for i in range(n_clusters):
                f.write(f"{i} 1\n")
        logger.info(f"Created {dict_file}")

        self._mark_stage_complete(stage_name)

    def stage_1_train(self):
        """Stage 1: Train HuBERT model."""
        stage_name = "stage1_train"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1: HuBERT Training")
        logger.info(f"{'='*60}")

        stage1_config = self.stages_config.get("stage1", {})
        label_dir = self.work_dir / "stage1" / "labels"
        save_dir = self.work_dir / "stage1" / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get training parameters
        world_size = self.training_config.get("distributed_world_size", 1)
        nproc_per_node = self.training_config.get("nproc_per_node", world_size)

        cmd = []

        if world_size > 1:
            # Multi-GPU training
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={self.training_config.get('master_port', 29501)}",
                "fairseq_cli/hydra_train.py",
            ]
        else:
            # Single GPU training
            cmd = ["python", "fairseq_cli/hydra_train.py"]

        cmd.extend([
            "--config-dir", "examples/hubert/config/pretrain",
            "--config-name", "hubert_base_librispeech",
            f"task.data={self.manifest_dir}",
            f"task.label_dir={label_dir}",
            "task.labels=[\"km\"]",
            "model.label_rate=100",
            f"checkpoint.save_dir={save_dir}",
            f"distributed_training.distributed_world_size={world_size}",
        ])

        # Add custom overrides
        overrides = stage1_config.get("train_overrides", {})
        for key, value in overrides.items():
            cmd.append(f"{key}={value}")

        self._run_command(cmd, "Stage 1 training", check=False)
        self._mark_stage_complete(stage_name)

    def stage_2_features(self):
        """Stage 2: Extract HuBERT features from layer 6."""
        stage_name = "stage2_features"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2: HuBERT Feature Extraction (Layer 6)")
        logger.info(f"{'='*60}")

        stage2_config = self.stages_config.get("stage2", {})
        nshard = stage2_config.get("nshard", 100)
        layer = stage2_config.get("layer", 6)

        stage1_ckpt = self.work_dir / "stage1" / "checkpoints" / "checkpoint_best.pt"
        if not stage1_ckpt.exists():
            stage1_ckpt = self.work_dir / "stage1" / "checkpoints" / "checkpoint_last.pt"

        if not stage1_ckpt.exists():
            logger.error(f"Stage 1 checkpoint not found in {stage1_ckpt.parent}")
            sys.exit(1)

        feat_dir = self.work_dir / "stage2" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "valid"]
        if self.data_config.get("has_test_split", False):
            splits.append("test")

        for split in splits:
            logger.info(f"Extracting HuBERT features for {split} split...")
            for rank in range(nshard):
                cmd = [
                    "python", "examples/hubert/simple_kmeans/dump_hubert_feature.py",
                    str(self.manifest_dir), split, str(stage1_ckpt), str(layer),
                    str(nshard), str(rank), str(feat_dir),
                ]
                self._run_command(cmd, f"HuBERT extraction {split} shard {rank}/{nshard}")

        self._mark_stage_complete(stage_name)

    def stage_2_kmeans(self):
        """Stage 2: K-means clustering on HuBERT features."""
        stage_name = "stage2_kmeans"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2: K-means Clustering (HuBERT L6)")
        logger.info(f"{'='*60}")

        stage2_config = self.stages_config.get("stage2", {})
        nshard = stage2_config.get("nshard", 100)
        n_clusters = stage2_config.get("n_clusters", 500)
        percent = stage2_config.get("percent", 0.1)

        feat_dir = self.work_dir / "stage2" / "features"
        km_model_path = self.work_dir / "stage2" / f"hubert_L6_km{n_clusters}.bin"

        cmd = [
            "python", "examples/hubert/simple_kmeans/learn_kmeans.py",
            str(feat_dir), "train", str(nshard), str(km_model_path),
            str(n_clusters), "--percent", str(percent),
        ]

        self._run_command(cmd, f"Training K-means (k={n_clusters})")
        self._mark_stage_complete(stage_name)

    def stage_2_labels(self):
        """Stage 2: Generate K-means labels."""
        stage_name = "stage2_labels"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2: Generate K-means Labels")
        logger.info(f"{'='*60}")

        stage2_config = self.stages_config.get("stage2", {})
        nshard = stage2_config.get("nshard", 100)
        n_clusters = stage2_config.get("n_clusters", 500)

        feat_dir = self.work_dir / "stage2" / "features"
        km_model_path = self.work_dir / "stage2" / f"hubert_L6_km{n_clusters}.bin"
        label_dir = self.work_dir / "stage2" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "valid"]
        if self.data_config.get("has_test_split", False):
            splits.append("test")

        for split in splits:
            logger.info(f"Generating labels for {split} split...")

            for rank in range(nshard):
                cmd = [
                    "python", "examples/hubert/simple_kmeans/dump_km_label.py",
                    str(feat_dir), split, str(km_model_path),
                    str(nshard), str(rank), str(label_dir),
                ]
                self._run_command(cmd, f"Label generation {split} shard {rank}/{nshard}")

            # Merge shards
            logger.info(f"Merging label shards for {split}...")
            output_file = label_dir / f"{split}.km"
            shard_files = [label_dir / f"{split}_{rank}_{nshard}.km" for rank in range(nshard)]

            with open(output_file, "w") as outf:
                for shard_file in shard_files:
                    if shard_file.exists():
                        with open(shard_file, "r") as inf:
                            outf.write(inf.read())
                        shard_file.unlink()

            logger.info(f"Created {output_file}")

        # Create dictionary
        dict_file = label_dir / "dict.km.txt"
        with open(dict_file, "w") as f:
            for i in range(n_clusters):
                f.write(f"{i} 1\n")
        logger.info(f"Created {dict_file}")

        self._mark_stage_complete(stage_name)

    def stage_2_train(self):
        """Stage 2: Train HuBERT model."""
        stage_name = "stage2_train"
        if self._is_stage_complete(stage_name):
            logger.info(f"Skipping '{stage_name}' (already completed)")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2: HuBERT Training")
        logger.info(f"{'='*60}")

        stage2_config = self.stages_config.get("stage2", {})
        label_dir = self.work_dir / "stage2" / "labels"
        save_dir = self.work_dir / "stage2" / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        stage1_ckpt = self.work_dir / "stage1" / "checkpoints" / "checkpoint_best.pt"
        if not stage1_ckpt.exists():
            stage1_ckpt = self.work_dir / "stage1" / "checkpoints" / "checkpoint_last.pt"

        world_size = self.training_config.get("distributed_world_size", 1)
        nproc_per_node = self.training_config.get("nproc_per_node", world_size)

        cmd = []

        if world_size > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc_per_node}",
                f"--master_port={self.training_config.get('master_port', 29502)}",
                "fairseq_cli/hydra_train.py",
            ]
        else:
            cmd = ["python", "fairseq_cli/hydra_train.py"]

        cmd.extend([
            "--config-dir", "examples/hubert/config/pretrain",
            "--config-name", "hubert_base_librispeech",
            f"task.data={self.manifest_dir}",
            f"task.label_dir={label_dir}",
            "task.labels=[\"km\"]",
            "model.label_rate=100",
            f"checkpoint.save_dir={save_dir}",
            f"checkpoint.finetune_from_model={stage1_ckpt}",
            "checkpoint.reset_optimizer=true",
            "checkpoint.reset_lr_scheduler=true",
            f"distributed_training.distributed_world_size={world_size}",
        ])

        overrides = stage2_config.get("train_overrides", {})
        for key, value in overrides.items():
            cmd.append(f"{key}={value}")

        self._run_command(cmd, "Stage 2 training", check=False)
        self._mark_stage_complete(stage_name)

    def run_pipeline(self, start_stage: Optional[str] = None, end_stage: Optional[str] = None):
        """Run the complete pipeline or specific stages."""
        stages = [
            ("validate", self.stage_0_validate_data),
            ("stage1_mfcc", self.stage_1_mfcc_features),
            ("stage1_kmeans", self.stage_1_kmeans),
            ("stage1_labels", self.stage_1_labels),
            ("stage1_train", self.stage_1_train),
            ("stage2_features", self.stage_2_features),
            ("stage2_kmeans", self.stage_2_kmeans),
            ("stage2_labels", self.stage_2_labels),
            ("stage2_train", self.stage_2_train),
        ]

        # Find start and end indices
        start_idx = 0
        end_idx = len(stages)

        if start_stage:
            for i, (name, _) in enumerate(stages):
                if name == start_stage:
                    start_idx = i
                    break

        if end_stage:
            for i, (name, _) in enumerate(stages):
                if name == end_stage:
                    end_idx = i + 1
                    break

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Tibetan HuBERT Training Pipeline")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"Running stages: {' -> '.join([s[0] for s in stages[start_idx:end_idx]])}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        # Run selected stages
        for stage_name, stage_func in stages[start_idx:end_idx]:
            try:
                stage_func()
            except Exception as e:
                logger.error(f"Error in stage '{stage_name}': {e}")
                raise

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline completed in {elapsed/3600:.2f} hours")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Tibetan HuBERT training pipeline"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "validate", "stage1", "stage2",
                 "stage1_mfcc", "stage1_kmeans", "stage1_labels", "stage1_train",
                 "stage2_features", "stage2_kmeans", "stage2_labels", "stage2_train"],
        help="Pipeline stage to run (stage1 includes validation)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed stage (skip completed stages)",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset pipeline state and start from scratch",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation stage (not recommended)",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create pipeline
    pipeline = HubertPipeline(config)

    # Reset state if requested
    if args.reset_state:
        pipeline.state = {"completed_stages": []}
        pipeline._save_state()
        logger.info("Pipeline state reset. Starting from scratch.")

    # If resume is set, pipeline will automatically skip completed stages
    # (this is the default behavior via _is_stage_complete checks)
    if args.resume:
        logger.info(f"Resuming pipeline. Completed stages: {pipeline.state['completed_stages']}")

    # Determine stages to run
    if args.stage == "all":
        pipeline.run_pipeline()
    elif args.stage == "validate":
        pipeline.stage_0_validate_data()
    elif args.stage == "stage1":
        # Stage1 now includes validation by default
        if args.skip_validation:
            pipeline.run_pipeline(start_stage="stage1_mfcc", end_stage="stage1_train")
        else:
            pipeline.run_pipeline(start_stage="validate", end_stage="stage1_train")
    elif args.stage == "stage2":
        pipeline.run_pipeline(start_stage="stage2_features", end_stage="stage2_train")
    else:
        # Run specific stage
        stage_map = {
            "stage1_mfcc": pipeline.stage_1_mfcc_features,
            "stage1_kmeans": pipeline.stage_1_kmeans,
            "stage1_labels": pipeline.stage_1_labels,
            "stage1_train": pipeline.stage_1_train,
            "stage2_features": pipeline.stage_2_features,
            "stage2_kmeans": pipeline.stage_2_kmeans,
            "stage2_labels": pipeline.stage_2_labels,
            "stage2_train": pipeline.stage_2_train,
        }
        if args.stage in stage_map:
            stage_map[args.stage]()


if __name__ == "__main__":
    main()
