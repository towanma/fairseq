#!/usr/bin/env python3
"""
Audio data validator and filter for HuBERT training.

This script checks audio files for potential issues that could cause
training instabilities (NaN gradients, inf values, etc.) and creates
a filtered manifest with only valid samples.

Usage:
    python scripts/audio_validator.py \
        --manifest /data/tibetan_manifest/train.tsv \
        --output /data/tibetan_manifest/train_filtered.tsv \
        --report /data/tibetan_manifest/validation_report.json \
        --num-workers 8
"""

import argparse
import json
import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Suppress warnings from audio libraries
warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("audio_validator")


@dataclass
class AudioValidationResult:
    """Result of audio file validation."""
    path: str
    valid: bool
    duration: float = 0.0
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0

    # Statistics
    mean_amplitude: float = 0.0
    max_amplitude: float = 0.0
    rms_energy: float = 0.0

    # Issues
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AudioValidator:
    """Validates audio files for HuBERT training."""

    def __init__(
        self,
        min_duration: float = 2.0,  # 32000 frames at 16kHz
        max_duration: float = 15.625,  # 250000 frames at 16kHz
        target_sample_rate: int = 16000,
        min_rms_energy: float = 0.001,  # Detect near-silent audio
        max_amplitude: float = 100.0,  # Detect clipped/corrupted audio
        check_nan_inf: bool = True,
        check_mfcc: bool = True,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        self.min_rms_energy = min_rms_energy
        self.max_amplitude = max_amplitude
        self.check_nan_inf = check_nan_inf
        self.check_mfcc = check_mfcc

    def validate_audio_file(self, audio_path: str) -> AudioValidationResult:
        """Validate a single audio file."""
        try:
            # Read audio file
            info = sf.info(audio_path)
            result = AudioValidationResult(
                path=audio_path,
                valid=True,
                duration=info.duration,
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
            )

            # Check sample rate
            if info.samplerate != self.target_sample_rate:
                result.valid = False
                result.error = f"Wrong sample rate: {info.samplerate} (expected {self.target_sample_rate})"
                return result

            # Check duration
            if info.duration < self.min_duration:
                result.valid = False
                result.error = f"Too short: {info.duration:.2f}s (min {self.min_duration}s)"
                return result

            if info.duration > self.max_duration:
                result.valid = False
                result.error = f"Too long: {info.duration:.2f}s (max {self.max_duration}s)"
                return result

            # Check channels
            if info.channels != 1:
                result.warnings.append(f"Multiple channels ({info.channels}), will use mean")

            # Load audio data for detailed checks
            audio, sr = sf.read(audio_path, dtype="float32")

            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Check for NaN or Inf
            if self.check_nan_inf:
                if np.any(np.isnan(audio)):
                    result.valid = False
                    result.error = "Contains NaN values"
                    return result

                if np.any(np.isinf(audio)):
                    result.valid = False
                    result.error = "Contains Inf values"
                    return result

            # Calculate statistics
            result.mean_amplitude = float(np.abs(audio).mean())
            result.max_amplitude = float(np.abs(audio).max())
            result.rms_energy = float(np.sqrt(np.mean(audio ** 2)))

            # Check for silence
            if result.rms_energy < self.min_rms_energy:
                result.valid = False
                result.error = f"Nearly silent (RMS: {result.rms_energy:.6f})"
                return result

            # Check for clipping/corruption
            if result.max_amplitude > self.max_amplitude:
                result.valid = False
                result.error = f"Extreme amplitude (max: {result.max_amplitude:.2f})"
                return result

            # Check if MFCC extraction works
            if self.check_mfcc:
                try:
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    mfcc = torchaudio.compliance.kaldi.mfcc(
                        waveform=audio_tensor,
                        sample_frequency=self.target_sample_rate,
                        use_energy=False,
                    )

                    if torch.any(torch.isnan(mfcc)) or torch.any(torch.isinf(mfcc)):
                        result.valid = False
                        result.error = "MFCC extraction produced NaN/Inf"
                        return result

                except Exception as e:
                    result.valid = False
                    result.error = f"MFCC extraction failed: {str(e)}"
                    return result

            return result

        except Exception as e:
            return AudioValidationResult(
                path=audio_path,
                valid=False,
                error=f"Failed to read: {str(e)}",
            )


def validate_single_file(args: Tuple[str, AudioValidator]) -> AudioValidationResult:
    """Worker function for parallel validation."""
    audio_path, validator = args
    return validator.validate_audio_file(audio_path)


def load_manifest(manifest_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """Load TSV manifest file."""
    with open(manifest_path, "r") as f:
        root = f.readline().strip()
        entries = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rel_path, num_frames = parts[0], int(parts[1])
                entries.append((rel_path, num_frames))
    return root, entries


def save_manifest(
    output_path: str,
    root: str,
    entries: List[Tuple[str, int]],
) -> None:
    """Save TSV manifest file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create dir if path has directory component
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{root}\n")
        for rel_path, num_frames in entries:
            f.write(f"{rel_path}\t{num_frames}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and filter audio data for HuBERT training"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Input TSV manifest file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV manifest file (filtered)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report file with validation results",
    )
    parser.add_argument(
        "--invalid-list",
        default=None,
        help="Text file listing all invalid audio paths",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.625,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--min-rms-energy",
        type=float,
        default=0.001,
        help="Minimum RMS energy (to filter silent audio)",
    )
    parser.add_argument(
        "--skip-mfcc-check",
        action="store_true",
        help="Skip MFCC extraction check (faster but less thorough)",
    )

    args = parser.parse_args()

    logger.info(f"Loading manifest: {args.manifest}")
    root, entries = load_manifest(args.manifest)
    logger.info(f"Found {len(entries)} audio files")
    logger.info(f"Audio root directory: {root}")

    # Create validator
    validator = AudioValidator(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_sample_rate=args.sample_rate,
        min_rms_energy=args.min_rms_energy,
        check_mfcc=not args.skip_mfcc_check,
    )

    # Prepare validation tasks
    audio_paths = [os.path.join(root, rel_path) for rel_path, _ in entries]
    tasks = [(path, validator) for path in audio_paths]

    # Validate files in parallel
    logger.info(f"Validating {len(tasks)} files with {args.num_workers} workers...")
    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(validate_single_file, task): i for i, task in enumerate(tasks)}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if i % 100 == 0:
                valid_count = sum(1 for r in results if r.valid)
                logger.info(f"Progress: {i}/{len(tasks)} ({valid_count} valid)")

    # Sort results by original order
    # Build a mapping from future to original index, then sort results
    future_to_idx = {f: futures[f] for f in futures}
    sorted_results = [None] * len(tasks)
    for future in futures:
        idx = future_to_idx[future]
        sorted_results[idx] = results[list(futures.keys()).index(future)]
    results = sorted_results

    # Analyze results
    valid_results = [r for r in results if r.valid]
    invalid_results = [r for r in results if not r.valid]

    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Summary:")
    logger.info(f"  Total files: {len(results)}")
    logger.info(f"  Valid files: {len(valid_results)} ({len(valid_results)/len(results)*100:.1f}%)")
    logger.info(f"  Invalid files: {len(invalid_results)} ({len(invalid_results)/len(results)*100:.1f}%)")

    # Count error types
    if invalid_results:
        error_types = {}
        for r in invalid_results:
            error_key = r.error.split(":")[0] if r.error else "Unknown"
            error_types[error_key] = error_types.get(error_key, 0) + 1

        logger.info(f"\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {error_type}: {count}")

    # Calculate statistics for valid files
    if valid_results:
        durations = [r.duration for r in valid_results]
        rms_values = [r.rms_energy for r in valid_results]

        logger.info(f"\nValid audio statistics:")
        logger.info(f"  Duration: min={min(durations):.2f}s, max={max(durations):.2f}s, "
                   f"mean={np.mean(durations):.2f}s")
        logger.info(f"  RMS Energy: min={min(rms_values):.6f}, max={max(rms_values):.6f}, "
                   f"mean={np.mean(rms_values):.6f}")
        logger.info(f"  Total duration: {sum(durations)/3600:.2f} hours")

    logger.info(f"{'='*60}\n")

    # Save filtered manifest
    valid_entries = [entries[i] for i, r in enumerate(results) if r.valid]
    save_manifest(args.output, root, valid_entries)
    logger.info(f"Saved filtered manifest: {args.output}")

    # Save validation report
    if args.report:
        report = {
            "summary": {
                "total_files": len(results),
                "valid_files": len(valid_results),
                "invalid_files": len(invalid_results),
                "valid_percentage": len(valid_results) / len(results) * 100,
            },
            "parameters": {
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "sample_rate": args.sample_rate,
                "min_rms_energy": args.min_rms_energy,
            },
            "all_results": [asdict(r) for r in results],
        }

        report_dir = os.path.dirname(args.report)
        if report_dir:  # Only create dir if path has directory component
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved validation report: {args.report}")

    # Save invalid file list
    if args.invalid_list and invalid_results:
        invalid_list_dir = os.path.dirname(args.invalid_list)
        if invalid_list_dir:  # Only create dir if path has directory component
            os.makedirs(invalid_list_dir, exist_ok=True)
        with open(args.invalid_list, "w") as f:
            for r in invalid_results:
                f.write(f"{r.path}\t{r.error}\n")
        logger.info(f"Saved invalid file list: {args.invalid_list}")

    # Exit with error code if too many invalid files
    if len(invalid_results) / len(results) > 0.5:
        logger.error(f"WARNING: More than 50% of files are invalid!")
        sys.exit(1)


if __name__ == "__main__":
    main()
