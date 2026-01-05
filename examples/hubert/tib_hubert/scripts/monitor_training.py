#!/usr/bin/env python3
"""
Training monitor for HuBERT training.

This script monitors training logs and provides:
1. Real-time loss tracking and visualization
2. NaN/Inf gradient detection
3. Validation performance tracking
4. Alert on training anomalies

Usage:
    # Monitor training in real-time
    python scripts/monitor_training.py \
        --log-dir /data/tibetan_hubert_work/stage1/checkpoints \
        --alert-on-nan

    # Analyze completed training
    python scripts/monitor_training.py \
        --log-dir /data/tibetan_hubert_work/stage1/checkpoints \
        --mode analyze \
        --output report.html
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("monitor_training")


class TrainingMonitor:
    """Monitors HuBERT training progress and detects anomalies."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics = defaultdict(list)
        self.anomalies = []

    def parse_hydra_log(self, log_file: Path) -> List[Dict]:
        """Parse Hydra training log file (JSON format)."""
        entries = []

        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Try to parse as JSON
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue

        return entries

    def extract_metrics(self, entries: List[Dict]) -> Dict[str, List]:
        """Extract training metrics from log entries."""
        metrics = defaultdict(list)

        for entry in entries:
            # Training metrics
            if 'train_loss' in entry:
                metrics['train_loss'].append({
                    'epoch': entry.get('epoch', 0),
                    'update': entry.get('num_updates', 0),
                    'loss': entry['train_loss'],
                    'lr': entry.get('lr', 0),
                    'wps': entry.get('wps', 0),
                    'ups': entry.get('ups', 0),
                    'wpb': entry.get('wpb', 0),
                    'clip': entry.get('clip', 0),
                    'gnorm': entry.get('gnorm', 0),
                })

            # Validation metrics
            if 'valid_loss' in entry:
                metrics['valid_loss'].append({
                    'epoch': entry.get('epoch', 0),
                    'update': entry.get('num_updates', 0),
                    'loss': entry['valid_loss'],
                })

        return metrics

    def detect_anomalies(self, metrics: Dict[str, List]) -> List[Dict]:
        """Detect training anomalies (NaN, sudden spikes, etc.)."""
        anomalies = []

        # Check training loss
        if 'train_loss' in metrics:
            losses = [(m['update'], m['loss'], m.get('gnorm', 0))
                      for m in metrics['train_loss']]

            for i, (update, loss, gnorm) in enumerate(losses):
                # Check for NaN or Inf
                if np.isnan(loss) or np.isinf(loss):
                    anomalies.append({
                        'type': 'nan_loss',
                        'update': update,
                        'epoch': metrics['train_loss'][i]['epoch'],
                        'message': f'NaN/Inf loss at update {update}',
                    })

                # Check for gradient explosion
                if gnorm > 1000:
                    anomalies.append({
                        'type': 'gradient_explosion',
                        'update': update,
                        'epoch': metrics['train_loss'][i]['epoch'],
                        'gnorm': gnorm,
                        'message': f'Large gradient norm ({gnorm:.2f}) at update {update}',
                    })

                # Check for sudden loss spike
                if i > 0:
                    prev_loss = losses[i-1][1]
                    if not np.isnan(prev_loss) and not np.isnan(loss):
                        if loss > prev_loss * 5:  # 5x increase
                            anomalies.append({
                                'type': 'loss_spike',
                                'update': update,
                                'epoch': metrics['train_loss'][i]['epoch'],
                                'prev_loss': prev_loss,
                                'curr_loss': loss,
                                'message': f'Loss spike at update {update}: {prev_loss:.4f} -> {loss:.4f}',
                            })

        return anomalies

    def plot_metrics(self, metrics: Dict[str, List], output_path: str):
        """Generate training metric plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        if 'train_loss' in metrics and metrics['train_loss']:
            updates = [m['update'] for m in metrics['train_loss']]
            losses = [m['loss'] for m in metrics['train_loss']]

            # Filter out NaN/Inf for plotting
            valid_idx = [i for i, loss in enumerate(losses)
                        if not np.isnan(loss) and not np.isinf(loss)]
            updates_clean = [updates[i] for i in valid_idx]
            losses_clean = [losses[i] for i in valid_idx]

            axes[0, 0].plot(updates_clean, losses_clean, 'b-', alpha=0.6)
            axes[0, 0].set_xlabel('Updates')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True)

            # Add moving average
            if len(losses_clean) > 100:
                window = 100
                ma = np.convolve(losses_clean, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(updates_clean[window-1:], ma, 'r-', linewidth=2, label='MA(100)')
                axes[0, 0].legend()

        # Validation loss
        if 'valid_loss' in metrics and metrics['valid_loss']:
            updates = [m['update'] for m in metrics['valid_loss']]
            losses = [m['loss'] for m in metrics['valid_loss']]

            axes[0, 1].plot(updates, losses, 'g-o', markersize=4)
            axes[0, 1].set_xlabel('Updates')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True)

        # Learning rate
        if 'train_loss' in metrics and metrics['train_loss']:
            updates = [m['update'] for m in metrics['train_loss']]
            lrs = [m['lr'] for m in metrics['train_loss']]

            axes[1, 0].plot(updates, lrs, 'r-', alpha=0.6)
            axes[1, 0].set_xlabel('Updates')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)

        # Gradient norm
        if 'train_loss' in metrics and metrics['train_loss']:
            updates = [m['update'] for m in metrics['train_loss']]
            gnorms = [m['gnorm'] for m in metrics['train_loss']]

            # Filter out zeros and extreme values
            valid_idx = [i for i, g in enumerate(gnorms) if 0 < g < 1000]
            updates_clean = [updates[i] for i in valid_idx]
            gnorms_clean = [gnorms[i] for i in valid_idx]

            if gnorms_clean:
                axes[1, 1].plot(updates_clean, gnorms_clean, 'purple', alpha=0.6)
                axes[1, 1].set_xlabel('Updates')
                axes[1, 1].set_ylabel('Gradient Norm')
                axes[1, 1].set_title('Gradient Norm')
                axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        plt.close()

    def generate_report(self, metrics: Dict[str, List], anomalies: List[Dict],
                       output_path: str):
        """Generate HTML training report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>HuBERT Training Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metric { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }
        .anomaly { background-color: #ffebee; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; }
        .anomaly-type { font-weight: bold; color: #f44336; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; }
        .summary-box { background-color: #e3f2fd; padding: 20px; border-radius: 8px; text-align: center; flex: 1; margin: 0 10px; }
        .summary-value { font-size: 32px; font-weight: bold; color: #1976d2; }
        .summary-label { color: #666; margin-top: 10px; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>HuBERT Training Report</h1>
"""

        # Summary statistics
        if 'train_loss' in metrics and metrics['train_loss']:
            final_loss = metrics['train_loss'][-1]['loss']
            total_updates = metrics['train_loss'][-1]['update']
            final_epoch = metrics['train_loss'][-1]['epoch']

            valid_losses = [m['loss'] for m in metrics['train_loss']
                          if not np.isnan(m['loss']) and not np.isinf(m['loss'])]
            best_loss = min(valid_losses) if valid_losses else float('nan')

            html += f"""
        <div class="summary">
            <div class="summary-box">
                <div class="summary-value">{total_updates}</div>
                <div class="summary-label">Total Updates</div>
            </div>
            <div class="summary-box">
                <div class="summary-value">{final_epoch}</div>
                <div class="summary-label">Epochs</div>
            </div>
            <div class="summary-box">
                <div class="summary-value">{final_loss:.4f}</div>
                <div class="summary-label">Final Loss</div>
            </div>
            <div class="summary-box">
                <div class="summary-value">{best_loss:.4f}</div>
                <div class="summary-label">Best Loss</div>
            </div>
        </div>
"""

        # Anomalies
        if anomalies:
            html += f"""
        <h2>⚠️ Detected Anomalies ({len(anomalies)})</h2>
"""
            for anomaly in anomalies:
                html += f"""
        <div class="anomaly">
            <span class="anomaly-type">{anomaly['type'].replace('_', ' ').title()}</span>
            - Update {anomaly['update']}, Epoch {anomaly['epoch']}<br>
            {anomaly['message']}
        </div>
"""
        else:
            html += """
        <h2>✓ No Anomalies Detected</h2>
        <div class="metric">Training proceeded smoothly without detected anomalies.</div>
"""

        # Validation performance
        if 'valid_loss' in metrics and metrics['valid_loss']:
            html += f"""
        <h2>Validation Performance</h2>
        <table>
            <tr>
                <th>Update</th>
                <th>Epoch</th>
                <th>Validation Loss</th>
            </tr>
"""
            for m in metrics['valid_loss'][-10:]:  # Last 10 validation points
                html += f"""
            <tr>
                <td>{m['update']}</td>
                <td>{m['epoch']}</td>
                <td>{m['loss']:.6f}</td>
            </tr>
"""
            html += """
        </table>
"""

        # Plots
        plot_path = output_path.replace('.html', '_plots.png')
        if Path(plot_path).exists():
            html += f"""
        <h2>Training Metrics</h2>
        <img src="{Path(plot_path).name}" alt="Training Metrics">
"""

        html += """
    </div>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Saved report to {output_path}")

    def monitor_realtime(self, log_file: Path, alert_on_nan: bool = True):
        """Monitor training log in real-time."""
        logger.info(f"Monitoring {log_file} (Press Ctrl+C to stop)")

        last_position = 0
        last_update = 0

        try:
            while True:
                if not log_file.exists():
                    time.sleep(5)
                    continue

                with open(log_file, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Display training progress
                        if 'train_loss' in entry:
                            update = entry.get('num_updates', 0)
                            if update > last_update:
                                loss = entry['train_loss']
                                lr = entry.get('lr', 0)
                                gnorm = entry.get('gnorm', 0)
                                wps = entry.get('wps', 0)

                                status = f"Update {update} | Loss: {loss:.4f} | LR: {lr:.2e} | " \
                                        f"GNorm: {gnorm:.2f} | WPS: {wps:.0f}"

                                # Check for anomalies
                                if np.isnan(loss) or np.isinf(loss):
                                    status += " ⚠️ NaN/Inf DETECTED!"
                                    if alert_on_nan:
                                        logger.error(status)
                                        logger.error("Training diverged! Check data or hyperparameters.")
                                elif gnorm > 1000:
                                    status += " ⚠️ Large gradient!"
                                    logger.warning(status)
                                else:
                                    logger.info(status)

                                last_update = update

                        # Display validation results
                        if 'valid_loss' in entry:
                            loss = entry['valid_loss']
                            update = entry.get('num_updates', 0)
                            logger.info(f"Validation at update {update} | Loss: {loss:.4f}")

                    except json.JSONDecodeError:
                        pass

                time.sleep(2)  # Check every 2 seconds

        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped")

    def analyze(self, output_dir: Optional[str] = None):
        """Analyze completed training and generate report."""
        # Find log file
        log_files = list(self.log_dir.glob("**/train.log"))

        if not log_files:
            logger.error(f"No train.log found in {self.log_dir}")
            return

        log_file = log_files[0]
        logger.info(f"Analyzing {log_file}")

        # Parse log
        entries = self.parse_hydra_log(log_file)
        logger.info(f"Parsed {len(entries)} log entries")

        # Extract metrics
        metrics = self.extract_metrics(entries)
        logger.info(f"Extracted metrics: {', '.join(metrics.keys())}")

        # Detect anomalies
        anomalies = self.detect_anomalies(metrics)

        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies")
            for anomaly in anomalies[:5]:  # Show first 5
                logger.warning(f"  - {anomaly['message']}")
        else:
            logger.info("No anomalies detected")

        # Generate outputs
        if output_dir is None:
            output_dir = self.log_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot metrics
        plot_path = output_dir / "training_metrics.png"
        self.plot_metrics(metrics, str(plot_path))

        # Generate report
        report_path = output_dir / "training_report.html"
        self.generate_report(metrics, anomalies, str(report_path))

        logger.info(f"\nAnalysis complete!")
        logger.info(f"  Plot: {plot_path}")
        logger.info(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Monitor HuBERT training")
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Training log directory (contains train.log)",
    )
    parser.add_argument(
        "--mode",
        choices=["monitor", "analyze"],
        default="monitor",
        help="Monitor mode: 'monitor' for real-time, 'analyze' for post-training",
    )
    parser.add_argument(
        "--alert-on-nan",
        action="store_true",
        help="Alert and stop on NaN/Inf detection (monitor mode only)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for analysis results (analyze mode only)",
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir)

    if args.mode == "monitor":
        log_file = Path(args.log_dir) / "train.log"
        monitor.monitor_realtime(log_file, alert_on_nan=args.alert_on_nan)
    else:
        monitor.analyze(output_dir=args.output)


if __name__ == "__main__":
    main()
