#!/bin/bash
# Quickstart script for Tibetan HuBERT training
# This script guides you through the setup and training process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

section() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Check if running from fairseq root
if [ ! -f "setup.py" ] || [ ! -d "fairseq" ]; then
    error "Please run this script from the fairseq root directory"
    exit 1
fi

section "Tibetan HuBERT Training - Quick Start"

info "This script will help you set up and start training a Tibetan HuBERT model."
echo ""

# Step 1: Check dependencies
section "Step 1: Checking Dependencies"

info "Checking Python packages..."

missing_packages=""

for pkg in soundfile torchaudio matplotlib pyyaml; do
    python -c "import $pkg" 2>/dev/null || missing_packages="$missing_packages $pkg"
done

if [ -n "$missing_packages" ]; then
    warn "Missing packages:$missing_packages"
    read -p "Install missing packages? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install$missing_packages
        info "Packages installed successfully"
    else
        error "Please install missing packages manually"
        exit 1
    fi
else
    info "All required packages are installed"
fi

# Step 2: Get paths from user
section "Step 2: Configure Paths"

# Audio directory
echo ""
read -p "Path to your audio files (16kHz mono WAV): " AUDIO_DIR
while [ ! -d "$AUDIO_DIR" ]; do
    error "Directory not found: $AUDIO_DIR"
    read -p "Path to your audio files: " AUDIO_DIR
done
info "Using audio directory: $AUDIO_DIR"

# Manifest directory
echo ""
read -p "Where to save manifest files [./data/tibetan_manifest]: " MANIFEST_DIR
MANIFEST_DIR=${MANIFEST_DIR:-./data/tibetan_manifest}
info "Manifest directory: $MANIFEST_DIR"

# Work directory
echo ""
read -p "Where to save intermediate files and checkpoints [./data/tibetan_work]: " WORK_DIR
WORK_DIR=${WORK_DIR:-./data/tibetan_work}
info "Work directory: $WORK_DIR"

mkdir -p "$MANIFEST_DIR"
mkdir -p "$WORK_DIR"

# Step 3: Generate manifests
section "Step 3: Generate Manifest Files"

if [ -f "$MANIFEST_DIR/train.tsv" ] && [ -f "$MANIFEST_DIR/valid.tsv" ]; then
    info "Found existing manifest files:"
    echo "  - $MANIFEST_DIR/train.tsv"
    echo "  - $MANIFEST_DIR/valid.tsv"
    read -p "Regenerate manifests? (y/n) " -n 1 -r
    echo
    REGEN=$REPLY
else
    REGEN="y"
fi

if [[ $REGEN =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Percentage of data for validation [1.0]: " VALID_PERCENT
    VALID_PERCENT=${VALID_PERCENT:-1.0}

    info "Generating manifests..."
    python examples/wav2vec/wav2vec_manifest.py \
        "$AUDIO_DIR" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent $(python -c "print($VALID_PERCENT / 100)")

    if [ -f "$MANIFEST_DIR/train.tsv" ]; then
        TRAIN_COUNT=$(wc -l < "$MANIFEST_DIR/train.tsv")
        VALID_COUNT=$(wc -l < "$MANIFEST_DIR/valid.tsv")
        info "Generated manifests:"
        echo "  - Training samples: $((TRAIN_COUNT - 1))"
        echo "  - Validation samples: $((VALID_COUNT - 1))"
    else
        error "Failed to generate manifests"
        exit 1
    fi
fi

# Step 4: Data validation
section "Step 4: Data Validation"

echo ""
info "It's strongly recommended to validate your data before training."
info "This will filter out corrupted or problematic audio files that could cause NaN gradients."
echo ""
read -p "Run data validation? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Running data validation (this may take a while)..."

    # Validate training data
    python examples/hubert/tib_hubert/scripts/audio_validator.py \
        --manifest "$MANIFEST_DIR/train.tsv" \
        --output "$MANIFEST_DIR/train_filtered.tsv" \
        --report "$WORK_DIR/validation_report_train.json" \
        --invalid-list "$WORK_DIR/invalid_train.txt" \
        --num-workers 8

    # Validate validation data
    python examples/hubert/tib_hubert/scripts/audio_validator.py \
        --manifest "$MANIFEST_DIR/valid.tsv" \
        --output "$MANIFEST_DIR/valid_filtered.tsv" \
        --report "$WORK_DIR/validation_report_valid.json" \
        --invalid-list "$WORK_DIR/invalid_valid.txt" \
        --num-workers 8

    # Show summary
    info "Validation complete. Summary:"
    python -c "
import json
with open('$WORK_DIR/validation_report_train.json') as f:
    report = json.load(f)
    summary = report['summary']
    print(f\"  Training: {summary['valid_files']}/{summary['total_files']} valid ({summary['valid_percentage']:.1f}%)\")

with open('$WORK_DIR/validation_report_valid.json') as f:
    report = json.load(f)
    summary = report['summary']
    print(f\"  Validation: {summary['valid_files']}/{summary['total_files']} valid ({summary['valid_percentage']:.1f}%)\")
"

    echo ""
    read -p "Use filtered manifests for training? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup originals
        cp "$MANIFEST_DIR/train.tsv" "$MANIFEST_DIR/train_original.tsv"
        cp "$MANIFEST_DIR/valid.tsv" "$MANIFEST_DIR/valid_original.tsv"

        # Use filtered versions
        cp "$MANIFEST_DIR/train_filtered.tsv" "$MANIFEST_DIR/train.tsv"
        cp "$MANIFEST_DIR/valid_filtered.tsv" "$MANIFEST_DIR/valid.tsv"

        info "Using filtered manifests (originals backed up with _original suffix)"
    fi
fi

# Step 5: Configure training
section "Step 5: Configure Training Parameters"

echo ""
read -p "Number of GPUs to use [1]: " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-1}

echo ""
read -p "Maximum updates for stage 1 [100000]: " MAX_UPDATE_1
MAX_UPDATE_1=${MAX_UPDATE_1:-100000}

echo ""
read -p "Maximum updates for stage 2 [100000]: " MAX_UPDATE_2
MAX_UPDATE_2=${MAX_UPDATE_2:-100000}

# Generate config file
CONFIG_FILE="$WORK_DIR/config.yaml"

info "Generating configuration file: $CONFIG_FILE"

cat > "$CONFIG_FILE" << EOF
# Auto-generated Tibetan HuBERT training configuration
# Generated by quickstart_tibetan_hubert.sh

data:
  manifest_dir: $MANIFEST_DIR
  work_dir: $WORK_DIR
  sample_rate: 16000
  has_test_split: false

  validation:
    num_workers: 8
    min_duration: 2.0
    max_duration: 15.625
    skip_mfcc_check: false

training:
  distributed_world_size: $NUM_GPUS
  nproc_per_node: $NUM_GPUS
  master_port: 29501

stages:
  stage1:
    nshard: 100
    n_clusters: 100
    percent: 0.1

    train_overrides:
      optimization.max_update: $MAX_UPDATE_1
      dataset.max_tokens: 1400000
      common.fp16: true
      optimization.clip_norm: 10.0
      optimization.clip_norm_type: l2
      common.log_interval: 100
      dataset.validate_interval: 1
      dataset.validate_interval_updates: 5000

  stage2:
    nshard: 100
    n_clusters: 500
    percent: 0.1
    layer: 6

    train_overrides:
      optimization.max_update: $MAX_UPDATE_2
      dataset.max_tokens: 1400000
      common.fp16: true
      optimization.clip_norm: 10.0
      optimization.clip_norm_type: l2
      common.log_interval: 100
      dataset.validate_interval: 1
      dataset.validate_interval_updates: 5000
EOF

info "Configuration saved to: $CONFIG_FILE"

# Step 6: Start training
section "Step 6: Start Training"

echo ""
info "Setup complete! You can now start training."
echo ""
echo "To start training, run:"
echo ""
echo "  python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \\"
echo "      --config $CONFIG_FILE \\"
echo "      --stage all"
echo ""
echo "To monitor training (in another terminal):"
echo ""
echo "  python examples/hubert/tib_hubert/scripts/monitor_training.py \\"
echo "      --log-dir $WORK_DIR/stage1/checkpoints \\"
echo "      --mode monitor \\"
echo "      --alert-on-nan"
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Starting training..."
    info "Tip: Use tmux or screen to keep training running in the background"
    echo ""

    python examples/hubert/tib_hubert/scripts/tibetan_hubert_pipeline.py \
        --config "$CONFIG_FILE" \
        --stage all 2>&1 | tee "$WORK_DIR/training.log"
else
    info "Training not started. You can start it manually using the commands above."
fi

section "Setup Complete"

info "For more information, see: examples/hubert/tib_hubert/README.md"
