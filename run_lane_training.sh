#!/bin/bash

# BiSeNet Lane Detection - Quick Start Script
# This script automates the complete workflow: conversion, training, and evaluation

set -e  # Exit on error

# Configuration
DATASET_DIR="../peter/fine-tune3/dataset/augmented1"
VAL_DIR="../peter/fine-tune3/dataset/original"  # Validation dataset (set to "" to disable)
OUTPUT_BASE="./outputs"
DATASET_OUTPUT="../peter/fine-tune3/dataset/bisenet"
N_CLASSES=2  # 2 for binary (lane vs background), 5 for multi-class (background + 4 lanes)
BINARY_MODE=true  # true for binary, false for multi-class
EPOCHS=100
BATCH_SIZE=8
# IMAGE_H=720
# IMAGE_W=1280
IMAGE_H=512
IMAGE_W=1024
THICKNESS=10

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}BiSeNet Lane Detection Training${NC}"
echo -e "${BLUE}======================================${NC}"

# Step 1: Convert dataset
echo -e "\n${GREEN}[Step 1/3] Converting dataset to segmentation masks...${NC}"

if [ "$BINARY_MODE" = true ]; then
    python convert_dataset.py \
        --input_dir "$DATASET_DIR" \
        --output_dir "$DATASET_OUTPUT" \
        --binary \
        --thickness $THICKNESS
    echo -e "${GREEN}✓ Binary segmentation masks created${NC}"
else
    python convert_dataset_multi.py \
        --input_dir "$DATASET_DIR" \
        --output_dir "$DATASET_OUTPUT" \
        --thickness $THICKNESS
    echo -e "${GREEN}✓ Multi-class segmentation masks created${NC}"
fi

echo -e "${YELLOW}Check sample visualizations in: $DATASET_OUTPUT/visualizations/${NC}"

# Step 2: Train model
echo -e "\n${GREEN}[Step 2/3] Training BiSeNet model...${NC}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="bisenet_lane_${TIMESTAMP}"
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"

VAL_ARG=""
if [ -n "$VAL_DIR" ]; then
    VAL_ARG="--val_dir $VAL_DIR"
    echo -e "${YELLOW}Using validation set: $VAL_DIR${NC}"
fi

python train_lane_detection.py \
    --train_dir "$DATASET_OUTPUT" \
    $VAL_ARG \
    --n_classes $N_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.005 \
    --image_size $IMAGE_H $IMAGE_W \
    --output_dir "$OUTPUT_DIR" \
    --use_ohem \
    --use_aux_loss \
    --experiment_name "$EXPERIMENT_NAME"

echo -e "${GREEN}✓ Training complete${NC}"
echo -e "${YELLOW}Model saved to: $OUTPUT_DIR/checkpoints/${NC}"

# Step 3: Evaluate model
echo -e "\n${GREEN}[Step 3/3] Evaluating model...${NC}"

CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
DIAG_DIR="$OUTPUT_DIR/diagnostics"

if [ -f "$CHECKPOINT" ]; then
    python diagnose.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATASET_OUTPUT" \
        --n_classes $N_CLASSES \
        --image_size $IMAGE_H $IMAGE_W \
        --visualize \
        --num_viz 20 \
        --error_analysis

    echo -e "${GREEN}✓ Evaluation complete${NC}"
    echo -e "${YELLOW}Results saved to: $DIAG_DIR${NC}"

    # Display metrics if available
    if [ -f "$DIAG_DIR/metrics.json" ]; then
        echo -e "\n${BLUE}Metrics Summary:${NC}"
        python -c "import json; metrics = json.load(open('$DIAG_DIR/metrics.json')); print(f\"Accuracy: {metrics['accuracy']:.4f}\"); print(f\"mIoU: {metrics['miou']:.4f}\")"
    fi
else
    echo -e "${YELLOW}⚠ Best model checkpoint not found, skipping evaluation${NC}"
fi

echo -e "\n${BLUE}======================================${NC}"
echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo -e "${BLUE}======================================${NC}"

echo -e "\n${YELLOW}Summary:${NC}"
echo -e "  Dataset: $DATASET_OUTPUT"
echo -e "  Model: $OUTPUT_DIR"
echo -e "  Diagnostics: $DIAG_DIR"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  - Review visualizations in $DIAG_DIR/visualizations/"
echo -e "  - Check metrics in $DIAG_DIR/metrics.json"
echo -e "  - Use model from $CHECKPOINT for inference"
