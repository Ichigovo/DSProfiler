#!/bin/bash

# ==============================================================================
# Pipeline Environment Setup
# ==============================================================================
# Source conda and activate the environment
eval "$(conda shell.bash hook)"
conda activate dsprofiler
# ==============================================================================
# Global Parameters 
# ==============================================================================
BASE_DIR="./example/score"
INPUT_CSV="$BASE_DIR/example_test.csv"
SOURCE_DIR="$BASE_DIR/structure"
OUTPUT_CSV="$BASE_DIR/example_results.csv"
MODEL_WEIGHT="./best_model.pth"
# ==============================================================================
# Step 1: Prepare Data (Extract Pockets and Copy Files)
# ==============================================================================
echo "Step 1: Running pre_processing.py"
python3 ./utils/pre_processing.py \
    --input_csv "$INPUT_CSV" \
    --source_dir "$SOURCE_DIR" \
    --base_dir "$BASE_DIR"

if [ $? -ne 0 ]; then echo "Error in pre_processing.py"; exit 1; fi
# ==============================================================================
# Step 2: Extract Protein Features
# ==============================================================================
echo "Step 2: Running protein_feature.py"
python3 ./utils/protein_feature.py \
    --input_csv "$INPUT_CSV" \
    --base_dir "$BASE_DIR"

if [ $? -ne 0 ]; then
    echo "Warning: protein_feature.py encountered errors, but continuing to next step."
fi

# ==============================================================================
# Step 3: Extract Pocket Features (Parallel Execution)
# ==============================================================================
echo "Step 3: Running pocket_feature.py"

python3 ./utils/pocket_feature_r.py --input_csv "$INPUT_CSV" --base_dir "$BASE_DIR" &
PID_R=$!

python3 ./utils/pocket_feature_a.py --input_csv "$INPUT_CSV" --base_dir "$BASE_DIR" &
PID_A=$!

# Wait for both background processes to finish
wait $PID_R
status_r=$?
wait $PID_A
status_a=$?

if [ $status_r -ne 0 ] || [ $status_a -ne 0 ]; then
    echo "Error in parallel pocket feature extraction."
fi

# ==============================================================================
# Step 4: Final Architecture Inference/Training
# ==============================================================================
echo "Step 4: Running DSProfiler_architecture.py"
python3 ./architecture/DSProfiler_architecture.py \
    --input_csv "$INPUT_CSV" \
    --base_dir "$BASE_DIR" \
    --model_weight_path "$MODEL_WEIGHT" \
    --output_csv "$OUTPUT_CSV"

if [ $? -ne 0 ]; then echo "Error in DSProfiler_architecture.py"; exit 1; fi

echo "=============================================================================="
echo "Pipeline Completed Successfully!"
echo "Final results saved to: $OUTPUT_CSV"
echo "=============================================================================="