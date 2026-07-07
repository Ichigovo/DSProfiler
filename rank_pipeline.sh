#!/bin/bash

# ==============================================================================
# Pipeline Environment Setup 
# ==============================================================================
eval "$(conda shell.bash hook)"

# ==============================================================================
# Global Parameters 
# ==============================================================================
BASE_DIR="./example/rank"
INPUT_CSV="$BASE_DIR/p2rank_pockets.csv"
SOURCE_DIR="$BASE_DIR/structure"
OUTPUT_CSV="$BASE_DIR/tmp_results.csv"
MODEL_WEIGHT="./best_model.pth"
FINAL_CSV="$BASE_DIR/example_results.csv"  

# ==============================================================================
# Step 0: Extract Pockets using P2Rank Parser
# ==============================================================================
echo "Step 0: Activating p2rank environment and running p2rank_parser.py"
conda activate p2rank

python3 ./run_p2rank/p2rank_parser.py \
    --input_csv "$INPUT_CSV" \
    --source_dir "$SOURCE_DIR" \
    --base_dir "$BASE_DIR"

if [ $? -ne 0 ]; then echo "Error in p2rank_parser.py"; exit 1; fi

# ==============================================================================
# Switch Environment for Deep Learning Model
# ==============================================================================
echo "Switching to environment for model inference..."
conda activate dsprofiler

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
echo "Step 3: Running pocket_feature.py (Parallel)"

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
# Step 4: Final Architecture Inference
# ==============================================================================
echo "Step 4: Running DSProfiler_architecture.py"
python3 ./architecture/DSProfiler_architecture.py \
    --input_csv "$INPUT_CSV" \
    --base_dir "$BASE_DIR" \
    --model_weight_path "$MODEL_WEIGHT" \
    --output_csv "$OUTPUT_CSV"

if [ $? -ne 0 ]; then echo "Error in DSProfiler_architecture.py"; exit 1; fi

# ==============================================================================
# Step 5: Post Processing (Re-ranking)
# ==============================================================================
echo "Step 5: Running post_processing.py"
python3 ./utils/post_processing.py \
    --output_csv "$OUTPUT_CSV" \
    --final_csv "$FINAL_CSV" \
    --base_dir "$BASE_DIR"

if [ $? -ne 0 ]; then echo "Error in post_processing.py"; exit 1; fi

echo "=============================================================================="
echo "Pipeline Completed Successfully!"
echo "Final results saved to: $FINAL_CSV"
echo "=============================================================================="