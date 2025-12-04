#!/bin/bash
#SBATCH --job-name=short_benchmark_profile
#SBATCH --output=short_benchmark_profile_%j.out
#SBATCH --error=short_benchmark_profile_%j.err
#SBATCH --partition=rtx-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

# ============================================================================
# FEM GPU Solver Benchmark Script
# Runs the benchmark shell script that parses result files and builds CSVs
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: Benchmark Only"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/10.1
module load gcc/6.3.0
module list
echo ""

# Check CUDA
echo "=== CUDA Check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo ""

# Navigate to working directory
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Make sure the benchmark script is executable
BENCH_SCRIPT="run_benchmark.sh"
if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found: $BENCH_SCRIPT"
    exit 1
fi

chmod +x "$BENCH_SCRIPT"

# ======================================================================
# RUN BENCHMARKS
# ======================================================================

echo "=========================================="
echo "  Running Benchmark Script: $BENCH_SCRIPT"
echo "=========================================="
echo ""

./"$BENCH_SCRIPT"

echo ""
echo "=========================================="
echo "  Benchmark Script Finished"
echo "=========================================="
echo ""

# Optional summary of outputs (assuming script writes these names)
RAW_CSV="benchmark_results_raw.csv"
AVG_CSV="benchmark_results_avg.csv"

echo "Generated files (top-level):"
ls -lh

if [ -f "$RAW_CSV" ]; then
    echo ""
    echo "Raw per-run CSV: $RAW_CSV"
    wc -l "$RAW_CSV"
fi

if [ -f "$AVG_CSV" ]; then
    echo ""
    echo "Averaged CSV: $AVG_CSV"
    wc -l "$AVG_CSV"
fi

echo ""
echo "If your fem_assembly wrote per-run result files under results/, you should see:"
if [ -d results ]; then
    find results -maxdepth 2 -type f | head
else
    echo "  (No results/ directory found)"
fi

echo ""
echo "All done!"
