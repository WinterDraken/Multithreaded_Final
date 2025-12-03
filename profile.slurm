#!/bin/bash
#SBATCH --job-name=fem_profile
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ============================================================================
# FEM GPU Solver Profiling Script
# Collects detailed GPU metrics for memory access patterns and SpMV performance
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver Profiling"
echo "  Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

# Load modules
module purge
module load cuda/10.1
module load gcc/6.3.0

cd $SLURM_SUBMIT_DIR

# Check if executable exists
if [ ! -f "./fem_assembly" ]; then
    echo "ERROR: Executable not found. Run benchmark script first or build manually."
    exit 1
fi

# Define meshes and methods (use subset for profiling - it's expensive)
MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
    "CPU/bracket_3d_xlarge.msh"
    "CPU/bracket_3d_xxlarge.msh"
)

METHODS=("none" "rcm" "amd")
PROFILE_DIR="profile_results"
mkdir -p $PROFILE_DIR

get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

echo "=========================================="
echo "  Running Profiling (this may take a while)"
echo "=========================================="
echo ""

for mesh_file in "${MESHES[@]}"; do
    if [ ! -f "$mesh_file" ]; then
        echo "WARNING: Mesh file not found: $mesh_file, skipping..."
        continue
    fi
    
    mesh_base=$(get_mesh_base "$mesh_file")
    echo "----------------------------------------"
    echo "Profiling mesh: $mesh_base"
    echo "----------------------------------------"
    
    for method in "${METHODS[@]}"; do
        echo "  Method: $method"
        
        profile_prefix="$PROFILE_DIR/${mesh_base}__${method}"
        
        # Use nsys for comprehensive profiling
        echo "    Running nsys profile..."
        nsys profile \
            --output="$profile_prefix" \
            --stats=true \
            --trace=cuda,nvtx \
            ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1
        
        # Extract key metrics using nvprof (if available) or parse nsys output
        echo "    Collecting detailed metrics..."
        
        # Use nvprof for detailed metrics (if available, otherwise skip)
        if command -v nvprof &> /dev/null; then
            nvprof --csv --log-file="$profile_prefix.metrics.csv" \
                --metrics=sm_efficiency,warp_efficiency,shared_efficiency,achieved_occupancy, \
                l2_cache_hit_rate,l2_read_throughput,l2_write_throughput, \
                dram_read_throughput,dram_write_throughput, \
                gld_throughput,gst_throughput, \
                atomic_transactions,atomic_transaction_per_request \
                ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1 || true
        fi
        
        # Also collect basic timing breakdown
        result_file="results/$method/${mesh_base}__${method}_results.txt"
        if [ -f "$result_file" ]; then
            cp "$result_file" "$profile_prefix.timing.txt"
        fi
        
        echo "    Profile saved: $profile_prefix"
    done
    echo ""
done

echo "=========================================="
echo "  Profiling Complete!"
echo "=========================================="
echo ""
echo "Profiles saved in: $PROFILE_DIR/"
echo ""
echo "To view profiles:"
echo "  nsys-ui $PROFILE_DIR/*.nsys-rep"