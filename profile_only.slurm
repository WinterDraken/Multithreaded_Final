#!/bin/bash
#SBATCH --job-name=fem_profile_only
#SBATCH --output=profile_only_%j.out
#SBATCH --error=profile_only_%j.err
#SBATCH --partition=rtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

# ============================================================================
# FEM GPU Solver Profiling-Only Script
# Runs profiling (nsys + ncu) for selected meshes and reordering methods
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: Profiling Only"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2  # Module with nsys/ncu tools
module load gcc/9.1.0
module list
echo ""

# Set LD_LIBRARY_PATH if needed
if [ -n "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH"
fi
echo ""

# Check CUDA and GPU details
echo "=== CUDA Check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo ""

# Navigate to working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Check if executable exists
if [ ! -f "./fem_assembly" ]; then
    echo "ERROR: Executable not found. Building..."
    make clean
    make
    if [ ! -f "./fem_assembly" ]; then
        echo "ERROR: Build failed - executable not found"
        exit 1
    fi
    echo "Build successful!"
    echo ""
fi

# Define meshes for profiling (use subset - profiling is expensive)
PROFILE_MESHES=(
    "CPU/bracket_3d.msh"
    "CPU/bracket_3d_large.msh"
    "CPU/bracket_3d_xlarge.msh"
    "CPU/bracket_3d_xxlarge.msh"
)

METHODS=("none" "rcm" "amd")
PROFILE_DIR="profile_results"
mkdir -p $PROFILE_DIR

# Function to extract mesh base name
get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

# Check tool availability
echo "=== Checking Profiling Tools ==="
NSYS_EXISTS=true
command -v nsys >/dev/null 2>&1 || NSYS_EXISTS=false

NCU_EXISTS=true
command -v ncu >/dev/null 2>&1 || NCU_EXISTS=false

if $NSYS_EXISTS; then
    echo "✓ nsys found: $(which nsys)"
    nsys --version
else
    echo "✗ nsys not found - system-level profiling will be skipped"
fi

if $NCU_EXISTS; then
    echo "✓ ncu found: $(which ncu)"
    ncu --version
else
    echo "✗ ncu not found - kernel-level metric profiling will be skipped"
fi
echo ""

# ============================================================================
# PROFILING PHASE
# ============================================================================

echo "=========================================="
echo "  Running Profiling"
echo "  Meshes: ${#PROFILE_MESHES[@]}"
echo "  Methods: ${METHODS[@]}"
echo "  Tools: nsys=$NSYS_EXISTS, ncu=$NCU_EXISTS"
echo "=========================================="
echo ""

for mesh_file in "${PROFILE_MESHES[@]}"; do
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
        
        # 1. Use nsys for comprehensive system-level profiling and timeline
        if $NSYS_EXISTS; then
            echo "    Running nsys profile..."
            nsys profile \
                --output="$profile_prefix" \
                --stats=true \
                --trace=cuda,nvtx \
                ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1
            
            if [ -f "${profile_prefix}.nsys-rep" ]; then
                echo "      ✓ nsys report generated: ${profile_prefix}.nsys-rep"
            else
                echo "      ✗ nsys report not found (may have failed)"
            fi
        else
            echo "    Skipping nsys profiling (tool not available)"
        fi
        
        # 2. Use ncu for detailed kernel-level metric profiling
        if $NCU_EXISTS; then
            echo "    Collecting detailed metrics with ncu..."
            # Use --set default instead of full to avoid resource exhaustion
            # Add --target-processes all to handle child processes
            # --csv outputs to stdout, redirect to file
            # --log-file captures ncu's own log messages
            # Capture stderr separately to see actual errors
            ncu --set default \
                --target-processes all \
                --csv --log-file="${profile_prefix}.ncu.log" \
                --force-overwrite \
                ./fem_assembly "$mesh_file" "$method" > "${profile_prefix}.ncu_metrics.csv" 2> "${profile_prefix}.ncu_stderr.log" || true
            
            if [ -f "${profile_prefix}.ncu_metrics.csv" ]; then
                echo "      ✓ ncu metrics generated: ${profile_prefix}.ncu_metrics.csv"
            else
                echo "      ✗ ncu metrics not found (may have failed or hit resource limits)"
                if [ -f "${profile_prefix}.ncu_stderr.log" ]; then
                    echo "      Error output:"
                    tail -20 "${profile_prefix}.ncu_stderr.log" | sed 's/^/        /'
                fi
            fi
        else
            echo "    Skipping ncu profiling (tool not available)"
        fi
        
        # 3. Also collect basic timing breakdown
        result_file="results/$method/${mesh_base}__${method}_results.txt"
        if [ -f "$result_file" ]; then
            cp "$result_file" "${profile_prefix}.timing.txt"
            echo "      ✓ Timing data copied: ${profile_prefix}.timing.txt"
        else
            echo "      ⚠ Timing file not found: $result_file"
        fi
        
        echo ""
    done
    echo ""
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  Profiling Complete!"
echo "=========================================="
echo ""
echo "Profile results saved in: $PROFILE_DIR/"
echo ""

# Count generated files
nsys_count=$(ls -1 $PROFILE_DIR/*.nsys-rep 2>/dev/null | wc -l)
ncu_count=$(ls -1 $PROFILE_DIR/*.ncu_metrics.csv 2>/dev/null | wc -l)
timing_count=$(ls -1 $PROFILE_DIR/*.timing.txt 2>/dev/null | wc -l)

echo "Summary:"
echo "  - nsys reports: $nsys_count"
echo "  - ncu metrics files: $ncu_count"
echo "  - timing files: $timing_count"
echo ""
echo "To view profiles:"
echo "  nsys-ui $PROFILE_DIR/*.nsys-rep"
echo "  (or download .nsys-rep files and open with Nsight Systems)"
echo ""
echo "To analyze ncu metrics:"
echo "  cat $PROFILE_DIR/*.ncu_metrics.csv"
echo ""

