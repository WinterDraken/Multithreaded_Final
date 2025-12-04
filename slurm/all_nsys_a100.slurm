#!/bin/bash
#SBATCH --job-name=fem_nsys_a100
#SBATCH --output=all_nsys_a100_%j.out
#SBATCH --error=all_nsys_a100_%j.err
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: Nsight Systems Profiling (A100)
# Runs nsys profiling for each mesh and method combination
# Focused on GPU solver region with insights on warp divergence and memory access
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: Nsight Systems Profiling (A100)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2  # Module with nsys tools
module load gcc/9.4.0
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

# Find all 3D mesh files in CPU directory
MESH_FILES=($(ls -1 CPU/bracket_3d*.msh 2>/dev/null | sort -V))

if [ ${#MESH_FILES[@]} -eq 0 ]; then
    echo "ERROR: No mesh files found in CPU directory"
    exit 1
fi

echo "Found ${#MESH_FILES[@]} mesh file(s):"
for mesh in "${MESH_FILES[@]}"; do
    echo "  - $mesh"
done
echo ""

METHODS=("none" "rcm" "amd")

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

if $NSYS_EXISTS; then
    echo "✓ nsys found: $(which nsys)"
    nsys --version
else
    echo "✗ nsys not found - profiling will be skipped"
    exit 1
fi
echo ""

# ============================================================================
# PHASE 1: PROFILING ALL MESHES
# ============================================================================

echo "=========================================="
echo "  PHASE 1: Running Nsight Systems Profiling"
echo "  Meshes: ${#MESH_FILES[@]} files"
echo "  Methods: ${METHODS[@]}"
echo "  Tool: nsys (focused on GPU solver region)"
echo "=========================================="
echo ""

# Create a100 base directory
A100_BASE_DIR="a100"
mkdir -p "$A100_BASE_DIR"

# Process each mesh
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    
    echo "========================================"
    echo "Processing mesh: $mesh_base"
    echo "========================================"
    echo ""
    
    # Create profile directory for this mesh under a100/
    PROFILE_DIR="$A100_BASE_DIR/${mesh_base}_profile_results"
    mkdir -p "$PROFILE_DIR"
    
    # Process each method
    for method in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Profiling mesh: $mesh_base, method: $method"
        echo "----------------------------------------"
        
        profile_prefix="$PROFILE_DIR/${mesh_base}__${method}"
        
        # Run nsys profiling with enhanced tracing for warp divergence and memory access patterns
        # NVTX markers will help identify CG_Solver region in the timeline
        if $NSYS_EXISTS; then
            echo "  Running nsys profile..."
            echo "  (Tracing: CUDA, NVTX, Memory, CPU context for warp divergence analysis)"
            nsys profile \
                --output="${profile_prefix}_nsys" \
                --stats=true \
                --trace=cuda,nvtx,osrt \
                --sample=cpu \
                --cuda-memory-usage=true \
                --cuda-graph-trace=node \
                ./fem_assembly "$mesh_file" "$method" > "${profile_prefix}_nsys_stdout.log" 2> "${profile_prefix}_nsys_stderr.log" || true
            
            if [ -f "${profile_prefix}_nsys.nsys-rep" ]; then
                echo "    ✓ nsys report generated: ${profile_prefix}_nsys.nsys-rep"
                # Get file size for info
                file_size=$(du -h "${profile_prefix}_nsys.nsys-rep" | awk '{print $1}')
                echo "    File size: $file_size"
            else
                echo "    ✗ nsys report not found (check ${profile_prefix}_nsys_stderr.log)"
            fi
            
            # Show nsys errors if any
            if [ -f "${profile_prefix}_nsys_stderr.log" ] && [ -s "${profile_prefix}_nsys_stderr.log" ]; then
                echo "    nsys stderr (last 10 lines):"
                tail -10 "${profile_prefix}_nsys_stderr.log" | sed 's/^/      /'
            fi
        else
            echo "  Skipping nsys profiling (tool not available)"
        fi
        
        echo ""
    done
    
    echo ""
done

echo "=========================================="
echo "  Phase 1 Complete: All Profiling Done!"
echo "=========================================="
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  Profiling Summary"
echo "=========================================="
echo ""

# Count reports for each mesh
total_nsys=0
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="$A100_BASE_DIR/${mesh_base}_profile_results"
    nsys_count=$(ls -1 $PROFILE_DIR/*_nsys.nsys-rep 2>/dev/null | wc -l)
    total_nsys=$((total_nsys + nsys_count))
    echo "  $mesh_base: $nsys_count nsys report(s) in $PROFILE_DIR/"
done

echo ""
echo "Total nsys reports generated: $total_nsys"
echo ""
echo "Results structure:"
echo "  - nsys reports: a100/{mesh}_profile_results/{mesh}__{method}_nsys.nsys-rep"
echo ""
echo "To view nsys profiles:"
echo "  nsys-ui a100/*_profile_results/*_nsys.nsys-rep"
echo "  (or download .nsys-rep files and open with Nsight Systems)"
echo ""
echo "To extract statistics:"
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="$A100_BASE_DIR/${mesh_base}_profile_results"
    echo "  nsys stats --report gpukernsum $PROFILE_DIR/*_nsys.nsys-rep"
done
echo ""
echo "Profiling insights available:"
echo "  - Warp divergence: Check GPU timeline for divergent branches"
echo "  - Memory access patterns: Check CUDA memory operations timeline"
echo "  - CG_Solver region: Look for NVTX 'CG_Solver' range in timeline"
echo "  - Kernel efficiency: Use Statistics view for kernel performance metrics"
echo ""

