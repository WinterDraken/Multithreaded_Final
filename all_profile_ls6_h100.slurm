#!/bin/bash
#SBATCH --job-name=fem_all_profile_h100
#SBATCH --output=all_profile_h100_%j.out
#SBATCH --error=all_profile_h100_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: All Meshes Profiling (H100)
# Runs nsys profiling for each mesh and method combination
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: All Meshes Profiling (H100)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2  # Module with nsys/ncu tools
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
echo "  PHASE 1: Running Profiling"
echo "  Meshes: ${#MESH_FILES[@]} files"
echo "  Methods: ${METHODS[@]}"
echo "  Tool: nsys"
echo "=========================================="
echo ""

# Process each mesh
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    
    echo "========================================"
    echo "Processing mesh: $mesh_base"
    echo "========================================"
    echo ""
    
    # Create profile directory for this mesh
    PROFILE_DIR="${mesh_base}_profile_results_h100"
    mkdir -p "$PROFILE_DIR"
    
    # Process each method
    for method in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Profiling mesh: $mesh_base, method: $method"
        echo "----------------------------------------"
        
        profile_prefix="$PROFILE_DIR/${mesh_base}__${method}"
        
        # Run nsys profiling
        if $NSYS_EXISTS; then
            echo "  Running nsys profile..."
            nsys profile \
                --output="$profile_prefix" \
                --stats=true \
                --trace=cuda,nvtx \
                ./fem_assembly "$mesh_file" "$method" > /dev/null 2>&1
            
            if [ -f "${profile_prefix}.nsys-rep" ]; then
                echo "    ✓ nsys report generated: ${profile_prefix}.nsys-rep"
                # Get file size for info
                file_size=$(du -h "${profile_prefix}.nsys-rep" | awk '{print $1}')
                echo "    File size: $file_size"
            else
                echo "    ✗ nsys report not found (may have failed)"
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

# Count nsys reports for each mesh
total_reports=0
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="${mesh_base}_profile_results_h100"
    report_count=$(ls -1 $PROFILE_DIR/*.nsys-rep 2>/dev/null | wc -l)
    total_reports=$((total_reports + report_count))
    echo "  $mesh_base: $report_count nsys report(s) in $PROFILE_DIR/"
done

echo ""
echo "Total nsys reports generated: $total_reports"
echo ""
echo "Results structure:"
echo "  - Profile reports: {mesh}_profile_results_h100/{mesh}__{method}.nsys-rep"
echo ""
echo "To view profiles:"
echo "  nsys-ui {mesh}_profile_results_h100/*.nsys-rep"
echo "  (or download .nsys-rep files and open with Nsight Systems)"
echo ""
echo "To extract statistics:"
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="${mesh_base}_profile_results_h100"
    echo "  nsys stats --report gpukernsum $PROFILE_DIR/*.nsys-rep"
done
echo ""

