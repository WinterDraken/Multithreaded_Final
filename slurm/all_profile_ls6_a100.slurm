#!/bin/bash
#SBATCH --job-name=fem_all_profile_h100
#SBATCH --output=all_profile_h100_%j.out
#SBATCH --error=all_profile_h100_%j.err
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: All Meshes Profiling (H100)
# Runs nsys and ncu profiling for each mesh and method combination
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
NCU_EXISTS=true
command -v nsys >/dev/null 2>&1 || NSYS_EXISTS=false
command -v ncu >/dev/null 2>&1 || NCU_EXISTS=false

if $NSYS_EXISTS; then
    echo "✓ nsys found: $(which nsys)"
    nsys --version
else
    echo "✗ nsys not found - profiling will be skipped"
    exit 1
fi

if $NCU_EXISTS; then
    echo "✓ ncu found: $(which ncu)"
    ncu --version
else
    echo "✗ ncu not found - ncu profiling will be skipped"
fi
echo ""

# ============================================================================
# PHASE 1: PROFILING ALL MESHES
# ============================================================================

echo "=========================================="
echo "  PHASE 1: Running Profiling"
echo "  Meshes: ${#MESH_FILES[@]} files"
echo "  Methods: ${METHODS[@]}"
echo "  Tools: nsys, ncu"
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
        
        # Run nsys profiling (NVTX markers will help identify CG_Solver region in UI)
        if $NSYS_EXISTS; then
            echo "  Running nsys profile..."
            nsys profile \
                --output="${profile_prefix}_nsys" \
                --stats=true \
                --trace=cuda,nvtx \
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
        
        # Run ncu profiling (profiles all kernels, NVTX markers help identify solver region)
        if $NCU_EXISTS; then
            echo "  Running ncu profile..."
            # Use basic metric set (compatible across GPU architectures)
            # Includes occupancy, speed-of-light metrics, and launch stats
            # Redirect application stdout/stderr separately, capture ncu CSV output
            ncu --set basic \
                --target-processes all \
                --csv \
                --log-file="${profile_prefix}_ncu.log" \
                --force-overwrite \
                ./fem_assembly "$mesh_file" "$method" > "${profile_prefix}_ncu_metrics.csv" 2> "${profile_prefix}_ncu_stderr.log" || true
            
            # If CSV contains application output (not metrics), it means ncu didn't profile any kernels
            # Check if file starts with CSV headers (contains "Kernel Name" or "ID")
            if [ -f "${profile_prefix}_ncu_metrics.csv" ] && [ -s "${profile_prefix}_ncu_metrics.csv" ]; then
                if grep -q "Kernel Name\|ID," "${profile_prefix}_ncu_metrics.csv" 2>/dev/null; then
                    echo "    ✓ ncu metrics generated: ${profile_prefix}_ncu_metrics.csv"
                    file_size=$(du -h "${profile_prefix}_ncu_metrics.csv" | awk '{print $1}')
                    echo "    File size: $file_size"
                else
                    echo "    ⚠ ncu CSV contains application output, not metrics (no kernels profiled)"
                    echo "    Moving application output to separate file..."
                    mv "${profile_prefix}_ncu_metrics.csv" "${profile_prefix}_app_output.log" 2>/dev/null || true
                    echo "    Check ${profile_prefix}_ncu.log and ${profile_prefix}_ncu_stderr.log for details"
                fi
            else
                echo "    ✗ ncu metrics file not found or empty (check ${profile_prefix}_ncu_stderr.log)"
            fi
            
            if [ -f "${profile_prefix}_ncu_metrics.csv" ]; then
                # Check if file has content (more than just headers/empty)
                if [ -s "${profile_prefix}_ncu_metrics.csv" ]; then
                    echo "    ✓ ncu metrics generated: ${profile_prefix}_ncu_metrics.csv"
                    file_size=$(du -h "${profile_prefix}_ncu_metrics.csv" | awk '{print $1}')
                    echo "    File size: $file_size"
                else
                    echo "    ⚠ ncu metrics file is empty (check ${profile_prefix}_ncu_stderr.log)"
                fi
            else
                echo "    ✗ ncu metrics file not found (check ${profile_prefix}_ncu_stderr.log)"
            fi
            
            # Show ncu errors/warnings if any
            if [ -f "${profile_prefix}_ncu_stderr.log" ] && [ -s "${profile_prefix}_ncu_stderr.log" ]; then
                echo "    ncu stderr (last 10 lines):"
                tail -10 "${profile_prefix}_ncu_stderr.log" | sed 's/^/      /'
            fi
            if [ -f "${profile_prefix}_ncu.log" ] && [ -s "${profile_prefix}_ncu.log" ]; then
                # Check for warnings about no kernels profiled
                if grep -q "No kernels were profiled\|WARNING" "${profile_prefix}_ncu.log" 2>/dev/null; then
                    echo "    ncu log warnings (last 5 lines):"
                    grep -i "warning\|error\|no kernels" "${profile_prefix}_ncu.log" | tail -5 | sed 's/^/      /'
                fi
            fi
        else
            echo "  Skipping ncu profiling (tool not available)"
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
total_ncu=0
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="${mesh_base}_profile_results_h100"
    nsys_count=$(ls -1 $PROFILE_DIR/*_nsys.nsys-rep 2>/dev/null | wc -l)
    ncu_count=$(ls -1 $PROFILE_DIR/*_ncu_metrics.csv 2>/dev/null | wc -l)
    total_nsys=$((total_nsys + nsys_count))
    total_ncu=$((total_ncu + ncu_count))
    echo "  $mesh_base: $nsys_count nsys report(s), $ncu_count ncu metric file(s) in $PROFILE_DIR/"
done

echo ""
echo "Total reports generated:"
echo "  - nsys: $total_nsys"
echo "  - ncu: $total_ncu"
echo ""
echo "Results structure:"
echo "  - nsys reports: {mesh}_profile_results_h100/{mesh}__{method}_nsys.nsys-rep"
echo "  - ncu metrics: {mesh}_profile_results_h100/{mesh}__{method}_ncu_metrics.csv"
echo ""
echo "To view nsys profiles:"
echo "  nsys-ui {mesh}_profile_results_h100/*_nsys.nsys-rep"
echo "  (or download .nsys-rep files and open with Nsight Systems)"
echo ""
echo "To extract nsys statistics:"
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    PROFILE_DIR="${mesh_base}_profile_results_h100"
    echo "  nsys stats --report gpukernsum $PROFILE_DIR/*_nsys.nsys-rep"
done
echo ""
echo "Note: Profiling is focused on the CG_Solver region (matrix solver only)"
echo ""