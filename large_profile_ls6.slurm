#!/bin/bash
#SBATCH --job-name=fem_large_profile
#SBATCH --output=large_profile_%j.out
#SBATCH --error=large_profile_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: large Mesh Run & Profile
# Runs solver firstto generate results, then profiles with nsys + ncu
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: large Run & Profile"
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

# Define mesh - only large
MESH_FILE="CPU/bracket_3d_large.msh"
MESH_BASE="bracket_3d_large"

METHODS=("none" "rcm" "amd")
BENCHMARK_DIR="large_benchmark_results_a100"
PROFILE_DIR="large_profile_results_a100"
mkdir -p $BENCHMARK_DIR
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
# PHASE 1: RUN SOLVER (Generate Results)
# ============================================================================

if [ ! -f "$MESH_FILE" ]; then
    echo "ERROR: Mesh file not found: $MESH_FILE"
    exit 1
fi

echo "=========================================="
echo "  PHASE 1: Running Solver"
echo "  Mesh: $MESH_BASE"
echo "  Methods: ${METHODS[@]}"
echo "=========================================="
echo ""

for method in "${METHODS[@]}"; do
    echo "----------------------------------------"
    echo "Running mesh: $MESH_BASE, method: $method"
    echo "----------------------------------------"
    
    # Run the solver 10 times, each saving to a unique file
    NUM_RUNS=10
    echo "  Running $NUM_RUNS benchmark iterations..."
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "    Run $run/$NUM_RUNS..."
        # Pass run number as third argument to get unique filenames
        ./fem_assembly "$MESH_FILE" "$method" "$run" > /dev/null 2>&1
        
        # Wait a moment to ensure file is written
        sleep 0.1
    done
    
    echo "  All runs completed. Reading results files..."
    
    # Now read all result files and assemble CSV
    # Arrays to store results from all runs
    declare -a cpu_values
    declare -a reorder_values
    declare -a h2d_values
    declare -a solve_values
    declare -a total_values
    
    # Read all result files for this method
    result_pattern="results/$method/${MESH_BASE}__${method}_run*_results.txt"
    result_files=($(ls -1 $result_pattern 2>/dev/null | sort -V))
    
    if [ ${#result_files[@]} -eq 0 ]; then
        echo "    ⚠ WARNING: No result files found matching pattern: $result_pattern"
        echo "    Available files in results/$method/:"
        ls -la "results/$method/" 2>&1 | sed 's/^/      /' || echo "      Directory does not exist"
        continue
    fi
    
    echo "    Found ${#result_files[@]} result file(s)"
    
    # Read each result file
    for result_file in "${result_files[@]}"; do
        if [ -f "$result_file" ]; then
            cpu=$(grep "CPU_Assembly_ms" "$result_file" | awk '{print $2}')
            reorder=$(grep "Reordering_ms" "$result_file" | awk '{print $2}')
            h2d=$(grep "HostToDevice_ms" "$result_file" | awk '{print $2}')
            solve=$(grep "GPU_Solve_ms" "$result_file" | awk '{print $2}')
            total=$(echo "$cpu + $reorder + $h2d + $solve" | bc -l)
            
            # Store values in arrays
            cpu_values+=("$cpu")
            reorder_values+=("$reorder")
            h2d_values+=("$h2d")
            solve_values+=("$solve")
            total_values+=("$total")
        fi
    done
    
    # Calculate averages
    num_found=${#cpu_values[@]}
    if [ $num_found -eq 0 ]; then
        echo "    ⚠ WARNING: No valid results found. Skipping CSV generation."
        continue
    fi
    
    cpu_sum=0
    reorder_sum=0
    h2d_sum=0
    solve_sum=0
    total_sum=0
    
    for i in $(seq 0 $((num_found - 1))); do
        cpu_sum=$(echo "$cpu_sum + ${cpu_values[$i]}" | bc -l)
        reorder_sum=$(echo "$reorder_sum + ${reorder_values[$i]}" | bc -l)
        h2d_sum=$(echo "$h2d_sum + ${h2d_values[$i]}" | bc -l)
        solve_sum=$(echo "$solve_sum + ${solve_values[$i]}" | bc -l)
        total_sum=$(echo "$total_sum + ${total_values[$i]}" | bc -l)
    done
    
    cpu_avg=$(echo "scale=6; $cpu_sum / $num_found" | bc -l)
    reorder_avg=$(echo "scale=6; $reorder_sum / $num_found" | bc -l)
    h2d_avg=$(echo "scale=6; $h2d_sum / $num_found" | bc -l)
    solve_avg=$(echo "scale=6; $solve_sum / $num_found" | bc -l)
    total_avg=$(echo "scale=6; $total_sum / $num_found" | bc -l)
    
    # Save to CSV with all runs and average
    output_file="$BENCHMARK_DIR/${MESH_BASE}__${method}.csv"
    echo "run,cpu_assembly_ms,reordering_ms,h2d_ms,gpu_solve_ms,total_ms" > "$output_file"
    
    # Write all runs
    for i in $(seq 0 $((num_found - 1))); do
        run_num=$((i + 1))
        echo "$run_num,${cpu_values[$i]},${reorder_values[$i]},${h2d_values[$i]},${solve_values[$i]},${total_values[$i]}" >> "$output_file"
    done
    
    # Write average row
    echo "average,$cpu_avg,$reorder_avg,$h2d_avg,$solve_avg,$total_avg" >> "$output_file"
    
    echo "  ✓ Results saved to: $output_file ($num_found runs + average)"
    echo ""
done

echo "=========================================="
echo "  Phase 1 Complete: Solver Runs Done!"
echo "=========================================="
echo ""

# ============================================================================
# PHASE 2: PROFILING
# ============================================================================

echo "=========================================="
echo "  PHASE 2: Running Profiling"
echo "  Mesh: $MESH_BASE"
echo "  Methods: ${METHODS[@]}"
echo "  Tools: nsys=$NSYS_EXISTS, ncu=$NCU_EXISTS"
echo "=========================================="
echo ""

for method in "${METHODS[@]}"; do
    echo "----------------------------------------"
    echo "Profiling mesh: $MESH_BASE, method: $method"
    echo "----------------------------------------"
    
    profile_prefix="$PROFILE_DIR/${MESH_BASE}__${method}"
    
    # 1. Use nsys for comprehensive system-level profiling and timeline
    if $NSYS_EXISTS; then
        echo "  Running nsys profile..."
        nsys profile \
            --output="$profile_prefix" \
            --stats=true \
            --trace=cuda,nvtx \
            ./fem_assembly "$MESH_FILE" "$method" > /dev/null 2>&1
        
        if [ -f "${profile_prefix}.nsys-rep" ]; then
            echo "    ✓ nsys report generated: ${profile_prefix}.nsys-rep"
        else
            echo "    ✗ nsys report not found (may have failed)"
        fi
    else
        echo "  Skipping nsys profiling (tool not available)"
    fi
    
    # 2. Use ncu for detailed kernel-level metric profiling
    if $NCU_EXISTS; then
        echo "  Collecting detailed metrics with ncu..."
        # Use --set speed-of-light for lighter metrics (faster than default)
        # Add --target-processes all to handle child processes
        # --kernel-regex targets only the custom compute kernels (faster profiling)
        #   - kernelKe_Tet4_3D: computes local element stiffness (most compute-intensive)
        #   - assembleCSR_atomic_kernel: assembles global CSR matrix
        # --csv outputs CSV to stdout
        # --log-file captures ncu's own log messages
        # Note: ncu captures application output, so we need to filter it
        # Run ncu and capture all output, then filter out application stdout
        ncu --set speed-of-light \
            --target-processes all \
            --kernel-regex "kernelKe_Tet4_3D|assembleCSR_atomic_kernel" \
            --csv --log-file="${profile_prefix}.ncu.log" \
            --force-overwrite \
            ./fem_assembly "$MESH_FILE" "$method" > "${profile_prefix}.ncu_metrics_raw.txt" 2> "${profile_prefix}.ncu_stderr.log" || true
        
        # Filter out application output lines (they start with specific patterns)
        # Keep only lines that look like CSV data (contain commas or are CSV headers)
        if [ -f "${profile_prefix}.ncu_metrics_raw.txt" ]; then
            # Extract CSV lines (lines with commas, or header-like lines)
            grep -E "^[^=]*," "${profile_prefix}.ncu_metrics_raw.txt" > "${profile_prefix}.ncu_metrics.csv" || \
            grep -E "^ID," "${profile_prefix}.ncu_metrics_raw.txt" > "${profile_prefix}.ncu_metrics.csv" || \
            grep -E "^Kernel Name|^Section Name" "${profile_prefix}.ncu_metrics_raw.txt" > "${profile_prefix}.ncu_metrics.csv" || \
            touch "${profile_prefix}.ncu_metrics.csv"
            
            # Clean up raw file if CSV was extracted
            if [ -s "${profile_prefix}.ncu_metrics.csv" ]; then
                rm -f "${profile_prefix}.ncu_metrics_raw.txt"
            fi
        fi
        
        if [ -f "${profile_prefix}.ncu_metrics.csv" ] && [ -s "${profile_prefix}.ncu_metrics.csv" ]; then
            echo "    ✓ ncu metrics generated: ${profile_prefix}.ncu_metrics.csv"
            echo "    File size: $(wc -l < "${profile_prefix}.ncu_metrics.csv") lines"
        else
            echo "    ✗ ncu metrics not found or empty (may have failed or no kernels profiled)"
            if [ -f "${profile_prefix}.ncu_stderr.log" ]; then
                echo "    Error output:"
                tail -20 "${profile_prefix}.ncu_stderr.log" | sed 's/^/      /'
            fi
            if [ -f "${profile_prefix}.ncu.log" ]; then
                echo "    ncu log warnings:"
                grep -i "warning\|error\|no metrics" "${profile_prefix}.ncu.log" | tail -5 | sed 's/^/      /'
            fi
        fi
    else
        echo "  Skipping ncu profiling (tool not available)"
    fi
    
    # 3. Copy timing data
    result_file="results/$method/${MESH_BASE}__${method}_results.txt"
    if [ -f "$result_file" ]; then
        cp "$result_file" "${profile_prefix}.timing.txt"
        echo "    ✓ Timing data copied: ${profile_prefix}.timing.txt"
    else
        echo "    ⚠ Timing file not found: $result_file"
    fi
    
    echo ""
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  All Phases Complete!"
echo "=========================================="
echo ""
echo "Benchmark results saved in: $BENCHMARK_DIR/"
echo "Profile results saved in: $PROFILE_DIR/"
echo ""

# Count generated files
benchmark_count=$(ls -1 $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l)
nsys_count=$(ls -1 $PROFILE_DIR/*.nsys-rep 2>/dev/null | wc -l)
ncu_count=$(ls -1 $PROFILE_DIR/*.ncu_metrics.csv 2>/dev/null | wc -l)
timing_count=$(ls -1 $PROFILE_DIR/*.timing.txt 2>/dev/null | wc -l)

echo "Summary:"
echo "  - Benchmark CSV files: $benchmark_count"
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
echo "To view benchmark results:"
echo "  cat $BENCHMARK_DIR/*.csv"
echo ""