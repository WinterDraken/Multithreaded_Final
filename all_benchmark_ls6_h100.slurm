#!/bin/bash
#SBATCH --job-name=fem_all_benchmark_h100
#SBATCH --output=all_benchmark_h100_%j.out
#SBATCH --error=all_benchmark_h100_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A TRA25006

# ============================================================================
# FEM GPU Solver: All Meshes Benchmark (H100)
# Runs 10 iterations for each mesh and method, saves results separately
# ============================================================================

set -e

echo "=========================================="
echo "  FEM GPU Solver: All Meshes Benchmark (H100)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load modules
echo "=== Loading Modules ==="
module purge
module load cuda/12.2
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
NUM_RUNS=10

# Function to extract mesh base name
get_mesh_base() {
    local mesh_file=$1
    local basename=$(basename "$mesh_file" .msh)
    echo "$basename"
}

# ============================================================================
# PHASE 1: BENCHMARKING ALL MESHES
# ============================================================================

echo "=========================================="
echo "  PHASE 1: Running Benchmarks"
echo "  Meshes: ${#MESH_FILES[@]} files"
echo "  Methods: ${METHODS[@]}"
echo "  Runs per combination: $NUM_RUNS"
echo "=========================================="
echo ""

# Create results_h100 directory structure
mkdir -p results_h100
for method in "${METHODS[@]}"; do
    mkdir -p "results_h100/$method"
done

# Create symlink so main.cpp saves to results_h100
# (main.cpp hardcodes "results" directory, so we use symlink)
if [ -L "results" ]; then
    rm "results"
elif [ -d "results" ] && [ ! -L "results" ]; then
    echo "WARNING: 'results' directory exists and is not a symlink"
    echo "Creating backup and symlink..."
    mv results results_backup_$(date +%s)
fi
ln -sfn results_h100 results

echo "Set up results directory structure (results -> results_h100)"
echo ""

# Process each mesh
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    
    echo "========================================"
    echo "Processing mesh: $mesh_base"
    echo "========================================"
    echo ""
    
    # Create benchmark directory for this mesh
    BENCHMARK_DIR="${mesh_base}_benchmark_results_h100"
    mkdir -p "$BENCHMARK_DIR"
    
    # Process each method
    for method in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Mesh: $mesh_base, Method: $method"
        echo "----------------------------------------"
        
        # Run the solver 10 times, each saving to a unique file
        echo "  Running $NUM_RUNS benchmark iterations..."
        
        for run in $(seq 1 $NUM_RUNS); do
            echo "    Run $run/$NUM_RUNS..."
            # Pass run number as third argument to get unique filenames
            # Results will be saved to results_h100/$method/ via symlink
            ./fem_assembly "$mesh_file" "$method" "$run" > /dev/null 2>&1
            
            # Wait a moment to ensure file is written
            sleep 0.1
        done
        
        echo "  All runs completed. Reading results files..."
        
        # Now read all result files and calculate average
        # Arrays to store results from all runs
        declare -a cpu_values
        declare -a reorder_values
        declare -a h2d_values
        declare -a solve_values
        declare -a total_values
        
        # Read all result files for this method and mesh
        result_pattern="results_h100/$method/${mesh_base}__${method}_run*_results.txt"
        result_files=($(ls -1 $result_pattern 2>/dev/null | sort -V))
        
        if [ ${#result_files[@]} -eq 0 ]; then
            echo "    ⚠ WARNING: No result files found matching pattern: $result_pattern"
            echo "    Available files in results_h100/$method/:"
            ls -la "results_h100/$method/" 2>&1 | sed 's/^/      /' || echo "      Directory does not exist"
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
        
        # Save to CSV with ONLY the average (as requested)
        output_file="$BENCHMARK_DIR/${mesh_base}__${method}.csv"
        echo "cpu_assembly_ms,reordering_ms,h2d_ms,gpu_solve_ms,total_ms" > "$output_file"
        echo "$cpu_avg,$reorder_avg,$h2d_avg,$solve_avg,$total_avg" >> "$output_file"
        
        echo "  ✓ Average results saved to: $output_file (from $num_found runs)"
        echo ""
    done
    
    echo ""
done

# Remove symlink (restore original if it existed)
if [ -L "results" ]; then
    rm "results"
    if [ -d "results_backup_"* ] 2>/dev/null; then
        echo "Restoring original results directory..."
        mv results_backup_* results
    fi
fi

echo "=========================================="
echo "  Phase 1 Complete: All Benchmarks Done!"
echo "=========================================="
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  Benchmark Summary"
echo "=========================================="
echo ""

# Count CSV files for each mesh
total_csvs=0
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    BENCHMARK_DIR="${mesh_base}_benchmark_results_h100"
    csv_count=$(ls -1 $BENCHMARK_DIR/*.csv 2>/dev/null | wc -l)
    total_csvs=$((total_csvs + csv_count))
    echo "  $mesh_base: $csv_count CSV file(s) in $BENCHMARK_DIR/"
done

echo ""
echo "Total CSV files generated: $total_csvs"
echo ""
echo "Results structure:"
echo "  - Individual run results: results_h100/{method}/"
echo "  - Average CSV files: {mesh}_benchmark_results_h100/"
echo ""
echo "To view results:"
for mesh_file in "${MESH_FILES[@]}"; do
    mesh_base=$(get_mesh_base "$mesh_file")
    BENCHMARK_DIR="${mesh_base}_benchmark_results_h100"
    echo "  cat $BENCHMARK_DIR/*.csv"
done
echo ""

