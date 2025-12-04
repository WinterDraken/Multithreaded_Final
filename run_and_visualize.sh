#!/bin/bash
# ============================================================================
# FEM Solver: Run and Visualize Script
# Step 1: Runs the solver for all reordering methods
# Step 2: Converts solution vectors to Gmsh format for visualization
# ============================================================================

set -e  # Exit on error

# Default values
MESH="bracket_3d"
MESH_DIR="CPU"
RESULTS_DIR="results"
OUTPUT_DIR="gmsh_results"
METHODS=("none" "rcm" "amd")
SOLVER="cg"

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mesh MESH          Mesh base name (default: bracket_3d)"
    echo "  --mesh-dir DIR       Directory with mesh files (default: CPU)"
    echo "  --results-dir DIR    Directory for results (default: results)"
    echo "  --output-dir DIR     Output directory for Gmsh files (default: gmsh_results)"
    echo "  --methods LIST       Comma-separated list of methods (default: none,rcm,amd)"
    echo "  --solver SOLVER      Solver to use: cg, chol, lu, or direct (default: cg)"
    echo "  --skip-run           Skip Step 1 (solver), only do visualization"
    echo "  --skip-viz           Skip Step 2 (visualization), only run solver"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mesh bracket_3d_large"
    echo "  $0 --mesh bracket_3d_xlarge --methods none,rcm"
    echo "  $0 --mesh bracket_3d --solver chol  # Use Cholesky solver"
    echo "  $0 --mesh bracket_3d --skip-run  # Only visualize existing results"
    exit 1
}

SKIP_RUN=false
SKIP_VIZ=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mesh)
            MESH="$2"
            shift 2
            ;;
        --mesh-dir)
            MESH_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --solver)
            SOLVER="$2"
            shift 2
            ;;
        --skip-run)
            SKIP_RUN=true
            shift
            ;;
        --skip-viz)
            SKIP_VIZ=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

MESH_FILE="${MESH_DIR}/${MESH}.msh"

echo "=========================================="
echo "  FEM Solver: Run and Visualize"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Mesh: $MESH"
echo "  Mesh file: $MESH_FILE"
echo "  Methods: ${METHODS[@]}"
echo "  Solver: $SOLVER"
echo "  Results dir: $RESULTS_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Check if mesh file exists
if [ ! -f "$MESH_FILE" ]; then
    echo "ERROR: Mesh file not found: $MESH_FILE"
    exit 1
fi

# Check if executable exists
if [ ! -f "./fem_assembly" ]; then
    echo "ERROR: Executable not found: ./fem_assembly"
    echo "Please build the project first: make"
    exit 1
fi

# Check if Python script exists
if [ ! -f "./gmsh_visualize.py" ]; then
    echo "ERROR: Visualization script not found: ./gmsh_visualize.py"
    exit 1
fi

# ============================================================================
# STEP 1: Run Solver
# ============================================================================

if [ "$SKIP_RUN" = false ]; then
    echo "=========================================="
    echo "  STEP 1: Running Solver"
    echo "=========================================="
    echo ""
    
    for method in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Running: $MESH with method: $method"
        echo "----------------------------------------"
        
        # Run the solver with specified solver option
        ./fem_assembly "$MESH_FILE" "$method" --solver="$SOLVER"
        
        # Check if solution file was created
        solution_file="${RESULTS_DIR}/${method}/${MESH}__${method}_solution.txt"
        if [ -f "$solution_file" ]; then
            echo "  ✓ Solution saved: $solution_file"
        else
            echo "  ✗ WARNING: Solution file not found: $solution_file"
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "  Step 1 Complete: Solver Runs Done!"
    echo "=========================================="
    echo ""
else
    echo "=========================================="
    echo "  STEP 1: Skipped (--skip-run flag)"
    echo "=========================================="
    echo ""
fi

# ============================================================================
# STEP 2: Convert to Gmsh Format
# ============================================================================

if [ "$SKIP_VIZ" = false ]; then
    echo "=========================================="
    echo "  STEP 2: Converting to Gmsh Format"
    echo "=========================================="
    echo ""
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    for method in "${METHODS[@]}"; do
        echo "----------------------------------------"
        echo "Converting: $MESH with method: $method"
        echo "----------------------------------------"
        
        # Check if solution file exists
        solution_file="${RESULTS_DIR}/${method}/${MESH}__${method}_solution.txt"
        if [ ! -f "$solution_file" ]; then
            echo "  ✗ ERROR: Solution file not found: $solution_file"
            echo "  Skipping visualization for this method."
            echo ""
            continue
        fi
        
        # Run the Python conversion script
        python3 gmsh_visualize.py \
            --mesh "$MESH" \
            --method "$method" \
            --mesh-dir "$MESH_DIR" \
            --results-dir "$RESULTS_DIR" \
            --output-dir "$OUTPUT_DIR"
        
        # Check if output file was created
        output_file="${OUTPUT_DIR}/${MESH}__${method}_with_displacement.msh"
        if [ -f "$output_file" ]; then
            echo "  ✓ Gmsh file created: $output_file"
        else
            echo "  ✗ WARNING: Gmsh file not found: $output_file"
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "  Step 2 Complete: Gmsh Files Created!"
    echo "=========================================="
    echo ""
else
    echo "=========================================="
    echo "  STEP 2: Skipped (--skip-viz flag)"
    echo "=========================================="
    echo ""
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "  All Steps Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Mesh: $MESH"
echo "  Methods processed: ${METHODS[@]}"
echo ""

# Count files
solution_count=0
gmsh_count=0

for method in "${METHODS[@]}"; do
    if [ -f "${RESULTS_DIR}/${method}/${MESH}__${method}_solution.txt" ]; then
        ((solution_count++))
    fi
    if [ -f "${OUTPUT_DIR}/${MESH}__${method}_with_displacement.msh" ]; then
        ((gmsh_count++))
    fi
done

echo "Files generated:"
echo "  - Solution files: $solution_count/${#METHODS[@]}"
echo "  - Gmsh files: $gmsh_count/${#METHODS[@]}"
echo ""

if [ "$SKIP_VIZ" = false ] && [ $gmsh_count -gt 0 ]; then
    echo "To visualize in Gmsh:"
    for method in "${METHODS[@]}"; do
        gmsh_file="${OUTPUT_DIR}/${MESH}__${method}_with_displacement.msh"
        if [ -f "$gmsh_file" ]; then
            echo "  gmsh $gmsh_file"
        fi
    done
    echo ""
    echo "In Gmsh GUI:"
    echo "  1. Go to View → Options → Visibility"
    echo "  2. Select 'Displacement_Magnitude' for scalar field"
    echo "  3. Or select 'Displacement_Vector' for vector arrows"
    echo "  4. Adjust colormap and scale as needed"
    echo ""
fi
