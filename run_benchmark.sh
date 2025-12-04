#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration
# =========================

# Labels for meshes (just tags for CSV)
MESH_TAGS=(
  "3d"
  "3d_large"
  "3d_large_coarse"
  "3d_large_fine"
  "3d_large_ultrafine"
)

# Corresponding mesh file paths
MESH_PATHS=(
  "CPU/bracket_3d.msh"
  "CPU/bracket_3d_large.msh"
  "CPU/bracket_3d_large_coarse.msh"
  "CPU/bracket_3d_large_fine.msh"
  "CPU/bracket_3d_large_ultrafine.msh"
)

# Reordering methods (must match reorder.cpp options)
METHODS=(
  "none"
  "rcm"
  "amd"
  "colamd"
)

# Use chol so we always get Cholesky_factor_ms in the result files
SOLVER="chol"

# Number of repetitions per (mesh, method)
RUNS=10

# Executable
EXE="./fem_assembly"

# Output CSV paths
RAW_CSV="benchmark_results_raw.csv"
AVG_CSV="benchmark_results_avg.csv"

# =========================
# Sanity checks
# =========================

if [[ ${#MESH_TAGS[@]} -ne ${#MESH_PATHS[@]} ]]; then
  echo "Error: MESH_TAGS and MESH_PATHS must have the same length." >&2
  exit 1
fi

if [[ ! -x "$EXE" ]]; then
  echo "Error: cannot execute $EXE (is it built and chmod +x'ed?)." >&2
  exit 1
fi

# =========================
# Prepare raw CSV header
# =========================
# Columns:
# 1: mesh_tag
# 2: mesh_file
# 3: method
# 4: solver
# 5: run
# 6: Bandwidth_orig
# 7: Bandwidth_reord
# 8: Reorder_ms
# 9: Cholesky_factor_ms
# 10: Solve_ms
echo "mesh_tag,mesh_file,method,solver,run,Bandwidth_orig,Bandwidth_reord,Reorder_ms,Cholesky_factor_ms,Solve_ms" > "$RAW_CSV"

# =========================
# Main benchmarking loop
# =========================

for ((mi = 0; mi < ${#MESH_TAGS[@]}; ++mi)); do
  mesh_tag="${MESH_TAGS[$mi]}"
  mesh_path="${MESH_PATHS[$mi]}"

  if [[ ! -f "$mesh_path" ]]; then
    echo "Warning: mesh file '$mesh_path' not found, skipping $mesh_tag." >&2
    continue
  fi

  for method in "${METHODS[@]}"; do
    echo "=== Mesh: $mesh_tag | Method: $method | Solver: $SOLVER ==="

    for ((run = 1; run <= RUNS; ++run)); do
      echo "  Run $run / $RUNS..."

      # Run fem_assembly and capture its stdout to get the "Wrote: <file>" line.
      # Don't pass -v so stdout is clean.
      output="$("$EXE" "$mesh_path" "$method" --solver="$SOLVER")" || {
        echo "    ERROR: fem_assembly failed for $mesh_tag, $method, run $run" >&2
        continue
      }

      # Extract result file path from the "Wrote: ..." line.
      result_file=$(echo "$output" | awk '/^Wrote: / {print $2}' | tail -n1)

      if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "    ERROR: Could not find result file for $mesh_tag, $method, run $run" >&2
        echo "    Output was:" >&2
        echo "$output" >&2
        continue
      fi

      # Parse values from result file
      mesh_file_val=$(awk '/^mesh / {print $2}' "$result_file")
      method_val=$(awk '/^reorder / {print $2}' "$result_file")
      solver_val=$(awk '/^solver/ {print $2}' "$result_file")

      bw_orig=$(awk '/^Bandwidth_orig / {print $2}' "$result_file")
      bw_reord=$(awk '/^Bandwidth_reord / {print $2}' "$result_file")
      reorder_ms=$(awk '/^Reorder_ms / {print $2}' "$result_file")
      chol_ms=$(awk '/^Cholesky_factor_ms / {print $2}' "$result_file")
      solve_ms=$(awk '/^Solve_ms / {print $2}' "$result_file")

      # Fallbacks (shouldn't usually trigger if solver=chol)
      : "${mesh_file_val:=$mesh_path}"
      : "${method_val:=$method}"
      : "${solver_val:=$SOLVER}"
      : "${bw_orig:=0}"
      : "${bw_reord:=0}"
      : "${reorder_ms:=0}"
      : "${chol_ms:=0}"
      : "${solve_ms:=0}"

      # Append one row to raw CSV
      echo "${mesh_tag},${mesh_file_val},${method_val},${solver_val},${run},${bw_orig},${bw_reord},${reorder_ms},${chol_ms},${solve_ms}" >> "$RAW_CSV"

      # Optional: if you worry about timestamp collisions in filenames, uncomment:
      # sleep 1
    done
  done
done

# =========================
# Compute averages
# =========================

# RAW_CSV fields:
# 1 mesh_tag
# 2 mesh_file
# 3 method
# 4 solver
# 5 run
# 6 Bandwidth_orig
# 7 Bandwidth_reord
# 8 Reorder_ms
# 9 Cholesky_factor_ms
# 10 Solve_ms

{
  echo "mesh_tag,mesh_file,method,solver,num_runs,avg_Bandwidth_orig,avg_Bandwidth_reord,avg_Reorder_ms,avg_Cholesky_factor_ms,avg_Solve_ms"
  awk -F',' 'NR>1 {
      key = $1","$2","$3","$4;  # mesh_tag,mesh_file,method,solver
      cnt[key]++
      sum_bw_orig[key]   += $6
      sum_bw_reord[key]  += $7
      sum_reorder[key]   += $8
      sum_chol[key]      += $9
      sum_solve[key]     += $10
  }
  END {
      for (k in cnt) {
          n = cnt[k]
          avg_bw_orig  = (n > 0 ? sum_bw_orig[k]  / n : 0)
          avg_bw_reord = (n > 0 ? sum_bw_reord[k] / n : 0)
          avg_reorder  = (n > 0 ? sum_reorder[k]  / n : 0)
          avg_chol     = (n > 0 ? sum_chol[k]     / n : 0)
          avg_solve    = (n > 0 ? sum_solve[k]    / n : 0)
          printf "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                 k, n, avg_bw_orig, avg_bw_reord, avg_reorder, avg_chol, avg_solve
      }
  }' "$RAW_CSV" | sort -t',' -k1,1 -k3,3 -k4,4
} > "$AVG_CSV"

echo
echo "Benchmarking complete."
echo "Raw per-run metrics:  $RAW_CSV"
echo "Averaged metrics:     $AVG_CSV"
