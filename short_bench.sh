#!/usr/bin/env bash
# Simple benchmarking script: run each reordering method twice and average runtimes
set -euo pipefail

ROOTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOTDIR"

MEShes=("CPU/bracket_3d.msh" "CPU/bracket_3d_large.msh")
METHODS=("none" "rcm" "amd")
NUM_RUNS=2
OUTDIR="short_results"
mkdir -p "$OUTDIR"

warn() { echo "[WARN] $*" >&2; }
info() { echo "[INFO] $*"; }

if [ ! -x "./fem_assembly" ]; then
  warn "Executable './fem_assembly' not found or not executable. The script will still run but expect failures unless you build/install it."
fi

for mesh in "${MEShes[@]}"; do
  if [ ! -f "$mesh" ]; then
    warn "Mesh file not found: $mesh -- skipping"
    continue
  fi

  mesh_base=$(basename "$mesh" .msh)
  summary_file="$OUTDIR/${mesh_base}__averages.csv"

  # Write header
  echo "method,avg_cpu_assembly_ms,avg_reordering_ms,avg_h2d_ms,avg_gpu_solve_ms,avg_total_ms,runs_completed" > "$summary_file"

  for method in "${METHODS[@]}"; do
    info "Mesh=$mesh_base Method=$method -> running $NUM_RUNS times"

    sum_cpu=0
    sum_reorder=0
    sum_h2d=0
    sum_gpu=0
    count=0

    for run in $(seq 1 $NUM_RUNS); do
      info "  Run $run/$NUM_RUNS"
      # Run the binary (silence stdout/stderr but allow it to create result files)
      if [ -x "./fem_assembly" ]; then
        ./fem_assembly "$mesh" "$method" >/dev/null 2>&1 || true
      else
        warn "Skipping execution because ./fem_assembly is missing"
      fi

      # Expected results file location (based on existing project layout)
      result_file="results/${method}/${mesh_base}__${method}_results.txt"
      if [ ! -f "$result_file" ]; then
        warn "Result file not found after run: $result_file"
        continue
      fi

      # Extract numeric values (fallback to 0 if not found)
      cpu=$(grep -E "CPU_Assembly_ms" "$result_file" | awk '{print $2}' || echo 0)
      reorder=$(grep -E "Reordering_ms" "$result_file" | awk '{print $2}' || echo 0)
      h2d=$(grep -E "HostToDevice_ms" "$result_file" | awk '{print $2}' || echo 0)
      gpu=$(grep -E "GPU_Solve_ms" "$result_file" | awk '{print $2}' || echo 0)

      # ensure numeric (replace empty with 0)
      cpu=${cpu:-0}
      reorder=${reorder:-0}
      h2d=${h2d:-0}
      gpu=${gpu:-0}

      # Use awk for floating point addition
      sum_cpu=$(awk -v a="$sum_cpu" -v b="$cpu" 'BEGIN{printf "%f", a+b}')
      sum_reorder=$(awk -v a="$sum_reorder" -v b="$reorder" 'BEGIN{printf "%f", a+b}')
      sum_h2d=$(awk -v a="$sum_h2d" -v b="$h2d" 'BEGIN{printf "%f", a+b}')
      sum_gpu=$(awk -v a="$sum_gpu" -v b="$gpu" 'BEGIN{printf "%f", a+b}')

      count=$((count+1))
    done

    if [ "$count" -eq 0 ]; then
      warn "No successful runs recorded for $mesh_base / $method"
      echo "$method, , , , , ,0" >> "$summary_file"
      continue
    fi

    avg_cpu=$(awk -v s="$sum_cpu" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
    avg_reorder=$(awk -v s="$sum_reorder" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
    avg_h2d=$(awk -v s="$sum_h2d" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
    avg_gpu=$(awk -v s="$sum_gpu" -v n="$count" 'BEGIN{printf "%.6f", s/n}')
    avg_total=$(awk -v a="$avg_cpu" -v b="$avg_reorder" -v c="$avg_h2d" -v d="$avg_gpu" 'BEGIN{printf "%.6f", a+b+c+d}')

    echo "$method,$avg_cpu,$avg_reorder,$avg_h2d,$avg_gpu,$avg_total,$count" >> "$summary_file"
    info "  Saved averages to $summary_file"
  done
done

info "All done. Summary files are in: $OUTDIR/"
