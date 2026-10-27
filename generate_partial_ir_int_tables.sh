#!/bin/bash
# All-integer partial-information-relaxation command tables: the same paths, post-warm-up states and
# 917-period arrival streams as generate_partial_ir_tables.sh, but every decision period integer.
# The recorded bound is the MIP best bound. Experiment names get the suffix _int.
# Only writes the three tables below in the project root; it does not run experiments.
set -euo pipefail
cd "$(dirname "$0")"
src_dir=outputs/partial_ir_20261026/stage
lb=table_toy_lb_partial_ir_int.dat
perf=table_toy_perf_partial_ir_int.dat
all=table_partial_ir_int_combined.dat
test ! -e "$lb" && test ! -e "$perf" && test ! -e "$all"   # the generator appends to an existing file

for panel in occupancy:768 type1_share:768 type2_target:512 type1_cost:512; do
  e=${panel%%:*}; groups=${panel##*:}
  python generate_params.py partial-ir --integer-future \
    --source-dat "$src_dir/source_lb_${e}.dat" \
    --source-results "experiments/results/toy_lb_${e}_alp_alpcoef" \
    --name "toy_lb_${e}_partial_ir" \
    --groups "$groups" \
    --dat "$lb"
done

for cond in fixed:256 alpwarmup:256 steady:681; do
  e=${cond%%:*}; groups=${cond##*:}
  python generate_params.py partial-ir --integer-future \
    --source-dat "$src_dir/source_perf_${e}.dat" \
    --source-results "experiments/results/toy_perf_${e}" \
    --name "toy_perf_${e}_partial_ir" \
    --groups "$groups" \
    --dat "$perf"
done

for dat in "$lb" "$perf"; do
  echo "$dat: $(wc -l < "$dat") lines, $(grep -o '"partial_information_relaxation": true' "$dat" | wc -l) paths"
  test -z "$(cut -d' ' -f1 "$dat" | sort -n | uniq -d)"
done
cat "$lb" "$perf" | awk '{sub(/^[0-9]+/, NR); print}' > "$all"
