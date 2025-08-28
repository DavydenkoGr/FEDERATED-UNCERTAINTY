#!/usr/bin/env bash
set -uo pipefail

# Toggle to preview commands without running: DRY_RUN=1 ./uncertainty_script.sh
DRY_RUN="${DRY_RUN:-0}"

# Datasets
DATASETS_IND=(cifar10 cifar100)
DATASETS_OOD=(cifar10 cifar100 tiny_imagenet svhn)

# Uncertainty types
U_TYPES=(Risk Mahalanobis GMM)

# Risk-specific knobs
GNAMES=(LogScore BrierScore ZeroOneScore SphericalScore)
RISK_TYPES=(TotalRisk ExcessRisk BayesRisk)
APPROXES=(outer inner central)
T_LIST=(1.0)

SUCC=0
FAILS=0
total=0

run() {
  echo ">>> $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    if ! "$@"; then
      rc=$?
      echo "!!! FAILED (exit $rc): $*" >&2
      ((FAILS++))
      return 0
    fi
  fi
  ((SUCC++))
}

for ind in "${DATASETS_IND[@]}"; do
  for ood in "${DATASETS_OOD[@]}"; do
    [[ "$ind" == "$ood" ]] && continue

    for utype in "${U_TYPES[@]}"; do
      if [[ "$utype" == "Risk" ]]; then
        for gname in "${GNAMES[@]}"; do
          for rtype in "${RISK_TYPES[@]}"; do
            for gt in "${APPROXES[@]}"; do
              for pred in "${APPROXES[@]}"; do
                for T in "${T_LIST[@]}"; do
                  print_name="Risk_${gname}_${rtype}_gt_${gt}_pred_${pred}"
                  cmd=(
                    uv run python compute_measures_1d.py
                    --ind_dataset "$ind"
                    --ood_dataset "$ood"
                    --uncertainty_measure_type "$utype"
                    --uncertainty_measure_print_name "$print_name"
                    --uncertainty_measure_gname "$gname"
                    --uncertainty_measure_risk_type "$rtype"
                    --uncertainty_measure_gt_approx "$gt"
                    --uncertainty_measure_pred_approx "$pred"
                    --uncertainty_measure_T "$T"
                  )
                  run "${cmd[@]}"
                  ((total++))
                done
              done
            done
          done
        done
      else
        # Mahalanobis / GMM: only pass the flags that apply
        print_name="$utype"
        cmd=(
          uv run python compute_measures_1d.py
          --ind_dataset "$ind"
          --ood_dataset "$ood"
          --uncertainty_measure_type "$utype"
          --uncertainty_measure_print_name "$print_name"
        )
        run "${cmd[@]}"
        ((total++))
      fi
    done
  done
done

echo "Done. Scheduled $total runs. Success: $SUCC, Failed: $FAILS"
