#!/usr/bin/env bash
set -uo pipefail

# Toggle to preview commands without running: DRY_RUN=1 ./uncertainty_script.sh
DRY_RUN="${DRY_RUN:-0}"

# In-distribution datasets
DATASETS_IND=(cifar10 cifar100 tiny_imagenet)

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

ood_list_for() {
  case "$1" in
    cifar10|cifar100) echo "cifar10 cifar100 tiny_imagenet svhn" ;;
    tiny_imagenet)    echo "imagenet_a imagenet_o imagenet_r tiny_imagenet" ;;
    *)                echo "" ;;
  esac
}

for ind in "${DATASETS_IND[@]}"; do
  IFS=' ' read -r -a OODS <<< "$(ood_list_for "$ind")"
  for ood in "${OODS[@]}"; do
    [[ "$ind" == "$ood" ]] && continue

    for utype in "${U_TYPES[@]}"; do
      if [[ "$utype" == "Risk" ]]; then
        for gname in "${GNAMES[@]}"; do
          for rtype in "${RISK_TYPES[@]}"; do
            for gt in "${APPROXES[@]}"; do

              if [[ "$rtype" == "BayesRisk" ]]; then
                for T in "${T_LIST[@]}"; do
                  print_name="Risk_${gname}_${rtype}_gt_${gt}"
                  cmd=(
                    uv run python scripts/compute_measures_1d.py
                    --ind_dataset "$ind"
                    --ood_dataset "$ood"
                    --uncertainty_measure_type "$utype"
                    --uncertainty_measure_print_name "$print_name"
                    --uncertainty_measure_gname "$gname"
                    --uncertainty_measure_risk_type "$rtype"
                    --uncertainty_measure_gt_approx "$gt"
                    --uncertainty_measure_T "$T"
                  )
                  run "${cmd[@]}"
                  ((total++))
                done
              else
                for pred in "${APPROXES[@]}"; do
                  for T in "${T_LIST[@]}"; do
                    print_name="Risk_${gname}_${rtype}_gt_${gt}_pred_${pred}"
                    cmd=(
                      uv run python scripts/compute_measures_1d.py
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
              fi

            done
          done
        done
      else
        print_name="$utype"
        cmd=(
          uv run python scripts/compute_measures_1d.py
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
