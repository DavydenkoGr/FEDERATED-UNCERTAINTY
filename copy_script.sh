# --- adjust these two paths to your layout ---------------------------------
SRC_ROOT="hdd/nikita/github/uncertainty_from_prop_scores/uncertainty_from_proper_scoring_rules/external_repos/pytorch_cifar10"
DST_ROOT="./model_weights/cifar10"
# ---------------------------------------------------------------------------

mkdir -p "$DST_ROOT"          # make sure destination exists
cd "$SRC_ROOT"                # work with paths relative to SRC_ROOT

# copy every CrossEntropy/{MODEL_ID} directory and keep its parents
for d in checkpoints_*/CrossEntropy/*; do
    cp -r --parents "$d" "$DST_ROOT"
done
