# multidimensional_uncertainty


To receive all results, first

1) make sure the embeddings (logits) for different datasets are downloaded and are in the resources/model_weights/{ind_dataset}

2) next, launch uncertainty_script.sh. It will save all 1d measures to the resources/results_cleaned

3) next, define interesting compositions in configs/interesting_compositions.py. They will be used for the OT-based method.

4) next, launch scripts/full_evaluation.py. It will compute all the measures on all the problems we considered.
