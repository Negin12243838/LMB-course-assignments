# mRMR hyperparameter ablation

This experiment evaluates 36 mRMR configurations on all 15 subjects, 315 original trials and 945 held-out windows, using the seed42 folds. The grid crosses feature counts5/10/15/20, redundancy penalties0.25/0.5/1, and MI neighborhood sizes3/5/7. The result archive also provides a nested parameter-selection evaluation.

Read [results and interpretation](RESULTS.md), [grid summary](results/grid_summary_percent.csv), [nested fold scores](results/nested_tuned_fold_scores.csv), and [verification](verification.json). The [original result ZIP](mrmr_hyperparameter_results.zip) preserves all predictions and selected-feature details. The [updated paper](../Q2_Report/LMB_HW1_Q2_Report.pdf) incorporates this experiment in Sections5.3-5.4 and the discussion.

The highest observed fixed setting is37.54% macro-F1; nested selection is34.56%. The best fixed setting is a development observation, not an independently selected winner. Our final spectral-temporal equal mixture remains38.91% on the same all-trial window endpoint.

## Reproduction scope

The notebook that produced this hyperparameter grid was not supplied. The earlier [fixed-top20 mRMR notebook](../Q2_Fixed_Fold_Comparison/LMB_Q2_mRMR_Ablation.ipynb) is a distinct experiment and should not be identified as the generator of these files. The inner split implementation cannot be certified from the archive. Every one of the1,710 grid fold scores was independently reconstructed from35,910 predictions; nested outer scores match their selected grid rows.

No raw EEG signals or pretrained checkpoints are included in this update. The original source archive hash is recorded in PROVENANCE.json. Uploaded results were supplied by the user; no classifier was retrained for this documentation update.
