# Homework 1

## Question 2: ear-EEG analysis

This folder currently contains the fixed-fold sensitivity comparison requested for Question 2. It is one analysis within Homework 1, not a claim that the complete homework or its separate report is finished.

- [Notebook](Q2_Fixed_Fold_Comparison/LMB_Q2_Fixed_Fold_Comparison.ipynb)
- [Important results, methods and limitations](Q2_Fixed_Fold_Comparison/RESULTS.md)
- [Execution and output guide](Q2_Fixed_Fold_Comparison/README.md)
- [Validation record](Q2_Fixed_Fold_Comparison/local_validation.json)

All extracted supporting result tables, the figure and configuration are in `Q2_Fixed_Fold_Comparison/results`.

## mRMR hyperparameter ablation and updated paper

- [Experiment guide](Q2_mRMR_Hyperparameter_Ablation/README.md)
- [Results and interpretation](Q2_mRMR_Hyperparameter_Ablation/RESULTS.md)
- [Question 2 paper PDF](Q2_Report/LMB_HW1_Q2_Report.pdf)
- [Paper source](Q2_Report/LMB_HW1_Q2_Report.tex)

The new experiment retains all original trials and distinguishes grid exploration from nested selection. Its supplied result archive is preserved; the notebook that generated the hyperparameter grid was not supplied. The earlier fixed-top20 mRMR notebook remains in Q2_Fixed_Fold_Comparison and is a separate experiment.

- [Main standalone Question 2 notebooks](Q2_Main_Notebooks/): fixed-fold reference, wavelet comparison, revised compact features, revised MoE, and foundation-assisted experts.
