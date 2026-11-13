# Fixed-fold comparison: retain versus exclude suspected trials

Open `LMB_Q2_Fixed_Fold_Comparison.ipynb` in Kaggle and run it from the beginning. Internet is needed for the archive download and the optional official SingLEM checkpoint. The original data archive is preserved. Outputs go to `/kaggle/working/results` and are packaged as `/kaggle/working/Fixed_Fold_Comparison_Results.zip`.

## What changes between the two policies

- **Include:** use all original training trials.
- **Exclude:** omit subject 1 / trial 6 and subject 7 / trial 21 whenever they occur in training.
- Both policies use the **same original folds** and the **same nonsuspected validation trials**. The saved original fold assignments are embedded in the notebook; the adjacent CSV is a readable reference.
- All three overlapping windows from a trial stay together. Scaler, feature selector, oversampling and classifier are refitted using each policy's training data. Identical training sets reuse the identical fitted pipeline.
- Removal is of the whole paired-ear trial, including the unaffected ear of subject 1 / trial 6. This preserves the original two-ear input definition.
- Suspected validation trials have a separate diagnostic report. They are excluded from both policies' primary validation scores, so a change in the test set cannot masquerade as a training improvement.

## Where to read the results

Start with `results/handcrafted_focus_Ftest_SVM.csv` and, after running the optional SingLEM stage, `results/singlem_focus_Ftest_SVM.csv`.

| Column | Meaning |
| --- | --- |
| `include_common_macro_f1` | Three-fold mean with suspected training trials retained, on the common validation set |
| `exclude_common_macro_f1` | Three-fold mean with suspected training trials removed, on that same validation set |
| `delta_exclude_minus_include` | Exclude minus include; positive favors exclusion |
| `delta_percentage_points` | The same difference multiplied by 100 |
| `original_full_validation_macro_f1` | Original baseline including suspected validation examples; a reproduction check, not the paired comparator |
| `subject_has_suspected_trial` | Identifies subjects 1 and 7 |

The PNG `*_paired_Ftest_SVM_delta.png` displays the subject differences. `*_paired_subject_results.csv` contains all eight selector/model combinations. `*_paired_cohort_summary.csv` reports descriptive results for all 15 subjects and separately for the two affected subjects. The other 13 subjects must have exactly zero differences.

Inspect `fixed_fold_comparison_audit.csv` for the exact training and validation trial IDs. `*_paired_fold_scores.csv` shows individual fold scores; `*_common_validation_predictions.csv` contains the predictions used to calculate them. `*_suspected_trial_diagnostics.csv` reports accuracy on each suspected held-out trial separately. For a suspected trial's validation fold, neither policy can train on that trial, so its two predictions should agree.

## Preprocessing and interpretation

The experiment starts again from all 21 original trials per subject rather than the globally excluded dataset. It subtracts each trial/channel mean by default, adds no extra notch by default, and extracts the same 80 handcrafted features. The optional SingLEM path uses the documented notebook settings for additional bandpass, resampling and frozen 32-feature extraction. Read the configuration cell before running: physical units and acquisition reference remain unverified, and the unit setting is an assumption rather than a measurement.

The two exclusions are a fixed, manually declared sensitivity analysis. There is no automatic 100-uV deletion rule. This experiment tests whether these two training trials affect this fixed evaluation. It cannot establish that every large amplitude is an artifact, or that exclusion generally improves EEG classification. Two affected subjects and overlapping windows do not justify a population significance claim. The preprocessing policy was reviewed after earlier results, so this is exploratory; an independent evaluation is needed for a confirmatory claim.

## Local execution

Local outputs are in this folder's `results` directory. `local_validation.json`, when present, records what was actually executed and checked. The local validation runs handcrafted models for all 15 subjects and disables SingLEM; no local SingLEM scores are claimed. Kaggle's notebook configuration enables SingLEM by default, and progress messages identify each stage and completed subject.

The validation also checks exact original-baseline reproduction, matching primary validation samples, trial grouping, all saved paired scores reconstructed from predictions, and identical outcomes for subjects without exclusions. Local runtime and library versions are recorded in `local_validation.json`.

### Observed local results

The complete handcrafted run passed in 203 seconds. All 120 original subject/model baseline scores reproduced to floating-point precision. All 720 paired fold scores were checked against their saved predictions; the 13 unaffected subjects had exactly zero differences.

For F-test + SVM, on the common validation samples:

| Subject | Include | Exclude | Difference, percentage points |
| --- | ---: | ---: | ---: |
| 1 | 0.261202 | 0.284942 | +2.374 |
| 7 | 0.374954 | 0.296572 | -7.838 |
| All 15, subject mean | 0.352910 | 0.349267 | -0.364 |

Thus, the previously reported decline for subject 1 does not persist in this comparison. Subject 7 still declines. This is an effect on this trained classifier and fixed evaluation: it does not prove that subject 7's large-amplitude trial is clean or contains useful neural activity. Removing it also changes the amount and class composition of training data, the fitted scaler, selected features and classifier.
