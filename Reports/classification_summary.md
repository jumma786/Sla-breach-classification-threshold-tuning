# Session 07 — Classification Summary

## Task
Tune a binary classifier decision threshold for SLA breach prediction and explain the trade-off in business terms.

## Chosen model
**Logistic regression** trained inside a Scikit-learn pipeline (`StandardScaler` for numerics, `OneHotEncoder` for categoricals, `class_weight='balanced'`).

## Chosen threshold
**0.30** (default 0.50 was rejected — see explanation below).

## Headline metrics on the test set (n = 225)

| Metric | Default 0.50 | **Chosen 0.30** |
|---|---|---|
| Precision | 0.811 | **0.768** |
| Recall | 0.673 | **0.918** |
| F1 | 0.735 | **0.837** |
| Accuracy | 0.658 | **0.747** |
| ROC-AUC | 0.716 | 0.716 |
| True positives | 107 | **146** |
| False positives | 25 | 44 |
| True negatives | 41 | 22 |
| False negatives | 52 | **13** |

ROC-AUC is threshold-independent so it is unchanged. Lowering the threshold trades 19 extra false alarms for 39 additional caught breaches.

## Confusion matrix at chosen threshold (0.30)

|  | Predicted no breach | Predicted breach |
|---|---|---|
| **Actual no breach** | 22 | 44 (FP) |
| **Actual breach** | 13 (FN) | 146 |

## Business explanation (258 words)

The classification task is to flag service requests that are likely to breach their SLA, so the operations team can prioritise human review before the deadline is missed. We trained four binary classifiers on 900 synthetic requests, splitting 75/25 with stratification on the breach label. The dataset is imbalanced toward breach (71% positive), so accuracy and F1 against a "predict-everything" baseline are misleading — the dummy classifier achieves F1 0.83 simply by predicting breach for every case. The honest separation signal comes from ROC-AUC: logistic regression scored 0.716, random forest 0.644, decision tree 0.603. We therefore selected logistic regression for its stronger underlying discrimination, its lower variance, and the interpretability that lets us explain individual predictions to stakeholders and auditors.

At the default threshold of 0.50, the model missed 52 of 159 actual breaches (a 33% miss rate) while raising 25 false alarms. That miss rate is unacceptable for an SLA-protection workflow, because every missed breach has direct contractual and customer-experience costs, whereas a false alarm only triggers a short human review by an analyst.

We swept thresholds from 0.20 to 0.80 and selected 0.30. At this threshold the model catches 146 of 159 breaches (recall 0.92), misses only 13, and raises 44 false alarms — precision 0.77 and F1 0.84. Compared to 0.50, we catch 39 more real breaches at the cost of only 19 additional reviews. Given the asymmetric cost between FN and FP, this is a clear net positive. The threshold should be re-tuned once operational costs are quantified, and reviewed monthly to monitor drift.

## Responsible AI caveat

**Class imbalance, subjective features, and the need for fairness slices and human review.** The training data is 71% positive, so headline F1 and accuracy can be inflated by trivial strategies — ROC-AUC and the confusion matrix at the chosen threshold are the metrics that should drive decisions. Two features carry known risks: `urgency_score` is assigned subjectively at submission and may encode agent bias, and `sentiment_score_at_submission` is derived from request text and could correlate with language register, channel, or region rather than true risk. Before deployment, precision and recall must be checked separately for each `region`, `customer_segment`, and `channel`; a global threshold that performs acceptably overall can still systematically under-protect a subgroup. The chosen 0.30 threshold should be revisited once FP and FN costs are quantified by the business owner, and reviewed on a regular cadence to monitor distribution drift. Finally, the output must remain a **decision-support signal that prioritises human review**, not an automated action — every flagged request should still be triaged by an analyst.

## Files in this submission
- `data/workplace_service_classification_dataset.csv` — synthetic dataset
- `notebooks/Session_07_Practical_Activity.ipynb` — executed notebook
- `reports/threshold_tradeoff_table.csv` — 11-threshold sweep with full confusion-matrix counts
- `reports/classification_summary.md` — this document
- `visuals/confusion_matrix.png` — confusion matrix at threshold 0.30
- `visuals/probability_distribution.png` — predicted probability histogram by true class
