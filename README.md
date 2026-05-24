# Session 07 — Classification Models

## Objective
Train binary and multiclass classifiers on a workplace service request dataset, inspect predicted probabilities, evaluate confusion matrices, and tune a decision threshold to manage the false-positive vs false-negative trade-off.

## Dataset
Synthetic workplace service request dataset (900 rows, 18 columns). No personal data. Provided in `data/`. Two prediction targets:
- **Binary**: `sla_breach` (1 = SLA breached, 0 = not breached). Base rate: 71% positive.
- **Multiclass**: `priority_class` (Low / Standard / Urgent).

The following columns are excluded as leakage (only known after the outcome):
`resolution_days_post_outcome`, `post_resolution_status`. `request_id` is also dropped as a pure identifier.

## Models
- **Dummy classifier** — `strategy='most_frequent'` baseline
- **Logistic regression** — `max_iter=1000`, `class_weight='balanced'`
- **Decision tree** — `max_depth=4`, `class_weight='balanced'`
- **Random forest** — `n_estimators=150`, `max_depth=7`, `class_weight='balanced'`

All models trained inside a Scikit-learn `Pipeline` with a `ColumnTransformer` (`StandardScaler` on numerics, `OneHotEncoder` on categoricals), so preprocessing is fitted only on training data.

## Key results (binary SLA breach, test set n=225)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy baseline | 0.707 | 0.707 | 1.000 | 0.828 | 0.500 |
| Random forest | 0.676 | 0.747 | 0.818 | 0.781 | 0.644 |
| Logistic regression | 0.658 | 0.811 | 0.673 | 0.735 | **0.716** |
| Decision tree | 0.627 | 0.791 | 0.642 | 0.708 | 0.603 |

The dummy's "high" F1 is an imbalance artefact (it predicts breach for everything). **Logistic regression has the strongest genuine signal (ROC-AUC 0.716)** and was selected for further analysis.

### Multiclass priority (random forest, test set n=225)
Overall accuracy 0.72; macro-F1 0.72. Errors are predominantly off-by-one between adjacent classes (Low↔Standard, Standard↔Urgent); Low and Urgent are almost never confused directly.

## Threshold decision
- **Chosen threshold: 0.30** (versus default 0.50).
- At 0.30: precision 0.77, recall 0.92, F1 0.84. 146 of 159 breaches caught; 13 missed; 44 false alarms.
- At 0.50: precision 0.81, recall 0.67, F1 0.74. 52 of 159 breaches missed.
- Lowering the threshold from 0.50 to 0.30 catches **39 additional real breaches** at the cost of **19 additional false alarms** — justified because a missed SLA breach is materially more costly to the business than a precautionary human review.

Full sweep across 11 thresholds (0.20–0.80) is in `reports/threshold_tradeoff_table.csv`.

## Responsible AI note
- **Class imbalance** (71% positive) means accuracy and F1 should not be the primary decision metric; use ROC-AUC and the confusion matrix at the chosen threshold.
- **Subjective features** — `urgency_score` is human-assigned and may encode agent bias; `sentiment_score_at_submission` is text-derived and may correlate with language or channel rather than true risk.
- **Fairness slices** — precision and recall must be checked per `region`, `customer_segment`, and `channel` before deployment.
- **Leakage exclusions** — `resolution_days_post_outcome` and `post_resolution_status` are post-outcome and excluded; this is documented in the data dictionary.
- **Human review** — model output is a decision-support signal, not an automated action. Every flagged request should still be triaged by an analyst, and the threshold should be re-tuned once operational FP/FN costs are quantified.

## Repository layout
```
session_07_classification_models/
├── data/
│   └── workplace_service_classification_dataset.csv
├── notebooks/
│   └── Session_07_Practical_Activity.ipynb     # executed top to bottom
├── reports/
│   ├── threshold_tradeoff_table.csv
│   └── classification_summary.md
├── visuals/
│   ├── confusion_matrix.png                    # at threshold 0.30
│   └── probability_distribution.png
└── README.md
```

## How to reproduce
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/Session_07_Practical_Activity.ipynb
```
Run all cells top to bottom. Random state is fixed (`RANDOM_STATE = 7`) so results are reproducible.
