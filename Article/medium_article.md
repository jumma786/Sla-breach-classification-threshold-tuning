# The Dummy Classifier Beat My Random Forest. Here's What I Learned.

### A practical guide to threshold tuning, ROC-AUC, and why F1 lies on imbalanced data

---

**TL;DR**

- On an imbalanced dataset, a dummy classifier that predicts "positive" for everything can score a higher F1 than a properly trained model. This is a feature of the metric, not a bug in the model.
- ROC-AUC is the honest measure of class separation — threshold-independent and immune to base-rate inflation.
- The default 0.50 decision threshold is a convention, not a business decision. Tuning it costs nothing and frequently changes the model's value entirely.
- On my project, dropping the threshold from 0.50 to 0.30 caught 39 additional real breaches for the cost of 19 extra false alarms — without retraining the model.

---

My dummy classifier — the one that predicts "breach" for every single service request, no learning involved — scored a higher F1 than my logistic regression and my random forest.

That's not a bug. It's the most important lesson I took from this project, and it's the lesson most classification tutorials skip past on their way to the next algorithm.

In this article, I want to walk through a binary classification project end to end: predicting whether a workplace service request will breach its SLA. Not because the dataset is special — it isn't, it's synthetic and small — but because it surfaces the kind of decisions that matter when you move a model from a notebook toward a real business workflow. Specifically: which metric to trust, which model to deploy, where to set the decision threshold, and what to put in a "responsible AI" section that isn't just box-ticking.

If you're working through your own classification project and finding that the textbook approach leaves you with no real story to tell stakeholders, this is for you.

## The business problem

A workplace service team receives customer requests through various channels. Each request has an **SLA** — a deadline by which it must be resolved. When a request breaches its SLA, the company pays: in contractual penalties, in customer goodwill, in churn risk.

The question I was trying to answer: *given everything we know about a request at the moment it arrives, can we predict whether it's going to breach — so the team can prioritise the at-risk cases before the deadline is missed?*

This is a binary classification problem. The label is `sla_breach`: 1 if the request breached, 0 if it didn't. The features include things known at submission time: urgency score, customer segment, channel, region, account age, recent case load on the team, request type, and a few others.

## The dataset (and an important problem with it)

900 rows, 18 columns. Three things mattered before I trained anything.

**First, the class balance.** 71% of requests breached. That's the opposite of what you usually imagine in fraud-style problems — the "interesting" class is actually the majority. This single fact is going to break two metrics later in the article, so hold onto it.

**Second, leakage columns.** Three columns in the dataset were known only *after* resolution: `resolution_days_post_outcome`, `post_resolution_status`, and an identifier. If I trained on these, the model would look fantastic in cross-validation and collapse in production, because at prediction time (when a new request just arrived) we don't yet have its outcome. I dropped them.

**Third, the dataset is synthetic.** I'll come back to what this means for what you can claim about the results.

## Setting up a leakage-safe pipeline

The most common mistake I see in beginner classification code is scaling the entire dataset before splitting it into train and test. That leaks test-set statistics into the scaler, and your test performance becomes optimistic in a way you can't detect by inspection.

The fix is to put preprocessing inside a scikit-learn `Pipeline`, so it gets fit on training data only:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
```

The pipeline is now one object you can `.fit()` and `.predict()` on. The `ColumnTransformer` standardises the numeric features and one-hot encodes the categoricals. Everything is fit on training data when `.fit()` is called, and the same fitted preprocessing is applied to test data.

The split itself uses stratification, to keep that 71/29 ratio consistent in train and test:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7)
```

This is dull, plumbing-level code. It's also where most real projects either succeed or quietly fail.

## Four models, one test set

I trained four classifiers on the same split:

- **Dummy classifier** (`strategy='most_frequent'`) — predicts the majority class for every input. The trivial baseline. Real models must beat this.
- **Logistic regression** — the interpretable linear baseline.
- **Decision tree** (`max_depth=4`) — visualisable as if/else rules.
- **Random forest** (150 trees, `max_depth=7`) — the "more is better" option.

All four used `class_weight='balanced'` to nudge them toward considering the minority class. Then I looked at the results.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy baseline | 0.707 | 0.707 | **1.000** | **0.828** |
| Random forest | 0.676 | 0.747 | 0.818 | 0.781 |
| Logistic regression | 0.658 | 0.811 | 0.673 | 0.735 |
| Decision tree | 0.627 | 0.791 | 0.642 | 0.708 |

**The dummy classifier has the highest F1.**

It also has the highest accuracy. It has perfect recall. The only metric it loses on is precision, and even there it ties with the base rate.

If I'd reported this table to a stakeholder and said "let's deploy the model with the highest F1," I would have deployed a constant function that flags every request as a breach. That model doesn't learn. It doesn't discriminate. It's not a model.

## Why F1 lies (and what to use instead)

Here's the mechanic. F1 is the harmonic mean of precision and recall, both computed at the **default 0.50 threshold**. The dummy classifier predicts "breach" for everything, which means:

- **Recall = 1.0** — it catches every actual breach (it catches everything).
- **Precision = 0.707** — that's just the base rate of breach in the data.

F1 = 2 × (0.707 × 1.0) / (0.707 + 1.0) = **0.828.**

The number is real. The model is fake. F1 isn't measuring discrimination — it's measuring how big the positive class is, with a slight penalty for not predicting positive enough.

The honest metric is **ROC-AUC**. It asks a different question entirely: *if I take a random positive case and a random negative case, what's the probability the model gives the positive one a higher score?* A perfect model = 1.0. A coin flip = 0.5. It's threshold-independent — it measures the underlying ability to separate classes, regardless of where you set the cut-off.

Here are the ROC-AUC values:

![ROC-AUC by model](roc_auc_comparison.png)

| Model | ROC-AUC |
|---|---|
| Dummy baseline | ~0.50 (random) |
| Decision tree | 0.603 |
| Random forest | 0.644 |
| **Logistic regression** | **0.716** |

The dummy classifier scores exactly random, as it should — it has zero discriminatory power. **Logistic regression has the strongest genuine signal**, beating random forest by a meaningful margin.

This was a surprise. Random forest is usually the "safe ensemble that wins" — and it didn't, even though it had higher F1.

## Why logistic regression beat random forest

When I dug into this, there were three reasons logistic regression won on this particular problem.

**Sample size.** I had 675 rows of training data. Random forests do their best work when there's a lot of data, ideally with strong non-linear interactions between features. With 675 rows and 150 trees of depth 7, the model has more capacity than the data can support — classic over-fitting territory. Logistic regression's lower capacity is actually a feature here: it can't memorise patterns that won't generalise.

**Hyperparameter discipline.** I used reasonable defaults for both, not optimised configurations. A properly cross-validated random forest with tuned `max_depth`, `min_samples_leaf`, and `n_estimators` might have closed the gap. If I were taking this to production I'd run the comparison — for an analytical project, the defaults are an honest representation of "what you get out of the box."

**Interpretability — and why it matters.** Logistic regression coefficients translate directly to business language: *"holding everything else constant, a one-unit increase in this feature multiplies the odds of breach by exp(β)."* You can show that to an ops manager. They can challenge it. You can defend it. Random forest is 150 trees averaged together — you can pull out feature importance, but you can't easily say which direction a feature pushes, or by how much, without bringing in SHAP and accepting another layer of complexity.

The professional rule of thumb the project taught me: **start with a baseline and logistic regression. Only deploy something more complex if it beats LR on a decision-relevant metric.** Random forest didn't, so it shouldn't be deployed — even if its F1 looked better at the default threshold.

## The real work: threshold tuning

Here's where the project shifted from "training models" to "making decisions."

A classifier doesn't really output a 0 or a 1. It outputs a probability between 0 and 1, and then a threshold (default 0.50) converts that into a class. The threshold is a convention, not a discovery. We can move it.

At the default threshold of 0.50, my logistic regression produced this confusion matrix on the test set:

```
                  Pred no breach   Pred breach
Actual no breach        41             25
Actual breach           52            107
```

The bottom-left cell is the painful one. **52 real breaches missed**, out of 159 actual breaches in the test set. A 33% miss rate. For an SLA-protection workflow, that is unacceptable — every miss is a customer with a broken commitment.

So I swept across thresholds and watched what changed:

![Precision, recall and F1 across thresholds](threshold_tradeoff_curve.png)

The pattern is the universal trade-off of binary classification:

- **Lower threshold** → flag more cases → catch more breaches (recall ↑) but more false alarms (precision ↓)
- **Higher threshold** → flag fewer cases → fewer false alarms (precision ↑) but more missed breaches (recall ↓)

A few specific points from the sweep:

| Threshold | Recall | Precision | False negatives (missed) | False positives (false alarms) |
|---|---|---|---|---|
| 0.20 | 0.975 | 0.728 | 4 | 58 |
| **0.30** | **0.918** | **0.768** | **13** | **44** |
| 0.50 (default) | 0.673 | 0.811 | 52 | 25 |
| 0.70 | 0.365 | 0.866 | 101 | 9 |

The chart makes one thing visually obvious: F1 stays roughly flat between thresholds 0.20 and 0.40. Within that plateau, **you can choose where to sit on the precision/recall trade-off without sacrificing balanced performance**.

## The threshold is a business decision, not a statistical one

Here is the bit I think gets undersold in most ML write-ups. There is no "correct" threshold you derive from the data. There is only the threshold that matches what the business actually values.

For SLA breach prediction, the cost is asymmetric:

- **False negative** = a request that *will* breach is not flagged. The breach happens. Penalty paid, customer angry. Hard cost.
- **False positive** = a request that wouldn't have breached gets reviewed by an analyst. A few minutes of human attention. Soft cost.

Under that asymmetry, recall is worth more than precision. The right threshold leans low.

I chose **0.30**. Compared to the default 0.50:

- 39 additional real breaches caught (52 misses → 13 misses)
- 19 additional false alarms (25 → 44)
- F1 actually improves (0.735 → 0.837)

That trade — 39 breach catches for 19 extra precautionary reviews — is one a real business would take in a heartbeat. The threshold should be revisited once someone has quantified the actual £ cost of a missed breach versus a false alarm, but 0.30 is defensible as a starting point.

This is the part of the project I'd put on a CV. Not "I trained a logistic regression with ROC-AUC 0.716" — that's a credential, not a story. The story is: I picked a metric that wouldn't lie, then I made a threshold decision tied to the actual cost structure of the problem, then I documented it so it could be challenged. That's analytical work. The model is just the engine.

## Responsible AI doesn't go at the end

The thing I'd most like to push back against in tutorial-style ML writing is the way "responsible AI" gets bolted on as a closing paragraph. Like a disclaimer at the bottom of a financial product. *Caveats: bias may apply, please review carefully.*

If you're going to talk about responsibility honestly, it has to be specific. Here is what I'd actually check on this project before any deployment:

**Subgroup performance.** A global threshold of 0.30 may behave very differently across `region`, `customer_segment`, and `channel`. Compute precision and recall *per subgroup* — a model that looks acceptable overall can still systematically under-protect a slice. This is the single most important pre-deployment check and it's the one most teams skip.

**Subjective features.** Two of my features carry known risk. `urgency_score` is assigned by a human at submission and may encode agent bias. `sentiment_score_at_submission` is derived from request text and may correlate with language register, channel, or region rather than true risk. Both need auditing before they go anywhere near a production decision.

**Reporting metrics.** Given the imbalance, accuracy and F1 are misleading. Internal dashboards should lead with ROC-AUC, recall, and the confusion matrix at the chosen threshold. If the dashboard reports accuracy and the accuracy is 70%, someone will eventually point at it as evidence the model is working when it's actually doing nothing useful.

**Human in the loop.** The output is a decision-support signal that prioritises which requests get a human review. It is not an automated action. Every flagged request still gets analyst eyes on it. The model adds a useful sort order; it does not replace judgement.

None of these are unique to this project. They are the same four things I'd ask about any classifier going into a customer-facing workflow.

## What I'd tell my past self

Four things, distilled from sitting in the project for a week.

**One.** Trust ROC-AUC, treat F1 with suspicion on imbalanced data. The dummy classifier example is a permanent reminder that a metric you compute at a single threshold can be fooled by base rates.

**Two.** The threshold matters more than the model choice, usually. The gap between logistic regression and random forest on AUC was ~0.07. The gap between recall at threshold 0.50 versus 0.30 was ~0.24. Tuning the threshold gave me three times the lift of switching algorithms, and it cost nothing.

**Three.** Pick the model you can defend in a room. The interpretable model that lets you explain individual predictions to stakeholders is often worth more than the slightly more accurate black box, especially in regulated or audit-heavy environments.

**Four.** Be honest about what you've shown. This dataset is synthetic. I haven't proven that logistic regression beats random forest in general — I've shown that under these conditions, with these defaults, on this dataset, it did. The transferable thing is the *workflow*: leakage-safe pipeline, ROC-AUC as the honest metric, threshold tuned to business cost, fairness checks before deployment. The numbers are particular to this project. The discipline isn't.

---

*Code, full notebook, and the threshold sweep table are on [GitHub](https://github.com/jumma786). If you found this useful, the most helpful thing is to apply it to your own classification project — try sweeping the threshold on a model you've already built and see what the trade-off looks like. You may be surprised at how much performance was sitting unused at the default 0.50.*
