 """Q1. Which model would you present first to a business stakeholder, and why?

Logistic regression. Three reasons grounded in our results:

- Strongest genuine signal: ROC-AUC of 0.716 beats random forest (0.644)
  and decision tree (0.603). ROC-AUC is threshold-independent, so it
  measures real class separation, not a quirk of the chosen cut-off.
- Interpretability: coefficients translate directly to 'feature X moves
  the probability of breach by Y', easy to explain and audit.
- Professional convention: start with a baseline and logistic regression,
  only deploy something more complex if it beats LR on a decision-relevant
  metric. The random forest did not, so the extra complexity is not
  justified here.

Note: the dummy classifier's F1 of 0.83 is an artefact of the 71%-positive
class imbalance (it predicts breach for everything). It is NOT a real
model and must not be presented as one."""

answer_2_costly_error = """Q2. Which error is more costly for SLA breach: false positive or false negative?

False negative.

- A false negative = a request that WILL breach SLA is not flagged.
  The breach actually happens. Contractual penalty + customer
  dissatisfaction + potential churn. Direct, hard cost.
- A false positive = a request that would not have breached is reviewed
  by an analyst. Cost of a few minutes of human attention. Soft cost.

The costs are asymmetric in favour of catching more positives, which
means RECALL matters more than PRECISION for this problem."""

answer_3_threshold = """Q3. Which threshold would you choose, and why?

Chosen threshold = 0.30.

                       Default 0.50     Chosen 0.30
Recall (breaches caught)   0.673          0.918
Precision                  0.811          0.768
F1                         0.735          0.837
False negatives (missed)   52             13
False positives (alarms)   25             44

Compared to the default, dropping to 0.30 catches 39 additional real
breaches at the cost of 19 extra false alarms. Given the asymmetric cost
established in Q2, this is a clear net positive: we eliminate 75% of
misses while growing the review queue by ~8% of the test set. F1 is
also higher at 0.30, so this is not a recall-greedy choice - it sits
near the optimum on the balanced metric too.

The threshold should be revisited once the business owner has quantified
the actual GBP cost of a missed breach versus a false alarm, and
reviewed on a regular cadence to monitor distribution drift."""

answer_4_responsible_ai = """Q4. What fairness or deployment caveat should be checked before using
this classifier?

Four things, in priority order:

1. Subgroup performance. A global threshold of 0.30 may perform very
   differently across region, customer_segment, and channel. Before
   deployment, compute precision and recall PER subgroup - an overall
   acceptable model can still systematically under-protect a slice
   (e.g. Public-sector or rural-region customers).

2. Subjective and text-derived features. urgency_score is assigned by
   a human at submission and may encode agent bias.
   sentiment_score_at_submission is derived from request text and may
   correlate with language register, channel or region rather than true
   risk. Both should be audited before production.

3. Class imbalance reporting. The training data is 71% positive.
   Headline accuracy and F1 are misleading - internal reporting should
   lead with ROC-AUC, recall and the confusion matrix at the chosen
   threshold, not accuracy.

4. Human-in-the-loop. The model output must remain a decision-support
   signal that prioritises human review, not an automated action. Every
   flagged request should still be triaged by an analyst before any
   customer-affecting action is taken."""


    print("=" * 72)
    print(text)
    print()

print("=" * 72)
print(f"Total answers stored in `reflection_answers`: {len(reflection_answers)}")
