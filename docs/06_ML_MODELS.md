# Models, formulas, and outputs

## 1. Principle

Launchly does not need one model per KPI. It uses one main model and several derived engines.

| Output | Correct type |
|---|---|
| Success Chance | Calibrated classification model |
| Model Uncertainty | Ensemble dispersion or conformal method |
| Similar Products | Embeddings + k-NN |
| Saturation | Formula over k-NN density |
| Decision Risk | Composite index, not a required model |
| Profit per Sale | Financial formula |
| Suggested Price | Model sweep + constraints |
| Monthly Profit | Formula + estimated demand or units |
| Review Sentiment | NLP model |
| Audience Age | Separate aggregate estimation |

## 2. Success target

Initial proxy by subcategory:

```text
success = 1 if:
  average_rating >= subcategory_rating_median
  AND log1p(rating_number) >= subcategory_log_reviews_p60
```

### Anti-leakage rule

`average_rating`, `rating_number`, same-product review sentiment, and post-launch review velocity cannot be pre-launch features.

## 3. Pre-launch features

- Subcategory.
- Proposed price and price-fit.
- Missing-price flag.
- Title, features, and description.
- Brand or store when available before launch.
- Item form, skin type, hair type, scent, material.
- Content length and completeness.
- Semantic embedding.

## 4. Candidate models

| Order | Model | Role |
|---|---|---|
| 1 | Logistic Regression + TF-IDF | Explainable baseline |
| 2 | Random Forest | Legacy non-linear baseline (pre-0.2.0) |
| 3 | **CatBoost** | **Current product model** (`success-catboost-0.2.0`) |

The winner is selected by PR-AUC, F1, Brier, ECE, and stability, not by prior preference. Phase 1 replaced RF with CatBoost; TF-IDF features are unchanged.

## 5. Calibration

Test `sigmoid` and `isotonic` on out-of-fold predictions. Select the method with the lowest Brier and ECE without a strong ranking degradation.

## 6. Comparables and saturation

1. Generate a normalized embedding for the product.
2. Filter by subcategory.
3. Find k neighbors by cosine distance.
4. Exclude the self-neighbor when validating the catalog.
5. Compute the mean similarity.
6. Convert it into a percentile inside the subcategory.

## 7. Risk Index

Initial version:

```text
risk = 100 x [0.5 x (1 - p_cal) + 0.3 x saturation + 0.2 x uncertainty]
```

- The three components are normalized to [0,1].
- The weights are versioned and validated.
- Show the breakdown.
- Never call it a "probability of failure."

## 8. Profit per Sale

```text
profit_per_sale =
  selling_price
  - unit_cost
  - fulfilment_cost
  - marketplace_fee
  - payment_fee
  - advertising_cost_per_unit
  - expected_return_cost
```

The current HTML does not ask for all these fields. The production form must include costs or clearly state defaults.

## 9. Expected Monthly Profit

```text
expected_monthly_profit = profit_per_sale x expected_monthly_units
```

`expected_monthly_units` requires one of these alternatives:

- A temporal model using reviews or sales proxy.
- A configurable reviews-to-sales ratio.
- Conservative, base, and optimistic scenarios entered by the user.

For P0, explicit scenarios are recommended instead of pretending to predict sales.

## 10. Suggested Price

- Sweep prices between the subcategory p10 and p90.
- Recompute price-dependent features.
- Recompute calibrated score, risk, and profit.
- Reject prices below the minimum margin.
- Do not extrapolate to unsupported regions.

Outputs:

- Most Profit.
- Safest Bet.
- Market-consistent range.
- Current vs recommended delta.

## 11. Reviews NLP

- Sentiment: VADER baseline and offline transformer as candidate.
- Topics: BERTopic or TF-IDF/NMF by subcategory.
- Keywords: frequency weighted by polarity.
- Evidence: review count and time window.

## 12. Validation

- Isolated train/validation/test.
- Group split by `parent_asin` or family.
- Temporal split if launch date is defined.
- Stratified K-fold for experiments.
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, Brier, log loss, ECE.
- Top 10 percent and top 20 percent lift vs base rate.
- Target sensitivity.
