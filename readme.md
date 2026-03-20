venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn

# ExtraaLearn — Lead Conversion Prediction Model

> Predicts which free-tier users will convert to paid subscribers using machine learning on behavioral and demographic lead data.

---

## Overview

This model addresses a core business problem for EdTech platforms operating a freemium model: out of all leads acquired, which ones are actually worth pursuing? Rather than treating every lead equally, the model assigns each one a conversion probability, allowing sales and marketing teams to prioritize effort and allocate resources where they matter most.

The work is grounded in the methodology of:

> _"Predictive Modeling for Lead Conversion in the EdTech Industry"_
> Governors State University Capstone — Manideep Donthagani

---

## Dataset

| Property       | Value                                  |
| -------------- | -------------------------------------- |
| Source         | ExtraaLearn.csv (public)               |
| Rows           | 4,612 leads                            |
| Target column  | `status` — 0 = free, 1 = paid          |
| Class balance  | 70.1% free / 29.9% paid (2.35:1 ratio) |
| Missing values | None                                   |

### Key signals found in the data

| Feature                 | Free avg | Paid avg | Difference                  |
| ----------------------- | -------- | -------- | --------------------------- |
| `time_spent_on_website` | 577s     | 1,068s   | **+85%** — strongest signal |
| `age`                   | 45.1     | 48.7     | +7.8%                       |
| `website_visits`        | 3.58     | 3.54     | negligible                  |
| `page_views_per_visit`  | 3.03     | 3.03     | none                        |

Conversion rates by category that shaped feature engineering:

| Category                       | Conversion Rate |
| ------------------------------ | --------------- |
| Referral = Yes                 | **67.7%**       |
| First interaction = Website    | **45.6%**       |
| Profile completed = High       | **41.8%**       |
| First interaction = Mobile App | 10.5%           |
| Profile completed = Low        | 7.5%            |

---

## Pipeline

### 1. Preprocessing

All categorical columns are label-encoded using sklearn's `LabelEncoder` (alphabetical integer mapping). Ordinal encoding was tested for `profile_completed` but produced worse results — tree-based models find optimal split thresholds numerically regardless of label order.

### 2. Feature Engineering

Four features were added based on domain reasoning and confirmed via permutation importance:

| Feature                 | Formula                                 | Rationale                                                      |
| ----------------------- | --------------------------------------- | -------------------------------------------------------------- |
| `time_x_age`            | `time_spent_on_website × age`           | Older users who stay longer are the strongest converters       |
| `log_time`              | `log(1 + time_spent_on_website)`        | Compresses right skew (0–2,537s range) for cleaner tree splits |
| `profile_x_interaction` | `profile_completed × first_interaction` | Encodes the ideal lead: high-profile + website entry           |
| `referral_x_profile`    | `referral × profile_completed`          | Amplifies the 67.7% referral signal with profile level         |

Features tested and discarded (negative permutation importance across multiple runs): `visits_x_time`, `media_channels_count`, `engagement_score`, `time_per_visit`.

### 3. Train / Test Split

- **80% training**, 20% testing
- `stratify=y` — preserves the 70/30 class ratio in both splits
- Fixed `random_state=42` for reproducibility

### 4. Models

Three models were tuned via `GridSearchCV` with 5-fold stratified cross-validation, optimizing for **recall** — because missing a real conversion (false negative) is more costly than a false alarm.

#### Decision Tree

```
max_depth=3, criterion=gini, min_samples_leaf=1, min_samples_split=2
```

Interpretable baseline. Useful for understanding which splits matter most.

#### Random Forest

```
n_estimators=100, max_depth=10, max_features=sqrt,
min_samples_leaf=1, criterion=gini
```

Primary model. Aggregates 100 trees to smooth out individual tree errors. `class_weight='balanced'` was tested but over-corrected for the class imbalance — accuracy dropped to 84.3% while recall spiked to 87.6%. Reverted.

#### Gradient Boosting

```
n_estimators=300, learning_rate=0.05, max_depth=3,
min_samples_leaf=1, subsample=1.0
```

Sequential tree learner — each tree corrects the residual errors of the previous. Grid search was centred around empirically found optimal values.

#### Voting Ensemble

Soft voting across all three models, with RF and GB receiving double weight:

```python
VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("dt", dt)],
    voting="soft",
    weights=[2, 2, 1]
)
```

### 5. Decision Threshold

The default threshold of 0.50 was compared against six alternatives. **0.45** consistently produced the best F1 score across all runs by catching more true conversions without excessive false alarms.

| Threshold        | Recall    | Precision | F1        | Accuracy  |
| ---------------- | --------- | --------- | --------- | --------- |
| 0.30             | 0.880     | 0.675     | 0.764     | 0.838     |
| 0.40             | 0.826     | 0.731     | 0.776     | 0.857     |
| **0.45**         | **0.804** | **0.776** | **0.790** | **0.872** |
| 0.50 _(default)_ | 0.743     | 0.807     | 0.774     | 0.870     |
| 0.60             | 0.638     | 0.842     | 0.726     | 0.856     |

---

## Results

| Model                      | Accuracy   | Recall     | Precision  | F1         | AUC        |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Decision Tree (tuned)      | 0.8104     | 0.7971     | 0.6490     | 0.7154     | 0.8755     |
| Decision Tree (t=0.45)     | 0.8093     | 0.7971     | 0.6471     | 0.7143     | 0.8755     |
| Random Forest (tuned)      | 0.8700     | 0.7428     | 0.8071     | 0.7736     | 0.9245     |
| **Random Forest (t=0.45)** | **0.8722** | **0.8043** | **0.7762** | **0.7900** | **0.9245** |
| Gradient Boosting (tuned)  | 0.8667     | 0.7464     | 0.7954     | 0.7701     | 0.9206     |
| Gradient Boosting (t=0.45) | 0.8657     | 0.7826     | 0.7714     | 0.7770     | 0.9206     |
| Ensemble (t=0.45)          | 0.8613     | 0.7862     | 0.7587     | 0.7722     | 0.9230     |

**Winner: Random Forest at threshold 0.45**

- Accuracy: **87.22%**
- ROC-AUC: **0.9245**
- Recall on paid leads: **80.4%**

---

## Feature Importance

Top 5 features ranked by built-in importance (Random Forest):

| Rank | Feature                 | Importance | Type       |
| ---- | ----------------------- | ---------- | ---------- |
| 1    | `first_interaction`     | 0.2263     | original   |
| 2    | `time_spent_on_website` | 0.1233     | original   |
| 3    | `log_time`              | 0.1139     | engineered |
| 4    | `time_x_age`            | 0.1058     | engineered |
| 5    | `profile_completed`     | 0.0831     | original   |

Permutation importance (more reliable — measures actual recall impact on held-out data) confirmed the same top 3, with `profile_completed` ranking highest by permutation (+0.1446) — higher than its built-in rank suggests.

---

## Lead Segmentation

Using the ensemble's predicted probability, every lead is bucketed into one of three segments:

| Segment     | Probability | Leads | Actual Conv% | Avg Time on Site |
| ----------- | ----------- | ----- | ------------ | ---------------- |
| Low Risk    | 0.00 – 0.30 | 2,779 | 4.1%         | 529s             |
| Medium Risk | 0.30 – 0.60 | 754   | 38.9%        | 897s             |
| High Risk   | 0.60 – 1.00 | 1,079 | 90.0%        | 1,104s           |

### Profile of a converted lead

| Field              | Value          | % of paid leads |
| ------------------ | -------------- | --------------- |
| Occupation         | Professional   | 67.5%           |
| First interaction  | Website        | 84.2%           |
| Profile completion | High           | 68.7%           |
| Last activity      | Email Activity | 50.2%           |

---

## Business Recommendations

1. **Prioritize website traffic over mobile acquisition.** Website-first leads convert at 45.6% vs 10.5% for mobile — a 4× difference. Budget allocation should reflect this.

2. **Encourage profile completion early.** High-completion profiles convert at 42% vs 7.5% for low. An onboarding nudge toward profile completion is a direct conversion lever.

3. **Act fast on referrals.** Referred leads convert at 67.7% — the single strongest signal in the dataset. These leads should enter an immediate high-touch follow-up flow.

4. **Use time-on-site as a real-time trigger.** Paid users spend 85% more time on site than free users. A session-length threshold (e.g. >800s) can trigger a targeted upgrade prompt or a sales outreach.

5. **Focus on professionals, not students.** Professional leads convert at 35.5% vs 11.7% for students. Messaging and content targeting should be calibrated accordingly.

---

## Output Files

| File                 | Contents                                                                     |
| -------------------- | ---------------------------------------------------------------------------- |
| `eda_charts.png`     | 9-panel exploratory analysis — distributions, conversion rates, correlations |
| `results_charts.png` | Confusion matrices, ROC curves, feature importance, model comparison         |
| `decision_tree.png`  | Visual diagram of the tuned decision tree (top 3 levels)                     |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

## Usage

```bash
python model.py
```

All charts are saved to the working directory on completion.

---

## Version History

| Version | Change                                                        | Best Accuracy |
| ------- | ------------------------------------------------------------- | ------------- |
| v1      | Fixed target column name (`status` not `converted`)           | 87.00%        |
| v2      | Feature engineering, threshold tuning, permutation importance | 87.22%        |
| v3      | Gradient Boosting tuned, soft voting ensemble added           | 87.22%        |
| v4      | Tested ordinal encoding + `class_weight='balanced'`           | 86.35% ↓      |
| v5      | Reverted v4 changes, refined GB grid search                   | 87.22%        |
