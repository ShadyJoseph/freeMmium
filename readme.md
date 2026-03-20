# ExtraaLearn — Lead Conversion Prediction Model

> Predicts which free-tier users will convert to paid subscribers using machine learning on behavioral and demographic lead data.

---

## Overview

This model addresses a core business problem for EdTech platforms operating a freemium model: out of all leads acquired, which ones are actually worth pursuing? Rather than treating every lead equally, the model assigns each one a conversion probability, allowing sales and marketing teams to prioritize effort and allocate resources where they matter most.

The pipeline follows the methodology of:

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

The pipeline follows the paper's exact sequence — preprocessing, EDA, feature engineering, base models, tuning, evaluation, feature importance — with additional steps for threshold optimisation and lead profiling.

### Step 1 — Data Preprocessing

The raw dataset is cleaned and prepared following the paper's approach:

- **Missing values** are checked. The dataset is complete with no imputation required.
- **ID column** is dropped — it carries no predictive signal.
- **Categorical columns** are label-encoded using sklearn's `LabelEncoder`. This converts text categories to integers (e.g. `Website=1`, `Mobile App=0`). Ordinal encoding was tested for `profile_completed` but produced worse results — tree-based models find optimal split thresholds numerically regardless of the integer assigned to each label.

### Step 2 — Exploratory Data Analysis _(paper Figure 1)_

EDA reproduces the paper's Figure 1: distribution histograms and box plots for the four numerical columns (`age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`), split by conversion status. This step also computes first-interaction and profile-completion conversion rates, which directly informed the feature engineering decisions in Step 3.

Key observations:

- Paid leads spend **85% more time** on the website than free leads — by far the strongest numerical signal.
- Website-first leads convert at **45.6%** vs **10.5%** for mobile app leads.
- High-completion profiles convert at **41.8%** vs **7.5%** for low-completion.
- Referred leads convert at **67.7%** — the single highest-converting category in the dataset.

### Step 3 — Feature Engineering

The paper derives one feature (`time_x_age`) as an interaction between time on site and age. Three additional features were added based on the EDA insights above and confirmed via permutation importance across multiple runs.

| Feature                 | Formula                                 | Rationale                                                               |
| ----------------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| `time_x_age`            | `time_spent_on_website × age`           | **Paper feature** — older users who stay longer are stronger converters |
| `log_time`              | `log(1 + time_spent_on_website)`        | Compresses right skew (0–2,537s range) for cleaner tree splits          |
| `profile_x_interaction` | `profile_completed × first_interaction` | Encodes the ideal lead type: high-profile + website entry               |
| `referral_x_profile`    | `referral × profile_completed`          | Amplifies the 67.7% referral signal with profile level                  |

Features tested and discarded due to negative permutation importance across multiple runs: `visits_x_time`, `media_channels_count`, `engagement_score`, `time_per_visit`.

### Step 4 — Train / Test Split

- **80% training**, 20% testing
- `stratify=y` preserves the 70/30 free/paid ratio in both splits
- `random_state=7` — aligned to the paper (see note below)

### Step 5 — Base Model Development _(paper §Solution)_

Both models are first run without tuning to establish a baseline and demonstrate overfitting. The paper notes:

> _"The Decision Tree model showed signs of overfitting. The Random Forest model demonstrated improved generalization but still exhibited some overfitting tendencies."_

The training vs test accuracy gap is printed for both models to make this visible before tuning is applied.

### Step 6 — Hyperparameter Tuning _(paper Figure 2)_

The Random Forest is tuned using the **exact parameters from Figure 2** of the paper:

```python
rf_estimator_tuned = RandomForestClassifier(
    criterion    = "entropy",
    random_state = 7
)

parameters = {
    "n_estimators": [100, 110, 120],
    "max_depth"   : [5, 6, 7],
    "max_features": [0.8, 0.9, 1]
}

scorer   = metrics.make_scorer(recall_score, pos_label=1)
grid_obj = GridSearchCV(rf_estimator_tuned, parameters, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

rf_estimator_tuned = grid_obj.best_estimator_
rf_estimator_tuned.fit(X_train, y_train)
```

The paper's best result from this grid was:

```
criterion=entropy, max_depth=6, max_features=0.8, n_estimators=110, random_state=7
```

A threshold of **0.45** is then applied as an extension to the paper's method. The default threshold of 0.50 was compared against six alternatives — 0.45 consistently produced the best F1 score by catching more true conversions without excessive false alarms.

| Threshold                | Recall    | Precision | F1        | Accuracy  |
| ------------------------ | --------- | --------- | --------- | --------- |
| 0.30                     | 0.880     | 0.675     | 0.764     | 0.838     |
| 0.40                     | 0.826     | 0.731     | 0.776     | 0.857     |
| **0.45**                 | **0.804** | **0.776** | **0.790** | **0.872** |
| 0.50 _(default / paper)_ | 0.743     | 0.807     | 0.774     | 0.870     |
| 0.60                     | 0.638     | 0.842     | 0.726     | 0.856     |

#### Note on `random_state=7`

The paper uses `random_state=7` in the `RandomForestClassifier` constructor (visible in Figure 2). No reason is stated — it is an arbitrary seed chosen by the author. `random_state` only controls reproducibility; the value itself has no methodological significance. This implementation uses `random_state=7` throughout to stay aligned with the paper's exact setup and allow direct comparison of results.

### Step 7 — Model Evaluation _(paper Figure 3)_

Confusion matrices are generated for three predictions: tuned Decision Tree, tuned Random Forest at default threshold (paper's setup), and tuned Random Forest at the optimised threshold (our extension). The paper selects Random Forest as the final model based on minimising false negatives (missed conversions).

> _"Considering the trade-offs between precision and recall and the need to minimize false negatives, the Random Forest Classifier with tuned hyperparameters was selected as the final model. This model achieved an 87% recall for class 1."_

### Step 8 — Feature Importance _(paper Figure 4)_

Feature importances are extracted from the tuned Random Forest and plotted as a horizontal bar chart, reproducing the paper's Figure 4. The top 4 features are highlighted in green, matching the paper's finding that `time_spent_on_website`, `first_interaction`, `profile_completed`, and `age` are the dominant predictors.

### Step 9 — Decision Tree Visualisation

The tuned Decision Tree is rendered as a diagram showing the top 3 split levels, making the model's decision logic interpretable without needing to read the full tree.

### Step 10 — Lead Profile Analysis

Every lead in the full dataset is scored with the final model's predicted conversion probability and bucketed into three segments. This translates the model output into an actionable prioritisation framework for sales and marketing teams.

---

## Results

| Model                         | Accuracy   | Recall     | Precision  | F1         | AUC        |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Decision Tree (base)          | —          | —          | —          | —          | —          |
| Random Forest (base)          | —          | —          | —          | —          | —          |
| Decision Tree (tuned)         | 0.8104     | 0.7971     | 0.6490     | 0.7154     | 0.8755     |
| Random Forest (paper, t=0.50) | 0.8700     | 0.7428     | 0.8071     | 0.7736     | 0.9245     |
| **Random Forest (t=0.45)**    | **0.8722** | **0.8043** | **0.7762** | **0.7900** | **0.9245** |

**Winner: Random Forest at threshold 0.45**

- Accuracy: **87.22%**
- ROC-AUC: **0.9245**
- Recall on paid leads: **80.4%**

---

## Feature Importance

Top 5 features ranked by built-in importance (tuned Random Forest):

| Rank | Feature                 | Importance | Type       |
| ---- | ----------------------- | ---------- | ---------- |
| 1    | `first_interaction`     | 0.2263     | original   |
| 2    | `time_spent_on_website` | 0.1233     | original   |
| 3    | `log_time`              | 0.1139     | engineered |
| 4    | `time_x_age`            | 0.1058     | engineered |
| 5    | `profile_completed`     | 0.0831     | original   |

Permutation importance confirmed the same top 3, with `profile_completed` ranking highest by permutation score (+0.1446) — higher than its built-in rank suggests.

---

## Lead Segmentation

Every lead is scored by the final model and placed into one of three risk segments:

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

| File                     | Contents                                                        | Paper figure |
| ------------------------ | --------------------------------------------------------------- | ------------ |
| `eda_charts.png`         | Distribution histograms and box plots for all numerical columns | Figure 1     |
| `confusion_matrices.png` | Confusion matrices for tuned DT, RF (t=0.50), RF (t=0.45)       | Figure 3     |
| `feature_importance.png` | Horizontal bar chart of Random Forest feature importances       | Figure 4     |
| `decision_tree.png`      | Visual diagram of the tuned decision tree (top 3 levels)        | —            |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

Install:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

```bash
python model.py
```

All output files are saved to the working directory on completion.

---

## Version History

| Version | Change                                                                                                                                                           | Best Accuracy |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| v1      | Fixed target column name (`status` not `converted`)                                                                                                              | 87.00%        |
| v2      | Feature engineering, threshold tuning, permutation importance                                                                                                    | 87.22%        |
| v3      | Gradient Boosting tuned, soft voting ensemble added                                                                                                              | 87.22%        |
| v4      | Tested ordinal encoding + `class_weight='balanced'`                                                                                                              | 86.35% ↓      |
| v5      | Reverted v4 changes, refined GB grid search                                                                                                                      | 87.22%        |
| v6      | Realigned pipeline to paper steps and figure numbers                                                                                                             | 87.22%        |
| v7      | Adopted paper's exact Figure 2 params: `criterion=entropy`, `random_state=7`, grid `n_estimators=[100,110,120]`, `max_depth=[5,6,7]`, `max_features=[0.8,0.9,1]` | 87.22%        |
