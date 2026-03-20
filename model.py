# ExtraaLearn — Lead Conversion Prediction Model
# Pipeline follows: "Predictive Modeling for Lead Conversion in the EdTech Industry"
# Governors State University Capstone — Manideep Donthagani
# See README.md for full documentation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/danieljordan2/"
    "Predicting-Potential-Customers/main/ExtraaLearn.csv"
)


# Our extension
RANDOM_STATE = 7
CV_FOLDS = 5
TEST_SIZE = 0.20
BEST_THRESHOLD = 0.45
TARGET_ACC = 0.870

CATEGORICAL_COLS = [
    "current_occupation", "first_interaction", "profile_completed",
    "last_activity", "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral",
]

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#f8f9fa"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4


# ==============================================================
# UTILITIES
# ==============================================================

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_metrics(y_true, y_pred, y_proba, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'-' * 40}")
    for name, val in [
        ("Accuracy",  accuracy_score(y_true, y_pred)),
        ("Recall",    recall_score(y_true, y_pred)),
        ("Precision", precision_score(y_true, y_pred)),
        ("F1 Score",  f1_score(y_true, y_pred)),
        ("ROC-AUC",   roc_auc_score(y_true, y_proba)),
    ]:
        flag = "  ✓" if name == "Accuracy" and val >= TARGET_ACC else ""
        print(f"  {name:<12} {val:.4f}{flag}")
    report = classification_report(y_true, y_pred, target_names=[
                                   "Free (0)", "Paid (1)"])
    indented = "\n".join("    " + line for line in report.splitlines())
    print(f"\n{indented}")


def build_score_row(label: str, y_true, y_pred, y_proba) -> dict:
    return {
        "Model": label,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
    }


# ==============================================================
# STEP 1 — DATA PREPROCESSING
# ==============================================================
print_section("STEP 1 — DATA PREPROCESSING")

raw_df = pd.read_csv(DATASET_URL)
print(f"\n  Loaded: {raw_df.shape[0]:,} rows   {raw_df.shape[1]} columns")

missing = raw_df.isnull().sum()
print(f"  Missing values: {'none' if missing.sum() == 0 else ''}")
if missing.sum() > 0:
    print(missing[missing > 0].to_string())

df = raw_df.copy()
df.drop(columns=["ID"], inplace=True)
print("  Dropped: ID column")

print("\n  Encoding categorical columns:")
for col in df.select_dtypes("object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mapping = {k: int(v)
               for k, v in zip(le.classes_, le.transform(le.classes_))}
    print(f"  {col:<25} {mapping}")

print(f"\n  Shape after preprocessing: {df.shape}")


# ==============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ==============================================================
print_section("STEP 2 — EXPLORATORY DATA ANALYSIS")

total = len(raw_df)
paid = int(raw_df["status"].sum())
free = total - paid
overall_conv_rate = raw_df["status"].mean() * 100

print(f"\n  Free (0): {free:,}  ({free/total*100:.1f}%)")
print(f"  Paid (1): {paid:,}  ({paid/total*100:.1f}%)")
print(f"  Overall conversion rate: {overall_conv_rate:.1f}%")

# Each column has a human-readable label and a unit suffix for printing.
# Units: age=years, website_visits=visits, time=seconds, page_views=pages/visit
COL_META = {
    "age": ("Age", "yrs"),
    "website_visits": ("Website visits", "visits"),
    "time_spent_on_website": ("Time on website", "sec"),
    "page_views_per_visit": ("Page views per visit", "pages"),
}

print(f"\n  Engagement averages — free vs paid:")
print(f"  {'Metric':<28} {'Unit':<8} {'Free':>8}  {'Paid':>8}  {'Diff%':>7}  Signal")
print("  " + "-" * 72)
for col, (label, unit) in COL_META.items():
    fa = raw_df[raw_df["status"] == 0][col].mean()
    pa = raw_df[raw_df["status"] == 1][col].mean()
    diff = ((pa - fa) / fa) * 100
    signal = "STRONG ←" if abs(diff) > 30 else (
        "moderate" if abs(diff) > 10 else "weak")
    print(f"  {label:<28} {unit:<8} {fa:>8.2f}  {pa:>8.2f}  {diff:>+7.1f}%  {signal}")

# Conversion rate by first interaction channel
print(f"\n  First interaction conversion rates:")
print(f"  {'Channel':<20} {'Conv%':>7}  {'Converted':>10}  {'Total':>7}")
print("  " + "-" * 48)
fi = raw_df.groupby("first_interaction")["status"].agg(["sum", "count"])
fi["rate"] = (fi["sum"] / fi["count"] * 100).round(1)
for cat, row in fi.iterrows():
    print(f"  {cat:<20} {row['rate']:>6.1f}%  "
          f"{int(row['sum']):>10,}  {int(row['count']):>7,}")

# Conversion rate by profile completion level
print(f"\n  Profile completion conversion rates:")
print(f"  {'Level':<20} {'Conv%':>7}  {'Converted':>10}  {'Total':>7}")
print("  " + "-" * 48)
pc = raw_df.groupby("profile_completed")["status"].agg(["sum", "count"])
pc["rate"] = (pc["sum"] / pc["count"] * 100).round(1)
for cat, row in pc.iterrows():
    print(f"  {cat:<20} {row['rate']:>6.1f}%  "
          f"{int(row['sum']):>10,}  {int(row['count']):>7,}")

# Figure 1: distribution histograms + box plots (paper Figure 1)
# Human-readable axis labels and units for each chart column
CHART_META = {
    "age": ("Age", "Age (years)", "#3498db"),
    "website_visits": ("Website Visits", "Visits (count)", "#2ecc71"),
    "time_spent_on_website": ("Time on Website", "Time (seconds)", "#e74c3c"),
    "page_views_per_visit": ("Page Views per Visit", "Pages per visit", "#9b59b6"),
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Figure 1 — Distribution and Box Plots of Different Variables",
             fontsize=14, fontweight="bold")

for i, (col, (title, axis_label, color)) in enumerate(CHART_META.items()):

    # Top row: histogram of the full column (all leads)
    axes[0, i].hist(raw_df[col], bins=30, color=color,
                    alpha=0.75, edgecolor="white")
    axes[0, i].set_title(f"{title}\nDistribution",
                         fontweight="bold", fontsize=9)
    axes[0, i].set_xlabel(axis_label, fontsize=8)
    axes[0, i].set_ylabel("Number of leads", fontsize=8)

    # Bottom row: side-by-side box plots split by conversion status
    # Two lists are passed — one per group — producing two boxes
    bp = axes[1, i].boxplot(
        [raw_df[raw_df["status"] == 0][col],
         raw_df[raw_df["status"] == 1][col]],
        patch_artist=True,               # required to fill boxes with color
        labels=["Free (0)", "Paid (1)"]
    )
    bp["boxes"][0].set_facecolor("#e74c3c")   # free = red
    bp["boxes"][1].set_facecolor("#2ecc71")   # paid = green
    axes[1, i].set_title(f"{title}\nBox Plot by Status",
                         fontweight="bold", fontsize=9)
    axes[1, i].set_ylabel(axis_label, fontsize=8)

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ eda_charts.png  (Figure 1)")


# ==============================================================
# STEP 3 — FEATURE ENGINEERING
# ==============================================================
print_section("STEP 3 — FEATURE ENGINEERING")

# Paper's exact derived feature
df["time_x_age"] = df["time_spent_on_website"] * df["age"]
print("  ✓ time_x_age            (paper feature)")

# Additional features — confirmed positive permutation importance
df["log_time"] = np.log1p(df["time_spent_on_website"])
df["profile_x_interaction"] = df["profile_completed"] * df["first_interaction"]
df["referral_x_profile"] = df["referral"] * df["profile_completed"]
print("  ✓ log_time")
print("  ✓ profile_x_interaction")
print("  ✓ referral_x_profile")

print(f"\n  Total features: {df.shape[1] - 1}")


# ==============================================================
# STEP 4 — TRAIN / TEST SPLIT
# ==============================================================
print_section("STEP 4 — TRAIN / TEST SPLIT")

X = df.drop(columns=["status"])
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(
    f"\n  Train : {X_train.shape[0]:,} rows  ({y_train.mean()*100:.1f}% paid)")
print(f"  Test  : {X_test.shape[0]:,} rows  ({y_test.mean()*100:.1f}% paid)")
print(f"  Features : {X.shape[1]}")


# ==============================================================
# STEP 5 — BASE MODEL DEVELOPMENT
# ==============================================================
print_section("STEP 5 — BASE MODEL DEVELOPMENT")

dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_base.fit(X_train, y_train)
dt_base_pred = dt_base.predict(X_test)
dt_base_proba = dt_base.predict_proba(X_test)[:, 1]
print_metrics(y_test, dt_base_pred, dt_base_proba, "Decision Tree (Base)")

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
rf_base.fit(X_train, y_train)
rf_base_pred = rf_base.predict(X_test)
rf_base_proba = rf_base.predict_proba(X_test)[:, 1]
print_metrics(y_test, rf_base_pred, rf_base_proba, "Random Forest (Base)")

print(f"\n  Overfitting check:")
print(f"  DT  train={accuracy_score(y_train, dt_base.predict(X_train)):.4f}  test={accuracy_score(y_test, dt_base_pred):.4f}")
print(f"  RF  train={accuracy_score(y_train, rf_base.predict(X_train)):.4f}  test={accuracy_score(y_test, rf_base_pred):.4f}")


# ==============================================================
# STEP 6 — HYPERPARAMETER TUNING
# ==============================================================
print_section("STEP 6 — HYPERPARAMETER TUNING  (Figure 2)")

# ── Decision Tree tuning ──────────────────────────────────────
dt_param_grid = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_leaf": [1, 5, 10, 20, 30],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10],
}
dt_scorer = metrics.make_scorer(recall_score, pos_label=1)
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_param_grid,
    scoring=dt_scorer,
    cv=CV_FOLDS
)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
dt_proba = dt_best.predict_proba(X_test)[:, 1]
print(f"\n  Decision Tree best params : {dt_grid.best_params_}")

# ── Random Forest tuning — exact paper Figure 2 ──────────────
# criterion="entropy" and random_state=7 are set directly on the
# estimator, matching the paper's classifier instantiation exactly.
rf_estimator_tuned = RandomForestClassifier(
    criterion="entropy",
    random_state=RANDOM_STATE
)

parameters = {
    "n_estimators": [100, 110, 120],       # paper's exact grid
    "max_depth": [5, 6, 7],             # paper's exact grid
    "max_features": [0.8, 0.9, 1],         # paper's exact grid
}

# paper's exact scorer: recall for class 1 (paid subscribers)
scorer = metrics.make_scorer(recall_score, pos_label=1)

grid_obj = GridSearchCV(
    rf_estimator_tuned,
    parameters,
    scoring=scorer,
    cv=CV_FOLDS
)
grid_obj = grid_obj.fit(X_train, y_train)

# paper's exact final step: set classifier to best found params
rf_estimator_tuned = grid_obj.best_estimator_
rf_estimator_tuned.fit(X_train, y_train)

rf_best = rf_estimator_tuned
rf_pred = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)[:, 1]

print(f"\n  Random Forest best params : {grid_obj.best_params_}")
print(f"  (paper result: criterion=entropy, max_depth=6,")
print(f"   max_features=0.8, n_estimators=110, random_state=7)")

print_metrics(y_test, dt_pred,  dt_proba,  "Decision Tree (Tuned)")
print_metrics(y_test, rf_pred,  rf_proba,
              "Random Forest (Tuned — paper Figure 2)")

# Threshold extension (beyond paper — see README)
rf_pred_tuned = (rf_proba >= BEST_THRESHOLD).astype(int)
print_metrics(y_test, rf_pred_tuned, rf_proba,
              f"Random Forest (t={BEST_THRESHOLD} — extended)")


# ==============================================================
# STEP 7 — MODEL EVALUATION                          [paper Figure 3]
# ==============================================================
print_section("STEP 7 — MODEL EVALUATION  (Figure 3)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 3 — Confusion Matrices", fontsize=14, fontweight="bold")

for ax, pred, title, cmap in [
    (axes[0], dt_pred,       "Decision Tree (Tuned)",                 "Oranges"),
    (axes[1], rf_pred,       "Random Forest (Tuned, t=0.50)",         "Greens"),
    (axes[2], rf_pred_tuned,
     f"Random Forest (t={BEST_THRESHOLD}) ← final", "Blues"),
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["Free (0)", "Paid (1)"]
    ).plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ confusion_matrices.png  (Figure 3)")

print(f"\n  Paper target : 87% recall on class 1 (paid)")
print(f"  Achieved     : {recall_score(y_test, rf_pred_tuned):.1%} recall  |  "
      f"{accuracy_score(y_test, rf_pred_tuned):.1%} accuracy")


# ==============================================================
# STEP 8 — FEATURE IMPORTANCE                        [paper Figure 4]
# ==============================================================
print_section("STEP 8 — FEATURE IMPORTANCE  (Figure 4)")

importances = pd.Series(
    rf_best.feature_importances_, index=X.columns
).sort_values(ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<28} {'Importance':>12}")
print("  " + "-" * 50)
for rank, (feat, val) in enumerate(importances.items(), 1):
    star = "  ← paper top 4" if rank <= 4 else ""
    print(f"  {rank:<6} {feat:<28} {val:>12.4f}{star}")

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#2ecc71" if i < 4 else "#bdc3c7" for i in range(len(importances))]
importances.sort_values().plot(
    kind="barh", ax=ax,
    color=list(reversed(colors)), edgecolor="black"
)
ax.set_title("Figure 4 — Feature Importance\n(top 4 in green — match paper findings)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ feature_importance.png  (Figure 4)")


# ==============================================================
# STEP 9 — DECISION TREE VISUALISATION
# ==============================================================
print_section("STEP 9 — DECISION TREE VISUALISATION")

fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt_best,
    feature_names=X.columns,
    class_names=["Free (0)", "Paid (1)"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
)
ax.set_title(
    f"Decision Tree — Top 3 Levels  "
    f"(depth={dt_best.get_depth()}, leaves={dt_best.get_n_leaves()})",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ decision_tree.png")


# ==============================================================
# STEP 10 — LEAD PROFILE ANALYSIS
# ==============================================================
print_section("STEP 10 — LEAD PROFILE ANALYSIS")

scored_df = raw_df.copy()
scored_df["conv_probability"] = rf_best.predict_proba(X)[:, 1]
scored_df["risk_segment"] = pd.cut(
    scored_df["conv_probability"],
    bins=[0.0, 0.30, 0.60, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

print(f"\n  {'Segment':<14} {'Leads':>7}  {'Actual Conv%':>13}  {'Avg Time(s)':>12}  {'Avg Age':>8}")
print("  " + "-" * 60)
for segment in ["Low Risk", "Medium Risk", "High Risk"]:
    seg = scored_df[scored_df["risk_segment"] == segment]
    print(f"  {segment:<14} {len(seg):>7}  {seg['status'].mean()*100:>12.1f}%  "
          f"{seg['time_spent_on_website'].mean():>12.0f}  {seg['age'].mean():>8.1f}")

paid_df = scored_df[scored_df["status"] == 1]
free_df = scored_df[scored_df["status"] == 0]

print(f"\n  {'Feature':<30} {'Paid':>10}  {'Free':>10}  {'Diff%':>7}")
print("  " + "-" * 60)
for col in ["age", "time_spent_on_website", "website_visits", "page_views_per_visit"]:
    pa = paid_df[col].mean()
    fa = free_df[col].mean()
    print(f"  {col:<30} {pa:>10.1f}  {fa:>10.1f}  {((pa-fa)/fa)*100:>+7.1f}%")

print(f"\n  Most common profile of a converted lead:")
for col in ["current_occupation", "first_interaction", "profile_completed", "last_activity"]:
    top = paid_df[col].mode()[0]
    pct = (paid_df[col] == top).mean() * 100
    print(f"  {col:<28}  {top}  ({pct:.1f}%)")


# ==============================================================
# FINAL SUMMARY
# ==============================================================
print_section("FINAL SUMMARY")

all_rows = [
    build_score_row("DT Base",                    y_test,
                    dt_base_pred,  dt_base_proba),
    build_score_row("RF Base",                    y_test,
                    rf_base_pred,  rf_base_proba),
    build_score_row("DT Tuned",                   y_test,
                    dt_pred,       dt_proba),
    build_score_row("RF Tuned (paper, t=0.50)",
                    y_test, rf_pred,       rf_proba),
    build_score_row(f"RF Tuned (t={BEST_THRESHOLD})",
                    y_test, rf_pred_tuned, rf_proba),
]

print(f"\n  {'Model':<30} {'Acc':>7} {'Rec':>7} {'Pre':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-" * 63)
for row in all_rows:
    flag = "  ✓" if row["Accuracy"] >= TARGET_ACC else ""
    print(f"  {row['Model']:<30} {row['Accuracy']:>7.4f} {row['Recall']:>7.4f} "
          f"{row['Precision']:>7.4f} {row['F1']:>7.4f} {row['AUC']:>7.4f}{flag}")

best = max(all_rows, key=lambda r: r["Accuracy"])
print(f"\n  Best : {best['Model']}  "
      f"Accuracy={best['Accuracy']:.4f}  AUC={best['AUC']:.4f}")

print(f"""
  Output files:
    eda_charts.png         — Figure 1: distributions and box plots
    confusion_matrices.png — Figure 3: confusion matrices
    feature_importance.png — Figure 4: feature importance chart
    decision_tree.png      — decision tree diagram
""")
