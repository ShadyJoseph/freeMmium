# ExtraaLearn — Lead Conversion Prediction Model
# See README.md for full documentation, decisions, and results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURATION
# ==============================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/danieljordan2/"
    "Predicting-Potential-Customers/main/ExtraaLearn.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
BEST_THRESHOLD = 0.45
TARGET_ACC = 0.870

ENGINEERED_FEATURES = [
    "time_x_age",
    "log_time",
    "profile_x_interaction",
    "referral_x_profile",
]

CATEGORICAL_COLS = [
    "current_occupation", "first_interaction", "profile_completed",
    "last_activity", "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral",
]

MEDIA_COLS = [
    "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral",
]

NUMERIC_COLS = [
    "age", "website_visits",
    "time_spent_on_website", "page_views_per_visit",
]

COLOR_FREE = "#e74c3c"
COLOR_PAID = "#2ecc71"
COLOR_NEUTRAL = "#3498db"
COLOR_BOOST = "#9b59b6"

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


def run_threshold_sweep(y_true, y_proba, label: str) -> float:
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print(f"\n  [{label}]")
    print(
        f"  {'Threshold':>10} {'Recall':>8} {'Precision':>11} {'F1':>8} {'Accuracy':>10}")
    print("  " + "-" * 55)
    best_t, best_f1 = 0.50, 0.0
    for t in thresholds:
        p = (y_proba >= t).astype(int)
        rec = recall_score(y_true, p)
        prec = precision_score(y_true, p)
        f1 = f1_score(y_true, p)
        acc = accuracy_score(y_true, p)
        note = "  ← default" if t == 0.50 else ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(
            f"  {t:>10.2f} {rec:>8.4f} {prec:>11.4f} {f1:>8.4f} {acc:>10.4f}{note}")
    print(f"  → best F1 at threshold {best_t}  (F1={best_f1:.4f})")
    return best_t


# ==============================================================
# STEP 1 — LOAD
# ==============================================================
print_section("STEP 1 — LOAD")

raw_df = pd.read_csv(DATASET_URL)
print(f"\n  {raw_df.shape[0]:,} rows   {raw_df.shape[1]} columns")


# ==============================================================
# STEP 2 — DATA UNDERSTANDING
# ==============================================================
print_section("STEP 2 — DATA UNDERSTANDING")

total_leads = len(raw_df)
paid_count = int(raw_df["status"].sum())
free_count = total_leads - paid_count
overall_conv_rate = raw_df["status"].mean() * 100

print(f"\n  Free (0) : {free_count:,}  ({free_count/total_leads*100:.1f}%)")
print(f"  Paid (1) : {paid_count:,}  ({paid_count/total_leads*100:.1f}%)")

print(f"\n  {'Column':<28} {'Free':>8}  {'Paid':>8}  {'Diff%':>7}   Signal")
print("  " + "-" * 60)
for col in NUMERIC_COLS:
    free_avg = raw_df[raw_df["status"] == 0][col].mean()
    paid_avg = raw_df[raw_df["status"] == 1][col].mean()
    diff_pct = ((paid_avg - free_avg) / free_avg) * 100
    signal = "STRONG" if abs(diff_pct) > 30 else (
        "moderate" if abs(diff_pct) > 10 else "weak")
    print(
        f"  {col:<28} {free_avg:>8.2f}  {paid_avg:>8.2f}  {diff_pct:>+7.1f}%   {signal}")

print(f"\n  Conversion rate by category:")
all_category_rates = []
for col in CATEGORICAL_COLS:
    print(f"\n  {col}:")
    group = raw_df.groupby(col)["status"].agg(["sum", "count"])
    group["conv_rate"] = (group["sum"] / group["count"] * 100).round(1)
    for category, row in group.iterrows():
        bar = "█" * int(row["conv_rate"] / 5)
        print(f"    {str(category):<25} {row['conv_rate']:>5.1f}%  {bar}")
        all_category_rates.append({
            "feature": col,
            "value": category,
            "conv_rate": row["conv_rate"],
            "total": row["count"],
        })

rates_df = pd.DataFrame(all_category_rates).sort_values(
    "conv_rate", ascending=False)
print(f"\n  Top 5 highest converting:")
print(rates_df.head(5).to_string(index=False))
print(f"\n  Top 5 lowest converting:")
print(rates_df.tail(5).to_string(index=False))


# ==============================================================
# STEP 3 — EDA CHARTS
# ==============================================================
print_section("STEP 3 — EDA CHARTS")

fig_eda = plt.figure(figsize=(20, 16))
fig_eda.suptitle("ExtraaLearn — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig_eda, hspace=0.45, wspace=0.35)

ax = fig_eda.add_subplot(gs[0, 0])
bars = ax.bar(["Free (0)", "Paid (1)"], [free_count, paid_count],
              color=[COLOR_FREE, COLOR_PAID], edgecolor="black", linewidth=1.2)
ax.set_title("Target Distribution", fontweight="bold")
ax.set_ylabel("Leads")
for bar, val in zip(bars, [free_count, paid_count]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val}\n({val/total_leads*100:.1f}%)", ha="center", fontweight="bold", fontsize=10)

ax = fig_eda.add_subplot(gs[0, 1])
for status, color, label in [(0, COLOR_FREE, "Free"), (1, COLOR_PAID, "Paid")]:
    subset = raw_df[raw_df["status"] == status]
    ax.hist(subset["age"], bins=20, alpha=0.65,
            color=color, label=label, edgecolor="white")
    ax.axvline(subset["age"].mean(), color=color, linestyle="--", linewidth=2,
               label=f"{label} avg: {subset['age'].mean():.0f}")
ax.set_title("Age Distribution by Status", fontweight="bold")
ax.set_xlabel("Age")
ax.legend(fontsize=8)

ax = fig_eda.add_subplot(gs[0, 2])
violin = ax.violinplot(
    [raw_df[raw_df["status"] == 0]["time_spent_on_website"].values,
     raw_df[raw_df["status"] == 1]["time_spent_on_website"].values],
    positions=[0, 1], showmedians=True, showmeans=True
)
for body, color in zip(violin["bodies"], [COLOR_FREE, COLOR_PAID]):
    body.set_facecolor(color)
    body.set_alpha(0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Free", "Paid"])
ax.set_title("Time on Website by Status", fontweight="bold")
ax.set_ylabel("Seconds")

ax = fig_eda.add_subplot(gs[1, 0])
interact_rates = raw_df.groupby("first_interaction")["status"].mean() * 100
interact_counts = raw_df.groupby("first_interaction")["status"].count()
bars_interact = ax.bar(
    interact_rates.index, interact_rates.values,
    color=[COLOR_PAID if v >
           overall_conv_rate else COLOR_FREE for v in interact_rates],
    edgecolor="black"
)
ax.axhline(overall_conv_rate, color="black", linestyle="--", linewidth=1.5,
           label=f"Avg {overall_conv_rate:.1f}%")
ax.legend(fontsize=8)
ax.set_title("Conversion by First Interaction", fontweight="bold")
ax.set_ylabel("Conv. Rate (%)")
for bar, rate, count in zip(bars_interact, interact_rates.values, interact_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}%\n(n={count})", ha="center", fontsize=8, fontweight="bold")

ax = fig_eda.add_subplot(gs[1, 1])
level_order = [x for x in ["Low", "Medium", "High"]
               if x in raw_df["profile_completed"].values]
profile_rates = raw_df.groupby("profile_completed")["status"].mean() * 100
profile_rates = profile_rates.reindex(level_order)
profile_counts = raw_df.groupby("profile_completed")[
    "status"].count().reindex(level_order)
bars_profile = ax.bar(
    profile_rates.index, profile_rates.values,
    color=[COLOR_FREE, COLOR_NEUTRAL, COLOR_PAID][:len(
        level_order)], edgecolor="black"
)
ax.axhline(overall_conv_rate, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Conversion by Profile Completion", fontweight="bold")
ax.set_ylabel("Conv. Rate (%)")
for bar, rate, count in zip(bars_profile, profile_rates.values, profile_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}%\n(n={count})", ha="center", fontsize=8, fontweight="bold")

ax = fig_eda.add_subplot(gs[1, 2])
occ_rates = raw_df.groupby("current_occupation")["status"].mean() * 100
bars_occ = ax.bar(
    occ_rates.index, occ_rates.values,
    color=[COLOR_PAID if v > overall_conv_rate else COLOR_FREE for v in occ_rates],
    edgecolor="black"
)
ax.axhline(overall_conv_rate, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Conversion by Occupation", fontweight="bold")
ax.set_ylabel("Conv. Rate (%)")
ax.tick_params(axis="x", rotation=15)
for bar, rate in zip(bars_occ, occ_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}%", ha="center", fontsize=8, fontweight="bold")

ax = fig_eda.add_subplot(gs[2, 0])
for status, color, label in [(0, COLOR_FREE, "Free"), (1, COLOR_PAID, "Paid")]:
    subset = raw_df[raw_df["status"] == status].sample(
        min(300, sum(raw_df["status"] == status)), random_state=RANDOM_STATE
    )
    ax.scatter(subset["website_visits"], subset["time_spent_on_website"],
               alpha=0.4, color=color, label=label, s=15)
ax.set_title("Website Visits vs Time Spent", fontweight="bold")
ax.set_xlabel("Website Visits")
ax.set_ylabel("Time (seconds)")
ax.legend()

ax = fig_eda.add_subplot(gs[2, 1])
temp_df = raw_df.drop(columns=["ID"]).copy()
for col in temp_df.select_dtypes("object").columns:
    temp_df[col] = LabelEncoder().fit_transform(temp_df[col])
corr_with_status = temp_df.corr()[["status"]].drop(
    "status").sort_values("status", ascending=False)
ax.barh(
    corr_with_status.index, corr_with_status["status"],
    color=[COLOR_PAID if v > 0 else COLOR_FREE for v in corr_with_status["status"]],
    edgecolor="black", alpha=0.8
)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Feature Correlation with Conversion", fontweight="bold")
ax.set_xlabel("Pearson r")

ax = fig_eda.add_subplot(gs[2, 2])
channel_rates = {
    col: raw_df[raw_df[col] == "Yes"]["status"].mean() * 100
    for col in MEDIA_COLS if "Yes" in raw_df[col].values
}
channel_rates["no channel"] = raw_df[(
    raw_df[MEDIA_COLS] == "No").all(axis=1)]["status"].mean() * 100
ax.bar(
    range(len(channel_rates)), list(channel_rates.values()),
    color=[COLOR_PAID if v >
           overall_conv_rate else COLOR_FREE for v in channel_rates.values()],
    edgecolor="black"
)
ax.set_xticks(range(len(channel_rates)))
ax.set_xticklabels([k.replace("_", "\n")
                   for k in channel_rates.keys()], fontsize=7)
ax.axhline(overall_conv_rate, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Conversion by Marketing Channel", fontweight="bold")
ax.set_ylabel("Conv. Rate (%)")

fig_eda.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
plt.close(fig_eda)
print("  ✓ eda_charts.png")


# ==============================================================
# STEP 4 — PREPROCESSING
# ==============================================================
print_section("STEP 4 — PREPROCESSING")

model_df = raw_df.copy()
model_df.drop(columns=["ID"], inplace=True)

for col in model_df.select_dtypes("object").columns:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    mapping = {k: int(v)
               for k, v in zip(le.classes_, le.transform(le.classes_))}
    print(f"  {col:<25} {mapping}")


# ==============================================================
# STEP 5 — FEATURE ENGINEERING
# ==============================================================
print_section("STEP 5 — FEATURE ENGINEERING")

model_df["time_x_age"] = model_df["time_spent_on_website"] * model_df["age"]
model_df["log_time"] = np.log1p(model_df["time_spent_on_website"])
model_df["profile_x_interaction"] = model_df["profile_completed"] * \
    model_df["first_interaction"]
model_df["referral_x_profile"] = model_df["referral"] * \
    model_df["profile_completed"]

for feat in ENGINEERED_FEATURES:
    print(f"  ✓ {feat}")
print(f"\n  Total features: {model_df.shape[1] - 1}")


# ==============================================================
# STEP 6 — TRAIN / TEST SPLIT
# ==============================================================
print_section("STEP 6 — TRAIN / TEST SPLIT")

X = model_df.drop(columns=["status"])
y = model_df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
cv_strategy = StratifiedKFold(
    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(
    f"\n  Train : {X_train.shape[0]:,} rows  ({y_train.mean()*100:.1f}% paid)")
print(f"  Test  : {X_test.shape[0]:,} rows  ({y_test.mean()*100:.1f}% paid)")
print(f"  Features : {X.shape[1]}")


# ==============================================================
# STEP 7 — BASELINE CROSS-VALIDATION
# ==============================================================
print_section("STEP 7 — BASELINE CROSS-VALIDATION (5-fold)")

baseline_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}

print(f"\n  {'Model':<25} {'Accuracy':>10} {'Recall':>8} {'F1':>8}")
print("  " + "-" * 57)
for name, model in baseline_models.items():
    acc = cross_val_score(model, X_train, y_train,
                          cv=cv_strategy, scoring="accuracy").mean()
    rec = cross_val_score(model, X_train, y_train,
                          cv=cv_strategy, scoring="recall").mean()
    f1 = cross_val_score(model, X_train, y_train,
                         cv=cv_strategy, scoring="f1").mean()
    print(f"  {name:<25} {acc:>10.4f} {rec:>8.4f} {f1:>8.4f}")


# ==============================================================
# STEP 8 — DECISION TREE (TUNED)
# ==============================================================
print_section("STEP 8 — DECISION TREE (TUNED)")

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    {"max_depth": [3, 5, 7, 10, 15, None],
     "min_samples_leaf": [1, 5, 10, 20, 30],
     "criterion": ["gini", "entropy"],
     "min_samples_split": [2, 5, 10]},
    cv=cv_strategy, scoring="recall", n_jobs=-1
)
dt_grid_search.fit(X_train, y_train)

dt_best_model = dt_grid_search.best_estimator_
dt_pred = dt_best_model.predict(X_test)
dt_proba = dt_best_model.predict_proba(X_test)[:, 1]

print(f"\n  Best params : {dt_grid_search.best_params_}")
print(
    f"  Depth {dt_best_model.get_depth()}  Leaves {dt_best_model.get_n_leaves()}")
print_metrics(y_test, dt_pred, dt_proba, "Decision Tree (Tuned)")


# ==============================================================
# STEP 9 — RANDOM FOREST (TUNED)
# ==============================================================
print_section("STEP 9 — RANDOM FOREST (TUNED)")

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    {"n_estimators": [100, 200, 300, 500],
     "max_depth": [5, 10, 15, 20, None],
     "min_samples_leaf": [1, 3, 5, 10],
     "criterion": ["gini", "entropy"],
     "max_features": ["sqrt", "log2"]},
    cv=cv_strategy, scoring="recall", n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)

rf_best_model = rf_grid_search.best_estimator_
rf_pred = rf_best_model.predict(X_test)
rf_proba = rf_best_model.predict_proba(X_test)[:, 1]

print(f"\n  Best params : {rf_grid_search.best_params_}")
print_metrics(y_test, rf_pred, rf_proba, "Random Forest (Tuned)")


# ==============================================================
# STEP 10 — GRADIENT BOOSTING (TUNED)
# ==============================================================
print_section("STEP 10 — GRADIENT BOOSTING (TUNED)")

gb_grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    {"n_estimators": [200, 300, 400],
     "max_depth": [3, 4, 5],
     "learning_rate": [0.03, 0.05, 0.08, 0.10],
     "min_samples_leaf": [1, 3, 5],
     "subsample": [0.7, 0.8, 0.9, 1.0]},
    cv=cv_strategy, scoring="recall", n_jobs=-1
)
gb_grid_search.fit(X_train, y_train)

gb_best_model = gb_grid_search.best_estimator_
gb_pred = gb_best_model.predict(X_test)
gb_proba = gb_best_model.predict_proba(X_test)[:, 1]

print(f"\n  Best params : {gb_grid_search.best_params_}")
print_metrics(y_test, gb_pred, gb_proba, "Gradient Boosting (Tuned)")


# ==============================================================
# STEP 11 — FEATURE IMPORTANCE
# ==============================================================
print_section("STEP 11 — FEATURE IMPORTANCE")

builtin_importance = pd.Series(
    rf_best_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

perm_result = permutation_importance(
    rf_best_model, X_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE, scoring="recall"
)
perm_importance = pd.Series(
    perm_result.importances_mean, index=X.columns
).sort_values(ascending=False)

builtin_ranks = {f: i + 1 for i, f in enumerate(builtin_importance.index)}
perm_ranks = {f: i + 1 for i, f in enumerate(perm_importance.index)}

print(f"\n  {'Feature':<25} {'Built-in':>10} {'Permutation':>13} {'Δ Rank':>8}")
print("  " + "-" * 62)
for feat in builtin_importance.index:
    rc = builtin_ranks[feat] - perm_ranks[feat]
    arrow = f"↑{abs(rc)}" if rc > 0 else (f"↓{abs(rc)}" if rc < 0 else "—")
    eng = "  ←" if feat in ENGINEERED_FEATURES else ""
    warn = "  ⚠" if perm_importance[feat] < 0 else ""
    print(
        f"  {feat:<25} {builtin_importance[feat]:>10.4f} {perm_importance[feat]:>13.4f} {arrow:>8}{eng}{warn}")


# ==============================================================
# STEP 12 — THRESHOLD ANALYSIS
# ==============================================================
print_section("STEP 12 — THRESHOLD ANALYSIS")

run_threshold_sweep(y_test, rf_proba, "Random Forest")
run_threshold_sweep(y_test, gb_proba, "Gradient Boosting")
run_threshold_sweep(y_test, dt_proba, "Decision Tree")

rf_pred_tuned = (rf_proba >= BEST_THRESHOLD).astype(int)
gb_pred_tuned = (gb_proba >= BEST_THRESHOLD).astype(int)
dt_pred_tuned = (dt_proba >= BEST_THRESHOLD).astype(int)

print(f"\n  → using threshold {BEST_THRESHOLD} for final predictions")


# ==============================================================
# STEP 13 — VOTING ENSEMBLE
# ==============================================================
print_section("STEP 13 — VOTING ENSEMBLE")

voting_ensemble = VotingClassifier(
    estimators=[
        ("random_forest",     rf_best_model),
        ("gradient_boosting", gb_best_model),
        ("decision_tree",     dt_best_model),
    ],
    voting="soft",
    weights=[2, 2, 1],
)
voting_ensemble.fit(X_train, y_train)

ensemble_pred = voting_ensemble.predict(X_test)
ensemble_proba = voting_ensemble.predict_proba(X_test)[:, 1]
ensemble_pred_tuned = (ensemble_proba >= BEST_THRESHOLD).astype(int)

print_metrics(y_test, ensemble_pred,       ensemble_proba, "Ensemble (t=0.50)")
print_metrics(y_test, ensemble_pred_tuned, ensemble_proba,
              f"Ensemble (t={BEST_THRESHOLD})")


# ==============================================================
# STEP 14 — RESULTS CHARTS
# ==============================================================
print_section("STEP 14 — RESULTS CHARTS")

fig_results, axes = plt.subplots(2, 3, figsize=(20, 12))
fig_results.suptitle("ExtraaLearn — Model Results",
                     fontsize=16, fontweight="bold")

for ax, pred, title, cmap in [
    (axes[0, 0], rf_pred_tuned,
     f"Random Forest (t={BEST_THRESHOLD})",     "Greens"),
    (axes[0, 1], gb_pred_tuned,
     f"Gradient Boosting (t={BEST_THRESHOLD})", "Purples"),
    (axes[0, 2], ensemble_pred_tuned,
     f"Ensemble (t={BEST_THRESHOLD})",           "Blues"),
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["Free", "Paid"]
    ).plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, fontweight="bold")

for proba, label, color, lw in [
    (dt_proba,       "Decision Tree",     COLOR_FREE,    1.8),
    (rf_proba,       "Random Forest",     COLOR_PAID,    2.5),
    (gb_proba,       "Gradient Boosting", COLOR_BOOST,   2.5),
    (ensemble_proba, "Ensemble",          COLOR_NEUTRAL, 2.0),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1, 0].plot(fpr, tpr, linewidth=lw, color=color,
                    label=f"{label}  AUC={roc_auc_score(y_test, proba):.3f}")
axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random  0.500")
axes[1, 0].fill_between(*roc_curve(y_test, rf_proba)
                        [:2], alpha=0.07, color=COLOR_PAID)
axes[1, 0].set_title("ROC Curves", fontweight="bold")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("Recall")
axes[1, 0].legend(fontsize=8)

gb_importance = pd.Series(gb_best_model.feature_importances_, index=X.columns)
top_10_features = builtin_importance.head(10).sort_values()
top_10_gb = gb_importance[top_10_features.index].reindex(top_10_features.index)
y_pos = np.arange(len(top_10_features))
axes[1, 1].barh(y_pos - 0.2, top_10_features.values, height=0.4,
                color=COLOR_PAID,  alpha=0.85, label="Random Forest",     edgecolor="black")
axes[1, 1].barh(y_pos + 0.2, top_10_gb.values,       height=0.4,
                color=COLOR_BOOST, alpha=0.85, label="Gradient Boosting", edgecolor="black")
axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(top_10_features.index, fontsize=8)
axes[1, 1].set_title("Feature Importance: RF vs GB", fontweight="bold")
axes[1, 1].set_xlabel("Importance Score")
axes[1, 1].legend()

all_score_rows = [
    build_score_row("DT Tuned",               y_test,
                    dt_pred,            dt_proba),
    build_score_row(f"DT t={BEST_THRESHOLD}", y_test,
                    dt_pred_tuned,      dt_proba),
    build_score_row("RF Tuned",               y_test,
                    rf_pred,            rf_proba),
    build_score_row(f"RF t={BEST_THRESHOLD}", y_test,
                    rf_pred_tuned,      rf_proba),
    build_score_row("GB Tuned",               y_test,
                    gb_pred,            gb_proba),
    build_score_row(f"GB t={BEST_THRESHOLD}", y_test,
                    gb_pred_tuned,      gb_proba),
    build_score_row(f"Ensemble t={BEST_THRESHOLD}",
                    y_test, ensemble_pred_tuned, ensemble_proba),
]
score_df = pd.DataFrame(all_score_rows)
x_pos = np.arange(len(score_df))
bar_width = 0.18

for i, (metric, color) in enumerate(zip(
    ["Accuracy", "Recall", "Precision", "F1"],
    [COLOR_FREE, COLOR_PAID, COLOR_NEUTRAL, COLOR_BOOST]
)):
    axes[1, 2].bar(x_pos + i * bar_width, score_df[metric], bar_width,
                   label=metric, color=color, edgecolor="black", alpha=0.85)
axes[1, 2].set_xticks(x_pos + bar_width * 1.5)
axes[1, 2].set_xticklabels(
    score_df["Model"], rotation=15, ha="right", fontsize=7)
axes[1, 2].set_ylim(0, 1.15)
axes[1, 2].set_ylabel("Score")
axes[1, 2].set_title("All Models — Metric Comparison", fontweight="bold")
axes[1, 2].axhline(TARGET_ACC, color="black", linestyle=":", linewidth=2,
                   label=f"Target {TARGET_ACC*100:.0f}%")
axes[1, 2].legend(fontsize=7)

fig_results.tight_layout()
fig_results.savefig("results_charts.png", dpi=150, bbox_inches="tight")
plt.close(fig_results)
print("  ✓ results_charts.png")

fig_tree, ax_tree = plt.subplots(figsize=(22, 9))
plot_tree(dt_best_model, feature_names=X.columns, class_names=["Free (0)", "Paid (1)"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
ax_tree.set_title(
    f"Decision Tree — Top 3 Levels  "
    f"(depth={dt_best_model.get_depth()}, leaves={dt_best_model.get_n_leaves()})",
    fontsize=14, fontweight="bold"
)
fig_tree.tight_layout()
fig_tree.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close(fig_tree)
print("  ✓ decision_tree.png")


# ==============================================================
# STEP 15 — LEAD PROFILE ANALYSIS
# ==============================================================
print_section("STEP 15 — LEAD PROFILE ANALYSIS")

scored_df = raw_df.copy()
scored_df["conv_probability"] = voting_ensemble.predict_proba(X)[:, 1]
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

print(f"\n  {'Model':<30} {'Acc':>7} {'Rec':>7} {'Pre':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-" * 70)
for row in all_score_rows:
    flag = "  ✓" if row["Accuracy"] >= TARGET_ACC else ""
    print(f"  {row['Model']:<30} {row['Accuracy']:>7.4f} {row['Recall']:>7.4f} "
          f"{row['Precision']:>7.4f} {row['F1']:>7.4f} {row['AUC']:>7.4f}{flag}")

best_row = max(all_score_rows, key=lambda r: r["Accuracy"])
print(f"\n  Best : {best_row['Model']}  "
      f"Accuracy={best_row['Accuracy']:.4f}  AUC={best_row['AUC']:.4f}")
