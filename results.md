.venv) PS D:\Projects\cindy> python model.py

============================================================
STEP 1 — DATA PREPROCESSING
============================================================

Loaded: 4,612 rows 15 columns

Missing values: none
Dropped: ID column

Encoding categorical columns:
current_occupation {'Professional': 0, 'Student': 1, 'Unemployed': 2}
first_interaction {'Mobile App': 0, 'Website': 1}
profile_completed {'High': 0, 'Low': 1, 'Medium': 2}
last_activity {'Email Activity': 0, 'Phone Activity': 1, 'Website Activity': 2}
print_media_type1 {'No': 0, 'Yes': 1}
print_media_type2 {'No': 0, 'Yes': 1}
digital_media {'No': 0, 'Yes': 1}
educational_channels {'No': 0, 'Yes': 1}
referral {'No': 0, 'Yes': 1}

Shape after preprocessing: (4612, 14)

============================================================
STEP 2 — EXPLORATORY DATA ANALYSIS
============================================================

Free (0): 3,235 (70.1%)
Paid (1): 1,377 (29.9%)

Engagement averages — free vs paid:
Metric Free Paid Diff%

---

age 45.15 48.66 +7.8%
website_visits 3.58 3.54 -1.1%
time_spent_on_website 577.42 1068.40 +85.0%
page_views_per_visit 3.03 3.03 +0.0%

First interaction conversion rates:
Mobile App 10.5% (n=2070)
Website 45.6% (n=2542)

Profile completion conversion rates:
High 41.8% (n=2264)
Low 7.5% (n=107)
Medium 18.9% (n=2241)

✓ eda_charts.png (Figure 1)

============================================================
STEP 3 — FEATURE ENGINEERING
============================================================
✓ time_x_age = time_spent_on_website × age (paper feature)
✓ log_time = log(1 + time_spent_on_website)
✓ profile_x_interaction = profile_completed × first_interaction
✓ referral_x_profile = referral × profile_completed

Total features: 17

============================================================
STEP 4 — TRAIN / TEST SPLIT
============================================================

Train : 3,689 rows (29.8% paid)
Test : 923 rows (29.9% paid)
Features : 17

============================================================
STEP 5 — MODEL DEVELOPMENT (BASE MODELS)
============================================================

Decision Tree (Base)

---

Accuracy 0.7996
Recall 0.6558
Precision 0.6679
F1 Score 0.6618
ROC-AUC 0.7583

                  precision    recall  f1-score   support

        Free (0)       0.85      0.86      0.86       647
        Paid (1)       0.67      0.66      0.66       276

        accuracy                           0.80       923
       macro avg       0.76      0.76      0.76       923
    weighted avg       0.80      0.80      0.80       923

Random Forest (Base)

---

Accuracy 0.8462
Recall 0.6993
Precision 0.7659
F1 Score 0.7311
ROC-AUC 0.9156

                  precision    recall  f1-score   support

        Free (0)       0.88      0.91      0.89       647
        Paid (1)       0.77      0.70      0.73       276

        accuracy                           0.85       923
       macro avg       0.82      0.80      0.81       923
    weighted avg       0.84      0.85      0.84       923

DT train accuracy : 0.9997
DT test accuracy : 0.7996 ← gap indicates overfitting
RF train accuracy : 0.9997
RF test accuracy : 0.8462

============================================================
STEP 6 — HYPERPARAMETER TUNING (Figure 2)
============================================================

Decision Tree best params : {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
Depth 3 Leaves 8

Random Forest best params : {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 100}

Decision Tree (Tuned)

---

Accuracy 0.8104
Recall 0.7971
Precision 0.6490
F1 Score 0.7154
ROC-AUC 0.8755

                  precision    recall  f1-score   support

        Free (0)       0.90      0.82      0.86       647
        Paid (1)       0.65      0.80      0.72       276

        accuracy                           0.81       923
       macro avg       0.78      0.81      0.79       923
    weighted avg       0.83      0.81      0.82       923

Random Forest (Tuned)

---

Accuracy 0.8700
Recall 0.7428
Precision 0.8071
F1 Score 0.7736
ROC-AUC 0.9245

                  precision    recall  f1-score   support

        Free (0)       0.89      0.92      0.91       647
        Paid (1)       0.81      0.74      0.77       276

        accuracy                           0.87       923
       macro avg       0.85      0.83      0.84       923
    weighted avg       0.87      0.87      0.87       923

Random Forest (t=0.45)

---

Accuracy 0.8722 ✓
Recall 0.8043
Precision 0.7762
F1 Score 0.7900
ROC-AUC 0.9245

                  precision    recall  f1-score   support

        Free (0)       0.92      0.90      0.91       647
        Paid (1)       0.78      0.80      0.79       276

        accuracy                           0.87       923
       macro avg       0.85      0.85      0.85       923
    weighted avg       0.87      0.87      0.87       923

============================================================
STEP 7 — MODEL EVALUATION (Figure 3)
============================================================
✓ confusion_matrices.png (Figure 3)

Final model: Random Forest (t=0.45)
Paper target: 87% recall on class 1 (paid)
Achieved : 80.4% recall on class 1

============================================================
STEP 8 — FEATURE IMPORTANCE (Figure 4)
============================================================

Rank Feature Importance

---

1 first_interaction 0.2263 ██████████████████████ ← paper top 4
2 time_spent_on_website 0.1233 ████████████ ← paper top 4
3 log_time 0.1139 ███████████ ← paper top 4
4 time_x_age 0.1058 ██████████ ← paper top 4
5 profile_completed 0.0831 ████████
6 age 0.0640 ██████
7 last_activity 0.0592 █████
8 profile_x_interaction 0.0589 █████
9 page_views_per_visit 0.0545 █████
10 current_occupation 0.0475 ████
11 website_visits 0.0306 ███
12 referral 0.0085
13 educational_channels 0.0076
14 print_media_type1 0.0050
15 digital_media 0.0048
16 print_media_type2 0.0036
17 referral_x_profile 0.0034

✓ feature_importance.png (Figure 4)

============================================================
STEP 9 — DECISION TREE VISUALISATION
============================================================
✓ decision_tree.png

============================================================
STEP 10 — LEAD PROFILE ANALYSIS
============================================================

Segment Leads Actual Conv% Avg Time(s) Avg Age

---

Low Risk 2801 2.2% 552 44.4
Medium Risk 657 43.1% 895 48.7
High Risk 1108 93.2% 1080 49.3

Behavioral averages — paid vs free:
Feature Paid Free Diff%

---

age 48.7 45.2 +7.8%
time_spent_on_website 1068.4 577.4 +85.0%
website_visits 3.5 3.6 -1.1%
page_views_per_visit 3.0 3.0 +0.0%

Most common profile of a converted lead:
current_occupation Professional (67.5%)
first_interaction Website (84.2%)
profile_completed High (68.7%)
last_activity Email Activity (50.2%)

============================================================
FINAL SUMMARY
============================================================

Model Acc Rec Pre F1 AUC

---

DT Base 0.7996 0.6558 0.6679 0.6618 0.7583
RF Base 0.8462 0.6993 0.7659 0.7311 0.9156
DT Tuned 0.8104 0.7971 0.6490 0.7154 0.8755
RF Tuned (t=0.50) 0.8700 0.7428 0.8071 0.7736 0.9245 ✓
RF Tuned (t=0.45) 0.8722 0.8043 0.7762 0.7900 0.9245 ✓

Best : RF Tuned (t=0.45) Accuracy=0.8722 AUC=0.9245

Output files:
eda_charts.png — Figure 1: distributions and box plots
confusion_matrices.png — Figure 3: confusion matrices
feature_importance.png — Figure 4: feature importance chart
decision_tree.png — decision tree diagram

(.venv) PS D:\Projects\cindy>
