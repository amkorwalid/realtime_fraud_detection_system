# Process of building robust ML/DL models for imbalance fraud dataset

### First try

I trained xgboost as base model evaluation 

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

what  i learned the accuracy is the wrong evaluation metric for imbalanced dataset
the important ones are f1_score recall and PR-AUC

### search for threshold that increases an evaluation metric

Best threshold for Average Precision: 0.39 with Average Precision: 0.71
TN: 872685, FP: 17, FN: 127, TP: 367
Precision - Non-Fraud: 1.00, Fraud: 0.96
Recall    - Non-Fraud: 1.00, Fraud: 0.74
F1 Score  - Non-Fraud: 1.00, Fraud: 0.84
F2 Score  - Non-Fraud: 1.00, Fraud: 0.78
F3 Score  - Non-Fraud: 1.00, Fraud: 0.76
ROC AUC: 0.87
Average Precision: 0.71
Log Loss: 0.01

### compute_sample_weight

from sklearn.utils.class_weight import compute_sample_weight


sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

TN: 867170, FP: 5532, FN: 0, TP: 494
Precision - Non-Fraud: 1.00, Fraud: 0.08
Recall    - Non-Fraud: 0.99, Fraud: 1.00
F1 Score  - Non-Fraud: 1.00, Fraud: 0.15
F2 Score  - Non-Fraud: 0.99, Fraud: 0.31
F3 Score  - Non-Fraud: 0.99, Fraud: 0.47
ROC AUC: 1.00
Average Precision: 0.08
Log Loss: 0.23

After threshold search

Best threshold for F2 score: 0.98 with F2 score: 0.80
TN: 872542, FP: 160, FN: 86, TP: 408
Precision - Non-Fraud: 1.00, Fraud: 0.72
Recall    - Non-Fraud: 1.00, Fraud: 0.83
F1 Score  - Non-Fraud: 1.00, Fraud: 0.77
F2 Score  - Non-Fraud: 1.00, Fraud: 0.80
F3 Score  - Non-Fraud: 1.00, Fraud: 0.81
ROC AUC: 0.91
Average Precision: 0.59
Log Loss: 0.01

### LightGBM
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model2 = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

model2.fit(x_train, y_train, sample_weight=sample_weights)

TN: 869338, FP: 3364, FN: 1, TP: 493
Precision - Non-Fraud: 1.00, Fraud: 0.13
Recall    - Non-Fraud: 1.00, Fraud: 1.00
F1 Score  - Non-Fraud: 1.00, Fraud: 0.23
F2 Score  - Non-Fraud: 1.00, Fraud: 0.42
F3 Score  - Non-Fraud: 1.00, Fraud: 0.59
ROC AUC: 1.00
Average Precision: 0.13
Log Loss: 0.14

After threshold search

TN: 872443, FP: 259, FN: 67, TP: 427
Precision - Non-Fraud: 1.00, Fraud: 0.62
Recall    - Non-Fraud: 1.00, Fraud: 0.86
F1 Score  - Non-Fraud: 1.00, Fraud: 0.72
F2 Score  - Non-Fraud: 1.00, Fraud: 0.80
F3 Score  - Non-Fraud: 1.00, Fraud: 0.83
ROC AUC: 0.93
Average Precision: 0.54
Log Loss: 0.01

### try model ensembeling
weight.P_xgb + (1 - weight).P_lgbm
Then apply a line search to find the best weight and threshold
TN: 872185, FP: 517, FN: 26, TP: 468
Precision - Non-Fraud: 1.00, Fraud: 0.48
Recall    - Non-Fraud: 1.00, Fraud: 0.95
F1 Score  - Non-Fraud: 1.00, Fraud: 0.63
F2 Score  - Non-Fraud: 1.00, Fraud: 0.79
F3 Score  - Non-Fraud: 1.00, Fraud: 0.86
ROC AUC: 0.97
Average Precision: 0.45
Log Loss: 0.02

