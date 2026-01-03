import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV

# Reading the source CSV file into a pandas DataFrame
df = pd.read_csv('data.csv')

# Converting all column names to lowercase to ensure consistency across the project
df.columns = df.columns.str.lower()

df.info()

df.head()

# Focus on continuous numerical variables where outliers are clinically significant
numerical_cols = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium', 'time'
]

print("\n--- Outlier Detection Summary (Interquartile Range Method) ---")
outlier_summary = {}

for col in numerical_cols:
    # Calculate Quartiles and IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define Bounds for Outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter Outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)

    print(f"{col:25}: {len(outliers)} outliers found (Range: {lower_bound:.2f} to {upper_bound:.2f})")

# Generating boxplots to visually inspect the distribution and outliers
plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.ylabel('Value')

plt.tight_layout()


# ADDRESSING OUTLIERS & SKEWNESS

# We apply Log Transformation to reduce the variance of highly skewed clinical markers without losing the critical information contained in the outliers.
skewed_features = ['creatinine_phosphokinase', 'serum_creatinine']
for col in skewed_features:
    df[col] = np.log1p(df[col])

# Visualizing relationships to identify primary predictors of mortality
plt.figure(figsize=(15, 5))
sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt='.2f')
plt.title('Clinical Feature Correlation Matrix')


# We use 'stratify' to ensure the train and test sets have the same ratio of death events
X = df.drop('death_event', axis=1)
y = df['death_event']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardizing features to ensure the model converges efficiently
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. INITIALIZE MODELS
# Logistic Regression for baseline, Random Forest for non-linear patterns
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 2. TRAINING AND PREDICTION LOOP
for name, model in models.items():
    # Training the model on scaled training data
    model.fit(X_train_scaled, y_train)

    # Making predictions on the unseen test set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n--- {name} Results ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# 3. FEATURE IMPORTANCE VISUALIZATION
# Extracting feature importance from the Random Forest model to see which
# clinical markers are the most predictive of mortality.
importances = models["Random Forest"].feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)
plt.title('Clinical Feature Importance (Random Forest)')

# HYPERPARAMETER TUNING FOR RECALL OPTIMIZATION
# 1. DEFINE PARAMETER GRID
# We prioritize 'class_weight' and 'min_samples_leaf' to improve minority class detection
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample'],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2']
}

# 2. GRID SEARCH FOR OPTIMAL RECALL
# By setting scoring='recall', we prioritize finding the best sensitivity for mortality
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# 3. EVALUATING THE OPTIMIZED MODEL
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_scaled)

print(f"Best Parameters: {grid_search.best_params_}")
print("\n--- Optimized Model Performance ---")
print(classification_report(y_test, y_pred_tuned))

# 4. VISUALIZING RECALL IMPROVEMENT
# Comparison between our baseline and the recall-optimized model

baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train_scaled, y_train)

baseline_recall = recall_score(y_test, baseline_rf.predict(X_test_scaled))
tuned_recall = recall_score(y_test, y_pred_tuned)

print(f"Baseline Recall: {baseline_recall:.2f}")
print(f"Tuned Recall:    {tuned_recall:.2f}")
print(f"Best Params:     {grid_search.best_params_}")

# Visualization of the improvement
comparison_df = pd.DataFrame({
    'Model': ['Baseline RF', 'Tuned RF'],
    'Recall': [baseline_recall, tuned_recall]
})

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=comparison_df, x='Model', y='Recall', hue='Model', palette='coolwarm')
if ax.get_legend() is not None: ax.get_legend().remove()
plt.title('Recall Improvement: Identifying High-Risk Patients')

# plt.savefig('baseline_vs_tuned_recall.png')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
import joblib

# 1. GENERATE PREDICTIONS (PROBABILITIES)
# We use the tuned Random Forest to get probability scores rather than binary labels
y_scores = best_rf.predict_proba(X_test_scaled)[:, 1]

# 2. PRECISION-RECALL CURVE ANALYSIS
# This helps us visualize how changing the threshold impacts our sensitivity
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall (Sensitivity)", linewidth=2)
plt.axvline(x=0.35, color='red', linestyle=':', label='Recommended Threshold (0.35)')
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Optimizing Threshold for Clinical Sensitivity")
plt.legend()
plt.grid(True)

# 3. IMPLEMENTING A CUSTOM THRESHOLD
# By lowering the threshold to 0.35, we capture significantly more mortality cases
custom_threshold = 0.35
y_pred_custom = (y_scores >= custom_threshold).astype(int)

print(f"--- Results at Custom Threshold ({custom_threshold}) ---")
print(f"Final Recall:    {recall_score(y_test, y_pred_custom):.2f}")
print(f"Final Precision: {precision_score(y_test, y_pred_custom):.2f}")

# 4. EXPORTING THE PROJECT ASSETS
# Serializing the model and scaler for deployment
joblib.dump(best_rf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')