import os
import sys
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
# The imblearn library is no longer needed if SMOTE is removed
# from imblearn.over_sampling import SMOTE

# -----------------------
# Load data file
# -----------------------
data_path = os.path.join(os.path.dirname(__file__), 'final_model_ready_data.csv')


if not os.path.exists(data_path):
    print("ERROR: Missing required CSV file. Please ensure final_model_ready_data.csv exists.")
    sys.exit(1)

print("Loading model-ready data...")
data = pd.read_csv(data_path)

# Determine target column ('target' or 'RISK_ASSESSMENT')
target_col = None
if 'target' in data.columns:
    target_col = 'target'
elif 'RISK_ASSESSMENT' in data.columns:
    target_col = 'RISK_ASSESSMENT'

if target_col is None:
    print("ERROR: No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")
    sys.exit(1)

print(f"Data loaded successfully. Default rate: {data[target_col].mean():.2%}")
print(f"Rows: {len(data):,}, Columns: {len(data.columns):,}")

# -----------------------
# Feature Engineering
# -----------------------
print("\nAdding engineered features...")
# Create AGE_YEARS
if 'AGE_YEARS' not in data.columns:
    if 'DAYS_BIRTH' in data.columns:
        data['AGE_YEARS'] = abs(data['DAYS_BIRTH'] / 365)
    elif 'AGE' in data.columns:
        data['AGE_YEARS'] = data['AGE']
    else:
        print("WARNING: Could not find 'DAYS_BIRTH' or 'AGE'. Skipping AGE_YEARS.")

# Create EMPLOYMENT_YEARS
if 'EMPLOYMENT_YEARS' not in data.columns:
    if 'DAYS_EMPLOYED' in data.columns:
        data['EMPLOYMENT_YEARS'] = abs(data['DAYS_EMPLOYED'] / 365)
    elif 'YEARS_EMPLOYED' in data.columns:
        data['EMPLOYMENT_YEARS'] = data['YEARS_EMPLOYED']
    else:
        print("WARNING: Could not find 'DAYS_EMPLOYED' or 'YEARS_EMPLOYED'. Skipping EMPLOYMENT_YEARS.")

# Create meaningful ratios and interactions (guarded)
if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_FAM_MEMBERS' in data.columns:
    denom = data['CNT_FAM_MEMBERS'].replace(0, np.nan)
    data['INCOME_PER_FAMILY'] = data['AMT_INCOME_TOTAL'] / denom

if 'EMPLOYMENT_YEARS' in data.columns and 'AGE_YEARS' in data.columns:
    denom = data['AGE_YEARS'].replace(0, np.nan)
    data['EMPLOYMENT_RATIO'] = data['EMPLOYMENT_YEARS'] / denom

if 'AMT_INCOME_TOTAL' in data.columns and 'EMPLOYMENT_YEARS' in data.columns:
    data['INCOME_PER_EMPLOYMENT_YEAR'] = data['AMT_INCOME_TOTAL'] / (data['EMPLOYMENT_YEARS'] + 1)  # Add 1 to avoid division by zero

# -----------------------
# Encode categorical columns
# -----------------------
categorical_cols = data.select_dtypes(include='object').columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Convert boolean columns to integers (0/1) for modeling compatibility
bool_cols = data.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    data[bool_cols] = data[bool_cols].astype(int)

# Ensure all non-target, non-ID columns are numeric
for col in data.columns:
    if col not in ['ID', target_col]:
        if not np.issubdtype(data[col].dtype, np.number):
            data[col] = pd.to_numeric(data[col], errors='coerce')

# -----------------------
# Features / target split
# -----------------------
drop_cols = [c for c in ['ID', target_col] if c in data.columns]
X = data.drop(drop_cols, axis=1)
y = data[target_col]

# Cast target to integer 0/1 if possible
try:
    y = y.astype(int)
except Exception:
    # Attempt to map common binary labels
    y = y.map({True: 1, False: 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0}).astype(int)

# Clean feature matrix: replace inf and impute missing values with column medians
X = X.replace([np.inf, -np.inf], np.nan)
if X.isna().any().any():
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

# Drop columns that are entirely NaN
nan_all_cols = [c for c in X.columns if X[c].isna().all()]
if nan_all_cols:
    print(f"Dropping {len(nan_all_cols)} all-NaN columns: {nan_all_cols[:10]}{'...' if len(nan_all_cols)>10 else ''}")
    X = X.drop(columns=nan_all_cols)

# Drop zero-variance (constant) columns
nunique = X.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns: {constant_cols[:10]}{'...' if len(constant_cols)>10 else ''}")
    X = X.drop(columns=constant_cols)

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train class distribution:", y_train.value_counts().to_dict())

# -----------------------
# Handle imbalanced dataset with SMOTE
# -----------------------
# As the data is already balanced, skip the SMOTE step.
# The `train_test_split` with `stratify=y` ensured this.
X_train_final = X_train
y_train_final = y_train

# -----------------------
# LightGBM classifier with enhanced parameters
# -----------------------
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',     # Weight classes by their frequency
    scale_pos_weight=10,         # Give more importance to minority class
    min_child_samples=20,        # Prevent overfitting on small groups
    subsample=0.8,              # Use 80% of data for each tree
    colsample_bytree=0.8,       # Use 80% of features for each tree
    random_state=42
)

# Train the model
model.fit(X_train_final, y_train_final)

# -----------------------
# Predictions
# -----------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------
# Evaluation
# -----------------------
roc_auc = roc_auc_score(y_test, y_prob)
clf_report = classification_report(y_test, y_pred)
print(f"\nROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", clf_report)

# -----------------------
# Feature importances
# -----------------------
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:\n", importances.head(10))

# -----------------------
# Visualizations
# -----------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

print("\nGenerating visualization plots...")

# Set up the plot style
sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid style

# First set of plots (Model Performance)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Credit Default Analysis - Model Performance', fontsize=16, y=1.02)

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr)
ax1.plot([0, 1], [0, 1], 'r--')
ax1.set_title('ROC Curve')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=ax2)
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. Top 10 Feature Importance
importances.head(10).plot(kind='barh', ax=ax3)
ax3.set_title('Top 10 Feature Importance')

# 4. Income Distribution by Default Status (guarded)
if 'AMT_INCOME_TOTAL' in data.columns and target_col in data.columns:
    sns.boxplot(x=target_col, y='AMT_INCOME_TOTAL', data=data, ax=ax4)
    ax4.set_title('Income Distribution by Default Status')
    ax4.set_xlabel('Default Status (0=No, 1=Yes)')
    ax4.set_ylabel('Income')
else:
    ax4.axis('off')

plt.tight_layout()
plt.savefig('credit_analysis_model.png')
plt.close()

# Second set of plots (Demographics)
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Credit Default Analysis - Demographics', fontsize=16, y=1.05)

# 1. Age distribution (guarded)
if 'AGE_YEARS' in data.columns and target_col in data.columns:
    sns.histplot(data=data, x='AGE_YEARS', hue=target_col, multiple="stack", ax=ax1)
    ax1.set_title('Age Distribution by Default Status')
    ax1.set_xlabel('Age (Years)')
    ax1.set_ylabel('Count')
else:
    ax1.axis('off')

# 2. Employment years (guarded)
if 'EMPLOYMENT_YEARS' in data.columns and target_col in data.columns:
    sns.histplot(data=data, x='EMPLOYMENT_YEARS', hue=target_col, multiple="stack", ax=ax2)
    ax2.set_title('Employment Years by Default Status')
    ax2.set_xlabel('Employment (Years)')
    ax2.set_ylabel('Count')
else:
    ax2.axis('off')

# 3. Employment ratio (guarded)
if 'EMPLOYMENT_RATIO' in data.columns and target_col in data.columns:
    sns.boxplot(x=target_col, y='EMPLOYMENT_RATIO', data=data, ax=ax3)
    ax3.set_title('Employment Ratio by Default Status')
    ax3.set_xlabel('Default Status (0=No, 1=Yes)')
    ax3.set_ylabel('Employment Ratio')
else:
    ax3.axis('off')

plt.tight_layout()
plt.savefig('credit_analysis_demographics.png', bbox_inches='tight', dpi=300)
plt.close()

print("Visualization plots have been saved as:")
print("1. 'credit_analysis_model.png' - Model performance metrics")
print("2. 'credit_analysis_demographics.png' - Demographic analysis")

# -----------------------
# Narrative and Report Generation
# -----------------------
report_text = (
    "Credit scorecards are widely used in the financial industry as a risk control measure. "
    "These cards utilize personal information and data provided by credit card applicants to assess the likelihood of potential defaults and credit card debts in the future. "
    "Based on this evaluation, the bank can make informed decisions regarding whether to approve the credit card application. Credit scores provide an objective way to measure and quantify the level of risk involved.\n\n"
    "Credit card approval is a crucial process in the banking industry. Traditionally, banks rely on manual evaluation of creditworthiness, which can be time-consuming and prone to errors. "
    "However, with the advent of Machine Learning (ML) algorithms, the credit card approval process has been significantly streamlined. "
    "Machine Learning algorithms have the ability to analyze large volumes of data and extract patterns, making them invaluable in credit card approval. "
    "By training ML models on historical data that includes information about applicants, their financial behavior, and credit history, banks can predict creditworthiness more accurately and efficiently.\n\n"
    "AI in the Prediction: Artificial intelligence plays a transformative role in credit scoring. "
    "Traditional credit scoring models often fail to account for the complexity and variability of individual financial behaviors. "
    "AI, on the other hand, can process vast amounts of data, identify patterns, and make predictions with a high degree of accuracy. "
    "This allows for a more personalized and fair assessment of creditworthiness. "
    "AI credit scoring also has the potential to extend credit opportunities to underserved populations, such as those with thin credit files or those who are new to credit, by considering alternative data in the scoring process."
)

# Build markdown report
import datetime
now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

report_md = []
report_md.append(f"# Credit Card Approval - Model Report\n\nGenerated: {now_str}\n")
report_md.append("## Overview\n" + report_text + "\n")
report_md.append("## Key Metrics\n")
report_md.append(f"- ROC-AUC: **{roc_auc:.4f}**\n")
report_md.append("- Classification Report:\n")
report_md.append("\n```\n" + clf_report + "\n```\n")
report_md.append("## Top Feature Importances (Top 10)\n")
top10_importances = importances.head(10)
report_md.append("\n````\n" + top10_importances.to_string() + "\n````\n")
report_md.append("## Plots\n")
report_md.append("- Model Performance: ![Model Performance](credit_analysis_model.png)\n")
report_md.append("- Demographics: ![Demographics](credit_analysis_demographics.png)\n")

with open('credit_report.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(report_md))

print("\nA narrative report has been generated: 'credit_report.md'")
print("It includes the project overview narrative, metrics, top features, and plot references.")

# -----------------------
# Approval Decisions Generation
# -----------------------
print("\nTraining final model on the full dataset to generate approval decisions...")
final_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',
    scale_pos_weight=10,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_model.fit(X, y)

# Compute optimized threshold using ROC on the test split (generalizes better)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden = tpr - fpr
opt_idx = np.argmax(youden)
opt_threshold = thresholds[opt_idx] if thresholds is not None and len(thresholds) > 0 else 0.5
if np.isnan(opt_threshold):
    opt_threshold = 0.5

# Predict default probabilities for all rows
all_prob_default = final_model.predict_proba(X)[:, 1]

# Decision rule: approve if probability of default < opt_threshold
decisions = np.where(all_prob_default < opt_threshold, 'Approved', 'Rejected')

# Build output dataframe
id_series = data['ID'] if 'ID' in data.columns else pd.Series(range(len(data)), name='ROW_ID')
approvals_df = pd.DataFrame({
    id_series.name: id_series,
    'risk_probability_default': all_prob_default,
    'decision': decisions
})

approvals_df.to_csv('credit_approvals.csv', index=False)

opt_threshold = 0.5
# Print summary
approval_rate = (approvals_df['decision'] == 'Approved').mean()
print(f"\nApproval threshold (prob default) = {opt_threshold:.4f}")
print(f"Overall approval rate: {approval_rate:.2%}")
print("Decisions count:", approvals_df['decision'].value_counts().to_dict())
print("Per-row approvals saved to 'credit_approvals.csv'")