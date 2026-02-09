"""
Stephen Curry Shot Prediction - Baseline Model
Predicts whether Curry makes or misses a shot based on location and context
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

 
# LOAD DATA

with open('data/shots.json', 'r') as f:
    data = json.load(f)

# Extract shot records
shots_data = data['resultSets'][0]['rowSet']
headers = data['resultSets'][0]['headers']

# Create DataFrame
df = pd.DataFrame(shots_data, columns=headers)

print(f"Loaded {len(df):,} shots")
print(f"Curry makes {df['SHOT_MADE_FLAG'].astype(int).mean():.1%} of shots")

 
# DATA PREPROCESSING
 

# Convert numeric columns
numeric_cols = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'PERIOD', 
                'MINUTES_REMAINING', 'SECONDS_REMAINING', 'SHOT_MADE_FLAG']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

 
# FEATURE ENGINEERING
 

print("\nEngineering features...")

# Shot angle (0-180 degrees)
df['SHOT_ANGLE'] = np.abs(np.arctan2(df['LOC_Y'], df['LOC_X']) * 180 / np.pi)

# Total time remaining in period (seconds)
df['TIME_REMAINING'] = df['MINUTES_REMAINING'] * 60 + df['SECONDS_REMAINING']

# Corner 3 indicator
df['IS_CORNER_3'] = df['SHOT_ZONE_BASIC'].str.contains('Corner', na=False).astype(int)

# 3-pointer indicator
df['IS_3PT'] = df['SHOT_TYPE'].str.contains('3PT', na=False).astype(int)

# Basic numeric features
basic_features = [
    'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_ANGLE',
    'PERIOD', 'TIME_REMAINING', 
    'IS_CORNER_3', 'IS_3PT'
]

# Categorical features to one-hot encode
categorical_features = ['SHOT_ZONE_BASIC', 'ACTION_TYPE']

# One-hot encode categorical variables
df_encoded = pd.get_dummies(
    df[basic_features + categorical_features + ['SHOT_MADE_FLAG']], 
    columns=categorical_features, 
    drop_first=True  # Avoid multicollinearity
)

print(f"Total features: {len(df_encoded.columns) - 1}")

# PREPARE TRAINING DATA

# Separate features (X) and target (y)
X = df_encoded.drop('SHOT_MADE_FLAG', axis=1)
y = df_encoded['SHOT_MADE_FLAG']

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain same make/miss ratio in both sets
)

print(f"\nTraining on {len(X_train):,} shots")
print(f"Testing on {len(X_test):,} shots")

# TRAIN MODEL
 

print("\nTraining Random Forest model...")

# Initialize Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,      # Number of decision trees
    max_depth=10,          # Maximum depth of each tree
    min_samples_split=50,  # Minimum samples to split a node
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

# Train the model
model.fit(X_train, y_train)
 
# EVALUATE MODEL

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.1%}")
print(f"Test Accuracy: {test_accuracy:.1%}")
print(f"Baseline (always predict 'make'): {y_test.mean():.1%}")
print(f"Improvement: {test_accuracy - y_test.mean():.1%}")

# Cross-validation for more robust estimate
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

# Detailed classification report
 
print("DETAILED METRICS")
 
print(classification_report(y_test, y_pred_test, target_names=['Miss', 'Make']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(f"                Predicted Miss  Predicted Make")
print(f"Actual Miss     {cm[0,0]:>14,}  {cm[0,1]:>14,}")
print(f"Actual Make     {cm[1,0]:>14,}  {cm[1,1]:>14,}")

 
# FEATURE IMPORTANCE
 

 
print("TOP 15 MOST IMPORTANT FEATURES")
 

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:40s} {row['importance']:.4f}")
 
# VISUALIZATIONS

# Set seaborn style
sns.set_style("whitegrid")

# Prepare test data for visualizations
df_test_viz = df.iloc[X_test.index].copy()
df_test_viz['predicted'] = y_pred_test

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# CONFUSION MATRIX
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Miss', 'Make'], 
            yticklabels=['Miss', 'Make'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title('Confusion Matrix - Shot Predictions', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('viz_1_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_1_confusion_matrix.png")

# FEATURE IMPORTANCE
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15).sort_values('importance')
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.savefig('viz_2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_2_feature_importance.png")

# MODEL PERFORMANCE COMPARISON
plt.figure(figsize=(10, 6))
performance_data = pd.DataFrame({
    'Model': ['Baseline\n(Always Make)', 'Random Forest\n(Train)', 'Random Forest\n(Test)', '5-Fold CV'],
    'Accuracy': [y_test.mean(), train_accuracy, test_accuracy, cv_scores.mean()]
})
bars = sns.barplot(data=performance_data, x='Model', y='Accuracy', palette='Set2')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)

# Add value labels on bars
for i, bar in enumerate(bars.patches):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{performance_data.iloc[i]["Accuracy"]:.1%}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('viz_3_model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_3_model_performance.png")

# PRECISION, RECALL & F1-SCORE
plt.figure(figsize=(10, 6))
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test)
metrics_data = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
    'Score': [precision[0], recall[0], f1[0], precision[1], recall[1], f1[1]],
    'Class': ['Miss']*3 + ['Make']*3
})
sns.barplot(data=metrics_data, x='Metric', y='Score', hue='Class', palette='Set1')
plt.title('Precision, Recall & F1-Score by Class', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.legend(title='Class', loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('viz_4_precision_recall_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_4_precision_recall_f1.png")

# MAKE RATE BY DISTANCE
plt.figure(figsize=(12, 6))
distance_bins = pd.cut(df['SHOT_DISTANCE'], bins=range(0, 35, 3))
actual_by_dist = df.groupby(distance_bins)['SHOT_MADE_FLAG'].mean()
pred_by_dist = df_test_viz.groupby(pd.cut(df_test_viz['SHOT_DISTANCE'], bins=range(0, 35, 3)))['predicted'].mean()

distance_labels = [f"{i}-{i+3}" for i in range(0, 32, 3)]
distance_data = pd.DataFrame({
    'Distance (ft)': distance_labels[:len(actual_by_dist)],
    'Actual': actual_by_dist.values,
    'Predicted': pred_by_dist.values
})
distance_melted = distance_data.melt(id_vars='Distance (ft)', var_name='Type', value_name='Make Rate')
sns.lineplot(data=distance_melted, x='Distance (ft)', y='Make Rate', hue='Type', 
             marker='o', linewidth=3, markersize=8)
plt.title('Make Rate by Shot Distance', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Distance from Basket (feet)', fontsize=12)
plt.ylabel('Make Rate', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='', loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_5_make_rate_by_distance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_5_make_rate_by_distance.png")

# PERFORMANCE BY SHOT TYPE
plt.figure(figsize=(10, 6))
shot_type_data = []
for shot_type in df['SHOT_TYPE'].unique():
    mask = df_test_viz['SHOT_TYPE'] == shot_type
    shot_type_data.append({
        'Shot Type': shot_type,
        'Actual': df_test_viz[mask]['SHOT_MADE_FLAG'].mean(),
        'Predicted': df_test_viz[mask]['predicted'].mean(),
        'Count': mask.sum()
    })
shot_type_df = pd.DataFrame(shot_type_data)
shot_type_melted = shot_type_df.melt(id_vars=['Shot Type', 'Count'], 
                                      value_vars=['Actual', 'Predicted'],
                                      var_name='Type', value_name='Make Rate')
sns.barplot(data=shot_type_melted, x='Shot Type', y='Make Rate', hue='Type', palette='Set2')
plt.title('Make Rate by Shot Type', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Make Rate', fontsize=12)
plt.ylim(0, 0.7)
plt.legend(title='', loc='upper right', fontsize=11)

# Add count labels
for i, row in shot_type_df.iterrows():
    plt.text(i, 0.05, f"n={row['Count']}", ha='center', fontsize=10, 
             color='white', fontweight='bold')
plt.tight_layout()
plt.savefig('viz_6_performance_by_shot_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_6_performance_by_shot_type.png")

# PERFORMANCE BY SHOT ZONE
plt.figure(figsize=(14, 6))
zone_data = []
for zone in df['SHOT_ZONE_BASIC'].value_counts().head(6).index:
    mask = df_test_viz['SHOT_ZONE_BASIC'] == zone
    zone_data.append({
        'Zone': zone,
        'Actual': df_test_viz[mask]['SHOT_MADE_FLAG'].mean(),
        'Predicted': df_test_viz[mask]['predicted'].mean(),
        'Count': mask.sum()
    })
zone_df = pd.DataFrame(zone_data).sort_values('Actual', ascending=False)
zone_melted = zone_df.melt(id_vars=['Zone', 'Count'], 
                            value_vars=['Actual', 'Predicted'],
                            var_name='Type', value_name='Make Rate')
sns.barplot(data=zone_melted, x='Zone', y='Make Rate', hue='Type', palette='coolwarm')
plt.title('Make Rate by Shot Zone (Top 6 Zones)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Make Rate', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=45, ha='right')
plt.legend(title='', loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('viz_7_performance_by_zone.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_7_performance_by_zone.png")

# ROC CURVE
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_8_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_8_roc_curve.png")

# PRECISION-RECALL CURVE
plt.figure(figsize=(8, 6))
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)

plt.plot(recall_curve, precision_curve, color='green', lw=3, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.axhline(y=y_test.mean(), color='navy', lw=2, linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_9_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_9_precision_recall_curve.png")

# SAVE MODELS

print("SAVING MODELS")
 

import pickle
with open('curry_shot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'curry_shot_model.pkl'")

# Save feature columns for future predictions
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature list saved as 'model_features.pkl'")
