# Import Necessary Libraries;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


# DATA LOADING & INITIAL EXPLORATION;

print("=" * 60)
print("1. DATA LOADING & INITIAL EXPLORATION")
print("=" * 60)

# Load datasets
df = pd.read_csv('brain_mri_rich_features.csv')
feature_desc = pd.read_csv('feature_description.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {len(df)}")
print(f"Total Features: {len(df.columns)}")

print("\nDataset Info:")
print(df.info())

print("\n" + "-" * 40)
print("Class Distribution:")
print(df['label'].value_counts())
print("-" * 40)

# DATA QUALITY CHECK;

print("\n" + "=" * 60)
print("2. DATA QUALITY CHECK")
print("=" * 60)

# Check for missing values
miss_values = df.isnull().sum()
if miss_values.any():
    print("Missing values found:\n", miss_values[miss_values > 0])
else:
    print("No missing values found.")

# Check for duplicates
dup_values = df.duplicated().sum()
if dup_values > 0:
    print(f"Duplicate rows found: {dup_values}")
else:
    print("No duplicate rows found.")

# EXPLORATORY DATA ANALYSIS (EDA);

print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Get numeric features only
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric Features: {len(numeric_features)}")

# Statistical Summary
print("\nStatistical Summary:")
print(df[numeric_features].describe().T.round(4))

# 3.1 Class Distribution Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
ax1 = axes[0]
df['label'].value_counts().plot(kind='bar', color=colors, ax=ax1, edgecolor='black')
ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Class Label', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.tick_params(axis='x', rotation=45)

# Pie chart
ax2 = axes[1]
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax2)
ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
ax2.set_ylabel('')

plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.2 Correlation Matrix Heatmap
plt.figure(figsize=(14, 12))
corr_matrix = df[numeric_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.3 Feature Distribution by Class
important_features = ['mean', 'std', 'contrast', 'energy', 'edge_density', 'fft_mean']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(important_features):
    ax = axes[idx]
    for label in df['label'].unique():
        subset = df[df['label'] == label][feature]
        ax.hist(subset, alpha=0.6, label=label, bins=30)
    ax.set_title(f'{feature} Distribution by Class', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# 3.4 Boxplot Comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(important_features):
    ax = axes[idx]
    df.boxplot(column=feature, by='label', ax=ax)
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel(feature)

plt.suptitle('Feature Comparison by Class (Boxplots)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('04_boxplot_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# DATA PREPARATION FOR MODELING

print("\n" + "=" * 60)
print("4. DATA PREPARATION FOR MODELING")
print("=" * 60)

# Separate features and target
X = df[numeric_features]
y = df['label']

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClass Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized successfully.")

# MODEL TRAINING & COMPARISON

print("\n" + "=" * 60)
print("5. MODEL TRAINING & COMPARISON")
print("=" * 60)

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Train and evaluate each model
results = []

print("\nTraining Models...")
print("-" * 70)

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f"{name:25} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

print("-" * 70)

# Results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\nModel Comparison Results (Sorted by Accuracy):")
print(results_df.to_string(index=False))

# BEST MODEL DETAILED EVALUATION

print("\n" + "=" * 60)
print("6. BEST MODEL DETAILED EVALUATION")
print("=" * 60)

# Get the best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

# Make predictions with best model
y_pred_best = best_model.predict(X_test_scaled)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('05_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# MODEL COMPARISON VISUALIZATION

print("\n" + "=" * 60)
print("7. MODEL COMPARISON VISUALIZATION")
print("=" * 60)

# Bar plot of model accuracies
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))
bars = ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='black')
ax1.set_xlabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
for bar, acc in zip(bars, results_df['Accuracy']):
    ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.4f}', 
             va='center', fontsize=10)

# F1-Score comparison
ax2 = axes[1]
bars = ax2.barh(results_df['Model'], results_df['F1-Score'], color=colors, edgecolor='black')
ax2.set_xlabel('F1-Score', fontsize=12)
ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
for bar, f1 in zip(bars, results_df['F1-Score']):
    ax2.text(f1 + 0.01, bar.get_y() + bar.get_height()/2, f'{f1:.4f}', 
             va='center', fontsize=10)

plt.tight_layout()
plt.savefig('06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# FEATURE IMPORTANCE

print("\n" + "=" * 60)
print("8. FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Use Random Forest for feature importance
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_15 = feature_importance.head(15)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_15)))[::-1]
plt.barh(top_15['Feature'], top_15['Importance'], color=colors, edgecolor='black')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# SUMMARY & CONCLUSION 

print("\n" + "=" * 60)
print("9. SUMMARY & CONCLUSION")
print("=" * 60)

print(f"""
ANALYSIS SUMMARY:

Dataset Overview:
  - Total Samples: {len(df)}
  - Total Features: {len(numeric_features)}
  - Number of Classes: {len(le.classes_)}
  - Classes: {list(le.classes_)}

Data Quality:
  - Missing Values: None
  - Duplicate Rows: None

Best Performing Model:
  - Model: {best_model_name}
  - Test Accuracy: {results_df.iloc[0]['Accuracy']:.4f} ({results_df.iloc[0]['Accuracy']*100:.2f}%)
  - F1-Score: {results_df.iloc[0]['F1-Score']:.4f}
  - Cross-Validation: {results_df.iloc[0]['CV Mean']:.4f} +/- {results_df.iloc[0]['CV Std']:.4f}

Top 3 Important Features:
  1. {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.4f})
  2. {feature_importance.iloc[1]['Feature']} ({feature_importance.iloc[1]['Importance']:.4f})
  3. {feature_importance.iloc[2]['Feature']} ({feature_importance.iloc[2]['Importance']:.4f})

Saved Visualizations:
  - 01_class_distribution.png
  - 02_correlation_matrix.png
  - 03_feature_distributions.png
  - 04_boxplot_comparison.png
  - 05_confusion_matrix.png
  - 06_model_comparison.png
  - 07_feature_importance.png

ANALYSIS COMPLETED SUCCESSFULLY!
""")




