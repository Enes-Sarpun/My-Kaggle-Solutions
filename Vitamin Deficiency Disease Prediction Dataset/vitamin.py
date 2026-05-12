# Import Necessary Libraries;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load Dataset;
df = pd.read_csv('Vitamin.csv')
print(df.head().T)
print(df.info())
print(df.describe().T)

# Data Preprocessing;
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

duplicate_values = df.duplicated().sum()
if duplicate_values > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicate_values} duplicate rows")

print(f"\nDataset shape after preprocessing: {df.shape}")

# Exploratory Data Analysis;
age_counts = df['age'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(age_counts.index, age_counts.values, marker='o', color='red', markersize=8, linewidth=2, label='Age Count', alpha=0.7, linestyle='--')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Categorical, Numerical Feature Analysis;
categorical_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

target = 'disease_diagnosis'
y = df[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nTarget classes: {label_encoder.classes_}")

categorical_features.remove(target)
if 'symptoms_list' in categorical_features:
    categorical_features.remove('symptoms_list')

# Visualization of Vitamin D Levels;
plt.figure(figsize=(12, 6))
sns.histplot(df['serum_vitamin_d_ng_ml'], bins=30, kde=True, color='blue')
plt.title('Distribution of Serum Vitamin D Levels')
plt.xlabel('Serum Vitamin D (ng/ml)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Correlation Analysis;
plt.figure(figsize=(16, 10))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature Selection - Add categorical variables with One-Hot Encoding;
X_numerical = df[numerical_features]
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
X = pd.concat([X_numerical, X_categorical], axis=1)

print(f"\nTotal features after encoding: {X.shape[1]}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features (after encoding): {X_categorical.shape[1]}")

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# Cross-Validation
cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
print(f"\n{'='*50}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*50}")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Model Evaluation
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix Visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature Importance Visualization
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', legend=False)
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10).to_string(index=False))


# Scale data for clustering;
print("K-Means Clustering Analysis")
X_scaled_full = scaler.fit_transform(X)

# Elbow Method to find optimal K;
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled_full)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled_full, kmeans.labels_))

# Plot Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal K')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Different K')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Optimal K (using number of actual classes)
optimal_k = len(label_encoder.classes_)
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled_full)

print(f"\nOptimal K (based on target classes): {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled_full, cluster_labels):.4f}")
print(f"Inertia: {kmeans_final.inertia_:.2f}")

# Cluster Distribution
cluster_df = pd.DataFrame({'Cluster': cluster_labels, 'Actual': y_encoded})
print("\nCluster Distribution:")
print(cluster_df['Cluster'].value_counts().sort_index())

# Visualize Clusters (using first 2 principal components for 2D visualization)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-Means Clustering Results')

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Actual Class')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Actual Disease Classes')

plt.tight_layout()
plt.show()


# Multiple ML Models Comparison;
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True)
}

# Results storage
results = []

for name, clf in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    clf.fit(x_train_scaled, y_train)
    y_pred_model = clf.predict(x_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred_model)
    f1_macro = f1_score(y_test, y_pred_model, average='macro')
    f1_weighted = f1_score(y_test, y_pred_model, average='weighted')
    precision = precision_score(y_test, y_pred_model, average='weighted')
    recall = recall_score(y_test, y_pred_model, average='weighted')
    
    # Cross-validation
    cv_score = cross_val_score(clf, X, y_encoded, cv=5, scoring='accuracy').mean()
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1 (Macro)': f1_macro,
        'F1 (Weighted)': f1_weighted,
        'Precision': precision,
        'Recall': recall,
        'CV Accuracy': cv_score
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1 (Weighted)', ascending=False)

print("\nModel Comparison Results:")
print(results_df.to_string(index=False))

# Visualize Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy & F1 Score Comparison
metrics_to_plot = ['Accuracy', 'F1 (Weighted)', 'CV Accuracy']
results_melted = results_df.melt(id_vars=['Model'], value_vars=metrics_to_plot, 
                                  var_name='Metric', value_name='Score')

sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=axes[0], palette='Set2')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_title('Model Performance Comparison')
axes[0].set_ylim(0, 1.1)
axes[0].legend(loc='lower right')

# Precision & Recall Comparison
metrics_to_plot2 = ['Precision', 'Recall']
results_melted2 = results_df.melt(id_vars=['Model'], value_vars=metrics_to_plot2, 
                                   var_name='Metric', value_name='Score')

sns.barplot(data=results_melted2, x='Model', y='Score', hue='Metric', ax=axes[1], palette='Set1')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_title('Precision & Recall Comparison')
axes[1].set_ylim(0, 1.1)
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

# Best Model Summary
best_model = results_df.iloc[0]
print(f"\n{'='*60}")
print("Best Model Summary:")
print(f"{'-'*60}")
print(f"Best Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"F1 Score (Weighted): {best_model['F1 (Weighted)']:.4f}")
print(f"F1 Score (Macro): {best_model['F1 (Macro)']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"Cross-Validation Accuracy: {best_model['CV Accuracy']:.4f}")

print(f"\nEnd of Analysis.")
# Finished.