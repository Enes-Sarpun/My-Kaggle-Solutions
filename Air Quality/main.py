# Import Necessary Libraries;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and Preprocess Data;
data = pd.read_csv("archive\Air_Quality.csv")

# Missing Values and Duplicates;
missing_values = data.isnull().sum()
duplicates = data.duplicated().sum()
if duplicates > 0:
    data = data.drop_duplicates()
if missing_values.any():
    data = data.dropna()    

print("Missing Values and Duplicates layers After Cleaning:\n", data.isnull().sum(), data.duplicated().sum())
print("Data Shape After Cleaning:", data.shape)
print("-" * 50)

print(data.info())
print(data.describe().T)
for idx, col in enumerate(data.columns):
    print(f"{idx}: {col}")

# Count Plots for Categorical Columns;
object_cols = data.select_dtypes(include=['object']).columns.tolist()
if len(object_cols) > 0:
    n_cols = 3
    n_rows = (len(object_cols) + n_cols - 1) // n_cols  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4*n_rows), dpi=100)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(object_cols):
        value_counts = data[col].value_counts().head(10)  
        axes[idx].bar(value_counts.index.astype(str), value_counts.values, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hidden unused subplots;
    for idx in range(len(object_cols), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.show()

# Box and Scatter Plots for Numeric Columns;
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) > 0:
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), dpi=100)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].scatter(data[col],data.index, color='teal', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Box Plots for Numeric Columns;
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), dpi=100)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(data[col].dropna(), patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', color='navy'))
        axes[idx].set_ylabel(col)
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
# Confusion Matrix for All Columns;
corr_matrix = data[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png',bbox_inches='tight', dpi=350)
plt.show()


# PM2.5 and PM10 based New Features;
data['PM_Combined'] = (data['PM2.5'] + data['PM10']) / 2  
data['PM_Ratio'] = data['PM2.5'] / (data['PM10'] + 1e-6) 

# Date and Time Based Features;
data['Date'] = pd.to_datetime(data['Date'])
data['Hour'] = data['Date'].dt.hour
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)  # Weekend?
data['Season'] = data['Month'].apply(lambda x: 1 if x in [12,1,2] else (2 if x in [3,4,5] else (3 if x in [6,7,8] else 4)))
# 1=Winter, 2=Spring, 3=Summer, 4=Autumn

# 3. Time of Day;
def time_of_day(hour):
    if 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    elif 18 <= hour < 22:
        return 3  # Evening
    else:
        return 4  # Night
data['TimeOfDay'] = data['Hour'].apply(time_of_day)

# 4. Pollution Interaction Features
data['CO_NO2_Interaction'] = data['CO'] * data['NO2']  # Traffic-related pollution indicator
data['SO2_O3_Interaction'] = data['SO2'] * data['O3']  # Industrial + photochemical pollution
data['Total_Gas_Pollution'] = data['CO'] + data['NO2'] + data['SO2'] + data['O3']  # Total gas pollution

# 5. Logarithmic transformations (for skewed distributions)
for col in ['CO', 'NO2', 'SO2', 'O3', 'PM_Combined']:
    data[f'{col}_log'] = np.log1p(data[col])  # log(1+x) - safe for zero and negative values

# 6. Normalized features (by city)
for col in ['CO', 'NO2', 'SO2', 'O3', 'PM_Combined']:
    data[f'{col}_city_norm'] = data.groupby('City')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

# 7. Pollution indices
data['Air_Pollution_Index'] = (data['CO']/1000 + data['NO2']/50 + data['SO2']/20 + data['O3']/100 + data['PM_Combined']/50) / 5
data['Health_Risk_Score'] = data['PM_Combined'] * 0.4 + data['O3'] * 0.3 + data['NO2'] * 0.2 + data['SO2'] * 0.1

# 8. City encoding (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['City_Encoded'] = le.fit_transform(data['City'])

# Drop original PM2.5 and PM10 (now we have PM_Combined)
data = data.drop(['PM2.5', 'PM10', 'Date'], axis=1)

print("\n" + "="*50)
print("FEATURE ENGINEERING COMPLETED!")
print("="*50)
print(f"New columns: {data.columns.tolist()}")
print(f"Total number of features: {data.shape[1]}")
print("\nStatistics of new features:")
print(data.describe().T)

# New Corelation Matrix;
numeric_cols_new = data.select_dtypes(include=['number']).columns.tolist()
corr_matrix_new = data[numeric_cols_new].corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix_new, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
            cbar_kws={"shrink": .8}, annot_kws={"size": 7})
plt.title('Correlation Matrix (After Feature Engineering)', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix_engineered.png', bbox_inches='tight', dpi=350)
plt.show()

# Feature ve Target Separation;
X = data.drop(['AQI', 'City'], axis=1)
y = data['AQI']

print(f"\nFeatures Shape: {X.shape}")
print(f"Target Shape: {y.shape}")
print(f"Feature Names: {X.columns.tolist()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ML Models;
print("\n" + "="*60)
print("CLASSICAL MACHINE LEARNING MODELS")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
}

results = {}
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"   MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

# Model Comparison Table;
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)
print(results_df.round(4))

# The best model;
best_model_name = results_df['R2'].idxmax()
print(f"\n🏆 Best Model: {best_model_name} (R² = {results_df.loc[best_model_name, 'R2']:.4f})")

# Model Comparison plots;
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² Score comparison;
colors = ['green' if name == best_model_name else 'steelblue' for name in results_df.index]
axes[0].barh(results_df.index, results_df['R2'], color=colors)
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison - R² Score')
axes[0].set_xlim(0, 1)

# RMSE Comparison;
axes[1].barh(results_df.index, results_df['RMSE'], color=colors)
axes[1].set_xlabel('RMSE')
axes[1].set_title('Model Comparison - RMSE (Lower is Better)')

plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=350)
plt.show()

# Feature Importance (Random Forest);
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15], color='teal')
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=350)
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Neural Network Model;
print("\n" + "="*60)
print("DEEP LEARNING MODEL (Neural Network)")
print("="*60)

# Model architecture;
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)  # Regression output
])

# Model Compile;
nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

nn_model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_nn_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Model training;
print("\n🚀 Training Neural Network...")
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# Neural Network Evaluation;
y_pred_nn = nn_model.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_rmse = np.sqrt(nn_mse)
nn_mae = mean_absolute_error(y_test, y_pred_nn)
nn_r2 = r2_score(y_test, y_pred_nn)

print("\n" + "="*60)
print("NEURAL NETWORK RESULTS")
print("="*60)
print(f"MSE: {nn_mse:.4f}")
print(f"RMSE: {nn_rmse:.4f}")
print(f"MAE: {nn_mae:.4f}")
print(f"R² Score: {nn_r2:.4f}")

# Training History Plot;
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training vs Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Training MAE', color='blue')
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Training vs Validation MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nn_training_history.png', bbox_inches='tight', dpi=350)
plt.show()

# Prediction vs Actual Plot;
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Best Classical Mode
axes[0].scatter(y_test, y_pred_best, alpha=0.5, color='steelblue', edgecolor='k', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual AQI')
axes[0].set_ylabel('Predicted AQI')
axes[0].set_title(f'{best_model_name}\nR² = {results_df.loc[best_model_name, "R2"]:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Neural Network;
axes[1].scatter(y_test, y_pred_nn, alpha=0.5, color='teal', edgecolor='k', linewidth=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual AQI')
axes[1].set_ylabel('Predicted AQI')
axes[1].set_title(f'Neural Network\nR² = {nn_r2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_vs_actual.png', bbox_inches='tight', dpi=350)
plt.show()

# Fınal Summary;
print("\n" + "="*60)
print("📊 FINAL MODEL SUMMARY")
print("="*60)

# Compare all models (including NN);
all_results = results.copy()
all_results['Neural Network'] = {'MSE': nn_mse, 'RMSE': nn_rmse, 'MAE': nn_mae, 'R2': nn_r2}
all_results_df = pd.DataFrame(all_results).T.sort_values('R2', ascending=False)

print("\nAll Models Ranked by R² Score:")
print(all_results_df.round(4))

final_best = all_results_df['R2'].idxmax()
print(f"\n🏆 OVERALL BEST MODEL: {final_best}")
print(f"   R² Score: {all_results_df.loc[final_best, 'R2']:.4f}")
print(f"   RMSE: {all_results_df.loc[final_best, 'RMSE']:.4f}")
print(f"   MAE: {all_results_df.loc[final_best, 'MAE']:.4f}")



