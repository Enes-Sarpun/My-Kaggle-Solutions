# Import Necessary Libraries;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load Dataset;
df = pd.read_csv('goldmansachs.csv')
if df.empty:
    raise ValueError("The dataset is empty. Please provide a valid dataset.")

# Display Basic Information about the Dataset;
print(df.head())
print(df.info())
print(df.describe().T)

# Check for Missing Values and Duplicate Entries;
missing_values = df.isnull().sum()
duplicate_entries = df.duplicated().sum()

if missing_values.any():
    print("-"*25)
    print("Missing values found:\n", missing_values)
    df = df.fillna(df.mean())
else:
    print("-"*25)
    print("No missing values found in the dataset.")

if duplicate_entries > 0:
    df = df.drop_duplicates()
    print("-"*25)
    print(f"Removed {duplicate_entries} duplicate entries.")
else:
    print("-"*25)
    print("No duplicate entries found in the dataset.")

# Date Processing;
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year 
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Weekday'] = df['date'].dt.weekday

print(df[['date', 'Year', 'Month', 'Day', 'Weekday']].head())

# Date Row Plots;
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
sns.lineplot(x='Year', y='close', data=df, ax=ax[0, 0])
ax[0, 0].set_xlabel('Year', fontsize=10,alpha=0.7)
ax[0, 0].set_ylabel('Close Price')
ax[0, 0].set_title('Year vs Close Price')   

sns.lineplot(x='Month', y='close', data=df, ax=ax[0, 1])
ax[0, 1].set_xlabel('Month', fontsize=10, alpha=0.7)   
ax[0, 1].set_ylabel('Close Price')
ax[0, 1].set_title('Month vs Close Price')

sns.lineplot(x='Day', y='close', data=df, ax=ax[1, 0])
ax[1, 0].set_xlabel('Day', fontsize=10, alpha=0.7)
ax[1, 0].set_ylabel('Close Price')
ax[1, 0].set_title('Day vs Close Price')

sns.lineplot(x='Weekday', y='close', data=df, ax=ax[1, 1])
ax[1, 1].set_xlabel('Weekday', fontsize=10, alpha=0.7)
ax[1, 1].set_ylabel('Close Price')
ax[1, 1].set_title('Weekday vs Close Price')

plt.tight_layout()
plt.show()

# Daily Price Change;
df['Daily_Change'] = df['close'] - df['open']

# Daily Price Change Percentage;
df['Daily_Change_Pct'] = ((df['close'] - df['open']) / df['open']) * 100

# Daily Intraday Volatility;
df['Intraday_Volatility'] = df['high'] - df['low']

# Previous Day's Close Price Change;
df['Prev_Close'] = df['close'].shift(1)
df['Price_Change'] = df['close'] - df['Prev_Close']
df['Price_Change_Pct'] = ((df['close'] - df['Prev_Close']) / df['Prev_Close']) * 100

# Accumulated Price Change over 7 and 30 days;
df['MA_7'] = df['close'].rolling(window=7).mean()
df['MA_30'] = df['close'].rolling(window=30).mean() 

# Target Variable: Next Day's Close Price;
df['Target_Next_Close'] = df['close'].shift(-1)

# Remove NaN values from the first rows (due to shift and rolling)
df = df.dropna()
print(df[['date', 'Daily_Change', 'Daily_Change_Pct', 'Intraday_Volatility', 
          'Price_Change', 'Price_Change_Pct', 'MA_7', 'MA_30']].head(10))

# Visualizations of New Features;
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['Daily_Change_Pct'], kde=True, ax=ax[0, 0], color='blue')
ax[0, 0].set_xlabel('Daily Change (%)', fontsize=10, alpha=0.7)
ax[0, 0].set_title('Distribution of Daily Percentage Change', fontsize=10, alpha=0.7)

sns.histplot(df['Price_Change_Pct'], kde=True, ax=ax[0, 1], color='green')
ax[0, 1].set_xlabel('Price Change (%)', fontsize=10, alpha=0.7)
ax[0, 1].set_title('Distribution of Change Compared to Previous Day', fontsize=10, alpha=0.7)

sns.lineplot(x=df.index, y='MA_7', data=df, ax=ax[1, 0], label='MA 7')
sns.lineplot(x=df.index, y='MA_30', data=df, ax=ax[1, 0], label='MA 30')
ax[1, 0].set_xlabel('Index', fontsize=10, alpha=0.7)
ax[1, 0].set_ylabel('Price', fontsize=10, alpha=0.7)
ax[1, 0].set_title('Moving Averages', fontsize=10, alpha=0.7)
ax[1, 0].legend()

sns.scatterplot(x='Intraday_Volatility', y='volume', data=df, ax=ax[1, 1], alpha=0.5)
ax[1, 1].set_xlabel('Intraday Volatility', fontsize=10, alpha=0.7)
ax[1, 1].set_ylabel('Trading Volume', fontsize=10, alpha=0.7)
ax[1, 1].set_title('Volatility vs Trading Volume', fontsize=10, alpha=0.7)

plt.tight_layout()
plt.show()

# Correlation Heatmap;
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=14, alpha=0.8, pad=20)
plt.tight_layout()
plt.show()

# Prepare Data for Modeling;
features = ['open', 'volume', 'Year', 'Month', 'Day', 'Weekday',
            'Prev_Close', 'Price_Change', 'Price_Change_Pct', 'MA_7', 'MA_30']
X = df[features]
y = df['Target_Next_Close']

# Split Data into Training and Testing Sets;
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train Linear Regression Model;
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions;
y_pred = model.predict(X_test)

# Evaluate Model Performance;
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-"*25)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot Actual vs Predicted Prices;
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Close Prices', fontsize=10, alpha=0.7)
plt.ylabel('Predicted Close Prices', fontsize=10, alpha=0.7)
plt.title('Actual vs Predicted Close Prices', fontsize=14, alpha=0.8)
plt.tight_layout()
plt.show()

# Feature Importance;
importance = pd.Series(model.coef_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=importance.index, palette='viridis')
plt.xlabel('Feature Importance', fontsize=10, alpha=0.7)
plt.ylabel('Features', fontsize=10, alpha=0.7)
plt.title('Feature Importance (Veri Sızıntısı Düzeltildi)', fontsize=14, alpha=0.8)
plt.tight_layout()
plt.show()




#Finished.