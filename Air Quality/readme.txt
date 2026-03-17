# Air Quality Index (AQI) Prediction Pipeline

## 🌍 Project Overview
This project aims to predict the **Air Quality Index (AQI)** across several major global cities using advanced machine learning and deep learning techniques. By analyzing various pollutants (PM2.5, PM10, CO, NO2, SO2, O3) and temporal factors, this pipeline performs extensive data preprocessing, feature engineering, and model training to accurately forecast air quality.

## 📂 Dataset & Source
The dataset comprises air quality readings from multiple global cities, including:
* `New_York_Air_Quality.csv`
* `London_Air_Quality.csv`
* `Dubai_Air_Quality.csv`
* `Cairo_Air_Quality.csv`
* `Brasilia_Air_Quality.csv`
* `Sydney_Air_Quality.csv`
* `Air_Quality.csv` (Combined dataset used for training)

**Data Source:** This dataset was obtained from Kaggle. 
**Target Variable:** `AQI` (Air Quality Index)
**Key Features:** Measurements of Carbon Monoxide (CO), Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), Ozone (O3), PM2.5, and PM10.

## 🛠️ Feature Engineering
To improve model performance, extensive feature engineering was performed:
* **Particulate Matter Combinations:** Created `PM_Combined` and `PM_Ratio` from PM2.5 and PM10.
* **Temporal Features:** Extracted `Hour`, `Day`, `Month`, `DayOfWeek`, `IsWeekend`, `Season`, and customized `TimeOfDay` categorizations.
* **Pollution Interactions:** Created interaction terms like `CO_NO2_Interaction`, `SO2_O3_Interaction`, and `Total_Gas_Pollution`.
* **Logarithmic Transformations:** Applied log transforms (`log1p`) to highly skewed pollutant variables.
* **City-Level Normalization:** Normalized pollutant levels grouped by city to account for baseline geographic differences.
* **Risk Indices:** Formulated custom metrics including `Air_Pollution_Index` and `Health_Risk_Score`.

## 📊 Exploratory Data Analysis (EDA)
Thorough EDA was conducted to understand distributions, handle missing values, and identify relationships between variables.

### Initial Correlation Matrix
*Before feature engineering, we analyzed the base relationships between primary pollutants.*

![Initial Correlation Matrix](correlation_matrix.jpg)

### Engineered Correlation Matrix
*After generating new features, we evaluated the correlation of our enriched dataset.*

![Engineered Correlation Matrix](correlation_matrix_engineered.jpg)

## 🤖 Modeling
The project compares standard regression models against a deep learning approach. The data was scaled using `StandardScaler` prior to training.

**Classical Machine Learning Models:**
1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor (Best Performing Classical Model)
4. Gradient Boosting Regressor
5. XGBoost Regressor

**Deep Learning Model:**
* A Sequential Artificial Neural Network (ANN) built with Keras.
* Architecture: Multiple Dense layers (128 -> 64 -> 32 -> 16 -> 1) interspersed with Dropout layers to prevent overfitting.
* Optimizer: Adam (learning rate = 0.001) with Early Stopping and Model Checkpointing.

## 📈 Results & Evaluation

### Model Comparison
Random Forest and XGBoost proved to be highly effective, alongside our custom Neural Network.

![Model Comparison](model_comparison.png)

### Feature Importance
The Random Forest model highlighted that the engineered `Health_Risk_Score`, `PM_Combined`, and `PM_Combined_log` were the strongest predictors of AQI.

![Feature Importance](feature_importance.png)

### Neural Network Training History
The model converged smoothly, minimizing both Mean Squared Error (MSE) and Mean Absolute Error (MAE) across 100 epochs.

![NN Training History](nn_training_history.jpg)

### Prediction vs Actual (Best Models)
*Comparing the R² scores and prediction accuracy of the Random Forest model (R² = 0.9441) against the Neural Network (R² = 0.8626).*

![Prediction vs Actual](prediction_vs_actual.jpg)

## 💻 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/air-quality-prediction.git](https://github.com/yourusername/air-quality-prediction.git)
   cd air-quality-prediction

2. **Install dependencies:**
```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```
Run the main pipeline:
Make sure the CSV files are placed in the archive/ directory as referenced in the code, then execute:                     

3. **Run the pipeline:**
```bash
   python main.py
```
Use the Pre-trained Model:
The best neural network weights are saved as best_nn_model.keras. You can load this using TensorFlow/Keras to make predictions on new data without retraining.

📜 License
This project is open-source and available under the MIT License. The original dataset is credited to its respective Kaggle author.                     
