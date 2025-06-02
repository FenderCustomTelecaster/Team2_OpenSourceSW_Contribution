# NYC Airbnb Price Prediction & Clustering System

## 📌 Overview

This project aims to analyze and predict Airbnb prices in New York City using clustering and regression models, particularly focusing on combining unsupervised and supervised learning techniques. We explored various data encoding and scaling strategies, applied multiple modeling approaches, and developed a robust architecture for cluster-based classification and price prediction.

---

## 🔧 Contribution Summary

### 1. Data Transformation & Preprocessing Experiments

To optimize feature processing, we tested **9 combinations** of categorical encoding and numerical scaling techniques:

- **Encodings**: One-hot, Label, Ordinal
- **Scalers**: Standard (Z-score), Min-Max, Robust

Label and Ordinal encoding produced the same results, effectively reducing our evaluation to **6 key configurations**. Comparative analysis was performed based on regression model results for each configuration.

### 2. Clustering

- **KMeans Clustering (k=4)** was applied based on the Elbow method.
- Cluster labels were appended to the dataset and used as new features.
- Provided valuable segmentation and interpretability.

### 3. Regression Models for Price Prediction

#### ▪ Multiple Linear Regression
- Baseline model on log-transformed prices.
- Limited performance for high-price listings.
- Showed heteroscedasticity in residuals.

#### ▪ Polynomial Regression (Degree=2)
- Better performance than linear regression.
- Handled non-linear trends slightly better.
- Still limited at higher price ranges.

#### ▪ XGBoost Regressor
- **Best performing model**.
- Strong R², low RMSE/MAE.
- Feature importance highlighted:
  - `room_type_Entire home/apt`
  - `cluster`
  - Neighborhood-related features

### 4. Cluster Classification Models

Tested models on predicting KMeans-generated clusters:

#### ▪ k-Nearest Neighbors (KNN)
- Lowest accuracy.
- Struggled with high-dimensional boundaries.

#### ▪ Random Forest (with OOB evaluation)
- Achieved ~99% accuracy in both holdout and K-Fold CV.
- OOB ≈ CV ≈ Test Accuracy → **No overfitting**.
- Chosen as the primary classifier.

#### ▪ AdaBoost
- Similar performance to RF.
- Evaluated via consistent CV/holdout metrics.
- Generalized well.

> 📌 Overall: Ensemble models were highly effective due to the distinctiveness of KMeans clusters.

---

## 🏗️ System Architecture

The system is divided into **User** and **Host** perspectives.

### 📍 User-side Workflow

1. **Preprocessing**
   - Drop irrelevant columns.
   - Normalize dates, encode categoricals.
   - Calculate distance from center.
   - Handle missing values.
   - Remove lat/lon post-clustering.

2. **Clustering**
   - StandardScaler → KMeans (k=4).
   - Store cluster in `df_scaled['cluster']`.

3. **Classification Modeling**
   - Label: `cluster`
   - Model: RandomForestClassifier
   - Evaluated via OOB, CV, and test split.

4. **Visualization**
   - PCA (2D clustering)
   - Confusion Matrix
   - Feature Importance
   - Accuracy comparison (OOB vs Test)

5. **New Data Prediction**
   - Predict clusters on new Airbnb entries.
   - Visualize clusters on a map.

### 🧑‍💼 Host-side Workflow

1. **Preprocessing**
   - Filter outliers (price < 0 or > 2000, nights < 1 or > 30).
   - Apply `log(price)` transformation.

2. **Clustering**
   - KMeans with optimal k=4 (Elbow method).

3. **Linear Regression**
   - Train on log(price).
   - Evaluate via RMSE, MAE, R².
   - Visualization: Predictions vs Actual, Residual plot.

4. **XGBoost Regression**
   - RandomizedSearchCV + 3-Fold CV for tuning.
   - Feature importance chart for explainability.
   - Strongest performance on both log and original scale prices.

---

## ✅ Final Model Selection & top 5 combination

-Modeling Approach(to find top 5 combination)
A top-level function (auto_train_and_evaluate) was created to:

- Automatically handle preprocessing.
- Train and evaluate multiple regression models.
- Use GridSearchCV with 5-fold cross-validation.
- Rank models based on R² score.
- Return the top N performing models with optimal hyperparameters.
Regression Models Used
- RandomForestRegressor: n_estimators, max_depth
- GradientBoostingRegressor: n_estimators, learning_rate
- SVR: C, kernel
- DecisionTreeRegressor: max_depth
- LinearRegression: No tuning
Evaluation Metric
- R² Score was used to evaluate model performance.
- The higher the R², the better the model explains variance in the data.
Results (Top 5 Models)
1. GradientBoostingRegressor – R²: 0.6153 – Best Params: {'n_estimators': 100, 'learning_rate': 0.1}
2. RandomForestRegressor – R²: 0.6042 – Best Params: {'n_estimators': 100, 'max_depth': 20}
3. SVR – R²: 0.5527 – Best Params: {'C': 1, 'kernel': 'rbf'}
4. DecisionTreeRegressor – R²: 0.5045 – Best Params: {'max_depth': 10}
5. LinearRegression – R²: 0.4651 – Best Params: {}


- **Best model**: XGBoost Regressor with KMeans clusters as features.
- **Best classifier**: RandomForestClassifier with OOB validation.
- **Key engineered features**: `cluster`, `distance_to_center`, `room_type`, and neighborhood information.
