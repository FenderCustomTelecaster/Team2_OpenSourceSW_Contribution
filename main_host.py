import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------/// Data load ///-----------------------
train_path = r"C:\Users\kgmin\Desktop\workspace\3-1\dataScience\Team2_OpenSourceSW_Contribution-main\AB_NYC_2019.csv"
df = pd.read_csv(train_path)
df.head()

# -----------------------/// Data preprocessing ///-----------------------

# haversine fuction
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood'], inplace=True)

df=pd.get_dummies(df,columns=['neighbourhood_group'])
df.head()

dummy_cols = [col for col in df.columns if col.startswith('neighbourhood_group_')]

#center latitude, longitude of each location
centers = {
    'neighbourhood_group_Bronx': (40.8448, -73.8648),
    'neighbourhood_group_Brooklyn': (40.6782, -73.9442),
    'neighbourhood_group_Manhattan': (40.7685, -73.9822),
    'neighbourhood_group_Queens': (40.7282, -73.7949),
    'neighbourhood_group_Staten Island': (40.5795, -74.1502),
}

#After examining the one-hot encoded t,f value of each row, find which sphere corresponds to and return-
#-the distance between the latitude longitude of the row and the latitude longitude of the center of the row
#shows at one column(feature)
def compute_distance(row):
    for col in dummy_cols:
        if row.get(col, False):
            clat, clon = centers[col]
            return haversine(row['latitude'], row['longitude'], clat, clon)
    return np.nan

df['distance_to_center'] = df.apply(compute_distance, axis=1)

df.head()

# Calculate review date differences (smaller value in recent days)
reference_date = pd.to_datetime("2019-12-01")
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['days_since_oldest_review'] = (reference_date - df['last_review']).dt.days

# Missing value processing: no review → considered older than the oldest value max + 30
temp_days = df['days_since_oldest_review'].copy()
df['days_since_oldest_review'] = temp_days.fillna(temp_days.max() + 30)

# Transform to be more recent
max_days = df['days_since_oldest_review'].max()
df['days_since_oldest_review'] = max_days - df['days_since_oldest_review']
df.drop(columns=['last_review'], inplace=True)

df[['days_since_oldest_review']].head()

#room type one-hot encoding
df['reviews_per_month'].fillna(0, inplace=True)
df=pd.get_dummies(df,columns=['room_type'])
df.head()

df.drop(columns=['latitude', 'longitude'], inplace=True)
df

# Remove price outlier (average: 152.72, minimum:0, maximum:10,000)
df = df[df['price'] > 0]
df = df[df['price'] < 2000] #수정가능

# log transform -> skewed distribution
df['log_price'] = np.log1p(df['price'])
df.drop(columns=['price'], inplace=True)

# Remove minimum day outlier (Average: 7, Minimum:1, Maximum:1250)
df = df[df['minimum_nights'] >= 1]
df = df[df['minimum_nights'] <= 30]

#data 48895 -> 48043

# -----------------------/// Modeling ///-----------------------

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# seperate feature/target
X = df.drop(columns=['log_price'])
y = df['log_price']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# without using scailing - XGBoost is less sensitive to scailing
X_to_train = X_train
X_to_test = X_test
feature_names_for_importance = X_train.columns

# define XGBoost hyperparameter exploration range
param_distributions_xgb = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.001, 0.01, 0.1],
    'reg_lambda': [0.1, 0.5, 1, 1.5, 2]
}

# generating RandomizedSearchCV object / training
xgb_base = xgb.XGBRegressor(objective='reg:squarederror',
                            random_state=42,
                            n_jobs=-1,
                           )

print(f"\n--- Hyperparameter Tuning with RandomizedSearchCV for XGBoost ---")

# setting RandomizedSearchCV
random_search_xgb = RandomizedSearchCV(estimator=xgb_base,
                                       param_distributions=param_distributions_xgb,
                                       n_iter=50,
                                       cv=3, 
                                       scoring='neg_mean_squared_error',
                                       verbose=2,
                                       random_state=42,
                                       n_jobs=-1)

random_search_xgb.fit(X_to_train, y_train)

print("\nXGBoost RandomizedSearchCV training complete.")
print("Best hyperparameters found: ", random_search_xgb.best_params_)
best_cv_rmse_xgb = np.sqrt(-random_search_xgb.best_score_)
print(f"Best CV RMSE (log_price) for XGBoost: {best_cv_rmse_xgb:.4f}")

# find best model / prediction & evaluation
best_xgb_model = random_search_xgb.best_estimator_

print("\n--- Model Evaluation (Best XGBoost Regressor from RandomizedSearchCV) ---")
y_pred_log_xgb_tuned = best_xgb_model.predict(X_to_test)

# evaluation (log)
mse_log_xgb_tuned = mean_squared_error(y_test, y_pred_log_xgb_tuned)
rmse_log_xgb_tuned = np.sqrt(mse_log_xgb_tuned)
mae_log_xgb_tuned = mean_absolute_error(y_test, y_pred_log_xgb_tuned)
r2_log_xgb_tuned = r2_score(y_test, y_pred_log_xgb_tuned)

print(f"Test RMSE (log_price): {rmse_log_xgb_tuned:.4f}")
print(f"Test MAE (log_price): {mae_log_xgb_tuned:.4f}")
print(f"Test R-squared (log_price): {r2_log_xgb_tuned:.4f}")

# evaluation (original)
y_test_original = np.expm1(y_test)
y_pred_original_xgb_tuned = np.expm1(y_pred_log_xgb_tuned)
y_pred_original_xgb_tuned[y_pred_original_xgb_tuned < 0] = 0

mse_original_xgb_tuned = mean_squared_error(y_test_original, y_pred_original_xgb_tuned)
rmse_original_xgb_tuned = np.sqrt(mse_original_xgb_tuned)
mae_original_xgb_tuned = mean_absolute_error(y_test_original, y_pred_original_xgb_tuned)

print(f"\nTest RMSE (original price): ${rmse_original_xgb_tuned:.2f}")
print(f"Test MAE (original price): ${mae_original_xgb_tuned:.2f}")

# feature importances visualization
print("\n--- Feature Importances (Best XGBoost Regressor) ---")
importances_tuned_xgb = best_xgb_model.feature_importances_
feature_importance_df_tuned_xgb = pd.DataFrame({'feature': feature_names_for_importance, 'importance': importances_tuned_xgb})
feature_importance_df_tuned_xgb = feature_importance_df_tuned_xgb.sort_values(by='importance', ascending=False)

print("Top 10 Feature Importances (Tuned XGBoost Model):")
print(feature_importance_df_tuned_xgb.head(10))

plt.figure(figsize=(10, 8))
top_n_features = 15
plt.barh(feature_importance_df_tuned_xgb['feature'][:top_n_features], feature_importance_df_tuned_xgb['importance'][:top_n_features])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(f"Top {top_n_features} Feature Importances from Tuned XGBoost Regressor")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# comparison prediction value with actual value
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original_xgb_tuned, alpha=0.3, label='Predicted vs Actual')
min_val = min(y_test_original.min(), y_pred_original_xgb_tuned.min())
max_val = max(y_test_original.max(), y_pred_original_xgb_tuned.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')

plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($) - Tuned XGBoost")
plt.title("Actual vs. Predicted Prices (Tuned XGBoost Regressor) - Original Scale")
plt.legend()
plt.grid(True)
plt.show()

# plot residual
residuals_xgb_tuned = y_test_original - y_pred_original_xgb_tuned

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_original_xgb_tuned, residuals_xgb_tuned, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price ($) - Tuned XGBoost")
plt.ylabel("Residuals (Actual - Predicted Price) ($)")
plt.title("Residual Plot (Tuned XGBoost Regressor) - Original Scale")
plt.grid(True)
plt.show()