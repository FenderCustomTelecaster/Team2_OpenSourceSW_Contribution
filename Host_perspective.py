import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# -----------------------/// Data load ///-----------------------
train_path = "AB_NYC_2019.csv"
df = pd.read_csv(train_path)
df.head()

# -----------------------/// Data Preprocessing ///-----------------------
# Haversine distance function to compute distance between two geo-locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Reload data
df = pd.read_csv(train_path)

# Drop unnecessary columns
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood'], inplace=True)

# One-hot encode neighbourhood group
df = pd.get_dummies(df, columns=['neighbourhood_group'])
dummy_cols = [col for col in df.columns if col.startswith('neighbourhood_group_')]

# Define geographical centers for each neighbourhood group
centers = {
    'neighbourhood_group_Bronx': (40.8448, -73.8648),
    'neighbourhood_group_Brooklyn': (40.6782, -73.9442),
    'neighbourhood_group_Manhattan': (40.7685, -73.9822),
    'neighbourhood_group_Queens': (40.7282, -73.7949),
    'neighbourhood_group_Staten Island': (40.5795, -74.1502),
}

# Compute distance from each listing to the center of its neighbourhood group
def compute_distance(row):
    for col in dummy_cols:
        if row.get(col, False):
            clat, clon = centers[col]
            return haversine(row['latitude'], row['longitude'], clat, clon)
    return np.nan

df['distance_to_center'] = df.apply(compute_distance, axis=1)

# Process review dates to compute how recent they are
reference_date = pd.to_datetime("2019-12-01")
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['days_since_oldest_review'] = (reference_date - df['last_review']).dt.days

# Handle missing review dates by assigning max + 30 days
temp_days = df['days_since_oldest_review'].copy()
df['days_since_oldest_review'] = temp_days.fillna(temp_days.max() + 30)

# Invert values so that more recent reviews have higher numbers
max_days = df['days_since_oldest_review'].max()
df['days_since_oldest_review'] = max_days - df['days_since_oldest_review']
df.drop(columns=['last_review'], inplace=True)

# Fill missing values in review frequency and one-hot encode room_type
df['reviews_per_month'].fillna(0, inplace=True)
df = pd.get_dummies(df, columns=['room_type'])

# Drop latitude and longitude columns
df.drop(columns=['latitude', 'longitude'], inplace=True)

# Remove price outliers
df = df[df['price'] > 0]
df = df[df['price'] < 2000]

# Log-transform price to reduce skewness
df['log_price'] = np.log1p(df['price'])
df.drop(columns=['price'], inplace=True)

# Remove outliers in minimum_nights
df = df[df['minimum_nights'] >= 1]
df = df[df['minimum_nights'] <= 30]

# Select all columns for scaling
features_to_scale = df.columns.tolist()

# Apply standard scaling
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[features_to_scale].values)

# Rebuild DataFrame with scaled values
df_std = pd.DataFrame(scaled_array, columns=features_to_scale, index=df.index)

print(df_std.head())

# -----------------------/// KMeans Clustering ///-----------------------
from sklearn.cluster import KMeans

df_standard = df_std.copy()
X_for_clustering = df_standard

sse = []
# Run KMeans for k from 1 to 10 and record SSE (inertia)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_for_clustering)
    sse.append(kmeans.inertia_)

# Plot Elbow method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.show()

# Final KMeans clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df_standard['cluster'] = kmeans.fit_predict(X_for_clustering)

# View cluster centroids
cluster_means = df_standard.groupby('cluster').mean(numeric_only=True)
print("\n--- Cluster Means ---")
print(cluster_means)

# Merge cluster info back to original df
df['cluster'] = df_standard['cluster']
df.head()

# -----------------------/// Feature/Target Split and Scaling ///-----------------------
X = df.drop(columns=['log_price'])
y = df['log_price']

# Scale features for linear regression
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

print("Scaled feature data:")
print(X_scaled.head())
print("Target (log_price) data:")
print(y.head())
print("Feature shape:", X_scaled.shape)
print("Target shape:", y.shape)

# -----------------------/// Linear Regression Modeling ///-----------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: X_train_lr={X_train_lr.shape}, y_train_lr={y_train_lr.shape}")
print(f"Test data shape: X_test_lr={X_test_lr.shape}, y_test_lr={y_test_lr.shape}")
print("\n--- Linear Regression Model Training and Evaluation ---")

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Predict on test set
y_pred_log_lr = lr_model.predict(X_test_lr)

# Evaluate in log scale
mse_log_lr = mean_squared_error(y_test_lr, y_pred_log_lr)
rmse_log_lr = np.sqrt(mse_log_lr)
mae_log_lr = mean_absolute_error(y_test_lr, y_pred_log_lr)
r2_log_lr = r2_score(y_test_lr, y_pred_log_lr)

print(f"Test RMSE (log_price): {rmse_log_lr:.4f}")
print(f"Test MAE (log_price): {mae_log_lr:.4f}")
print(f"Test R-squared (log_price): {r2_log_lr:.4f}")

# Evaluate in original scale
y_test_original = np.expm1(y_test_lr)
y_pred_original_lr = np.expm1(y_pred_log_lr)
y_pred_original_lr[y_pred_original_lr < 0] = 0

mse_original_lr = mean_squared_error(y_test_original, y_pred_original_lr)
rmse_original_lr = np.sqrt(mse_original_lr)
mae_original_lr = mean_absolute_error(y_test_original, y_pred_original_lr)

print(f"\nTest RMSE (original price): ${rmse_original_lr:.2f}")
print(f"Test MAE (original price): ${mae_original_lr:.2f}")

# Plot actual vs predicted (original scale)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original_lr, alpha=0.3, label='Predicted vs Actual')
min_val = min(y_test_original.min(), y_pred_original_lr.min())
max_val = max(y_test_original.max(), y_pred_original_lr.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($) - Linear Regression")
plt.title("Actual vs. Predicted Prices (Linear Regression) - Original Scale")
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals
residuals_lr = y_test_original - y_pred_original_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_original_lr, residuals_lr, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price ($) - Linear Regression")
plt.ylabel("Residuals (Actual - Predicted Price) ($)")
plt.title("Residual Plot (Linear Regression) - Original Scale")
plt.grid(True)
plt.show()

# -----------------------/// XGBoost Modeling and Evaluation ///-----------------------
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Train-test split for XGBoost (no scaling required)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for XGBoost
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

# Initialize XGBoost model
xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

print(f"\n--- Hyperparameter Tuning with RandomizedSearchCV for XGBoost ---")

# Perform randomized hyperparameter search
random_search_xgb = RandomizedSearchCV(estimator=xgb_base,
                                       param_distributions=param_distributions_xgb,
                                       n_iter=50,
                                       cv=3,
                                       scoring='neg_mean_squared_error',
                                       verbose=2,
                                       random_state=42,
                                       n_jobs=-1)

random_search_xgb.fit(X_train_xgb, y_train_xgb)

print("\nXGBoost RandomizedSearchCV training complete.")
print("Best hyperparameters found: ", random_search_xgb.best_params_)
best_cv_rmse_xgb = np.sqrt(-random_search_xgb.best_score_)
print(f"Best CV RMSE (log_price) for XGBoost: {best_cv_rmse_xgb:.4f}")

# Evaluate best XGBoost model
best_xgb_model = random_search_xgb.best_estimator_
y_pred_log_xgb_tuned = best_xgb_model.predict(X_test_xgb)

mse_log_xgb_tuned = mean_squared_error(y_test_xgb, y_pred_log_xgb_tuned)
rmse_log_xgb_tuned = np.sqrt(mse_log_xgb_tuned)
mae_log_xgb_tuned = mean_absolute_error(y_test_xgb, y_pred_log_xgb_tuned)
r2_log_xgb_tuned = r2_score(y_test_xgb, y_pred_log_xgb_tuned)

print(f"Test RMSE (log_price): {rmse_log_xgb_tuned:.4f}")
print(f"Test MAE (log_price): {mae_log_xgb_tuned:.4f}")
print(f"Test R-squared (log_price): {r2_log_xgb_tuned:.4f}")

# Evaluate in original scale
y_test_original = np.expm1(y_test_xgb)
y_pred_original_xgb_tuned = np.expm1(y_pred_log_xgb_tuned)
y_pred_original_xgb_tuned[y_pred_original_xgb_tuned < 0] = 0

mse_original_xgb_tuned = mean_squared_error(y_test_original, y_pred_original_xgb_tuned)
rmse_original_xgb_tuned = np.sqrt(mse_original_xgb_tuned)
mae_original_xgb_tuned = mean_absolute_error(y_test_original, y_pred_original_xgb_tuned)

print(f"\nTest RMSE (original price): ${rmse_original_xgb_tuned:.2f}")
print(f"Test MAE (original price): ${mae_original_xgb_tuned:.2f}")

# Feature importance plot
print("\n--- Feature Importances (Best XGBoost Regressor) ---")
feature_names_for_importance = X_train_xgb.columns
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

# Prediction vs actual
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

# Residual plot
residuals_xgb_tuned = y_test_original - y_pred_original_xgb_tuned
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_original_xgb_tuned, residuals_xgb_tuned, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price ($) - Tuned XGBoost")
plt.ylabel("Residuals (Actual - Predicted Price) ($)")
plt.title("Residual Plot (Tuned XGBoost Regressor) - Original Scale")
plt.grid(True)
plt.show()