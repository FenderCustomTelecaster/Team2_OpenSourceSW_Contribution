import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# -----------------------/// Linear Regression ///-----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_log_lr = lr.predict(X_test_scaled)

mse_log_lr = mean_squared_error(y_test, y_pred_log_lr)
rmse_log_lr = np.sqrt(mse_log_lr)
mae_log_lr = mean_absolute_error(y_test, y_pred_log_lr)
r2_log_lr = r2_score(y_test, y_pred_log_lr)

print(f"\n[Linear Regression log_price] RMSE: {rmse_log_lr:.4f}, MAE: {mae_log_lr:.4f}, R²: {r2_log_lr:.4f}")

y_pred_orig_lr = np.expm1(y_pred_log_lr)
y_pred_orig_lr[y_pred_orig_lr < 0] = 0

mse_orig_lr = mean_squared_error(y_test_orig, y_pred_orig_lr)
rmse_orig_lr = np.sqrt(mse_orig_lr)
mae_orig_lr = mean_absolute_error(y_test_orig, y_pred_orig_lr)

print(f"[Linear Regression original price] RMSE: ${rmse_orig_lr:.2f}, MAE: ${mae_orig_lr:.2f}")
[]
# -----------------------/// Visualization ///-----------------------
# Linear Regression Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred_orig_lr, alpha=0.3)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (Linear Regression)")
plt.title("Actual vs Predicted Price - Linear Regression")
plt.grid(True)
plt.show()
