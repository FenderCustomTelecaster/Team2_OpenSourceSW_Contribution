import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- (이전 코드 그대로 사용) ---
# 데이터 로드 및 초기 전처리
df = pd.read_csv(r"C:\Users\kgmin\Desktop\workspace\3-1\dataScience\Team2_OpenSourceSW_Contribution-main\AB_NYC_2019.csv")

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
# df.head() # 주석 처리

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

# df.head() # 주석 처리

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

# df[['days_since_oldest_review']].head() # 주석 처리

#room type one-hot encoding
df['reviews_per_month'].fillna(0, inplace=True)
df=pd.get_dummies(df,columns=['room_type'])
# df.head() # 주석 처리

df.drop(columns=['latitude', 'longitude'], inplace=True)
# df # 주석 처리

# Columns to scale (all remaining columns in df now)
features_to_scale = df.columns.tolist() # 스케일링할 피처 목록

# StandardScaler apply
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[features_to_scale].values)

# Rebuild to DataFrame
df_std = pd.DataFrame(scaled_array, columns=features_to_scale, index=df.index)

# checking result
print(df_std.head())

from sklearn.cluster import KMeans

df_standard = df_std.copy() # 원본 df_std를 유지하기 위해 copy() 사용
X_for_clustering = df_standard # 클러스터링을 위한 데이터

sse = []  # Sum of Squared Errors

# Perform KMeans by changing the K value from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_for_clustering)
    sse.append(kmeans.inertia_)  # inertia_ == SSE

# present graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

# Use silhouette analysis to search for optimal K values
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto') # n_init='auto' 추가
    kmeans.fit(X_for_clustering)
    score = silhouette_score(X_for_clustering, kmeans.labels_)
    silhouette_scores.append(score)

# present graph
plt.figure(figsize=(8, 5)) # 새로운 figure 생성
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# Cluster with K=3
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto') # n_init='auto' 추가
df_standard['cluster'] = kmeans.fit_predict(X_for_clustering) # 클러스터 라벨 추가

cluster_means = df_standard.groupby('cluster').mean(numeric_only=True)
print("\n--- Cluster Means ---")
print(cluster_means) # display 대신 print 사용 (VS Code 일반 실행 시)

# import seaborn as sns # 이미 위에서 import 됨

# Create boxplot based on df_scaled with cluster column
features_for_boxplot = df_standard.columns.drop('cluster')  # exept 'cluster'

# present box plot
plt.figure(figsize=(20, 30))
for i, feature in enumerate(features_for_boxplot):
    plt.subplot(len(features_for_boxplot) // 2 + 1, 2, i + 1)
    sns.boxplot(x='cluster', y=feature, data=df_standard)
    plt.title(f'{feature} by Cluster')
    plt.tight_layout()
plt.show() # plt.show() 추가

from sklearn.decomposition import PCA

# PCA 2 deminsion visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_standard.drop('cluster', axis=1))

# present
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_standard['cluster'], cmap='viridis')
plt.title('PCA Visualization of Clusters (K=3)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show() # plt.show() 추가

# -----------------------/// Modeling ///-----------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import randint

print("\n--- Random Forest Classifier with RandomizedSearchCV ---")

# seperate feature/target
X = df_standard.drop('cluster', axis=1)
y = df_standard['cluster']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target class distribution:\n{y.value_counts()}")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain set shape for RandomizedSearchCV: {X_train.shape}, {y_train.shape}")
print(f"Test set shape for final evaluation: {X_test.shape}, {y_test.shape}")


# initializing model
rf_model = RandomForestClassifier(random_state=42)

# 3. define hyperparameter exploration range
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_features': ['sqrt', 'log2', None],
    'max_depth': randint(10, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

# generating RandomizedSearchCV object
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=skf,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# training model
print("\nStarting RandomizedSearchCV...")
random_search.fit(X_train, y_train)
print("RandomizedSearchCV finished.")

# 6. find best model
print("\n--- Best Parameters and Score ---")
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(random_search.best_score_))

best_rf_model = random_search.best_estimator_

print("\n--- Final Evaluation on Test Set with Best Model ---")
y_pred_final = best_rf_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_final))

print("\n--- Feature Importances from Best Random Forest Model ---")
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)
print(sorted_importances.head(10))

# visualizing feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances.head(10).values, y=sorted_importances.head(10).index)
plt.title('Top 10 Feature Importances for Cluster Prediction (Best Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- ROC Curve Plotting ---
print("\n--- ROC Curve (One-vs-Rest) ---")

# 1. Get prediction probabilities
y_proba = best_rf_model.predict_proba(X_test)

# 2. Binarize the true labels for multi-class ROC
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)

# 3. Plot ROC curve for each class
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green'] # Assuming 3 classes (0, 1, 2)

for i, class_label in enumerate(lb.classes_):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'ROC curve for class {class_label} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
