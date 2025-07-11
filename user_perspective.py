import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from datetime import datetime

RANDOM_STATE = 42
REFERENCE_DATE = pd.to_datetime("2019-12-01")


#A function of calculating the distance between the location of each 
#accommodation and the location of the city center (central) in the area
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

centers = {
    'Bronx': (40.8448, -73.8648),
    'Brooklyn': (40.6782, -73.9442),
    'Manhattan': (40.7685, -73.9822),
    'Queens': (40.7282, -73.7949),
    'Staten Island': (40.5795, -74.1502),
}

#Analyze the properties of each cluster and label the cluster according 
#to the characteristics of each of the four clusters
cluster_names = {
    0: "A high-end hotel located in the city center",
    1: "A low-cost room outside",
    2: "Accommodation suitable for mid-priced sightseeing",
    3: "A long-term rental-oriented outer-style accommodation",
}

#Preprocessing function
#one-hot encoding for 'neighbourhood', 'room_type', drop few columns
def preprocess(df):
    df = df.copy()
    #Drop all features that do not affect the result
    df.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood'], errors='ignore', inplace=True)

    if 'neighbourhood_group' in df.columns:
        df['distance_to_center'] = df.apply(lambda row: haversine(
            row['latitude'], row['longitude'],
            *centers.get(row['neighbourhood_group'], (0, 0))
        ), axis=1)
        df = pd.get_dummies(df, columns=['neighbourhood_group'])

    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['days_since_oldest_review'] = (REFERENCE_DATE - df['last_review']).dt.days
    
    #Missing value processing: no review → considered older than the oldest value max + 30
    df['days_since_oldest_review'] = df['days_since_oldest_review'].fillna(df['days_since_oldest_review'].max() + 30)
    max_days = df['days_since_oldest_review'].max()
    df['days_since_oldest_review'] = max_days - df['days_since_oldest_review']
    df.drop(columns=['last_review'], inplace=True)

    df['reviews_per_month'].fillna(0, inplace=True)
    df = pd.get_dummies(df, columns=['room_type'])
    df.drop(columns=['latitude', 'longitude'], inplace=True)

    return df

# ========================
# Training Phase
# ========================

df = pd.read_csv("AB_NYC_2019.csv")
df_proc = preprocess(df)

X_for_clustering = df_proc.copy()  
X_for_clustering.drop(columns='cluster', inplace=True, errors='ignore')  

sse = []  # Sum of Squared Errors

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_proc), columns=df_proc.columns)

#
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

# Use silhouette analysis to search for optimal K values
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X_for_clustering)
    score = silhouette_score(X_for_clustering, kmeans.labels_)
    silhouette_scores.append(score)

# present graph
plt.figure(figsize=(8, 5)) 
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
#


kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

X = df_scaled.drop('cluster', axis=1)
y = df_scaled['cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Tuned RandomForestClassifier(low depth)
# Lower depth because of OVERFITTING
# Hpyerparameters are set by RandomCV
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    oob_score=True,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#cross-validation accuracy(to check that high OOB score means overfitting, or great score of model)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", cv_scores)
print("\nMean Cross-Validation Accuracy:", np.mean(cv_scores))

print("\nCLssification report",classification_report(y_test, y_pred))

# ========================
# Visualization
# ========================

def visualize_results(X, y, y_pred, model):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    axs[0, 0].set_title("PCA: Clusters")
    axs[0, 0].set_xlabel("PC 1")
    axs[0, 0].set_ylabel("PC 2")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1])
    axs[0, 1].set_title("Confusion Matrix")
    axs[0, 1].set_xlabel("Predicted")
    axs[0, 1].set_ylabel("Actual")

    # Feature importances
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feat_imp[:10], y=feat_imp[:10].index, ax=axs[1, 0])
    axs[1, 0].set_title("Top 10 Feature Importances")

    # OOB vs Test Accuracy
    acc = accuracy_score(y, y_pred)
    axs[1, 1].bar(["OOB Score", "Test Accuracy"], [model.oob_score_, acc], color=["orange", "blue"])
    axs[1, 1].set_ylim(0.5, 1.0)
    axs[1, 1].set_title("Overfitting Check: OOB vs Test Accuracy")

    plt.tight_layout()
    plt.show()

visualize_results(X_test, y_test, y_pred, rf)

# ========================
# Prediction Interface
# ========================

def predict_cluster_from_csv(csv_path):
    new_df = pd.read_csv(csv_path)
    new_df = new_df.dropna(subset=["latitude", "longitude"])
    coords = new_df[['latitude', 'longitude']].copy()
    proc_new = preprocess(new_df)
    proc_new = proc_new.reindex(columns=X.columns, fill_value=0)
    proc_new_scaled = scaler.transform(proc_new)
    cluster_pred = rf.predict(proc_new_scaled)
    new_df['predicted_cluster'] = cluster_pred
    new_df['cluster_name'] = new_df['predicted_cluster'].map(cluster_names)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, row in new_df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=colors[row['predicted_cluster'] % len(colors)],
                fill=True,
                fill_opacity=0.6,
                popup=f"{row['cluster_name']}"
            ).add_to(m)
    m.save("cluster_map.html")

    return new_df[['latitude', 'longitude', 'predicted_cluster', 'cluster_name']]


clusters = predict_cluster_from_csv("sample_new_airbnb_data.csv")
print(clusters)