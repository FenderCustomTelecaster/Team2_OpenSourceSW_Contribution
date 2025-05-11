import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# haversine(대원거리) fuction
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df = pd.read_csv("AB_NYC_2019.csv")

df.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood'], inplace=True)

# 리뷰 날짜 차이 계산 (최근일수록 작은 값)
reference_date = pd.to_datetime("2019-12-01")
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['days_since_last_review'] = (reference_date - df['last_review']).dt.days

# 결측치 처리: 리뷰 없음 → 가장 오래된 값보다 더 오래된 것으로 간주 max + 30
temp_days = df['days_since_last_review'].copy()
df['days_since_last_review'] = temp_days.fillna(temp_days.max() + 30)

# 최근일수록 큰 값이 되도록 변환
max_days = df['days_since_last_review'].max()
df['days_since_last_review'] = max_days - df['days_since_last_review']
df.drop(columns=['last_review'], inplace=True)

df['reviews_per_month'].fillna(0, inplace=True)

# 2) pandas map으로 수동 매핑하기
# 매핑 순서는 원하는 대로 정의 가능해
group_map = {
    'Bronx': 0,
    'Brooklyn': 1,
    'Manhattan': 2,
    'Queens': 3,
    'Staten Island': 4
}
room_map = {
    'Entire home/apt': 0,
    'Private room': 1,
    'Shared room': 2
}

df['neighbourhood_group'] = df['neighbourhood_group'].map(group_map)
df['room_type']          = df['room_type'].map(room_map)

# 1) 아까 정의했던 group_map 과 똑같은 순서로 centers_ord 정의
centers_ord = {
    0: (40.8448, -73.8648),  # Bronx
    1: (40.6782, -73.9442),  # Brooklyn
    2: (40.7685, -73.9822),  # Manhattan
    3: (40.7282, -73.7949),  # Queens
    4: (40.5795, -74.1502),  # Staten Island
}

# 2) 각 row 의 neighbourhood_group 값을 기준으로 중심지 좌표 꺼내서 거리 계산
def compute_distance_ord(row):
    grp = row['neighbourhood_group']
    if np.isnan(grp):
        return np.nan
    clat, clon = centers_ord[int(grp)]
    return haversine(row['latitude'], row['longitude'], clat, clon)

df['distance_to_center'] = df.apply(compute_distance_ord, axis=1)

df.drop(columns=['latitude', 'longitude'], inplace=True)


# 스케일링할 컬럼들 (지금 df에 남아있는 모든 컬럼)
features = df.columns.tolist()

# 스케일러 정의
scalers = {
    'standard': StandardScaler(),
    'minmax':   MinMaxScaler(),
    'robust':   RobustScaler()
}

# 결과를 담을 dict
scaled_dfs = {}

for name, scaler in scalers.items():
    # fit_transform 결과를 DataFrame으로 만들기
    scaled_array = scaler.fit_transform(df.values)
    scaled_dfs[name] = pd.DataFrame(scaled_array, columns=features, index=df.index)

# 각각 꺼내 쓰기
df_std = scaled_dfs['standard']  # 평균0, 분산1 스케일
df_mm  = scaled_dfs['minmax']    # 0~1 스케일
df_rb  = scaled_dfs['robust']    # 중앙값0, IQR기반 스케일


print(df_std.head())
print(df_mm.head())
print(df_rb.head())


# 시각화 (박스플롯)
# Standard
plt.figure()
plt.boxplot(df_std.values)
plt.xticks(range(1, len(features)+1), features, rotation=45, ha='right')
plt.title('Standard Scaler')
plt.tight_layout()
plt.show()

# MinMax
plt.figure()
plt.boxplot(df_mm.values)
plt.xticks(range(1, len(features)+1), features, rotation=45, ha='right')
plt.title('MinMax Scaler')
plt.tight_layout()
plt.show()

# Robust
plt.figure()
plt.boxplot(df_rb.values)
plt.xticks(range(1, len(features)+1), features, rotation=45, ha='right')
plt.title('Robust Scaler')
plt.tight_layout()
plt.show()