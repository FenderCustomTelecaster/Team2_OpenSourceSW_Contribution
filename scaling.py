import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

df = pd.read_csv("AB_NYC_2019.csv")

df = pd.read_csv(csv_path, encoding='cp949')

df.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood',
                 'latitude', 'longitude'], inplace=True, errors='ignore')

df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=False)

df['reviews_per_month'].fillna(0, inplace=True)

# 리뷰 날짜 처리
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    ref_date = pd.to_datetime("2019-12-01")
    df['days_since_last_review'] = (ref_date - df['last_review']).dt.days
    df['days_since_last_review'].fillna(df['days_since_last_review'].max() + 30, inplace=True)
    df.drop(columns=['last_review'], inplace=True)

# 수치형 컬럼만 추출
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 스케일러 정의
scalers = {
    'zscore': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

scaled_dfs = {}
for name, scaler in scalers.items():
    scaled_array = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols)
    scaled_dfs[name] = scaled_df
    print(f"\n {name.upper()} Scaling 결과:")
    print(scaled_df.head())
