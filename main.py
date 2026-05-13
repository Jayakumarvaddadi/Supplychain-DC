import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2

# 1. Haversine distance for real-world accuracy
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

# 2. Load Data (Adjusted for your specific file names)
df = pd.read_excel("saavu2.xlsx")
df.columns = df.columns.str.lower().str.strip()

# 3. Weighted K-Means Logic
X = df[['lat', 'long']].values
weights = df['sales'].values
MAX_DIST = 700

for k in range(1, 21):
    # Cluster stores; sales act as gravity
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X, sample_weight=weights)
    
    temp_df = df.copy()
    temp_df['cluster'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Snap DCs to the nearest real store location
    snapped_centroids = []
    for center in centroids:
        distances = np.sqrt(((X[:, 0] - center[0])**2) + ((X[:, 1] - center[1])**2))
        snapped_centroids.append(X[np.argmin(distances)])
    snapped_centroids = np.array(snapped_centroids)
    
    # Map DC coordinates back to stores
    temp_df['dc_lat'] = temp_df['cluster'].apply(lambda x: snapped_centroids[x][0])
    temp_df['dc_long'] = temp_df['cluster'].apply(lambda x: snapped_centroids[x][1])
    
    # Calculate Distances
    temp_df['distance_km'] = temp_df.apply(
        lambda r: haversine(r['lat'], r['long'], r['dc_lat'], r['dc_long']), axis=1
    )
    
    # Check if this K satisfies your 700km rule
    if temp_df['distance_km'].max() <= MAX_DIST:
        print(f"Optimal Network found with {k} DCs.")
        # Save the specific output you requested
        output_cols = ['store', 'lat', 'long', 'sales', 'cluster', 'dc_lat', 'dc_long', 'distance_km']
        temp_df[output_cols].to_excel("clustered_output.xlsx", index=False)
        break
