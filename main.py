import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# 1. CORE UTILITIES
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

# =====================================================
# 2. DATA LOADING & CLEANING
# =====================================================

def load_data():
    # Use the filenames as they appear in your folder
    df = pd.read_excel("saavu2.xlsx")
    
    # Cleaning column names to be easier to work with
    df.columns = df.columns.str.lower().str.strip()
    
    # We rename the demand column for easier coding
    if 'order demand ( in cft)' in df.columns:
        df = df.rename(columns={'order demand ( in cft)': 'demand_cft'})
        
    print(f"Loaded {len(df)} stores.")
    return df

# =====================================================
# 3. PHASE 1: WEIGHTED K-MEANS (DC LOCATION)
# =====================================================

def find_dc_network(df, max_dist_km=700):
    X = df[['lat', 'long']].values
    weights = df['sales'].values
    
    best_k = None
    best_df = None
    best_centroids = None

    # We try increasing K until all stores are within 700km of their DC
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X, sample_weight=weights)
        
        temp_df = df.copy()
        temp_df['cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # Snap Centroids to the nearest real store
        real_centroids = []
        for center in centroids:
            # Find closest real store location to the mathematical center
            distances = np.sqrt(((X[:, 0] - center[0])**2) + ((X[:, 1] - center[1])**2))
            real_centroids.append(X[np.argmin(distances)])
        
        snapped_centroids = np.array(real_centroids)
        
        # Assign DC coordinates to the stores
        temp_df['dc_lat'] = temp_df['cluster'].apply(lambda x: snapped_centroids[x][0])
        temp_df['dc_long'] = temp_df['cluster'].apply(lambda x: snapped_centroids[x][1])
        
        # Calculate Haversine distance for each store to its DC
        temp_df['dist_to_dc'] = temp_df.apply(
            lambda r: haversine(r['lat'], r['long'], r['dc_lat'], r['dc_long']), axis=1
        )
        
        if temp_df['dist_to_dc'].max() <= max_dist_km:
            print(f"Success! Found {k} DCs where max distance is {temp_df['dist_to_dc'].max():.2f} km")
            return temp_df, snapped_centroids
            
    return None, None

# Run the logic
store_df = load_data()
clustered_df, dc_coords = find_dc_network(store_df)

# Save result to verify
if clustered_df is not None:
    clustered_df.to_excel("clustered_output.xlsx", index=False)
    print("Clustering complete. 'clustered_output.xlsx' created.")
