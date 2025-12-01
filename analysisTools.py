import pandas as pd
import numpy as np
from scipy import stats

def detect_anomalies(df, metric='discharge', percentile=95):
    # Detect anomalies using percintile method - yields better analysis than the Z-score method mentioned in my proposal
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    def flag_outliers_percentile(group):
        upper_threshold = group[metric].quantile(percentile / 100)
        lower_threshold = group[metric].quantile((100 - percentile) / 100)
        
        group['is_anomaly'] = (group[metric] > upper_threshold) | (group[metric] < lower_threshold)
        group['anomaly_type'] = None
        group.loc[group[metric] > upper_threshold, 'anomaly_type'] = 'high'
        group.loc[group[metric] < lower_threshold, 'anomaly_type'] = 'low'
        
        return group
    
    df = df.groupby('month', group_keys=False).apply(flag_outliers_percentile, include_groups=False)
    
    return df

def find_significant_trends(df, metric='discharge', window=30, min_r2=0.7, min_slope_pct=5):
    # Identify trends by comparing slopes (of LR) within specified windoes
     
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df_clean = df[[metric, 'date']].dropna()
    
    if len(df_clean) < window:
        return []
    
    trends = []
    
    for i in range(len(df_clean) - window + 1):
        window_df = df_clean.iloc[i:i+window].copy()
        
        if len(window_df) < window * 0.8:  
            continue
        
        window_df['days'] = (window_df['date'] - window_df['date'].min()).dt.days
        
        x = window_df['days'].values
        y = window_df[metric].values
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            start_val = y[0]
            end_val = y[-1]
            pct_change = abs((end_val - start_val) / start_val * 100) if start_val != 0 else 0
            
            if r_squared >= min_r2 and pct_change >= min_slope_pct and p_value < 0.05:
                
                overlaps = False
                for existing in trends:
                    if (window_df['date'].min() <= existing['end_date'] and 
                        window_df['date'].max() >= existing['start_date']):
                        if r_squared > existing['r_squared']:
                            trends.remove(existing)
                        else:
                            overlaps = True
                            break
                
                if not overlaps:
                    trend_line = pd.DataFrame({
                        'date': window_df['date'],
                        'trend_value': slope * x + intercept
                    })
                    
                    trends.append({
                        'start_date': window_df['date'].min(),
                        'end_date': window_df['date'].max(),
                        'slope': slope,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'pct_change': pct_change,
                        'trend_line': trend_line
                    })
        except:
            continue
    
    trends = sorted(trends, key=lambda x: x['r_squared'], reverse=True)
    
    return trends[:5]

def calculate_overall_trend(df, metric='discharge'):
    # Compute trend with linear regression
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df_clean = df[[metric, 'date']].dropna()
    
    if len(df_clean) < 2:
        return None
    
    df_clean = df_clean.sort_values('date')
    df_clean['days'] = (df_clean['date'] - df_clean['date'].min()).dt.days
    
    x = df_clean['days'].values
    y = df_clean[metric].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    trend_line = pd.DataFrame({
        'date': df_clean['date'],
        'trend_value': slope * x + intercept
    })
    
    if p_value > 0.05: 
        trend_direction = 'stable'
    elif slope > 0:
        trend_direction = 'increasing'
    else:
        trend_direction = 'decreasing'
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'trend_line': trend_line,
        'trend_direction': trend_direction,
        'std_err': std_err
    }

def cluster_anomalies(df, metric='discharge'):
    # Cluster consecutive anomalies
    # This is important because lets say that 1/1/2024 is abnormally low. It is reasonable to assume that 1/2/2024-1/?/2024 would also be low
    # Water metrics change gradually. This prevents multiple anomalies from essentially corresponding to the same event
    # ONLY KEEPS MOST EXTREME VALUE/VALUE DATE IN CLUSTER
    
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        return anomalies
    
    anomalies = anomalies.sort_values('date').reset_index(drop=True)
    
    anomalies['days_diff'] = anomalies['date'].diff().dt.days
    anomalies['new_cluster'] = (anomalies['days_diff'].isna()) | (anomalies['days_diff'] > 5)
    anomalies['cluster_id'] = anomalies['new_cluster'].cumsum()
    
    peak_anomalies = []
    for cluster_id in anomalies['cluster_id'].unique():
        cluster = anomalies[anomalies['cluster_id'] == cluster_id]
        
        if cluster['anomaly_type'].iloc[0] == 'high':
            peak_idx = cluster[metric].idxmax()
        else:
            peak_idx = cluster[metric].idxmin()
        
        peak_anomalies.append(cluster.loc[peak_idx])
    
    return pd.DataFrame(peak_anomalies)

def analyze_dataset(df, metric='discharge', anomaly_percentile=95):
    # Run analysis functions on data
    
    df_with_anomalies = detect_anomalies(df, metric, anomaly_percentile)
    
    clustered_anomalies = cluster_anomalies(df_with_anomalies, metric)

    df_agg = df.groupby('date')[metric].mean().reset_index()
    trend_results = find_significant_trends(df_agg, metric)
    
    anomaly_count = len(clustered_anomalies)
    high_anomalies = (clustered_anomalies['anomaly_type'] == 'high').sum() if len(clustered_anomalies) > 0 else 0
    low_anomalies = (clustered_anomalies['anomaly_type'] == 'low').sum() if len(clustered_anomalies) > 0 else 0
    
    summary_stats = {
        'total_records': len(df),
        'anomaly_count': int(anomaly_count),
        'high_anomalies': int(high_anomalies),
        'low_anomalies': int(low_anomalies),
        'anomaly_percentage': (anomaly_count / len(df) * 100) if len(df) > 0 else 0,
        'trend_count': len(trend_results)
    }
    
    return clustered_anomalies, trend_results, summary_stats
