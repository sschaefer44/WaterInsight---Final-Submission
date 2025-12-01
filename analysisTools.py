"""
Anomaly Detection and Trend Analysis Tools
For WaterInsight Dashboard
"""

import pandas as pd
import numpy as np
from scipy import stats

def detect_anomalies(df, metric='discharge', percentile=95):
    """
    Detect anomalies using percentile method (simpler and more robust)
    
    Parameters:
    -----------
    df : DataFrame with columns ['date', metric]
    metric : str - 'discharge' or 'gage_height'
    percentile : float - percentile threshold (default 95 = top/bottom 5%)
    
    Returns:
    --------
    DataFrame with additional columns:
        - 'is_anomaly': boolean flag for anomaly
        - 'anomaly_type': 'high', 'low', or None
    """
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # Calculate seasonal percentiles (by month)
    def flag_outliers_percentile(group):
        # Calculate percentile thresholds for this month
        upper_threshold = group[metric].quantile(percentile / 100)
        lower_threshold = group[metric].quantile((100 - percentile) / 100)
        
        # Flag values beyond thresholds
        group['is_anomaly'] = (group[metric] > upper_threshold) | (group[metric] < lower_threshold)
        group['anomaly_type'] = None
        group.loc[group[metric] > upper_threshold, 'anomaly_type'] = 'high'
        group.loc[group[metric] < lower_threshold, 'anomaly_type'] = 'low'
        
        return group
    
    # Apply percentile method by month for seasonal adjustment
    df = df.groupby('month', group_keys=False).apply(flag_outliers_percentile, include_groups=False)
    
    return df

def find_significant_trends(df, metric='discharge', window=30, min_r2=0.7, min_slope_pct=5):
    """
    Find windows with significant trends (steep slopes)
    
    Parameters:
    -----------
    df : DataFrame with columns ['date', metric]
    metric : str - 'discharge' or 'gage_height'
    window : int - rolling window size in days (default 30)
    min_r2 : float - minimum R² to consider trend significant (default 0.7)
    min_slope_pct : float - minimum percent change to flag (default 5%)
    
    Returns:
    --------
    list of dicts with trend segments:
        - 'start_date': start of trend
        - 'end_date': end of trend
        - 'slope': rate of change
        - 'r_squared': goodness of fit
        - 'direction': 'increasing' or 'decreasing'
        - 'pct_change': percent change over window
        - 'trend_line': DataFrame with dates and values
    """
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove NaN
    df_clean = df[[metric, 'date']].dropna()
    
    if len(df_clean) < window:
        return []
    
    trends = []
    
    # Sliding window to find significant trends
    for i in range(len(df_clean) - window + 1):
        window_df = df_clean.iloc[i:i+window].copy()
        
        # Skip if not enough data
        if len(window_df) < window * 0.8:  # Allow 20% missing data
            continue
        
        # Calculate trend for this window
        window_df['days'] = (window_df['date'] - window_df['date'].min()).dt.days
        
        x = window_df['days'].values
        y = window_df[metric].values
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Calculate percent change
            start_val = y[0]
            end_val = y[-1]
            pct_change = abs((end_val - start_val) / start_val * 100) if start_val != 0 else 0
            
            # Check if trend is significant
            if r_squared >= min_r2 and pct_change >= min_slope_pct and p_value < 0.05:
                
                # Check if this overlaps with existing trends
                overlaps = False
                for existing in trends:
                    if (window_df['date'].min() <= existing['end_date'] and 
                        window_df['date'].max() >= existing['start_date']):
                        # If overlap, keep the one with higher R²
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
    
    # Sort by R² and return top trends
    trends = sorted(trends, key=lambda x: x['r_squared'], reverse=True)
    
    return trends[:5]  # Return top 5 trends

def calculate_overall_trend(df, metric='discharge'):
    """
    Calculate overall linear regression trend for the entire date range
    (Keep for reference, but won't use on dashboard)
    """
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove NaN values
    df_clean = df[[metric, 'date']].dropna()
    
    if len(df_clean) < 2:
        return None
    
    # Convert dates to numeric (days since start)
    df_clean = df_clean.sort_values('date')
    df_clean['days'] = (df_clean['date'] - df_clean['date'].min()).dt.days
    
    # Perform linear regression
    x = df_clean['days'].values
    y = df_clean[metric].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Create trend line for plotting
    trend_line = pd.DataFrame({
        'date': df_clean['date'],
        'trend_value': slope * x + intercept
    })
    
    # Determine trend direction
    if p_value > 0.05:  # Not statistically significant
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
    """
    Cluster consecutive anomalies and keep only the most extreme point in each cluster
    
    Parameters:
    -----------
    df : DataFrame with 'date', metric, and 'is_anomaly' columns
    metric : str - 'discharge' or 'gage_height'
    
    Returns:
    --------
    DataFrame with only peak anomalies (one per event)
    """
    
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        return anomalies
    
    # Sort by date
    anomalies = anomalies.sort_values('date').reset_index(drop=True)
    
    # Identify clusters (consecutive days within 5 days of each other - more aggressive)
    anomalies['days_diff'] = anomalies['date'].diff().dt.days
    anomalies['new_cluster'] = (anomalies['days_diff'].isna()) | (anomalies['days_diff'] > 5)
    anomalies['cluster_id'] = anomalies['new_cluster'].cumsum()
    
    # For each cluster, keep only the most extreme value
    peak_anomalies = []
    for cluster_id in anomalies['cluster_id'].unique():
        cluster = anomalies[anomalies['cluster_id'] == cluster_id]
        
        # Determine if this is a high or low anomaly cluster
        if cluster['anomaly_type'].iloc[0] == 'high':
            # Keep the max value
            peak_idx = cluster[metric].idxmax()
        else:
            # Keep the min value
            peak_idx = cluster[metric].idxmin()
        
        peak_anomalies.append(cluster.loc[peak_idx])
    
    return pd.DataFrame(peak_anomalies)

def analyze_dataset(df, metric='discharge', anomaly_percentile=95):
    """
    Complete analysis: anomalies + significant trends
    
    Parameters:
    -----------
    anomaly_percentile : float - percentile for anomaly detection (95 = top/bottom 5%)
    """
    
    # Detect anomalies using percentile method
    df_with_anomalies = detect_anomalies(df, metric, anomaly_percentile)
    
    # Cluster consecutive anomalies to get one per event
    clustered_anomalies = cluster_anomalies(df_with_anomalies, metric)
    
    # Find significant trend windows
    # Aggregate by date first
    df_agg = df.groupby('date')[metric].mean().reset_index()
    trend_results = find_significant_trends(df_agg, metric)
    
    # Summary statistics
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