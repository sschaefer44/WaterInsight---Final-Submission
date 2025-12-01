import database
import pandas as pd
import numpy as np
import loadData
import time

def temporalFeatures(df):
    """Create features based on time"""

    # derive date features from dt column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayOfYear'] = df['date'].dt.dayofyear
    df['dayOfWeek'] = df['date'].dt.dayofweek
    df['weekOfYear'] = df['date'].dt.isocalendar().week

    # Seasonal features (metorological seasons)
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    })

    # Cycle encoding
    # The purpose of this is so that the model understands months (e.g) 1 and 12 are next to each other. If month was just stored as an integer,
    # the model wouldn't learn that 1 and 12 are close, 11 and 1 are close, etc
    df['dayOfYearSine'] = np.sin(2 * np.pi * df['dayOfYear'] /365.25)
    df['dayOfYearCosine'] = np.cos(2 * np.pi * df['dayOfYear'] / 365.25)
    df['monthSine'] = np.sin(2 * np.pi * df['month'] / 12)
    df['monthCosine'] = np.cos(2 * np.pi * df['month'] / 12)

    print(f"    Added Temporal Features")
    return df

def lagFeatures(df):
    """Create lagged features by site"""

    lagVals = [1, 7, 14, 30] 

    for lagVal in lagVals:
        df[f'gageHeightLag{lagVal}'] = df.groupby('site_code')['gage_height'].shift(lagVal)

    print(f" Added {len(lagVals)} lagged features")
    return df

def rollingFeatures(df):
    """Create rolling window statistics by site"""

    windows = [7, 14, 30]

    for window in windows:
        
        df[f'gageHeightRollingMean{window}'] = (
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).mean())
        )

        df[f'gageHeightRollingStd{window}'] = (
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).std())
        )

        df[f'gageHeightRollingMin{window}'] = (
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).min())
        )
        
        df[f'gageHeightRollingMax{window}'] = (
            df.groupby('site_code')['gage_height'].transform(lambda x: x.rolling(window, min_periods = 1).max())
        )

    print(f"   Added {len(windows) * 4} rolling features.")
    return df

def calculateTrend(series):
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    y = series.values
    if np.any(np.isnan(y)):
        return 0
    try:
        slope = np.polyfit(x, y, 1)[0]
        return slope
    except:
        return 0

def trendFeatures(df):
    """Create trend and delta features"""

    df['gageHeightDelta1D'] = df.groupby('site_code')['gage_height'].diff(1)
    df['gageHeightDelta7D'] = df.groupby('site_code')['gage_height'].diff(7)
    df['gageHeightDelta14D'] = df.groupby('site_code')['gage_height'].diff(14)
    df['gageHeightDelta30D'] = df.groupby('site_code')['gage_height'].diff(30)

    df['gageHeightTrend7D'] = df.groupby('site_code')['gage_height'].transform(
        lambda x: x.rolling(7, min_periods = 2).apply(calculateTrend, raw= False)
    )

    print("Added trend features")
    return df

def climatologyFeatures(df, train_df=None): 
    """ Create climatology features """
    
    clim_data = train_df if train_df is not None else df
        
    climatologyGageHeight = clim_data.groupby(['site_code', 'dayOfYear'])['gage_height'].agg([
        ('gageHClimateMean', 'mean'),
        ('gageHClimateStd', 'std'),
        ('gageHClimateMin', 'min'),
        ('gageHClimateMax', 'max'),
        ('gageHClimateQuant25', lambda x: x.quantile(0.25)),
        ('gageHClimateQuant75', lambda x: x.quantile(0.75))
    ]).reset_index()

    df = df.merge(climatologyGageHeight, on = ['site_code', 'dayOfYear'], how = 'left')
    
    df['gageHeightAnomaly'] = df['gage_height'] - df['gageHClimateMean']
    df['gageHeightZScore'] = (df['gage_height'] - df['gageHClimateMean']) / (df['gageHClimateStd'] + 0.000001)

    print("     Added 8 climatology features")
    return df

def siteFeatures(df, train_df=None):
    """ Create site based features """
    
    stats_data = train_df if train_df is not None else df

    siteStats = stats_data.groupby('site_code').agg({
        'gage_height': ['mean', 'std', 'min', 'max'],
        'stream_elevation': 'first',
        'latitude': 'first',
        'longitude': 'first'
    })

    siteStats.columns = ['siteGageHeightMean', 'siteGageHeightStd', 'siteGageHeightMin', 'siteGageHeightMax',
                       'siteElevation', 'siteLatitude', 'siteLongitude']
    
    siteStats = siteStats.reset_index()

    df = df.merge(siteStats, on = 'site_code', how = 'left')

    monthlyGageHeight = stats_data.groupby(['site_code', 'month'])['gage_height'].mean().reset_index()
    seasonalityGageHeight = monthlyGageHeight.groupby('site_code')['gage_height'].std().reset_index()
    seasonalityGageHeight.columns = ['site_code', 'siteGageHeightSeasonality']
    df = df.merge(seasonalityGageHeight, on = 'site_code', how = 'left')

    df['latitudeSine'] = np.sin(np.radians(df['siteLatitude']))
    df['latitudeCosine'] = np.cos(np.radians(df['siteLatitude']))
    df['longitudeSine'] = np.sin(np.radians(df['siteLongitude']))
    df['longitudeCosine'] = np.cos(np.radians(df['siteLongitude']))

    centroidLatitude = df['siteLatitude'].mean()
    centroidLongitude = df['siteLongitude'].mean()
    print(f"    Sites Centroid: {centroidLatitude:.2f}°N, {centroidLongitude:.2f}°W")

    df['distanceFromCentroid'] = np.sqrt(
        (df['siteLatitude'] - centroidLatitude) **2 +
        (df['siteLongitude'] - centroidLongitude) **2
    )

    print(f"    Added by-site features")
    return df

def crossVarFeatures(df):
    df['temperatureLag1'] = df.groupby('site_code')['temperature'].shift(1)
    df['temperatureRolling7'] = df.groupby('site_code')['temperature'].transform(
        lambda x: x.rolling(7, min_periods = 1).mean()
    )
    df['tempAboveFreezing'] = (df['temperature'] > 0).astype(float)

    print("     Added cross variable features")

    return df

def categoricalEncoding(df, train_df=None):
    """ Encode sites by average GAGE HEIGHT """
    
    encoding_data = train_df if train_df is not None else df
    
    siteEncoding = encoding_data.groupby('site_code')['gage_height'].mean().to_dict()
    df['encodedSiteCode'] = df['site_code'].map(siteEncoding)

    print("     Added encoded feature")
    return df