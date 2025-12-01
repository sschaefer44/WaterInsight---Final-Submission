from featureEngineering import *
import loadData
import pandas as pd 
import time

def createFeatures(df, train_df=None):
    # The function that creates all the features -> utilizing functions lower in the file

    print("Feature Engineering Pipeline")
    print(f"Initial DF Shape: {df.shape}\n")

    df = df.sort_values(['site_code', 'date']).reset_index(drop = True)

    df = temporalFeatures(df)
    df = lagFeatures(df)
    df = rollingFeatures(df)
    df = trendFeatures(df)
    df = climatologyFeatures(df, train_df=train_df)
    df = siteFeatures(df, train_df=train_df)
    df = crossVarFeatures(df)
    df = categoricalEncoding(df, train_df=train_df)    

    print(f"Final Shape: {df.shape}\n")
    print(f"Features Added: {df.shape[1] - 10}")

    return df

def getFeatureCols(df):
    # Identifies feature Cols

    excludedCols = ['site_code', 'date', 'discharge', 'temperature', 
                    'dissolved_oxygen', 'latitude', 'longitude', 'state']
    
    featureCols = [col for col in df.columns if col not in excludedCols]
    return featureCols

def cleanMissingFeatures(df, featureCols):
    # Handles missing features

    print("Handling Missing Feature Values")


    missing = df[featureCols].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending = False)

    if len(missing) > 0:
        print(f"\nFeatures with missing Values:")
        print(missing.head(15))

        criticalLagFeatures = ['gageHeightLag1', 'gageHeightLag7']

        print(f"\nRows before dropping NaN lag features: {len(df):,}")
        cleanedDf = df.dropna(subset = criticalLagFeatures)
        print(f"Rows after dropping NaN lag features: {len(cleanedDf):,}")
        print(f"Dropped: {len(df) - len(cleanedDf):,} rows ({(len(df) - len(cleanedDf))/len(df)*100:.2f}%)")
        
        cleanedDf[featureCols] = cleanedDf[featureCols].fillna(0)

        remainingNaN = cleanedDf[featureCols].isnull().sum().sum()
        print(f"Remaining NaN: {remainingNaN}")
        
        return cleanedDf
    else:
        print("No missing values found!")
        return df
    
def timeSeriesTrainTestSplit(df, targetCol):
    # Split function for train test val sets. Cant use scikit-learn TrainTestSplit() for timeseries data

    print("Splitting into Train/Val/Test sets")
    trainDF = df[df['year'] <= 2022].copy()
    valDF = df[df['year'] == 2023].copy()
    testDF = df[df['year'] == 2024].copy()

    print(f"\nTrain (2016-2022): {len(trainDF):,} rows ({len(trainDF)/len(df)*100:.1f}%)")
    print(f"Validation (2023): {len(valDF):,} rows ({len(valDF)/len(df)*100:.1f}%)")
    print(f"Test (2024):       {len(testDF):,} rows ({len(testDF)/len(df)*100:.1f}%)")

    print(f"\nTarget ({targetCol}) Statistics")
    print(f"  Train: mean={trainDF[targetCol].mean():.2f}, std={trainDF[targetCol].std():.2f}")
    print(f"  Val:   mean={valDF[targetCol].mean():.2f}, std={valDF[targetCol].std():.2f}")
    print(f"  Test:  mean={testDF[targetCol].mean():.2f}, std={testDF[targetCol].std():.2f}")
    
    return trainDF, valDF, testDF

def saveFeatureData(df, featureCols, filename = '/Users/sschaefer/Desktop/WaterInsight/CSV Backups/engineeredFeatures.csv'):
    # Saves engineered features to .csv file

    print(f"\nSaving to {filename}")

    df.to_csv(filename, index = False)

    with open('featureColumns.txt', 'w') as f:
        for col in featureCols:
            f.write(f"{col}\n")
    
    print(f"Saved {len(featureCols)} feature names to featureColumns.txt")

def summarizeFeatures(feature_cols):
    # Feature Summary
    print("FEATURE SUMMARY")

    temporal = [c for c in feature_cols if any(t in c for t in ['year', 'month', 'day', 'season', 'week'])]
    lagged = [c for c in feature_cols if 'lag' in c or 'Lag' in c]
    rolling = [c for c in feature_cols if 'rolling' in c or 'Rolling' in c]
    trend = [c for c in feature_cols if 'trend' in c or 'Trend' in c or 'Delta' in c]
    climatology = [c for c in feature_cols if 'climate' in c or 'Climate' in c or 'anomaly' in c or 'Anomaly' in c or 'ZScore' in c]
    site = [c for c in feature_cols if 'site' in c or 'latitude' in c or 'longitude' in c or 'distance' in c or 'elevation' in c or 'seasonality' in c or 'Seasonality' in c]
    cross = [c for c in feature_cols if 'ratio' in c or 'temp' in c or 'Temp' in c or 'freezing' in c or 'Freezing' in c]
    encoded = [c for c in feature_cols if 'encoded' in c or 'Encoded' in c]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"  Temporal:     {len(temporal)}")
    print(f"  Lagged:       {len(lagged)}")
    print(f"  Rolling:      {len(rolling)}")
    print(f"  Trend:        {len(trend)}")
    print(f"  Climatology:  {len(climatology)}")
    print(f"  Site:         {len(site)}")
    print(f"  Cross-var:    {len(cross)}")
    print(f"  Encoded:      {len(encoded)}")

def balancedTimeSeriesSplit(df, targetCol):
    # Different split function that balances years. 15% of each year (2016-2023) is held out for validation. Entire 2024 is test set
    
    if 'year' not in df.columns:
        raise ValueError("DataFrame must have 'year' column")
    
    print("Splitting into Train/Val/Test sets (Balanced)")
    
    testDF = df[df['year'] == 2024].copy()
    
    trainValDF = df[df['year'] < 2024].copy()
    
    valDFs = []
    trainDFs = []
    
    for year in sorted(trainValDF['year'].unique()):
        yearDF = trainValDF[trainValDF['year'] == year]
        valSize = int(len(yearDF) * 0.15)
        
        valYear = yearDF.sample(n=valSize, random_state=42)
        trainYear = yearDF.drop(valYear.index)
        
        valDFs.append(valYear)
        trainDFs.append(trainYear)
    
    trainDF = pd.concat(trainDFs).reset_index(drop=True)
    valDF = pd.concat(valDFs).reset_index(drop=True)
    testDF = testDF.reset_index(drop=True)
    
    print(f"\nTrain (2016-2023, 85% each year): {len(trainDF):,} rows ({len(trainDF)/len(df)*100:.1f}%)")
    print(f"Validation (15% from each year):  {len(valDF):,} rows ({len(valDF)/len(df)*100:.1f}%)")
    print(f"Test (2024):                       {len(testDF):,} rows ({len(testDF)/len(df)*100:.1f}%)")
    
    print(f"\nTarget ({targetCol}) Statistics")
    print(f"  Train: mean={trainDF[targetCol].mean():.2f}, std={trainDF[targetCol].std():.2f}")
    print(f"  Val:   mean={valDF[targetCol].mean():.2f}, std={valDF[targetCol].std():.2f}")
    print(f"  Test:  mean={testDF[targetCol].mean():.2f}, std={testDF[targetCol].std():.2f}")
    
    return trainDF, valDF, testDF

# MAIN -> full workflow

if __name__ == "__main__":
    startTime = time.time()

    print("Water Discharge Feature Engineering")

    print("\nStep 1: Loading raw data")
    df = loadData.loadData()
    
    print("\nStep 2: Splitting data with balanced sampling (BEFORE feature engineering)")
    
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    # USING BALANCED SPLIT
    trainDf, valDf, testDf = balancedTimeSeriesSplit(df, 'discharge')
    
    print("\nStep 3: Engineering features")
    
    print("\n>>> Processing TRAIN set:")
    trainDfEngineered = createFeatures(trainDf)
    
    print("\n>>> Processing VALIDATION set (using train statistics):")
    valDfEngineered = createFeatures(valDf, train_df=trainDfEngineered)
    
    print("\n>>> Processing TEST set (using train statistics):")
    testDfEngineered = createFeatures(testDf, train_df=trainDfEngineered)
    
    featureCols = getFeatureCols(trainDfEngineered)
    
    print("\n>>> Cleaning TRAIN set:")
    trainCleaned = cleanMissingFeatures(trainDfEngineered, featureCols)
    
    print("\n>>> Cleaning VALIDATION set:")
    valCleaned = cleanMissingFeatures(valDfEngineered, featureCols)
    
    print("\n>>> Cleaning TEST set:")
    testCleaned = cleanMissingFeatures(testDfEngineered, featureCols)
    
    trainCleaned.to_csv('CSV Backups/train_features.csv', index=False)
    valCleaned.to_csv('CSV Backups/val_features.csv', index=False)
    testCleaned.to_csv('CSV Backups/test_features.csv', index=False)
    
    summarizeFeatures(featureCols)
    print("FINAL DATASET SIZES")
    print(f"Train:      {len(trainCleaned):,} rows")
    print(f"Validation: {len(valCleaned):,} rows")
    print(f"Test:       {len(testCleaned):,} rows")

    elapsed = time.time() - startTime

    print(f"\nTotal time: {elapsed/60:.2f} minutes")