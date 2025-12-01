import pandas as pd
import database
from sqlalchemy import text

STATE_ABBREV_MAP = {
    'Minnesota': 'MN',
    'Wisconsin': 'WI',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Michigan': 'MI'
}

def _convert_states_to_abbrev(states):
    if not states:
        return None
    return [STATE_ABBREV_MAP.get(state, state) for state in states]

def loadHistoricalData(startDate, endDate, sites, states):
    print(f"\n--- loadHistoricalData called ---")
    print(f"startDate: {startDate}")
    print(f"endDate: {endDate}")
    print(f"sites: {sites}")
    print(f"states: {states}")
    
    engine = database.getSQLAlchemyEngine()

    query = """
        SELECT 
            w.site_code,
            w.date,
            w.discharge,
            w.gage_height,
            l.state,
            l.latitude,
            l.longitude,
            l.site_name
        FROM waterdata w
        LEFT JOIN sitelocations l on w.site_code = l.site_code
        WHERE w.discharge IS NOT NULL    
    """
    
    parameters = {}

    if startDate:
        query += " AND w.date >= :startDate"
        parameters['startDate'] = startDate.split('T')[0] if isinstance(startDate, str) else startDate
    if endDate:
        query += " AND w.date <= :endDate"
        parameters['endDate'] = endDate.split('T')[0] if isinstance(endDate, str) else endDate
    if sites:
        query += " AND w.site_code IN :sites"
        parameters['sites'] = tuple(sites)
        print(f"Filtering by sites: {sites}")
    elif states:
        state_abbrevs = _convert_states_to_abbrev(states)
        query += " AND l.state IN :states"
        parameters['states'] = tuple(state_abbrevs)
        print(f"Converted states {states} to abbreviations {state_abbrevs}")
    
    query += " ORDER BY w.date, w.site_code"
    
    print(f"\nFinal query:\n{query}")
    print(f"Parameters: {parameters}")

    df = pd.read_sql(text(query), engine, params=parameters)
    print(f"Query returned {len(df)} rows")
    
    df['date'] = pd.to_datetime(df['date'])
    df['data_type'] = 'historical'

    return df

def loadPredictions(startDate, endDate, sites, states):
    print(f"\n--- loadPredictions called ---")
    print(f"sites: {sites}")
    print(f"states: {states}")
    
    columnsToKeep = ['site_code', 'date', 'state', 'latitude', 'longitude', 'predicted_discharge']
    df = pd.read_csv('c2Model/2024_predictions.csv', usecols=columnsToKeep)
    print(f"Loaded {len(df)} rows from CSV")
    
    df['site_code'] = df['site_code'].astype(str).str.zfill(8)
    print(f"Converted site_codes to string format")
    print(f"Sample site_codes from predictions: {df['site_code'].unique()[:5]}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'predicted_discharge': 'discharge'})
    df['data_type'] = 'predicted'

    if startDate:
        start_dt = pd.to_datetime(startDate.split('T')[0] if isinstance(startDate, str) else startDate)
        df = df[df['date'] >= start_dt]
        print(f"After start date filter: {len(df)} rows")
    if endDate:
        end_dt = pd.to_datetime(endDate.split('T')[0] if isinstance(endDate, str) else endDate)
        df = df[df['date'] <= end_dt]
        print(f"After end date filter: {len(df)} rows")
    if sites and len(sites) > 0:
        print(f"Filtering predictions by sites: {sites}")
        print(f"Checking if sites exist in predictions...")
        for site in sites:
            count = df[df['site_code'] == site].shape[0]
            print(f"  Site {site}: {count} rows")
        df = df[df['site_code'].isin(sites)]
        print(f"After site filter: {len(df)} rows")
    elif states and len(states) > 0:
        state_abbrevs = _convert_states_to_abbrev(states)
        print(f"Before state filter (converted {states} to {state_abbrevs}): {len(df)} rows")
        df = df[df['state'].isin(state_abbrevs)]
        print(f"After state filter: {len(df)} rows")
    
    print(f"Returning {len(df)} rows")
    return df

def getAvailSites():
    engine = database.getSQLAlchemyEngine()

    query = "SELECT DISTINCT site_code FROM sitelocations ORDER BY site_code"
    sites = pd.read_sql(query, engine)
    return sites['site_code'].to_list()

def getAvailSiteNames():
    # Get tuple of site codes, state
    engine = database.getSQLAlchemyEngine()
    
    query = """
        SELECT DISTINCT l.site_code, l.site_name, l.state
        FROM sitelocations l
        INNER JOIN waterdata w ON l.site_code = w.site_code
        WHERE w.discharge IS NOT NULL
        ORDER BY l.state, l.site_name
    """
    sites = pd.read_sql(query, engine)
    
    return [(row['site_code'], f"{row['site_name']} ({row['state']})") 
            for _, row in sites.iterrows()]

def getAvailStates():
    return ['Minnesota', 'Wisconsin', 'Illinois', 'Indiana', 'Michigan']
