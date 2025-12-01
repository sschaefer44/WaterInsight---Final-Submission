from dash import *
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime#, timedelta
import pandas as pd
import numpy as np
import loadDataDashboard
import analysisTools

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

states = loadDataDashboard.getAvailStates()
site_names = loadDataDashboard.getAvailSiteNames()


COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'historical': '#3498db',
    'predicted': '#e74c3c',
    'accent': '#2ecc71',
    'text': '#2c3e50',
    'background': '#f0f2f5',
    'card': '#ffffff',
    'sidebar': '#ffffff',
    'border': '#e0e0e0'
}

SIDEBAR_STYLE = {
    'backgroundColor': COLORS['sidebar'],
    'padding': '20px',
    'height': '100vh',
    'overflowY': 'auto',
    'boxShadow': '2px 0 10px rgba(0,0,0,0.05)',
    'borderRight': f"1px solid {COLORS['border']}"
}

CONTENT_STYLE = {
    'padding': '20px',
    'backgroundColor': COLORS['background']
}

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("WaterInsight Dashboard", 
                       className="text-white mb-0",
                       style={
                           'fontWeight': 'bold',
                           'fontSize': '2.5rem',
                           'textShadow': '2px 2px 4px rgba(0,0,0,0.2)'
                       })
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                'padding': '30px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)'
            })
        ])
    ], className="mb-4 mt-3"),
    
    dbc.Row([
        # Filter Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Filters", className="mb-0", style={'fontWeight': '600', 'color': COLORS['text']}),
                    style={'backgroundColor': '#f8f9fa', 'border': 'none'}
                ),
                dbc.CardBody([
                    html.Label("Filter By:", className="fw-bold mt-2", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='filter-type',
                        options=[
                            {'label': ' State', 'value': 'state'},
                            {'label': ' Site', 'value': 'site'}
                        ],
                        value='state',
                        inline=True,
                        className="mb-3"
                    ),
                    
                    html.Div([
                        html.Label("Select State(s):", className="fw-bold", style={'color': COLORS['text']}),
                        dcc.Dropdown(
                            id='state-dropdown',
                            options=[{'label': state, 'value': state} for state in states],
                            value=['Wisconsin'],
                            multi=True,
                            className="mb-3",
                            style={'borderRadius': '5px'}
                        )
                    ], id='state-filter-div'),
                    
                    html.Div([
                        html.Label("Select Site(s):", className="fw-bold", style={'color': COLORS['text']}),
                        dcc.Dropdown(
                            id='site-dropdown',
                            options=[{'label': name, 'value': code} for code, name in site_names],
                            value=None,
                            multi=True,
                            placeholder="Select specific sites",
                            className="mb-3",
                            style={'borderRadius': '5px'}
                        )
                    ], id='site-filter-div', style={'display': 'none'}),
                    
                    html.Hr(style={'borderColor': COLORS['border']}),
                    
                    html.Label("Date Range:", className="fw-bold", style={'color': COLORS['text']}),
                    html.Div(
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=datetime(2024, 1, 1),
                            end_date=datetime(2024, 12, 31),
                            min_date_allowed=datetime(2016, 1, 1),
                            max_date_allowed=datetime(2024, 12, 31),
                            display_format='YYYY-MM-DD',
                            className="mb-3"
                        ),
                        style={'marginTop': '10px'}
                    ),
                    
                    html.Hr(style={'borderColor': COLORS['border']}),
                    
                    html.H5("Analysis Options", 
                           className="mt-3",
                           style={
                               'fontWeight': '600',
                               'color': COLORS['primary'],
                               'borderBottom': f'2px solid {COLORS["primary"]}',
                               'paddingBottom': '8px',
                               'marginBottom': '15px'
                           }),
                    
                    html.Label("Show Anomalies:", className="fw-bold mt-2", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='show-anomalies',
                        options=[
                            {'label': ' Yes', 'value': 'yes'},
                            {'label': ' No', 'value': 'no'}
                        ],
                        value='no',
                        className="mb-3"
                    ),
                    
                    html.Label("Show Trend Line:", className="fw-bold", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='show-trend',
                        options=[
                            {'label': ' Yes', 'value': 'yes'},
                            {'label': ' No', 'value': 'no'}
                        ],
                        value='no',
                        className="mb-3"
                    ),
                    
                    html.Hr(style={'borderColor': COLORS['border']}),
                    
                    html.Label("Display Data:", className="fw-bold", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='data-type-toggle',
                        options=[
                            {'label': ' Historical Only', 'value': 'historical'},
                            {'label': ' Historical + Predicted', 'value': 'both'}
                        ],
                        value='both',
                        className="mb-3"
                    ),
                    
                    html.Hr(style={'borderColor': COLORS['border']}),
                    
                    html.H5("Map Options", 
                           className="mt-3",
                           style={
                               'fontWeight': '600',
                               'color': COLORS['primary'],
                               'borderBottom': f'2px solid {COLORS["primary"]}',
                               'paddingBottom': '8px',
                               'marginBottom': '15px'
                           }),
                    
                    html.Label("Marker Size Based On:", className="fw-bold mt-2", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='map-metric',
                        options=[
                            {'label': ' Discharge', 'value': 'discharge'},
                            {'label': ' Gage Height', 'value': 'gage_height'}
                        ],
                        value='discharge',
                        className="mb-3"
                    ),
                    
                    html.Label("Show Predicted on Hover:", className="fw-bold", style={'color': COLORS['text']}),
                    dbc.RadioItems(
                        id='map-show-predicted',
                        options=[
                            {'label': ' Yes', 'value': 'yes'},
                            {'label': ' No', 'value': 'no'}
                        ],
                        value='yes',
                        className="mb-3"
                    ),
                    
                    html.Hr(style={'borderColor': COLORS['border']}),
                    
                    dbc.Button(
                        "Update Dashboard",
                        id='update-button',
                        size="lg",
                        className="w-100 mt-3",
                        n_clicks=0,
                        style={
                            'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                            'border': 'none',
                            'borderRadius': '8px',
                            'padding': '12px',
                            'fontSize': '1.1rem',
                            'fontWeight': 'bold',
                            'boxShadow': '0 4px 10px rgba(102, 126, 234, 0.3)',
                            'transition': 'transform 0.2s, box-shadow 0.2s'
                        }
                    )
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 2px 8px rgba(0,0,0,0.08)', 'border': 'none', 'borderRadius': '10px'})
        ], width=3, style=SIDEBAR_STYLE),
        
        # Graphs
        dbc.Col([
            # Discharge Graph
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-discharge",
                        type="default",
                        children=dcc.Graph(id='discharge-timeseries'),
                        color=COLORS['primary']
                    )
                ], style={'padding': '15px'})
            ], className="mb-4", style={
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
                'border': 'none',
                'borderRadius': '10px',
                'backgroundColor': COLORS['card']
            }),
            
            # Gage Height Graph
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-gage",
                        type="default",
                        children=dcc.Graph(id='gage-height-timeseries'),
                        color=COLORS['primary']
                    )
                ], style={'padding': '15px'})
            ], className="mb-4", style={
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
                'border': 'none',
                'borderRadius': '10px',
                'backgroundColor': COLORS['card']
            }),
            
            # Map Visualization
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-map",
                        type="default",
                        children=dcc.Graph(id='site-map'),
                        color=COLORS['primary']
                    )
                ], style={'padding': '15px'})
            ], style={
                'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
                'border': 'none',
                'borderRadius': '10px',
                'backgroundColor': COLORS['card']
            })
        ], width=9, style=CONTENT_STYLE)
    ])
], fluid=True, style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
})


@app.callback(
    [Output('state-filter-div', 'style'),
     Output('site-filter-div', 'style')],
    [Input('filter-type', 'value')]
)
def toggle_filter_visibility(filter_type):
    if filter_type == 'state':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    [Output('discharge-timeseries', 'figure'),
     Output('gage-height-timeseries', 'figure'),
     Output('site-map', 'figure')],
    [Input('update-button', 'n_clicks')],
    [State('filter-type', 'value'),
     State('state-dropdown', 'value'),
     State('site-dropdown', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('data-type-toggle', 'value'),
     State('map-metric', 'value'),
     State('map-show-predicted', 'value'),
     State('show-anomalies', 'value'),
     State('show-trend', 'value')]
)
def update_dashboard(n_clicks, filter_type, selected_states, selected_sites, start_date, end_date, 
                     data_type, map_metric, show_predicted, show_anomalies, show_trend):
    if filter_type == 'state':
        selected_sites = None
    else:
        selected_states = None
    
    print(f"Filter type: {filter_type}")
    print(f"Selected states: {selected_states}")
    print(f"Selected sites: {selected_sites}")
    
    historical_df = None
    predicted_df = None
    
    if data_type in ['historical', 'both']:
        historical_df = loadDataDashboard.loadHistoricalData(
            startDate=start_date,
            endDate=end_date,
            sites=selected_sites,
            states=selected_states
        )
    
    if data_type == 'both':
        predicted_df = loadDataDashboard.loadPredictions(
            startDate=start_date,
            endDate=end_date,
            sites=selected_sites,
            states=selected_states
        )
    
    if historical_df is not None and predicted_df is not None:
        combined_df = pd.concat([historical_df, predicted_df])
    elif historical_df is not None:
        combined_df = historical_df
    else:
        combined_df = predicted_df
    
    discharge_fig = create_discharge_timeseries(combined_df, show_anomalies, show_trend)
    gage_height_fig = create_gage_height_timeseries(combined_df, show_anomalies, show_trend)
    map_fig = create_site_map(combined_df, map_metric, show_predicted)
    
    return discharge_fig, gage_height_fig, map_fig

def create_discharge_timeseries(df, show_anomalies='no', show_trend='no'):
    unique_sites = df['site_code'].nunique()
    
    # Logic for comparing individual sites
    if unique_sites <= 5 and unique_sites > 1:
        daily_avg = df.groupby(['date', 'site_code', 'data_type'])['discharge'].mean().reset_index()
        
        if 'site_name' in df.columns:
            site_names = df.groupby('site_code')['site_name'].first().to_dict()
            daily_avg['site_label'] = daily_avg['site_code'].map(site_names)
        else:
            daily_avg['site_label'] = daily_avg['site_code']
        
        fig = px.line(
            daily_avg,
            x='date',
            y='discharge',
            color='site_label',
            line_dash='data_type',
            title=f'Discharge Comparison - {unique_sites} Sites',
            labels={'discharge': 'Discharge (cfs)', 'date': 'Date', 'site_label': 'Site'},
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                title='Sites',
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.01,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            font=dict(family='Arial, sans-serif', size=12, color=COLORS['text']),
            title_font=dict(size=18, color=COLORS['text'], family='Arial, sans-serif'),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            )
        )
    else:
        # Aggregate version of graph
        daily_avg = df.groupby(['date', 'data_type'])['discharge'].mean().reset_index()
        
        fig = px.line(
            daily_avg,
            x='date',
            y='discharge',
            color='data_type',
            title='Average Discharge Over Time',
            labels={'discharge': 'Discharge (cfs)', 'date': 'Date'},
            color_discrete_map={'historical': COLORS['historical'], 'predicted': COLORS['predicted']}
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                title='Data Type',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            font=dict(family='Arial, sans-serif', size=12, color=COLORS['text']),
            title_font=dict(size=18, color=COLORS['text'], family='Arial, sans-serif'),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            )
        )
        
        # Adds anomaly detection if selected in sidebar
        if show_anomalies == 'yes':
            historical_data = df[df['data_type'] == 'historical'].copy()
            if len(historical_data) > 0:
                daily_avg_for_anomaly = historical_data.groupby('date')['discharge'].mean().reset_index()
                daily_avg_for_anomaly['site_code'] = 'aggregated'  # Dummy site_code for function
                
                df_anomalies, _, _ = analysisTools.analyze_dataset(daily_avg_for_anomaly, 'discharge')
                anomalies = df_anomalies[df_anomalies['is_anomaly'] == True]
                
                if len(anomalies) > 0:
                    fig.add_scatter(
                        x=anomalies['date'],
                        y=anomalies['discharge'],
                        mode='markers',
                        marker=dict(size=12, color='#e74c3c', symbol='diamond', line=dict(width=2, color='#c0392b')),
                        name=f'Anomalies ({len(anomalies)})',
                        showlegend=True
                    )
        
        # Adds trend lines if selected in sidebar
        if show_trend == 'yes':
            historical_data = df[df['data_type'] == 'historical'].copy()
            if len(historical_data) > 0:
                trend_df = historical_data.groupby('date')['discharge'].mean().reset_index()
                _, trend_results, _ = analysisTools.analyze_dataset(historical_data, 'discharge')
                
                if trend_results and len(trend_results) > 0:
                    colors = ['#f39c12', '#27ae60', '#9b59b6', '#e67e22', '#16a085']
                    for idx, trend in enumerate(trend_results):
                        color = colors[idx % len(colors)]
                        fig.add_scatter(
                            x=trend['trend_line']['date'],
                            y=trend['trend_line']['trend_value'],
                            mode='lines',
                            line=dict(color=color, width=4, dash='dash'),
                            name=f"{trend['direction'].capitalize()} ({trend['pct_change']:.1f}%)",
                            showlegend=True
                        )
    
    return fig

def create_gage_height_timeseries(df, show_anomalies='no', show_trend='no'):
    historical_only = df[df['data_type'] == 'historical'].copy()
    
    historical_only = historical_only[
        (historical_only['gage_height'].isna()) | 
        ((historical_only['gage_height'] >= 0) & (historical_only['gage_height'] <= 100))
    ].copy()
    
    unique_sites = historical_only['site_code'].nunique()
        
        # Logic for comparing individual sites
    if unique_sites <= 5 and unique_sites > 1:
        daily_avg = historical_only.groupby(['date', 'site_code'])['gage_height'].mean().reset_index()
        
        if 'site_name' in historical_only.columns:
            site_names = historical_only.groupby('site_code')['site_name'].first().to_dict()
            daily_avg['site_label'] = daily_avg['site_code'].map(site_names)
        else:
            daily_avg['site_label'] = daily_avg['site_code']
        
        fig = px.line(
            daily_avg,
            x='date',
            y='gage_height',
            color='site_label',
            title=f'Gage Height Comparison - {unique_sites} Sites',
            labels={'gage_height': 'Gage Height (ft)', 'date': 'Date', 'site_label': 'Site'}
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                title='Sites',
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.01,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            font=dict(family='Arial, sans-serif', size=12, color=COLORS['text']),
            title_font=dict(size=18, color=COLORS['text'], family='Arial, sans-serif'),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            )
        )
    else:
        # Aggregate ver.
        daily_avg = historical_only.groupby('date')['gage_height'].mean().reset_index()
        
        fig = px.line(
            daily_avg,
            x='date',
            y='gage_height',
            title='Average Gage Height Over Time',
            labels={'gage_height': 'Gage Height (ft)', 'date': 'Date'}
        )
        
        fig.update_traces(line=dict(color=COLORS['historical'], width=2.5))
        
        fig.update_layout(
            hovermode='x unified',
            font=dict(family='Arial, sans-serif', size=12, color=COLORS['text']),
            title_font=dict(size=18, color=COLORS['text'], family='Arial, sans-serif'),
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#e0e0e0'
            )
        )
        
        # Adds anomaly detection if selected in sidebar
        if show_anomalies == 'yes' and len(historical_only) > 0:
            # Now use the SAME clean data that's already filtered
            # Aggregate FIRST, then detect anomalies
            daily_avg_for_anomaly = historical_only.groupby('date')['gage_height'].mean().reset_index()
            daily_avg_for_anomaly['site_code'] = 'aggregated'
            
            df_anomalies, _, _ = analysisTools.analyze_dataset(daily_avg_for_anomaly, 'gage_height')
            anomalies = df_anomalies[df_anomalies['is_anomaly'] == True]
            
            if len(anomalies) > 0:
                fig.add_scatter(
                    x=anomalies['date'],
                    y=anomalies['gage_height'],
                    mode='markers',
                    marker=dict(size=12, color='#e74c3c', symbol='diamond', line=dict(width=2, color='#c0392b')),
                    name=f'Anomalies ({len(anomalies)})',
                    showlegend=True
                )
                
                fig.update_layout(showlegend=True)
        
        # Adds trend lines if selected in sidebar
        if show_trend == 'yes' and len(historical_only) > 0:
            # Aggregate for trend
            trend_df = historical_only.groupby('date')['gage_height'].mean().reset_index()
            _, trend_results, _ = analysisTools.analyze_dataset(historical_only, 'gage_height')
            
            if trend_results and len(trend_results) > 0:
                colors = ['#f39c12', '#27ae60', '#9b59b6', '#e67e22', '#16a085']
                for idx, trend in enumerate(trend_results):
                    color = colors[idx % len(colors)]
                    fig.add_scatter(
                        x=trend['trend_line']['date'],
                        y=trend['trend_line']['trend_value'],
                        mode='lines',
                        line=dict(color=color, width=4, dash='dash'),
                        name=f"{trend['direction'].capitalize()} ({trend['pct_change']:.1f}%)",
                        showlegend=True
                    )
                
                fig.update_layout(showlegend=True)
    
    return fig

def create_site_map(df, map_metric='discharge', show_predicted='yes'):
    if 'site_name' not in df.columns:
        import database
        engine = database.getSQLAlchemyEngine()
        site_info = pd.read_sql("SELECT site_code, site_name FROM sitelocations", engine)
        df = df.merge(site_info, on='site_code', how='left')
    
    historical_df = df[df['data_type'] == 'historical'].copy()
    
    site_avg = historical_df.groupby(['site_code', 'latitude', 'longitude', 'state']).agg({
        'discharge': 'mean',
        'gage_height': 'mean'
    }).reset_index()
    
    if 'site_name' in df.columns:
        site_names = df.groupby('site_code')['site_name'].first()
        site_avg = site_avg.merge(site_names, on='site_code', how='left')
    
    if show_predicted == 'yes' and 'predicted' in df['data_type'].values:
        predicted_df = df[df['data_type'] == 'predicted'].copy()
        predicted_avg = predicted_df.groupby('site_code')['discharge'].mean().reset_index()
        predicted_avg.columns = ['site_code', 'predicted_discharge']
        site_avg = site_avg.merge(predicted_avg, on='site_code', how='left')
    
    site_avg = site_avg.dropna(subset=[map_metric])
    
    site_avg['size_value'] = np.power(site_avg[map_metric] + 1, 0.35)
    
    metric_label = 'Discharge (cfs)' if map_metric == 'discharge' else 'Gage Height (ft)'
    
    fig = px.scatter_map(
        site_avg,
        lat='latitude',
        lon='longitude',
        size='size_value',
        color='state',
        hover_name='site_code',
        title=f'Site Locations - Marker Size by {metric_label}',
        zoom=5,
        height=600,
        size_max=10
    )
    # Default layout of map point is hard to read and just isn't good
    # This template fixes that issue
    hover_template = '<b>%{hovertext}</b><br><br>'
    hover_template += '<b>Site Information</b><br>'
    hover_template += 'Site Name: %{customdata[0]}<br>'
    hover_template += 'State: %{customdata[1]}<br>'
    hover_template += '<br>'
    hover_template += '<b>Water Metrics</b><br>'
    hover_template += 'Avg Discharge: %{customdata[2]:.2f} cfs<br>'
    hover_template += 'Avg Gage Height: %{customdata[3]:.2f} ft<br>'
    
    customdata_cols = ['site_name', 'state', 'discharge', 'gage_height']
    
    if show_predicted == 'yes' and 'predicted_discharge' in site_avg.columns:
        hover_template += 'Predicted Discharge: %{customdata[4]:.2f} cfs<br>'
        customdata_cols.append('predicted_discharge')
    
    hover_template += '<br>'
    hover_template += '<b>Coordinates</b><br>'
    hover_template += 'Latitude: %{customdata[' + str(len(customdata_cols)) + ']:.4f}<br>'
    hover_template += 'Longitude: %{customdata[' + str(len(customdata_cols) + 1) + ']:.4f}'
    hover_template += '<extra></extra>'
    
    customdata_cols.extend(['latitude', 'longitude'])
    
    fig.update_traces(
        customdata=site_avg[customdata_cols].values,
        hovertemplate=hover_template,
        marker=dict(
            sizemode='diameter',
            sizemin=5,
            opacity=0.8
        )
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend=dict(
            title='State',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        font=dict(family='Arial, sans-serif', size=12, color=COLORS['text']),
        title_font=dict(size=18, color=COLORS['text'], family='Arial, sans-serif')
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)