import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import joblib
import gc
st.set_page_config(layout="wide")
import ast

def initialize_session_state():
    # Initialize session state variables for data and timestamps
    if 'df_workorder_pkl_path' not in st.session_state:
        st.session_state.df_workorder_pkl_path = None
    if "start_date" not in st.session_state:
         st.session_state.start_date = None
    if "end_date" not in st.session_state:
        st.session_state.end_date = None

initialize_session_state()

@st.cache_data
def load_workorder_file(file_name):
    temp_df = pd.read_csv(file_name)
    temp_df['workorder_date'] = pd.to_datetime(temp_df['workorder_date']).dt.date
    joblib.dump(temp_df, "./data/output/dash/Workorder_Anomalies.pkl")
    del temp_df
    gc.collect()
    return "./data/output/dash/Workorder_Anomalies.pkl"



st.session_state.df_workorder_pkl_path = load_workorder_file('./data/output/Agglomerative_Clustering_Anomalies_05_08_2025_threshold_30.csv')

df_workorder = joblib.load(st.session_state.df_workorder_pkl_path)

df = df_workorder.copy()
st.title('Workorder Anomalies Dashboard')

show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=True)

repair_code_filter = st.sidebar.multiselect('Filter by Repair Code', options=df['repairCode'].unique(), default=df['repairCode'].unique())
shop_code_filter = st.sidebar.multiselect('Filter by Shop Code', options=df['shop_code'].unique(), default=df['shop_code'].unique())
repair_type_filter = st.sidebar.multiselect('Filter by Repair Type Code', options=df['RepairTypeCode'].unique(), default=df['RepairTypeCode'].unique())
cdx_damage_filter = st.sidebar.multiselect('Filter by cdx_damage', options=df['cdx_damage'].unique(), default=df['cdx_damage'].unique())
cdx_location_filter = st.sidebar.multiselect('Filter by cdx_location', options=df['cdx_location'].unique(), default=df['cdx_location'].unique())

# Add cluster filter if cluster column exists and has values
if 'cluster' in df.columns:
    cluster_options = [x for x in df['cluster'].unique() if pd.notna(x)]
    if cluster_options:
        cluster_filter = st.sidebar.multiselect('Filter by Cluster', options=cluster_options, default=cluster_options)
    else:
        cluster_filter = None
else:
    cluster_filter = None 

# Filter for start and end date
st.session_state.start_date = st.sidebar.date_input('Select a Start date (optional):', None)
st.session_state.end_date = st.sidebar.date_input('Select an End date (optional):', None)

df_filtered = df[
    (df['shop_code'].isin(shop_code_filter)) &
    (df['RepairTypeCode'].isin(repair_type_filter)) &
    (df['cdx_damage'].isin(cdx_damage_filter)) &
    (df['cdx_location'].isin(cdx_location_filter)) &
    (df['repairCode'].isin(repair_code_filter))
]

# Apply cluster filter if available
if cluster_filter is not None:
    df_filtered = df_filtered[
        (df_filtered['cluster'].isin(cluster_filter)) | (df_filtered['cluster'].isna())
    ]
    
if st.session_state.start_date and st.session_state.end_date:
    if st.session_state.start_date > st.session_state.end_date:
        st.error("Start Date can not be greater than End Date")

if st.session_state.start_date:

    start_date = st.session_state.start_date

    df_filtered = df_filtered[
        (df_filtered['workorder_date'] >= start_date)
    ]

if st.session_state.end_date:

    end_date = st.session_state.end_date

    df_filtered = df_filtered[
        (df_filtered['workorder_date'] <= end_date)
    ]

if show_anomalies_only:
    df_filtered = df_filtered[df_filtered['anomaly'] == True]

columns_to_remove = ['upper_range_grouped', 'lower_range']
df_filtered = df_filtered.drop(columns=[col for col in columns_to_remove if col in df_filtered.columns])

st.write("Filtered Data", df_filtered)

graph_width = st.sidebar.slider("Adjust Graph Width", min_value=800, max_value=2000, value=1200, step=100)

if 'workorder_date' in df_filtered.columns and 'total_repair_hours' in df_filtered.columns:
    df_filtered['workorder_date'] = pd.to_datetime(df_filtered['workorder_date'])
    df_filtered.sort_values(by='workorder_date', inplace=True)
    df_filtered.set_index('workorder_date', inplace=True)

    for repair_code in df_filtered['repairCode'].unique():
        df_repair_code = df_filtered[df_filtered['repairCode'] == repair_code]

        colors = ['red' if anomaly else 'blue' for anomaly in df_repair_code['anomaly']]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_repair_code.index, y=df_repair_code['total_repair_hours'], mode='markers+lines',
                                      marker=dict(color=colors), name=f'Total Repair Hours - {repair_code}'))

        fig_line.update_layout(
            title=f'Total Repair Hours Throughout Workorder Date for Repair Code {repair_code}',
            xaxis_title='Workorder Date',
            yaxis_title='Total Repair Hours',
            width=graph_width,
            height=600,
            xaxis={
                'rangeselector': {'buttons': list([{'step': 'all', 'label': 'All'}, {'step': 'month', 'count': 1, 'label': '1m'}, {'step': 'year', 'count': 1, 'label': '1y'}])},
                'rangeslider': {'visible': True},
                'type': 'date'
            }
        )

        st.plotly_chart(fig_line, use_container_width=True)
else:
    # st.write("Required columns 'workorder_date' and 'total_repair_hours' are not available in the dataset.")
    pass
