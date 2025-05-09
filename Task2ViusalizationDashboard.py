import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Global Water Consumption Dashboard",
    page_icon="💧",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        color: #0D47A1;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #1565C0;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .insight-text {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Global Water Consumption Dashboard (2000-2024)</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Analyzing Global Water Management: Challenges and Opportunities</div>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_global_water_consumption.csv")
    
    if 'CountryYearTotal Water Consumption (Billion Cubic Meters)' in df.columns:
        df = pd.read_csv("cleaned_global_water_consumption.csv", header=None)
        df.columns = ["Country", "Year", "Total Water Consumption (Billion Cubic Meters)", 
                      "Per Capita Water Use (Liters per Day)", "Agricultural Water Use (%)",
                      "Industrial Water Use (%)", "Household Water Use (%)",
                      "Rainfall Impact (Annual Precipitation in mm)", 
                      "Groundwater Depletion Rate (%)", "Water Scarcity Level"]
        
    return df

df = load_data()

st.sidebar.markdown("## Dashboard Filters")

all_countries = sorted(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    all_countries,
    default=all_countries
)

min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

scarcity_levels = sorted(df['Water Scarcity Level'].unique())
selected_scarcity = st.sidebar.multiselect(
    "Filter by Water Scarcity Level:",
    scarcity_levels,
    default=scarcity_levels
)

filtered_df = df[
    (df['Country'].isin(selected_countries)) &
    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
    (df['Water Scarcity Level'].isin(selected_scarcity))
]

if filtered_df.empty:
    st.error("No data available with the selected filters. Please adjust your selection.")
    st.stop()

st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_consumption = filtered_df['Total Water Consumption (Billion Cubic Meters)'].mean()
    st.metric("Avg. Water Consumption", f"{avg_consumption:.2f} B m³")

with col2:
    avg_per_capita = filtered_df['Per Capita Water Use (Liters per Day)'].mean()
    st.metric("Avg. Per Capita Use", f"{avg_per_capita:.2f} L/day")

with col3:
    avg_agri = filtered_df['Agricultural Water Use (%)'].mean()
    st.metric("Avg. Agricultural Use", f"{avg_agri:.2f}%")

with col4:
    avg_depletion = filtered_df['Groundwater Depletion Rate (%)'].mean()
    st.metric("Avg. Groundwater Depletion", f"{avg_depletion:.2f}%")

tab1, tab2 = st.tabs(["Global & Sector Analysis", "Trends & Rankings"])

with tab1:
    st.markdown("<div class='section-header'>1. Global Water Consumption Map</div>", unsafe_allow_html=True)
    map_data = filtered_df.groupby('Country')['Total Water Consumption (Billion Cubic Meters)'].mean().reset_index()
    fig_map = px.choropleth(
        map_data,
        locations='Country',
        locationmode='country names',
        color='Total Water Consumption (Billion Cubic Meters)',
        hover_name='Country',
        color_continuous_scale=[
            "#cce5ff", "#66b3ff", "#3399ff", "#0073e6", "#0047b3"
        ],
        title='Average Water Consumption by Country',
        labels={'Total Water Consumption (Billion Cubic Meters)': 'Avg. Consumption (B m³)'}
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            showframe=False,
            showcountries=True,
            projection_type='kavrayskiy7'
        )
    )
    
    fig_map.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Average Consumption: %{z:.2f} B m³<extra></extra>'
    )
    
    fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div class='insight-text'> The map reveals significant disparities in water consumption across different regions, with larger economies and agricultural producers typically consuming more water.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>2. Sector Usage Comparison</div>", unsafe_allow_html=True)

    selected_year = st.slider("Select specific year for sector analysis:", min_year, max_year, max_year)
    
    year_df = filtered_df[filtered_df['Year'] == selected_year]
    
    if not year_df.empty:
        year_df = year_df.sort_values(by="Total Water Consumption (Billion Cubic Meters)", ascending=True)
        
        fig_sectors = go.Figure()
        fig_sectors.add_trace(go.Bar(
            y=year_df['Country'],
            x=year_df['Agricultural Water Use (%)'],
            name='Agricultural',
            orientation='h',
            marker=dict(color='#4CAF50'),
            hovertemplate='Agricultural: %{x:.0f}%<extra></extra>'
        ))
        
        fig_sectors.add_trace(go.Bar(
            y=year_df['Country'],
            x=year_df['Industrial Water Use (%)'],
            name='Industrial',
            orientation='h',
            marker=dict(color='#2196F3'),
            hovertemplate='Industrial: %{x:.0f}%<extra></extra>'
        ))
        
        fig_sectors.add_trace(go.Bar(
            y=year_df['Country'],
            x=year_df['Household Water Use (%)'],
            name='Household',
            orientation='h',
            marker=dict(color='#FF9800'),
            hovertemplate='Household: %{x:.0f}%<extra></extra>'
        ))

        fig_sectors.update_layout(
            barmode='stack',
            title=f'Water Usage by Sector in {selected_year}',
            xaxis_title='Percentage (%)',
            yaxis_title='Country',
            legend_title='Sector',
            height=600
        )
        
        st.plotly_chart(fig_sectors, use_container_width=True)

        st.markdown("<div class='insight-text'> Agricultural water use typically dominates in most countries, but the balance between sectors varies significantly based on economic development stage and natural resources.</div>", unsafe_allow_html=True)
    else:
        st.warning(f"No data available for year {selected_year}. Please select another year.")

    st.markdown("<div class='section-header'>3. Rainfall vs. Consumption Analysis</div>", unsafe_allow_html=True)

    fig_rainfall = px.scatter(
            filtered_df,
            x='Rainfall Impact (Annual Precipitation in mm)',
            y='Total Water Consumption (Billion Cubic Meters)',
            color='Water Scarcity Level',
            size='Per Capita Water Use (Liters per Day)',
            hover_name='Country',
            hover_data=['Year'],
            color_discrete_map={'Low': '#2196F3', 'Moderate': '#FFC107', 'High': '#F44336', 'Severe': '#B71C1C'},
            title='Relationship between Rainfall and Water Consumption',
            labels={
                'Rainfall Impact (Annual Precipitation in mm)': 'Annual Rainfall (mm)',
                'Total Water Consumption (Billion Cubic Meters)': 'Total Consumption (B m³)'
            }
        )
    fig_rainfall.update_layout(height=500)
    st.plotly_chart(fig_rainfall, use_container_width=True)
    
    st.markdown("<div class='insight-text'> The correlation analysis reveals important relationships between rainfall patterns and consumption behaviors, with notable impacts on groundwater depletion rates.</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-header'>4. Groundwater Depletion Trends</div>", unsafe_allow_html=True)
    
    country_depletion = filtered_df.pivot_table(
        index='Year', 
        columns='Country', 
        values='Groundwater Depletion Rate (%)'
    ).reset_index()
    
    fig_depletion = px.line(
        country_depletion, 
        x='Year', 
        y=country_depletion.columns[1:],
        title='Groundwater Depletion Rate Trends by Country',
        labels={'value': 'Depletion Rate (%)', 'variable': 'Country'}
    )
    fig_depletion.update_layout(height=500, xaxis_title='Year', yaxis_title='Depletion Rate (%)')
    st.plotly_chart(fig_depletion, use_container_width=True)

    st.markdown("<div class='insight-text'> Several countries show alarming trends in groundwater depletion, indicating unsustainable water management practices that require immediate attention.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>5. Per Capita Consumption Ranking</div>", unsafe_allow_html=True)

    selected_year_capita = st.slider(
        "Select year for per capita analysis:", 
        min_year, 
        max_year, 
        max_year,
        key="per_capita_year"
    )
    
    capita_df = filtered_df[filtered_df['Year'] == selected_year_capita]
    
    if not capita_df.empty:
        capita_df = capita_df.sort_values(by='Per Capita Water Use (Liters per Day)', ascending=False)
        
        fig_capita = px.bar(
            capita_df,
            x='Country',
            y='Per Capita Water Use (Liters per Day)',
            color='Water Scarcity Level',
            color_discrete_map={'Low': '#2196F3', 'Moderate': '#FFC107', 'High': '#F44336', 'Severe': '#B71C1C'},
            title=f'Per Capita Water Use Ranking in {selected_year_capita}',
            labels={'Per Capita Water Use (Liters per Day)': 'Per Capita Use (L/day)'}
        )
        fig_capita.update_layout(height=500, xaxis_title='Country', yaxis_title='Per Capita Use (L/day)', yaxis=dict(showgrid=False))
        st.plotly_chart(fig_capita, use_container_width=True)
    else:
        st.warning(f"No data available for year {selected_year_capita}. Please select another year.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Explanation")
st.sidebar.markdown("""
This dashboard explores global water consumption patterns from 2000-2024, focusing on:

- Global consumption patterns
- Sectoral water usage
- Rainfall and consumption relationships
- Groundwater depletion trends
- Per capita water usage rankings

Use the filters above to customize your view.
""")