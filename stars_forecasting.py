import streamlit as st
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
import plotly.graph_objects as go

# Create the data dictionary from the previous image data
data = {
    "Star Year": [2025, 2024, 2023, 2022, 2021, 2020],
    "2 Stars": [53, 48, 43, 42, 50, 50],
    "3 Stars": [67, 63, 62, 61, 66, 66],
    "4 Stars": [75, 71, 70, 69, 76, 76],
    "5 Stars": [82, 79, 77, 76, 83, 83],
    "Avg Star": [73, 72, 70, 71, 75, 75],
    "Avg Score": [3.4, 3.7, 3.7, 3.9, 3.5, 3.5]
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Convert 'Star Year' to datetime format
df['Star Year'] = pd.to_datetime(df['Star Year'], format='%Y')

# Sort the DataFrame by 'Star Year' in ascending order
df = df.sort_values(by='Star Year')

# Streamlit UI
st.title('Health Care Data Forecasting')

# Display the head of the dataset
st.write('### Dataset Preview')
st.write(df.head())

# Dropdown for column selection
column_to_forecast = st.selectbox('Select a column to forecast', df.columns[1:])

# Load and configure ChronosPipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Prepare the data for forecasting (using all available historical data)
context = torch.tensor(df[column_to_forecast].values, dtype=torch.float32).unsqueeze(0)
prediction_length = 3  # Set to 3 for the next 3 years
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Get the next three years
last_year = df['Star Year'].dt.year.max()
forecast_years = pd.date_range(start=f'{last_year + 1}', periods=3, freq='Y').year

# Calculate the quantiles for low, median, and high predictions
low, median, high = np.quantile(forecast.squeeze(0).numpy(), [0.1, 0.5, 0.9], axis=0)

fig = go.Figure()

# Add historical data trace
fig.add_trace(go.Scatter(
    x=df['Star Year'].dt.year,
    y=df[column_to_forecast],
    mode='lines',
    name='Historical Data'
))

# Add median forecast trace for the next 3 years
fig.add_trace(go.Scatter(
    x=forecast_years,
    y=median,
    mode='lines',
    name='Median Forecast',
    line=dict(color='tomato')
))

# Add confidence interval
fig.add_trace(go.Scatter(
    x=forecast_years,
    y=low,
    fill=None,
    mode='lines',
    line=dict(color='tomato', width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast_years,
    y=high,
    fill='tonexty',
    mode='lines',
    line=dict(color='tomato', width=0),
    name='80% Prediction Interval'
))

# Update layout
fig.update_layout(
    title=f'Forecast for {column_to_forecast} (Next 3 Years)',
    xaxis_title='Year',
    yaxis_title=column_to_forecast,
    template='plotly_white'
)

# Display the plot
st.plotly_chart(fig)
