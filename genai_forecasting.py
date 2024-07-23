import streamlit as st
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv('health_care_data_2023_2024.csv')

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

# Prepare the data for forecasting
context = torch.tensor(df[column_to_forecast].values, dtype=torch.float32).unsqueeze(0)
prediction_length = 12
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast.squeeze(0).numpy(), [0.1, 0.5, 0.9], axis=0)

fig = go.Figure()

# Add historical data trace
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df[column_to_forecast],
    mode='lines',
    name='Historical Data'
))

# Add median forecast trace
fig.add_trace(go.Scatter(
    x=pd.date_range(df['Date'].iloc[-1], periods=prediction_length+1, freq='M')[1:], 
    y=median,
    mode='lines',
    name='Median Forecast',
    line=dict(color='tomato')
))

# Add confidence interval
fig.add_trace(go.Scatter(
    x=pd.date_range(df['Date'].iloc[-1], periods=prediction_length+1, freq='M')[1:], 
    y=low,
    fill=None,
    mode='lines',
    line=dict(color='tomato', width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=pd.date_range(df['Date'].iloc[-1], periods=prediction_length+1, freq='M')[1:], 
    y=high,
    fill='tonexty',
    mode='lines',
    line=dict(color='tomato', width=0),
    name='80% Prediction Interval'
))

# Update layout
fig.update_layout(
    title=f'Forecast for {column_to_forecast}',
    xaxis_title='Date',
    yaxis_title=column_to_forecast,
    template='plotly_white'
)

# Display the plot
st.plotly_chart(fig)
