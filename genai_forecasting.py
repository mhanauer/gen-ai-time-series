import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline

# Load dataset
df = pd.read_csv('health_care_data_2023_2024_monthly.csv')

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

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df[column_to_forecast], color="royalblue", label="historical data")
ax.plot(forecast_index, median, color="tomato", label="median forecast")
ax.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
ax.legend()
ax.grid()

st.pyplot(fig)
