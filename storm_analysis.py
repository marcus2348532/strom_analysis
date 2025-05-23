import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Title of the Streamlit app
st.title('Climate and Storm Data Analysis')

# Function to load and preprocess data
def load_and_process_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Upload Storm Data
st.subheader('Upload Storm Data (CSV)')
uploaded_storm = st.file_uploader('Choose a Storm data file', type=['csv'])
st.subheader('Upload Extreme Temperature Data (CSV)')
uploaded_temperature = st.file_uploader('Choose an Extreme Temperature data file', type=['csv'])
st.subheader('Upload Global Warming Data (CSV)')
uploaded_global_warming = st.file_uploader('Choose a Global Warming data file', type=['csv'])
st.subheader('Upload Methane Data (CSV)')
uploaded_methane = st.file_uploader('Choose a Methane data file', type=['csv'])
st.subheader('Upload Carbon Dioxide Data (CSV)')
uploaded_co2 = st.file_uploader('Choose a Carbon Dioxide data file', type=['csv'])
st.subheader('Upload Ocean Warming Data (CSV)')
uploaded_ocean_warming = st.file_uploader('Choose an Ocean Warming data file', type=['csv'])

final_data = pd.DataFrame()
if uploaded_storm and uploaded_temperature and uploaded_global_warming and uploaded_methane and uploaded_co2 and uploaded_ocean_warming:

    storm = load_and_process_data(uploaded_storm)
    le = LabelEncoder()
    storm["Indicator"] = le.fit_transform(storm["Indicator"])
    country = storm["Country"]
    x = 0
    for i in storm.columns:
        if x < 10:
            storm.drop(i, axis=1, inplace=True)
            x += 1
    storm_india = pd.DataFrame()
    storm_india = storm.T
    storm_india.columns = country
    storm_india.replace(np.nan, 0, inplace=True)
    x = range(1980, 2023)
    storm_india["Years"] = x
    x_storm = storm_india["India"]
    y_storm = storm_india["Years"]
    storm_india.reset_index(inplace=True)

    # Display storm data
    st.subheader("Storm Data")
    st.write(storm_india)

   
   


    df_temp = pd.DataFrame()
    temperature = load_and_process_data(uploaded_temperature)
    temperature["Indicator"] = le.fit_transform(temperature["Indicator"])
    country = temperature["Country"]
    x = 0
    for i in temperature.columns:
        if x < 11:
            temperature.drop(i, axis=1, inplace=True)
            x += 1
    
    df_temp = temperature.T
    df_temp.columns = country
    df_temp.replace(np.nan, 0, inplace=True)
    x = range(1980, 2023)
    df_temp["Years"] = x
    df_temp.reset_index(inplace=True)
    st.subheader("Extreame Temperature")
    st.write(temperature)


# Upload Global Warming Data

    global_warming = load_and_process_data(uploaded_global_warming)
    x = global_warming.iloc[100:, 1:2]
    for i in range(0, 100):
        global_warming.drop(i, axis=0, inplace=True)
    global_warming = global_warming.reset_index(drop=True)
    st.subheader("Global warming")
    st.write(global_warming)
  

# Upload Methane Data

    st.write(uploaded_methane)
    methane_data = load_and_process_data(uploaded_methane)
    st.subheader("Methane data")
    st.write(methane_data)


# Upload Carbon Dioxide Data

    dt = load_and_process_data(uploaded_co2)
    dt['Year'] = pd.to_numeric(dt['Year'], errors='coerce')
    dt = dt[(dt['Year'] >= 1980) & (dt['Year'] <= 2022)]
    carbondioxide = dt.groupby('Year').mean().reset_index()
    st.subheader("Co2")
    st.write(carbondioxide)

    
   

# Upload Ocean Warming Data

    dt = load_and_process_data(uploaded_ocean_warming)
    dt['Year'] = pd.to_numeric(dt['Year'], errors='coerce')
    dt = dt[(dt['Year'] >= 1980) & (dt['Year'] <= 2022)]
    column_average = dt.groupby('Year').mean().reset_index()
    st.subheader("Ocean warming")
    st.write(column_average)
    st.write("Years (y_storm):", y_storm)
    st.write("Storm Count (x_storm):", x_storm)
    
    
final_data = pd.DataFrame({
    'year': storm_india['Years'],
    'storm_count': storm_india['India'],
    'extreme_temperature': df_temp['India'],
    'GW_lowess': global_warming['Lowess'],
    'methane_mean': methane_data['mean'],
    'CO2_deseasionalized_mean': carbondioxide['de-season alized'],
    'Ocean_warming': column_average['Temperature']
})


st.write("The final Columns")
st.write(final_data.columns)

# Data Cleaning (Replace missing values and adjust values)
max_ocean_temp = final_data['Ocean_warming'].max()
std_ocean_temp = final_data['Ocean_warming'].std()
final_data.loc[40, 'Ocean_warming'] = max_ocean_temp + std_ocean_temp
final_data.loc[41, 'Ocean_warming'] = final_data.loc[40, 'Ocean_warming'] + std_ocean_temp
final_data.loc[42, 'Ocean_warming'] = final_data.loc[41, 'Ocean_warming'] + std_ocean_temp
final_data['Ocean_warming'].replace(0.000000, final_data['Ocean_warming'].min(), inplace=True)

final_data.loc[3, 'methane_mean'] = final_data.loc[4, 'methane_mean']-13
final_data.loc[2, 'methane_mean'] = final_data.loc[3, 'methane_mean']-13
final_data.loc[1, 'methane_mean'] = final_data.loc[2, 'methane_mean']-13
final_data.loc[0, 'methane_mean'] = final_data.loc[1, 'methane_mean']-13

st.subheader("Final Data")
st.write(final_data)

# Display descriptive statistics
st.subheader("Descriptive Statistics")
st.write(final_data.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(final_data.corr(), cmap='Blues', annot=True, ax=ax)
st.pyplot(fig)

# Line Plot for Storm Count Over the Years
st.subheader("Storm Count Over the Years")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(final_data['year'], final_data['storm_count'])
ax.set_xlabel('Year')
ax.set_ylabel('Storm Count')
ax.set_title('Storm Count Over the Years')
ax.grid(True)
st.pyplot(fig)

# Prepare data for model
x = final_data.drop(columns=['storm_count'])
y = final_data['storm_count']

xtest = x.tail(5)
ytest = y.tail(5)

xtrain = x.head(37)
ytrain = y.head(37)

# Decision Tree Regressor Model
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(xtrain, ytrain)

ypred = dt_regressor.predict(xtest)

# Decision Tree Evaluation
mse = mean_squared_error(ytest, ypred)
st.write(f"Mean Squared Error (Decision Tree): {mse}")

# Plot Actual vs Predicted Storm Count
st.subheader("Actual vs Predicted Storm Count (Decision Tree)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(xtest['year'], ytest, label='Actual Storm Count', marker='o')
ax.plot(xtest['year'], ypred, label='Predicted Storm Count', marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Storm Count')
ax.set_title('Actual vs Predicted Storm Count (Decision Tree)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Time Series Analysis - ACF and PACF plots
st.subheader("ACF and PACF plots")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(final_data['storm_count'], lags=20, ax=axes[0])
plot_pacf(final_data['storm_count'], lags=20, ax=axes[1])
st.pyplot(fig)

# ARIMA Model
train_data = final_data['storm_count'][:-5]
test_data = final_data['storm_count'][-5:]
model = ARIMA(train_data, order=(4, 1, 0))
model_fit = model.fit()
predictions = model_fit.predict(start=len(train_data), end=len(final_data['storm_count']) - 1)
mse_arima = mean_squared_error(test_data, predictions)
st.write(f"Mean Squared Error (ARIMA): {mse_arima}")

# ARIMA Predictions Plot
st.subheader("Actual vs ARIMA Predicted Storm Count")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(final_data['year'][-5:], test_data, label='Actual Storm Count', marker='o')
ax.plot(final_data['year'][-5:], predictions, label='ARIMA Predictions', marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Storm Count')
ax.set_title('Actual vs ARIMA Predicted Storm Count')
ax.legend()
ax.grid(True)
st.pyplot(fig)
