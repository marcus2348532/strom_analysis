# strom_analysis

Storm Prediction and Climate Trends Analysis

This project focuses on analyzing climate and stormrelated data to predict storm counts and understand their relationship with various environmental factors. The application is implemented using Streamlit for an interactive user interface and utilizes machine learning and time series analysis techniques for predictive modeling and insights.

Features

Data Upload: Upload multiple datasets such as Storm Data, Extreme Temperature Data, Global Warming Data, Methane Data, Carbon Dioxide Data, and Ocean Warming Data in CSV format.

Data Processing: Cleans and preprocesses data to ensure compatibility across datasets. Handles missing values and adjusts values for consistency.

Exploratory Data Analysis: Displays descriptive statistics for key variables. Visualizes relationships using correlation heatmaps. Line plots for storm counts over the years.

Machine Learning Prediction: Implements a Decision Tree Regressor to predict storm counts based on climate factors. Evaluates predictions using Mean Squared Error (MSE).

Time Series Analysis: Generates ACF and PACF plots for storm counts. Builds an ARIMA model to predict storm counts and evaluates its performance using MSE.

Visualization: Heatmaps, line plots, and comparative plots for actual vs predicted storm counts.

Requirements

Python Libraries: pandas numpy seaborn matplotlib scikitlearn statsmodels Streamlit

How to Use

Clone this repository to your local machine: git clone

Navigate to the project directory: cd stormprediction

Install the required dependencies: pip installr requirements.txt

Run the Streamlit application: streamlit run storm_analysis.py

Upload the necessary CSV files when prompted in the application.

Input Data

Storm Data: Historical storm counts and related indicators. Extreme Temperature Data: Records of extreme temperature events over time. Global Warming Data: Trends and lowessfiltered data related to global temperature anomalies. Methane Data: Methane concentration levels across years. Carbon Dioxide Data: CO2 levels with seasonal adjustments. Ocean Warming Data: Average temperature changes in ocean environments.

Acknowledgments

This project uses publicly available datasets for analysis and visualization from Nasa's site : https://climate.nasa.gov/%C2%A0%C2%A0/
