# Explainable Demand Forecasting System
### Sri Lankan Grocery Retail - XGBoost + SHAP + Streamlit



## What This Project Does
Predicts daily unit sales for 11 products across 2 Sri Lankan retail 
stores and explains each prediction using SHAP (SHapley Additive 
exPlanations).

## Models
- XGBoost (primary) - R2 88.63%, WMAPE 21.78% on 2017 test data
- Linear Regression, Facebook Prophet, LSTM, Moving Average (baselines)

## Dataset
Corporacion Favorita from Ecuador.
2 stores mapped to Store Colombo and Store Gampaha.
Sri Lankan holiday calendar with 301 entries applied.

## How To Run The Dashboard
cd src
streamlit run dashboard.py

## If You Only Want XGBoost
cd src
python xgboost_model.py
python generate_test_predictions.py
streamlit run dashboard.py

## Full Pipeline (optional)
cd src
python create_sri_lanka_holidays.py
python data_processing.py
python merge_environmental.py
python xgboost_model.py
python linear_regression.py
python moving_average.py
python prophet_model.py
python shap_explainability.py
python generate_test_predictions.py
streamlit run dashboard.py

## Project Structure
data/        raw and processed datasets  
models/      saved trained model files  
results/     predictions and SHAP values  
src/         all Python scripts and dashboard

## Tech Stack
Python 3.12, XGBoost, SHAP, Streamlit, Plotly, Pandas, Prophet, TensorFlow