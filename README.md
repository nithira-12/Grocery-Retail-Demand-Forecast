# Explainable Demand Forecasting System
### Sri Lankan Grocery Retail - XGBoost + SHAP + Streamlit



## What This Project Does
Predicts daily unit sales for 11 products across 2 Sri Lankan retail 
stores and explains each prediction using SHAP (SHapley Additive 
exPlanations).

## Models
- XGBoost (primary) - R squared 88.76%, WMAPE 21.95% on 2017 test data
- Linear Regression, Facebook Prophet, Moving Average (baselines)

## selected procedure 
dataset corporacion favorita from ecuadore 
2 stores mapped to store colombo and store gampaha

## How To Run The Dashboard
cd src
streamlit run dashboard.py

## How To Retrain Models (optional)
Pre-trained models are included in /models directory.
To retrain from scratch:
cd src
python retrain_models.py

## Project Structure
data/        — raw and processed datasets  
models/      — saved trained model files  
results/     — predictions and SHAP values  
src/         — all Python scripts and dashboard

## Tech Stack
Python 3.12, XGBoost, SHAP, Streamlit, Plotly, Pandas