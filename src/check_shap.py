import pandas as pd

importance = pd.read_csv('../results/shap/shap_feature_importance.csv')
print(importance[['rank', 'feature', 'mean_abs_shap', 'normalized']].to_string(index=False))