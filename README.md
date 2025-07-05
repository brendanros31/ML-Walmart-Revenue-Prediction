# ML-Walmart-Revenue-Prediction

This project focuses on forecasting Walmart sales for a 12-week period using advanced machine learning models, including Linear Regression, Random Forest Regressor, and XGBoost Regressor.

Key components include comprehensive data analysis, data visualization, feature engineering, model training, and evaluation using metrics such as r2 score, MSE, RMSE, MAE, and RMAE. 

The project also provides forecast visualizations to support insights and decision-making.


## How to Use
```bash
pip install -r requirements.txt
python main.ipynb
```

## Project Structure
```
data/
  raw/
    Walmart_data.csv

src/
  data_loader.py
  evaluate.py
  features.py
  model.py
  utils.py

EDA.ipynb
main.ipynb
Walmart_RevenuePred_Report
config/config.yaml
```