from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost



def build_model(model_type, params=None):
    if model_type == 'LinearRegression':
        return LinearRegression(**(params or {}))
    
    elif model_type =='RandomForestRegressor':
        return RandomForestRegressor(**(params or {}))
        
    elif model_type == 'XGBoost':
        return xgboost.XGBRegressor(**(params or {}))



def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)
