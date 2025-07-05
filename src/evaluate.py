import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics



# Predictions - Future weeks
def pred(model, X_future, y_test):
    y_pred = model.predict(X_future)
    return y_pred


# Prediction Stats
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)   # Predictions

    print(model)
    print(f'\nr2: {metrics.r2_score(y_test, y_pred)}')
    print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
    print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')
    print(f'RMAE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}')



# Actual vs Predicted Values
def pred_values(model, X_test, y_test):
    y_pred = model.predict(X_test)   # Predictions

    difference_dt = pd.DataFrame(
        np.c_[y_test, y_pred],
        columns = ['Actual Value', 'Predicted Value']
    )
    print(difference_dt)



# Plotting Actual and Predicted values
def pred_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)   # Predictions

    plt.figure(figsize=(10,5))

    sns.histplot(y_test, label='Actual', color="skyblue", alpha=0.8, kde=True)
    sns.histplot(y_pred, label='Predicted', color="orange", alpha=0.3, kde=True)

    plt.title('Actual vs Predicted values - Decision Tree')
    plt.xlabel('Ticket price')
    plt.ylabel('Count')

    plt.legend()
    plt.show()



# Plotting the Predictions - Future weeks
def future_plot(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Predicted_Sales', hue='Model', marker='o')

    plt.title('Predicted Weekly Sales from Different Models', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Predicted Sales (Millions)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.grid(True)

    plt.show()