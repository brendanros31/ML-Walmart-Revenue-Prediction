import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Heatmap for selected Features
def heatmap(df, show = True):
    if show == True:
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

        plt.show()
