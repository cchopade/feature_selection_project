# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
def percentile_k_features(df,k = 20):
    import numpy as np
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    select = SelectPercentile(f_regression, k)
    X_new = select.fit_transform(X,y)

    columns = X.columns.values
    support = select.get_support()

    return np.ndarray.tolist(columns[support])

# Write your solution here:
