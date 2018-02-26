# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def percentile_k_features(df,k = 20):

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    select = SelectPercentile(f_regression, k)
    X_new = select.fit_transform(X,y)

    scores = select.scores_
    selected_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:X_new.shape[1]]
    selected_predictors = [X.columns[i] for i in selected_index]

    return selected_predictors
# Write your solution here:
