# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(df):
    import numpy as np
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    rfc = RandomForestClassifier()
    nfeatures = df.shape[1]/2

    rfe = RFE(rfc, n_features_to_select = nfeatures)
    rfe = rfe.fit(X,y)

    columns = X.columns.values
    support = rfe.support_

    return np.ndarray.tolist(columns[support])
# Your solution code here
