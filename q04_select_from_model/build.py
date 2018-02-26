# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(df):
    import numpy as np

    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    rfc = RandomForestClassifier(random_state = 9)
    sfm = SelectFromModel(rfc)
    sfm = sfm.fit(X,y)

    columns = X.columns.values
    support = sfm.get_support()
    return np.ndarray.tolist(columns[support])
# Your solution code here
