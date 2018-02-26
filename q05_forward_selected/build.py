# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()
def forward_selected(df,model):
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    outer_i = X.shape[1]

    features = []
    features_r2 = []
    X_new = pd.DataFrame(index = X.index)
    X_try = X.copy()

    for i in range(outer_i):
        temp_features = []
        temp_r2 = []

        for j in range(X_try.shape[1]):
            X_fit = pd.concat([X_new,X_try.iloc[:,j].to_frame()],axis = 1)#X_new.merge(X_try.iloc[:,j].to_frame(),left_index = True, right_index = True)

            model.fit(X_fit,y)
            score = model.score(X_fit,y)
            temp_r2.append(score)
            temp_features.append(X_try.iloc[:,j].name)

        n = temp_r2.index(max(temp_r2))
        features.append(temp_features[n])
        features_r2.append(temp_r2[n])

        X_new = X[features]
        X_left = X[X.columns.difference(features)]


    return features, features_r2

# Your solution code here
