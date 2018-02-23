# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')

def plot_corr(df, size = 11):
    import seaborn as sns
    fig, ax = subplots(figsize=(size, size))
    sns.heatmap(data.corr(), cmap = 'YlOrRd')
# Write your solution here:
