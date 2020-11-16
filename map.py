import pandas as pd
import requests

df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

df.drop(
    df.columns.difference(['Customer_Gender',  'Product_Category', 'Revenue', 'State', 'Country', 'Year']),
    1, inplace=True)

data = df.loc[(df["Country"] == "France")]

data = data.groupby(['State', 'Customer_Gender', 'Product_Category']).agg({'Revenue': 'sum'})
#data = pd.DataFrame(data.sum().reset_index())

state_pcts = data.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

data['Perc'] = state_pcts

data.reset_index(inplace=True)

print(data)