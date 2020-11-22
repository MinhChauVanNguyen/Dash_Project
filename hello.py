import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')


data = df.loc[(df["Country"] == "France") & (df["Year"].isin(['2011', '2012', '2013', '2014']))]

data = data.groupby(['State', 'Age_Group', 'Product_Category'])
data = pd.DataFrame(data.sum().reset_index())
data.drop(data.columns.difference(['State', 'Product_Category', 'Revenue', 'Age_Group']), 1, inplace=True)

data = data.pivot(index=['State', 'Age_Group'], columns=['Product_Category'], values='Revenue')
data = data.fillna(0)

data.reset_index(level=['State', 'Age_Group'], inplace=True)

data = data[data['Age_Group'] == 'Adults (35-64)']

for c in data:
    if type(data[c]) != 'object':
        data['Revenue'] = data.sum(axis=1)


try:
    selected_year is None
except ValueError:
    print("Please select at least a year")
