import pandas as pd

df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

df.drop(
    df.columns.difference(['Customer_Gender', 'Age_Group', 'Product_Category', 'Revenue', 'State', 'Country', 'Year', 'Profit']),
    1, inplace=True)

value = ['Age_Group', 'Customer_Gender', 'Year', 'Profit']

data = df.groupby(value).agg('sum')

#str1 = ','.join('"{0}"'.format(w) for w in value)
#print(str1)
