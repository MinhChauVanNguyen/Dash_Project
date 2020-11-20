import pandas as pd
import sklearn

df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

df.drop(
    df.columns.difference(['Customer_Gender', 'Revenue', 'State', 'Country', 'Year']),
    1, inplace=True)

data = df.loc[(df["Country"] == "France") & (df["State"] == "Seine Saint Denis")]

# 2016 data
data_2016 = data[data["Year"] == 2016]

data = data.set_index("Year").drop(2016, axis=0)

print(data)
