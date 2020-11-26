import pandas as pd
import plotly.figure_factory as ff


df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')


print(df.columns)

selected_variable = ['Age_Group', 'Customer_Gender', 'Year', 'Profit']


data = df[["Country", "State", selected_variable[0], selected_variable[1], selected_variable[2], selected_variable[3], "Revenue"]]

data = data.loc[(data["Country"] == "France") & (data["State"] == "Loiret")]

print(data)
#data = data[(data["Country"] == "France") & (data["State"] == "Loiret")]

