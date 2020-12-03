import pandas as pd

# Read data
my_df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

print(my_df.dtypes)