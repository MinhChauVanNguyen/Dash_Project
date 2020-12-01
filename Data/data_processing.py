import pandas as pd

# Read data
my_df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

# Remove UK data since it only has one state
uk = 'United Kingdom'
mask = my_df['Country'].str.contains(uk, na=False, regex=True, case=False)
my_df = my_df[~mask]

# fix the France state name so that it corresponds to the geojson file
my_df = my_df.replace({'State': {
    'Seine Saint Denis': 'Seine-Saint-Denis',  #
    'Loir et Cher': 'Loir-et-Cher',
    'Seine (Paris)': 'Paris',  #
    'Hauts de Seine': 'Hauts-de-Seine',
    "Val d'Oise": "Val-d'Oise",
    'Seine et Marne': 'Seine-et-Marne',
    'Val de Marne': 'Val-de-Marne',
    'Pas de Calais': 'Pas-de-Calais',
    'Garonne (Haute)': 'Haute-Garonne',
    'Yveline': 'Yvelines'}}
)

tab12_df = my_df.copy()

# subset data based on the following columns
tab12_df.drop(
    tab12_df.columns.difference(['Customer_Gender', 'Age_Group', 'Product_Category', 'Revenue', 'State', 'Country', 'Year', 'Profit']),
    1, inplace=True)

# sort data from the least to most recent year
tab12_df = tab12_df.sort_values(by=["Country", "Year"])

# rename the levels of Gender column
tab12_df['Customer_Gender'] = tab12_df['Customer_Gender'].replace({'F': 'Female', 'M': 'Male'})

