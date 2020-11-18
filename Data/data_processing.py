import pandas as pd

# Read data
df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

# Remove UK data since it only has one state
uk = 'United Kingdom'
mask = df['Country'].str.contains(uk, na=False, regex=True, case=False)
df = df[~mask]

# fix the France state name so that it corresponds to the geojson file
df = df.replace({'State': {
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
# subset data based on the following columns
df.drop(
    df.columns.difference(['Customer_Gender', 'Age_Group', 'Product_Category', 'Revenue', 'State', 'Country', 'Year']),
    1, inplace=True)

# sort data from the least to most recent year
df = df.sort_values(by=["Country", "Year"])

# rename the levels of Gender column
df['Customer_Gender'] = df['Customer_Gender'].replace({'F': 'Female', 'M': 'Male'})

df = df

