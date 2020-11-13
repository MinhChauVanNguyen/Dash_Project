import pandas as pd
import requests
import plotly
import plotly
print(plotly.__version__)



df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')


uk = ['United Kingdom', 'United States']
mask = df['Country'].str.contains(uk, na=False, regex=True, case=False)
df = df[~mask]
print(df.Country.unique())

canada_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson'
france_url = 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson'
germany_url = 'https://gist.githubusercontent.com/oscar6echo/4423770/raw/990e602cd47eeca87d31a3e25d2d633ed21e2005/dataBundesLander.json'
australia_url = 'https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.json'


data = requests.get(canada_url).json()


def my_function(country, country_url):
    data = requests.get(country_url).json()

    for i in data["features"]:
        property = i['properties']
        if country == "Australia":
            property['id'] = property.pop('STATE_NAME')
        elif country == "Canada":
            property['id'] = property.pop('name')
        elif country == "France":
            property['id'] = property.pop('nom')
        else:
            property['id'] = property.pop('NAME_1')

    return data["features"][0]


my_function(country='Australia',
            country_url='https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.json')

# print(data["features"][0])

