import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objs as go

df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')


def label_code(row):
    if row['State'] == 'Oregon':
        return 'OR'
    elif row['State'] == 'California':
        return 'CA'
    elif row['State'] == 'Washington':
        return 'WA'
    elif row['State'] == 'Kentucky':
        return 'KY'
    elif row['State'] == 'Texas':
        return 'TX'
    elif row['State'] == 'New York':
        return 'NY'
    elif row['State'] == 'Florida':
        return 'FL'
    elif row['State'] == 'Illinois':
        return 'IL'
    elif row['State'] == 'South Carolina':
        return 'SC'
    elif row['State'] == 'North Carolina':
        return 'NC'
    elif row['State'] == 'Georgia':
        return 'GA'
    elif row['State'] == 'Virginia':
        return 'VA'
    elif row['State'] == 'Ohio':
        return 'OH'
    elif row['State'] == 'Wyoming':
        return 'WY'
    elif row['State'] == 'Missouri':
        return 'MO'
    elif row['State'] == 'Montana':
        return 'MT'
    elif row['State'] == 'Utah':
        return 'UT'
    elif row['State'] == 'Minnesota':
        return 'MN'
    elif row['State'] == 'Mississippi':
        return 'MS'
    elif row['State'] == 'Arizona':
        return 'AZ'
    elif row['State'] == 'Alabama':
        return 'AL'
    else:
        return 'MA'


def update_map(selected_country, json_state, scope, country_url):

    data = df.loc[df["Country"] == selected_country]

    data = data.groupby(['State', 'Product_Category'])
    data = pd.DataFrame(data.sum().reset_index())

    data.drop(data.columns.difference(['State', 'Product_Category', 'Revenue']), 1, inplace=True)

    data = data.pivot(index='State', columns='Product_Category', values='Revenue')

    data = data.fillna(0)

    data.reset_index(level=0, inplace=True)

    if selected_country == 'United States':
        data['state_code'] = data.apply(lambda row: label_code(row), axis=1)
    else:
        data = data

    data['Revenue'] = data['Accessories'] + data['Bikes'] + data['Clothing']

    data.rename(columns={'State': json_state}, inplace=True)

    for col in data.columns:
        data[col] = data[col].astype(str)

    data['text'] = 'State: ' + data[json_state] + '<br>' + \
                      'Accessories Rev: $' + data['Accessories'] + '<br>' + \
                      'Bikes Rev: $' + data['Bikes'] + '<br>' + \
                      'Clothing Rev: $' + data['Clothing'] + '<br>' + \
                      'Total Rev: $' + data['Revenue']

    data['Revenue'] = data['Revenue'].apply(pd.to_numeric)

    if selected_country == 'United States':
        my_map = go.Figure(
            data=[
                go.Choropleth(
                    colorbar=dict(title='Revenue', ticklen=3),
                    locationmode='USA-states',
                    locations=data['state_code'],
                    z=data["Revenue"],
                    colorscale='Reds',
                    text=data['text'],
                ),
            ],
            layout=dict(geo={'subunitcolor': 'black'})
        )

        my_map.update_layout(
            title_text='USA map',
            geo_scope=scope,
            dragmode=False
        )
    #
    else:
        json_data = requests.get(country_url).json()

        tuple_feature_id = ('properties', json_state)

        my_map = px.choropleth(
            data_frame=data,
            geojson=json_data,
            locations=json_state,
            featureidkey=".".join(tuple_feature_id),
            color='Revenue',
            hover_data=['text'],
            color_continuous_scale="Magma",
            scope=scope)

        my_map.update_geos(
            visible=False,
            showcountries=True,
            showcoastlines=False,
            showland=False,
            fitbounds="locations",
            showsubunits=True,
            subunitcolor="Blue",
            resolution=110)

        my_map.update_layout(dragmode=False)

    my_map.show()

    return my_map


# update_map(
#      selected_country="United States",
#      country_url='',
#      json_state='State',
#      scope='usa')

update_map(
    country_url='https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.json',
    selected_country="Australiaa",
    json_state='STATE_NAME',
    scope='world')

# update_map(
#     selected_country='France',
#     country_url='https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson',
#     json_state='nom',
#     scope='europe')