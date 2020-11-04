import json
import urllib.request
import pandas as pd
import plotly.graph_objects as go

import random

df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

canada = df.loc[(df["Country"] == "Canada")]

canada = canada.groupby(["Country", "State", "Product_Category"])

canada = pd.DataFrame(canada.sum().reset_index())


def read_geojson(url):
    with urllib.request.urlopen(url) as url:
        jdata = json.loads(url.read().decode())
    return jdata


#
#irish_url = 'https://gist.githubusercontent.com/pnewall/9a122c05ba2865c3a58f15008548fbbd/raw/5bb4f84d918b871ee0e8b99f60dde976bb711d7c/ireland_counties.geojson'

canada_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson'

data = read_geojson(canada_url)
#print(type(data["features"]))


for i in data["features"]:
    property = i['properties']
    property['id'] = property.pop('cartodb_id')


# locations=['British Columbia', 'Alberta', 'Ontario']
# z = [20, 15, 4]

fig = go.Figure(
    data=[go.Choropleth(
        geojson=data,
        locations=canada["State"],
        text=canada["State"],
        z=canada["Revenue"],
        colorscale='reds',
        colorbar=dict(thickness=20, ticklen=3),
        hoverinfo='all',
        marker_line_width=1,
        marker_opacity=0.75)]
)
#
fig.update_layout(title_text='Canada Map',
                  title_x=0.5, width=700, height=700,
                  geo=dict(
                    lataxis=dict(range=[40, 70]),
                    lonaxis=dict(range=[-130, -55]),
                    scope="north america")
                  )
fig.show()

#print(df.columns)