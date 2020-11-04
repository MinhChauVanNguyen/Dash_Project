import plotly.express as px
import pandas as pd
import plotly.express as px  # (version 4.7.0)

pd.options.mode.chained_assignment = None  # default='warn'

#import psutil
import plotly.io as pio

#pio.renderers.default = 'png'

df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')
#
#
# year_list = df["Year"].unique().tolist()
#
# #grouped_df = df.loc[(df["Country"] == "Canada") & (df["State"] == "British Columbia") & (df["Year"].isin([2011, 2012, 2013, 2014, 2015]))]
#
# usa = df.loc[(df["Country"] == "United States")]

#print(usa[["State", "Revenue"]])

#print(usa["State"].unique())



# usa['state_code'] = usa.apply(lambda row: label_code(row), axis=1)
#
# usa = usa[usa["Product_Category"] == "Bikes"]

# fig = px.choropleth(
#         data_frame=usa,
#         locationmode='USA-states',
#         locations='state_code',
#         scope="usa",
#         color='Revenue',
#         hover_data=['State', 'Revenue'],
#         color_continuous_scale="Viridis")
# #
# fig.show()



import plotly.graph_objects as go
import json
import urllib.request
import random

canada = df.loc[(df["Country"] == "Canada")]

print(canada["State"].unique())


def read_geojson(url):
    with urllib.request.urlopen(url) as url:
        jdata = json.loads(url.read().decode())
    return jdata


#
irish_url = 'https://gist.githubusercontent.com/pnewall/9a122c05ba2865c3a58f15008548fbbd/raw/5bb4f84d918b871ee0e8b99f60dde976bb711d7c/ireland_counties.geojson'

canada_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson'

jdata = read_geojson(canada_url)
print(jdata["features"][0])

data = read_geojson(irish_url)
list = data['features']
location = [item['id'] for item in list]
#print(location)

#locations = canada["State"].unique()
# locations = ['British Columbia', 'Alberta', 'Ontario']
# z = [20, 15, 4]
# randomlist = []
# for i in range(0, 26):
#     n = random.randint(0, 10)
#     randomlist.append(n)
#
# z = randomlist
# print(z)

#
#
# fig = go.Figure(go.Choroplethmapbox(z=z,  # This is the data.
#                                     locations=locations,
#                                     colorscale='reds',
#                                     colorbar=dict(thickness=20, ticklen=3),
#                                     geojson=jdata,
#                                     text=locations,
#                                     hoverinfo='all',
#                                     marker_line_width=1,
#                                     marker_opacity=0.75))
# #
# #
# fig.update_layout(title_text='Symptom Map',
#                   title_x=0.5, width=700, height=700,
#                   mapbox=dict(center= dict(lat=54, lon=-120),
#                               style='carto-positron',
#                               zoom=5.6,
#                               ))
# fig.show()

