import plotly.graph_objects as go
import pandas as pd

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


#print(df.columns)
usa = df.loc[(df["Country"] == "United States")]

usa_df1 = usa.groupby('State')
usa_df1 = pd.DataFrame(usa_df1.sum().reset_index())

usa_df2 = usa.groupby(['State', 'Product_Category'])
usa_df2 = pd.DataFrame(usa_df2.sum().reset_index())

#print(isinstance(usa, pd.DataFrame))


#new = usa.loc[usa.Product_Category == 'Bikes'][['State', 'Revenue']]

#print(new['Revenue'])

#usa_df2['state_code'] = usa_df2.apply(lambda row: label_code(row), axis=1)
#bikes_df2 = usa_df2[usa_df2["Product_Category"] == 'Bikes']
#accessories_df2 = usa_df2[usa_df2["Product_Category"] == 'Accessories']
#clothing_df2 = usa_df2[usa_df2["Product_Category"] == 'Clothing']




# Kentucky, Virginia, South Carolina
#print(bikes_df2["State"][~bikes_df2["State"].isin(usa_df2["State"])])


usa_df2.drop(
    usa_df2.columns.difference(['Product_Category', 'Revenue', 'State']),
    1, inplace=True)

usa_df2 = usa_df2.pivot(index='State', columns='Product_Category', values='Revenue')

usa_df2 = usa_df2.fillna(0)

usa_df2.reset_index(level=0, inplace=True)
usa_df2['state_code'] = usa_df2.apply(lambda row: label_code(row), axis=1)

usa_df2['Revenue'] = usa_df2['Accessories'] + usa_df2['Bikes'] + usa_df2['Clothing']

for col in usa_df2.columns:
    usa_df2[col] = usa_df2[col].astype(str)
#
usa_df2['text'] = 'State: ' + usa_df2['State'] + '<br>' + \
                  'Accessories Rev: $' + usa_df2['Accessories'] + '<br>' + \
                  'Bikes Rev: $' + usa_df2['Bikes'] + '<br>' + \
                  'Clothing Rev: $' + usa_df2['Clothing'] + '<br>' + \
                  'Total Rev: $' + usa_df2['Revenue']


# #
# #
print(usa_df2["Revenue"])
#
usa_map = go.Figure(
            data=[go.Choropleth(
                locationmode='USA-states',
                locations=usa_df2['state_code'],
                z=usa_df2["Revenue"],
                colorscale='Reds',
                text=usa_df2['text']
            )],
)
# # #
#
usa_map.update_layout(
    title_text='2011 US Agriculture Exports by State',
    geo_scope='usa',
    dragmode=False)


#
# #
usa_map.show()


##################################
import pandas as pd
import json
import urllib.request
import requests
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

def read_geojson(url):
    with urllib.request.urlopen(url) as url:
        jdata = json.loads(url.read().decode())
    return jdata


canada_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson'

data = read_geojson(canada_url)

for i in data["features"]:
    print(i["properties"])

for i in data["features"]:
    property = i['properties']
    property['id'] = property.pop('cartodb_id')

canada = df.loc[(df["Country"] == "Canada")]

canada = df.groupby(["Country", "State", "Product_Category"])

canada = pd.DataFrame(canada.sum().reset_index())

my_map = go.Figure(
            data=[go.Choropleth(
                geojson=data,
                locations=canada["State"],
                text=canada["State"],
                z=canada["Revenue"],
                colorscale='reds',
                colorbar=dict(
                    title='Revenue',
                    thickness=20,
                    ticklen=3),
                hoverinfo='all',
                marker_line_width=1,
                marker_opacity=0.75)]
        )

        my_map.update_layout(title_text='Canada Map',
                          title_x=0.5, width=700, height=700,
                          geo=dict(
                              lataxis=dict(range=[40, 70]),
                              lonaxis=dict(range=[-130, -55]),
                              scope="north america")
                          )
