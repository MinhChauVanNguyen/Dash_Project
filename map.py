import pandas as pd
import requests
import plotly.express as px

germany_url = 'https://gist.githubusercontent.com/oscar6echo/4423770/raw/990e602cd47eeca87d31a3e25d2d633ed21e2005/dataBundesLander.json'

germany_regions_geo = requests.get(germany_url).json()

print(germany_regions_geo['features'][0]['properties'])

# df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')
#
# germany = df.loc[(df["Country"] == "Germany")]
#
# germany = germany.groupby(['State', 'Product_Category'])
# germany = pd.DataFrame(germany.sum().reset_index())
#
# germany.drop(germany.columns.difference(['State', 'Product_Category', 'Revenue']), 1, inplace=True)
#
# germany = germany.pivot(index='State', columns='Product_Category', values='Revenue')
#
# germany = germany.fillna(0)
#
# germany.reset_index(level=0, inplace=True)
#
# germany['Revenue'] = germany['Accessories'] + germany['Bikes'] + germany['Clothing']
#
# germany.rename(columns={'State': 'NAME_1'}, inplace=True)
#
# for col in germany.columns:
#     germany[col] = germany[col].astype(str)
# #
# germany['text'] = 'State: ' + germany['NAME_1'] + '<br>' + \
#                   'Accessories Rev: $' + germany['Accessories'] + '<br>' + \
#                   'Bikes Rev: $' + germany['Bikes'] + '<br>' + \
#                   'Clothing Rev: $' + germany['Clothing'] + '<br>' + \
#                   'Total Rev: $' + germany['Revenue']
#
# germany['Revenue'] = germany['Revenue'].apply(pd.to_numeric)
#
# fig = px.choropleth(data_frame=germany,
#                     geojson=germany_regions_geo,
#                     locations='NAME_1',
#                     featureidkey='properties.NAME_1',
#                     color='Revenue',
#                     hover_data=['text'],
#                     color_continuous_scale="Magma",
#                     scope="europe")
#
# fig.update_geos(
#     scope="europe",
#     visible=False,
#     showcountries=True,
#     showcoastlines=False,
#     showland=False,
#     fitbounds="locations",
#     showsubunits=True,
#     subunitcolor="Blue",
#     resolution=110)
#
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()