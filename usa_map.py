import plotly.graph_objects as go
import plotly.express as px

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')




usa = df.loc[(df["Country"] == "United States")]

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


usa_df = usa.groupby(['State', 'Product_Category'])
usa_df = pd.DataFrame(usa_df.sum().reset_index())





# Kentucky, Virginia, South Carolina
#print(bikes_df2["State"][~bikes_df2["State"].isin(usa_df2["State"])])


usa_df.drop(
    usa_df.columns.difference(['Product_Category', 'Revenue', 'State']),
    1, inplace=True)

usa_df = usa_df.pivot(index='State', columns='Product_Category', values='Revenue')

usa_df = usa_df.fillna(0)

usa_df.reset_index(level=0, inplace=True)
usa_df['state_code'] = usa_df.apply(lambda row: label_code(row), axis=1)

usa_df['Revenue'] = usa_df['Accessories'] + usa_df['Bikes'] + usa_df['Clothing']

for col in usa_df.columns:
    usa_df[col] = usa_df[col].astype(str)
#
usa_df['text'] = 'State: ' + usa_df['State'] + '<br>' + \
                  'Accessories Rev: $' + usa_df['Accessories'] + '<br>' + \
                  'Bikes Rev: $' + usa_df['Bikes'] + '<br>' + \
                  'Clothing Rev: $' + usa_df['Clothing'] + '<br>' + \
                  'Total Rev: $' + usa_df['Revenue']

usa_map = px.choropleth(
        data_frame=usa_df,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Revenue',
        hover_data=['text'],
        template='plotly_dark'
    )

# usa_map = go.Figure(
#             data=[go.Choropleth(
#                 locationmode='USA-states',
#                 locations=usa_df2['state_code'],
#                 z=usa_df2["Revenue"],
#                 colorscale='Reds',
#                 text=usa_df2['text']
#             )],
# )
#
# usa_map.update_layout(
#     title_text='2011 US Agriculture Exports by State',
#     geo_scope='usa',
#     dragmode=False)


usa_map.show()


