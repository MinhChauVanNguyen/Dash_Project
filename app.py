"""
Created on Tues, 27 Oct 2020
@author: Minh Chau Van Nguyen
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import json
import requests

#from draft import update_map

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

# Read data
df = pd.read_csv(
    'https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')



df.drop(
    df.columns.difference(['Customer_Gender', 'Age_Group', 'Product_Category', 'Revenue', 'State', 'Country', 'Year']),
    1, inplace=True)

df = df.sort_values(by="Year")

year_list = df["Year"].unique().tolist()

# Convert Country and State data into dictionary
country_df = df[["Country", "State"]]

my_dict = dict()

for i in country_df['Country'].unique().tolist():
    country = country_df[country_df['Country'] == i]
    my_dict[i] = country['State'].unique().tolist()


def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H3("Title"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
            html.Br(),
            html.P("Select Country"),
            dcc.Dropdown(id="slct_country",
                         options=[{'label': c, 'value': c} for c in my_dict.keys()],
                         multi=False,
                         value="Canada",
                         clearable=False
                         ),
            html.Br(),
            html.Div(id='state_label', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(id="slct_state", clearable=False),
            html.Br(),
            html.P("Grouped by:"),
            dcc.Dropdown(
                id='slct_group',
                options=[
                    {'label': 'Age group', 'value': 'Age_Group'},
                    {'label': 'Gender', 'value': 'Customer_Gender'},
                ],
                value='Age_Group',
                clearable=False
            ),
            html.Br(),
            html.P("Select year"),
            dcc.Dropdown(id="slct_year",
                         options=[{"label": i, "value": i} for i in year_list],
                         value=year_list[:],
                         multi=True,
                         clearable=False
                         ),
        ],
    )


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    id="app-container",

    children=[

        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("Logo.png")),
                html.H5("TITLE")
            ],
        ),

        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card()]
        ),

        # Right column
        html.Div(
            id="right-column",
            className="nine columns",
            children=[
                # State Bar graph
                html.Div(
                    id="bike_volume_card",
                    children=[
                        html.B("Total bike related products bought by selected country, state and group"),
                        html.Hr(),
                        html.Div(children=[
                            html.Div(
                                children=[dcc.Graph(id="bar_graph")],
                                className="five columns"),
                            html.Div(
                                children=[html.Div(id='data-table')],
                                style={'margin-top': '20px'},
                                className="four columns offset-by-one tableDiv")
                        ], className="row")
                    ]
                ),
                html.Div(
                    id="map_card",
                    children=[
                        html.Div(children=[
                            dcc.Graph(id="map")
                        ])
                    ]
                )
            ],

        ),
    ],
)


def read_geojson(url):
    with urllib.request.urlopen(url) as url:
        jdata = json.loads(url.read().decode())
    return jdata

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# chained callbacks for Country and State
@app.callback(
    Output(component_id='slct_state', component_property='options'),
    [Input(component_id='slct_country', component_property='value')])
def set_states_options(selected_country):
    return [{'label': i, 'value': i} for i in my_dict[selected_country]]


@app.callback(
    Output(component_id='slct_state', component_property='value'),
    [Input(component_id='slct_state', component_property='options')])
def set_states_value(country_options):
    return country_options[0]['value']


# callbacks for outputs
@app.callback(
    [Output(component_id='state_label', component_property='children'),
     Output(component_id='bar_graph', component_property='figure'),
     Output(component_id='data-table', component_property='children'),
     Output(component_id='map', component_property='figure')
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_group', component_property='value'),
     Input(component_id='slct_year', component_property='value')
     ]
)
def update_output(selected_country, selected_state, selected_group, selected_year):
    if selected_country == 'France':
        container = "Select Department"
    elif (selected_country == 'Australia' or selected_country == 'United Kingdom'):
        container = "Select Region"
    elif selected_country == 'Canada':
        container = "Select Province"
    else:
        container = "Select State"

    if selected_year is None:
        # PreventUpdate prevents ALL outputs updating
        raise dash.exceptions.PreventUpdate

    grouped_df = df.loc[
        (df["Country"] == selected_country) & (df["State"] == selected_state) & (df["Year"].isin(selected_year))]

    grouped_df = grouped_df.groupby(["Country", "State", selected_group, "Product_Category"])

    grouped_df = pd.DataFrame(grouped_df.sum().reset_index())

    if selected_group == "Age_Group":
        name_sort = {'Youth (<25)': 0, 'Young Adults (25-34)': 1, 'Adults (35-64)': 2, 'Seniors (64+)': 3}
        grouped_df['name_sort'] = grouped_df['Age_Group'].map(name_sort)

        grouped_df = grouped_df.sort_values(by='name_sort', ascending=True)
        grouped_df = grouped_df.drop('name_sort', axis=1)

    else:
        grouped_df = grouped_df

    grouped_df = grouped_df

    # Table
    table = html.Div([
        dt.DataTable(
            columns=[
                {'name': 'Age group' if selected_group == "Age_Group" else 'Gender', 'id': selected_group,
                 'type': 'text', 'editable': True},
                {'name': 'Product Category', 'id': 'Product_Category', 'type': 'text', 'editable': True},
                {'name': 'Revenue', 'id': 'Revenue', 'type': 'numeric', 'editable': True}
            ],
            # columns=[{"name": i, "id": i} for i in grouped_df.columns],
            data=grouped_df.to_dict('records'),
            style_table={
                'maxHeight': '20%',
                # 'overflowY': 'scroll',
                'width': '30%',
                'minWidth': '10%',
            },
            editable=True,
            style_header={'backgroundColor': 'rgb(30, 30, 30)',
                          'color': 'white'},
            style_data={'whiteSpace': 'auto', 'height': 'auto', 'width': 'auto'},
            style_cell={'textAlign': 'left'},
            style_data_conditional=([
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Adults (35-64)"',
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Seniors (64+)"',
                    },
                    'backgroundColor': 'hotpink',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Young Adults (25-34)"',
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Youth (<25)"',
                    },
                    'backgroundColor': '#7FDBFF',
                    'color': 'white'
                } if selected_group == "Age_Group" else
                {
                    'if': {
                        'filter_query': '{Customer_Gender} eq "F"',
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Customer_Gender} eq "M"',
                    },
                    'backgroundColor': '#7FDBFF',
                    'color': 'white'
                }
            ])
        )
    ])

    # Plotly Express
    fig = px.bar(
        data_frame=grouped_df,
        x=selected_group,
        y='Revenue',
        color='Product_Category',
        opacity=0.6,
        color_discrete_map={
            'Accessories': '#0059b2',
            'Bikes': '#4ca6ff',
            'Clothing': '#99ccff'},
        hover_data={'Revenue': ':,.0f'},
        labels={
            'Age_Group' if selected_group == 'Age_Group' else 'Customer_Gender': '<b>Age group</b>' if selected_group == 'Age_Group' else '<b>Gender</b>',
            'Revenue': '<b>Revenue</b>',
            'Product_Category': '<b>Product category</b>'}
    )
    #
    fig.update_layout(
        width=400,
        height=500,
        yaxis_tickprefix='$',
        # plot_bgcolor='rgba(0,0,0,0)',
        legend_traceorder="reversed",
        legend=dict(yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    orientation="h"),
        # bordercolor="Black",
        # borderwidth=1.5),
        xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1.5, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True),
        margin=dict(l=50, r=20, t=30, b=20),
        title_x=0.53,
        font=dict(family="Helvetica Neue, sans-serif"),
        hoverlabel=dict(
            bgcolor="white",
            font_family="Helvetica Neue, sans-serif"
        )
    )

    fig.update_traces(
        marker_line_color='black',  # bar border color
        marker_line_width=1.5,
        hovertemplate='<b>Age group</b>: %{x} <br>'
                      # '<b>Product category</b>: %{name} <br>'
                        '<b>Revenue</b>: %{y}')
    # opacity=0.6)

    ### USA
    usa = df.loc[(df["Country"] == "United States") & (df["Year"].isin(selected_year))]

    usa = usa.groupby(['State', 'Product_Category'])
    usa = pd.DataFrame(usa.sum().reset_index())

    usa.drop(
        usa.columns.difference(['Product_Category', 'Revenue', 'State']),
        1, inplace=True)

    usa = usa.pivot(index='State', columns='Product_Category', values='Revenue')

    usa = usa.fillna(0)

    usa.reset_index(level=0, inplace=True)

    usa['state_code'] = usa.apply(lambda row: label_code(row), axis=1)

    usa['Revenue'] = usa['Accessories'] + usa['Bikes'] + usa['Clothing']

    for col in usa.columns:
        usa[col] = usa[col].astype(str)

    usa['text'] = '<b>State</b>: ' + usa['State'] + '<br>' + \
                      '<b>Accessories Rev</b>: $' + usa['Accessories'] + '<br>' + \
                      '<b>Bikes Rev</b>: $' + usa['Bikes'] + '<br>' + \
                      '<b>Clothing Rev</b>: $' + usa['Clothing'] + '<br>' + \
                      '<b>Total Rev</b>: $' + usa['Revenue']

    if selected_country == "United States":
        my_map = go.Figure(
            data=[
                go.Choropleth(
                    colorbar=dict(title='Revenue', ticklen=3),
                    locationmode='USA-states',
                    locations=usa['state_code'],
                    z=usa["Revenue"],
                    colorscale='Reds',
                    text=usa['text'],
            ),
            ],
            layout=dict(geo={'subunitcolor': 'black'})
        )

        my_map.update_layout(
            title_text='USA map',
            geo_scope='usa',
            dragmode=False
        )
    elif selected_country == "Canada":
        my_map = update_map(
            selected_country='Canada',
            country_url='https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson',
            json_state='name',
            scope='north america'
        )
    elif selected_country == "Germany":
        my_map = update_map(
            selected_country='Germany',
            country_url='https://gist.githubusercontent.com/oscar6echo/4423770/raw/990e602cd47eeca87d31a3e25d2d633ed21e2005/dataBundesLander.json',
            json_state='NAME_1',
            scope='europe'
        )
    elif selected_country == "France":
        my_map = update_map(
            selected_country='France',
            country_url='https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson',
            json_state='nom',
            scope='europe')
    elif selected_country == "Australia":
        my_map = update_map(
            selected_country="Australia",
            country_url='https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.json',
            json_state='STATE_NAME',
            scope='world')
    else:
        return None

    return container, fig, table, my_map


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
