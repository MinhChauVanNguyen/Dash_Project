"""
Created on Tues, 27 Oct 2020
@author: Minh Chau Van Nguyen
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import plotly.express as px
import requests
from Data.helper_function import label_code
from Data.data_processing import *

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

year_list = df["Year"].unique().tolist()

# Convert Country and State data into dictionary
country_df = df[["Country", "State"]]

my_dict = dict()

for i in country_df['Country'].unique().tolist():
    country = country_df[country_df['Country'] == i]
    my_dict[i] = country['State'].unique().tolist()

all_options = {
    'Age_Group': df['Age_Group'].unique().tolist(),
    'Customer_Gender': df['Customer_Gender'].unique().tolist()
}

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = False


def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H3("Title"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score."
            ),
            html.Br(),
            html.P("Select Country"),
            dcc.Dropdown(id="slct_country",
                         options=[{'label': c, 'value': c} for c in my_dict.keys()],
                         multi=False,
                         value="France",
                         clearable=False
                         ),
            html.Br(),
            html.Div(id='state_label', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(id="slct_state", clearable=False),
            html.Br(),
            html.P("Select year"),
            dcc.Dropdown(id="slct_year",
                         options=[{"label": i, "value": i} for i in year_list],
                         value=year_list[:],
                         multi=True,
                         clearable=False
                         ),
            html.Br(),
            html.P("Select Group"),
            dcc.Dropdown(
                id='slct_group',
                options=[{'label': k, 'value': k} for k in all_options.keys()],
                value='Age_Group',
                clearable=False
            ),
            html.Br(),
            html.P("Select sub group"),
            dcc.RadioItems(id='slct_subgrp'),
        ],
    )


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("Logo.png")),
                html.H5("TITLE")
            ],
        ),
        # html.Div(
        #     id="left-column",
        #     className="three columns",
        #     children=[
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Tab one', value='tab-1'),
            dcc.Tab(label='Tab two', value='tab-2'),
        ]),
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card()]
        ),
        html.Div(id='tabs-content')
    ]
)


@app.callback(Output(component_id='tabs-content', component_property='children'),
              [Input(component_id='tabs', component_property='value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
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
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

# chained callbacks for Country and State
@app.callback(
    Output(component_id='slct_state', component_property='options'),
    [Input(component_id='slct_country', component_property='value')])
def set_states_options(selected_country):
    return [{'label': i, 'value': i} for i in my_dict[selected_country]]


@app.callback(
    Output(component_id='slct_subgrp', component_property='options'),
    [Input(component_id='slct_group', component_property='value')])
def set_group_options(selected_group):
    return [{'label': i, 'value': i} for i in all_options[selected_group]]


@app.callback(
    Output(component_id='slct_state', component_property='value'),
    [Input(component_id='slct_state', component_property='options')])
def set_states_value(country_options):
    return country_options[0]['value']


@app.callback(
    Output(component_id='slct_subgrp', component_property='value'),
    [Input(component_id='slct_subgrp', component_property='options')])
def set_subgroup_value(group_option):
    return group_option[0]['value']


# callbacks for outputs
@app.callback(
    [Output(component_id='state_label', component_property='children'),
     Output(component_id='bar_graph', component_property='figure'),
     Output(component_id='data-table', component_property='children')
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
    elif selected_country == 'Australia':
        container = "Select Region"
    elif selected_country == 'Canada':
        container = "Select Province"
    else:
        container = "Select State"

    # if selected_year is None:
    #     # PreventUpdate prevents ALL outputs updating
    #     raise dash.exceptions.PreventUpdate

    grouped_df = df.loc[
        (df["Country"] == selected_country) & (df["State"] == selected_state) & (df["Year"].isin(selected_year))]

    # grouped_df = grouped_df.groupby(["State", selected_group, "Product_Category"])

    grouped_df = grouped_df.groupby(['State', selected_group, 'Product_Category']).agg({'Revenue': 'sum'})

    # grouped_df = pd.DataFrame(grouped_df.sum().reset_index())

    state_pcts = grouped_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

    grouped_df['Perc'] = state_pcts

    grouped_df.reset_index(inplace=True)

    if selected_group == "Age_Group":
        name_sort = {'Youth (<25)': 0, 'Young Adults (25-34)': 1, 'Adults (35-64)': 2, 'Seniors (64+)': 3}
        grouped_df['name_sort'] = grouped_df['Age_Group'].map(name_sort)

        grouped_df = grouped_df.sort_values(by='name_sort', ascending=True)
        grouped_df = grouped_df.drop('name_sort', axis=1)
    else:
        grouped_df = grouped_df

    table_data = grouped_df.copy()
    table_data['Revenue'] = table_data['Revenue'].map('${:,.0f}'.format)

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
            data=table_data.to_dict('records'),
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
                        'filter_query': '{Customer_Gender} eq "Female"',
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Customer_Gender} eq "Male"',
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
        y='Revenue',
        x=selected_group,
        color='Product_Category',
        text='Perc',
        category_orders={"Product_Category": ["Clothing", "Bikes", "Accessories"]},
        opacity=0.6,
        color_discrete_map={
            'Accessories': '#0059b2',
            'Bikes': '#4ca6ff',
            'Clothing': '#99ccff'},
        hover_data={'Revenue': ':$,.0f', 'Product_Category': True},
        labels={
            'Age_Group' if selected_group == 'Age_Group' else 'Customer_Gender': '<b>Age group</b>' if selected_group == 'Age_Group' else '<b>Gender</b>',
            'Revenue': '<b>Revenue</b>',
            'Product_Category': '<b>Product Category</b>'
        },
    )

    fig.update_layout(
        width=400,
        height=500,
        yaxis_tickprefix='$',
        # plot_bgcolor='rgba(0,0,0,0)',
        legend_traceorder="reversed",
        legend=dict(yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    orientation="h"),
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
        texttemplate='%{text:.2f}%'
    )
    # opacity=0.6)

    return container, fig, table


@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_group', component_property='value'),
     Input(component_id='slct_subgrp', component_property='value')
     ]
)
def update_my_map(selected_country, selected_year, selected_group, selected_subgroup):
    if selected_country == 'France':
        container = "Select Department"
    elif selected_country == 'Australia':
        container = "Select Region"
    elif selected_country == 'Canada':
        container = "Select Province"
    else:
        container = "Select State"

    # container = update_output()[0]
    container = container.split(' ')[1]

    data = df.loc[(df["Country"] == selected_country) & (df["Year"].isin(selected_year))]

    data = data.groupby(['State', selected_group, 'Product_Category'])
    data = pd.DataFrame(data.sum().reset_index())

    data.drop(data.columns.difference(['State', 'Product_Category', 'Revenue', selected_group]), 1, inplace=True)

    data = data.pivot(index=['State', selected_group], columns=['Product_Category'], values='Revenue')

    data = data.fillna(0)

    data.reset_index(level=['State', selected_group], inplace=True)

    data = data[data[selected_group] == selected_subgroup]

    if selected_country == 'United States':
        data['state_code'] = data.apply(lambda row: label_code(row), axis=1)

    data['Accessories2'] = data['Accessories'].groupby(data['State']).transform('sum')

    data['Bikes2'] = data['Bikes'].groupby(data['State']).transform('sum')

    data['Clothing2'] = data['Clothing'].groupby(data['State']).transform('sum')

    data['Revenue'] = data['Accessories2'] + data['Bikes2'] + data['Clothing2']

    data.rename(columns={'State': 'id'}, inplace=True)

    data = data.sort_values(by=['Revenue'], ascending=False)

    data['Revenue'] = data['Revenue'].map('${:,.0f}'.format)

    data['Revenue'] = data['Revenue'].astype(str)

    if selected_country == "United States":
        my_map = px.choropleth(
            data_frame=data,
            locationmode='USA-states',
            locations='state_code',
            scope="usa",
            color='Revenue',
            hover_name="id",
            hover_data={'id': False,
                        'state_code': False,
                        'Accessories2': ':$,.0f',
                        'Bikes2': ':$,.0f',
                        'Clothing2': ':$,.0f'},
            labels={'Revenue': 'Total Revenue',
                    'Accessories2': 'Accessories',
                    'Bikes2': 'Bikes',
                    'Clothing2': 'Clothing'
                    },
            title='<b>USA map</b>'
            # template='plotly_dark'
        )

        my_map.add_scattergeo(
            locationmode='USA-states',
            locations=data['state_code'],
            text=data['state_code'],
            mode='text',
            hoverinfo='none',
            showlegend=False
        )

    else:
        if selected_country == 'Canada':
            country_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson'
            scope = 'north america'
        elif selected_country == 'France':
            country_url = 'https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson'
            scope = 'europe'

        elif selected_country == 'Germany':
            country_url = 'https://gist.githubusercontent.com/oscar6echo/4423770/raw/990e602cd47eeca87d31a3e25d2d633ed21e2005/dataBundesLander.json'
            scope = 'europe'
        else:
            country_url = 'https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.json'
            scope = 'world'

        json_data = requests.get(country_url).json()

        for i in json_data["features"]:
            property = i['properties']
            if selected_country == "Australia":
                property['id'] = property.pop('STATE_NAME')
            elif selected_country == "Canada":
                property['id'] = property.pop('name')
            elif selected_country == "France":
                property['id'] = property.pop('nom')
            else:
                property['id'] = property.pop('NAME_1')

        map_title = (selected_country, 'map')

        my_map = px.choropleth(
            data_frame=data,
            geojson=json_data,
            locations='id',
            featureidkey='properties.id',
            color='id' if selected_country == 'France' else 'Revenue',
            hover_data={
                'Accessories2': ':$,.0f',
                'Bikes2': ':$,.0f',
                'Clothing2': ':$,.0f',
                'Revenue': True},
            labels={
                'Revenue': 'Total Revenue',
                'id': container,
                'Accessories2': 'Accessories',
                'Bikes2': 'Bikes',
                'Clothing2': 'Clothing'
            },
            color_continuous_scale='Magma',
            scope=scope,
            title='<b>' + " ".join(map_title) + '</b>',
            center=dict(lat=48.864716, lon=2.349014) if selected_country == 'France' else None,
        )

        my_map.update_geos(
            visible=False,  # hide the base map and frame
            showcountries=True,
            showcoastlines=False,
            showland=False,
            fitbounds="locations",  # automatically zoom the map to show just the area of interest
            resolution=110
        )

        my_map.add_scattergeo(
            geojson=json_data,
            locations=data['id'],
            text=None if selected_country == 'France' else data['id'],
            showlegend=False,
            featureidkey="properties.id",
            mode='text',
            hoverinfo='none'
        )

    my_map.update_traces(
        marker_line_width=1,
        marker_line_color='white'
    )

    my_map.update_layout(
        # dragmode=False,
        margin=dict(l=20, r=0, t=50, b=20),
        legend_title_text='<b>Total Revenue</b>',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=12)
        ),
        title_x=0.5,
        uniformtext_minsize=12,
        uniformtext_mode='hide'
    )

    return my_map


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)