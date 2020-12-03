import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, Output
import plotly.express as px
import requests

from app import app
from Data.helper_functions import label_code, label_state
from Data.data_processing import tab12_df

df = tab12_df

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            dbc.Row(
                children=[
                    html.Div(
                        html.B("Switch between graph and table outputs"),
                        style={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 15}),
                    dcc.RadioItems(
                        id='slct_tab1_output',
                        options=[
                            {'label': "Graph", 'value': "Graph"},
                            {'label': "Table", 'value': "Table"},
                        ],
                        value='Table',
                        labelStyle={'display': 'inline-block', 'marginRight': 10},
                        inputStyle={"marginRight": 6}
                    )]
            ),
            html.Hr(),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(
                    children=[
                        html.Div(
                            id='tab1_bar',
                            children=[
                                dcc.Loading(dcc.Graph(id="bar_graph"))
                            ],
                            style={'display': 'block'}
                        )
                    ]
                )
            ]
            ),
            html.Div(
                id='table_title',
                style={'text-align': 'center'}
            ),
            dbc.Row(
                children=[
                    dbc.Col(),
                    dbc.Col(
                        children=[
                            html.Div(
                                id='data_table',
                                className='center',
                                style={'display': 'block'}
                            )
                        ],
                    ),
                    dbc.Col()
                ],
            )
        ]
    ),
    html.Br(),
    html.Div(
        id="bike_volume_card",
        children=[
            html.B('Map of Revenue'),
            html.Hr(),
            dcc.Loading(
                children=[dcc.Graph(id="map")],
                type="graph"
            )
        ]
    )
])


# callbacks for outputs
@app.callback(
    [Output(component_id='bar_graph', component_property='figure'),
     Output(component_id='data_table', component_property='children'),
     Output(component_id='tab1_bar', component_property='style'),
     Output(component_id='data_table', component_property='style'),
     Output(component_id='table_title', component_property='children')
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_group', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_tab1_output', component_property='value')
     ]
)
def update_output(selected_country, selected_state, selected_group, selected_year, selected_value):
    grouped_df = df.loc[
        (df["Country"] == selected_country) & (df["State"] == selected_state) & (df["Year"].isin(selected_year))]

    # try:
    #     grouped_df = grouped_df[(grouped_df["Year"].isin(selected_year))]
    #     if grouped_df.empty:
    #         raise Exception("Data is empty")
    # except Exception:
    #     print("Data is empty")

    grouped_df = grouped_df.groupby(['State', selected_group, 'Product_Category']).agg({'Revenue': 'sum'})

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
                {'name': 'Age_Group' if selected_group == "Age_Group" else 'Gender', 'id': selected_group,
                 'type': 'text', 'editable': True},
                {'name': 'Product Category', 'id': 'Product_Category', 'type': 'text', 'editable': True},
                {'name': 'Revenue', 'id': 'Revenue', 'type': 'numeric', 'editable': True}
            ],
            data=table_data.to_dict('records'),
            style_table={
                'maxHeight': '20%',
                'width': '30%',
                'minWidth': '10%',
            },
            editable=True,
            style_header={'backgroundColor': '#d0e3e8', 'color': '#2c8cff', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'auto', 'height': 'auto', 'width': 'auto'},
            style_cell={'textAlign': 'left', 'font-family': 'Helvetica Neue, sans-serif'},
            style_data_conditional=([
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Adults (35-64)"',
                    },
                    'backgroundColor': '#f9b464',
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Seniors (64+)"',
                    },
                    'backgroundColor': '#e59050',
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Young Adults (25-34)"',
                    },
                    'backgroundColor': '#e0b184',
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Youth (<25)"',
                    },
                    'backgroundColor': '#f3d1ae',
                }
                if selected_group == "Age_Group" else
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
        title=f'<b>Revenue by Product types and {selected_group} for {selected_state} in {selected_country}<b>',
        data_frame=grouped_df,
        y='Revenue',
        x=selected_group,
        color='Product_Category',
        text='Perc',
        category_orders={"Product_Category": ["Accessories", "Bikes", "Clothing"]},
        # opacity=0.6,
        color_discrete_map={
            'Accessories': '#e59050',
            'Bikes': '#e0b184',
            'Clothing': '#f3d1ae'},
        hover_data={
            'Revenue': ':$,.0f',
            'Product_Category': True,
            'Perc': False
        },
        labels={
            'Age_Group' if selected_group == 'Age_Group' else 'Customer_Gender': '<b>Age group</b>' if selected_group == 'Age_Group' else '<b>Gender</b>',
            'Revenue': '<b>Revenue</b>',
            'Product_Category': '<b>Product Category</b>'
        }
    )

    fig.update_layout(
        yaxis_tickprefix='$',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            xanchor="right",
            x=1,
            bgcolor='#E2E2E2',
            bordercolor='#828184',
            borderwidth=1
        ),
        xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
        margin=dict(l=20, r=0, t=40, b=20),
        title_x=0.53,
        font=dict(family="Helvetica Neue, sans-serif"),
        hoverlabel=dict(
            bgcolor="#E2E2E2",
            font_family="Helvetica Neue, sans-serif"
        ),
        barmode="group",
        title_font_size=19,
        title_font_color='#2c8cff'
    )

    fig.update_traces(
        marker_line_color='black',  # bar border color
        marker_line_width=1.5,
        texttemplate='%{text:.1f}%',
        textposition='outside',
        textfont_size=14,
        width=0.25,
        textfont=dict(
            color="#2c8cff"
        )
    )

    table_title = html.H5(f'Revenue by Product types and {selected_group} for {selected_state} in {selected_country}')

    if selected_value == 'Graph':
        return fig, [], {'display': 'block'}, {'display': 'none'}, []
    else:
        return dash.no_update, table, {'display': 'none'}, {'display': 'flex'}, table_title


@app.callback(
     Output(component_id='map', component_property='figure'),
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_group', component_property='value'),
     Input(component_id='slct_subgrp', component_property='value')
     ]
)
def update_my_map(selected_country, selected_year, selected_group, selected_subgroup):
    container = label_state(country=selected_country)

    container = container.split(' ')[1]

    data = df.loc[(df["Country"] == selected_country) & (df["Year"].isin(selected_year))]

    data = data.groupby(['State', selected_group, 'Product_Category'])

    data = pd.DataFrame(data.sum().reset_index())
    data.drop(data.columns.difference(['State', 'Product_Category', 'Revenue', selected_group]), 1, inplace=True)

    data = data.pivot(index=['State', selected_group], columns=['Product_Category'], values='Revenue')

    data = data.fillna(0)

    data.reset_index(level=['State', selected_group], inplace=True)

    data = data[data[selected_group] == selected_subgroup]

    for c in data:
        if type(data[c]) != 'object':
            data['Revenue'] = data.sum(axis=1)

    data.rename(columns={'State': 'id'}, inplace=True)

    data = data.sort_values(by=['Revenue'], ascending=False)
    data['Revenue'] = data['Revenue'].map('${:,.0f}'.format)
    data['Revenue'] = data['Revenue'].astype(str)

    data2 = data.copy()

    hover_data = {}
    for c in data2.columns[2:-1]:
        hover_data[c] = ':$,.0f'

    colors = ['#7794ac', '#f0af46', '#acbdca', '#6b7077', '#b0ada7', '#d0ada7', '#d0e1e8',
              '#d5d1cd', '#f9b464', '#afc2ca', '#8ca1a4', '#968a92', '#c4bbbe', '#5dbcd2',
              '#e59050', '#d0dfe1', '#f3d1ae', '#acadb0', '#828184']

    map_title = f'<b>Geographical distribution of Revenue by {selected_subgroup} in {selected_country}<b>'

    if selected_country == "United States":
        data['state_code'] = data.apply(lambda row: label_code(row), axis=1)
        my_map = px.choropleth(
            data_frame=data,
            locationmode='USA-states',
            locations='state_code',
            scope="usa",
            color='Revenue',
            hover_name="id",
            color_discrete_sequence=colors,
            hover_data=hover_data,
            labels={'Revenue': 'Total Revenue'},
            title=map_title
        )

        my_map.add_scattergeo(
            locationmode='USA-states',
            locations=data['state_code'],
            text='<b>' + data['state_code'] + '</b>',
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

        my_map = px.choropleth(
            data_frame=data,
            geojson=json_data,
            locations='id',
            featureidkey='properties.id',
            color='Revenue',
            hover_data=hover_data,
            labels={'Revenue': 'Total Revenue', 'id': container},
            scope=scope,
            color_discrete_sequence=colors,
            title=map_title,
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
            text=None if selected_country == 'France' else '<b>' + data['id'] + '</b>',
            showlegend=False,
            featureidkey="properties.id",
            mode='text',
            hoverinfo='none'
            # textfont=dict(
            #     color="black"
            # )
        )

    my_map.update_traces(
        marker_line_width=1,
        marker_line_color='white'
    )

    my_map.update_layout(
        margin=dict(l=40, r=0, t=50, b=20),
        legend_title_text='<b>Total Revenue</b>',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=12),
        ),
        title_x=0.5,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        title_font_size=19,
        title_font_color='#2c8cff',
        hoverlabel=dict(
            bgcolor="#E2E2E2",
            font_family="Helvetica Neue, sans-serif"
        )
    )

    return my_map
