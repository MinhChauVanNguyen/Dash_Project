import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app import app

from Data import data_processing

df = data_processing.df

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

layout = html.Div(children=[
    dbc.Row(children=[
      dbc.Col(
        html.Div(
          children=[
            html.H3("Title"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score."),
            html.Br(),
            html.P("Select Country"),
            dcc.Dropdown(id="slct_country",
                         options=[{'label': c, 'value': c} for c in my_dict.keys()],
                         multi=False,
                         value="France",
                         clearable=False),
            html.Br(),
            html.Div(id='state_label', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(id="slct_state", clearable=False),
            html.Br(),
            html.P("Select Year"),
            dcc.Dropdown(id="slct_year",
                         options=[{"label": i, "value": i} for i in year_list],
                         value=year_list[:],
                         multi=True,
                         clearable=False),
            html.Br(),
            html.P("Select Group"),
            dcc.Dropdown(
                id='slct_group',
                options=[
                    {'label': "Age group", 'value': "Age_Group"},
                    {'label': "Gender", 'value': "Customer_Gender"},
                ],
                value='Age_Group',
                clearable=False),
            html.Br(),
            html.P("Select Sub group"),
            dcc.RadioItems(id='slct_subgrp', inputStyle={"margin-right": "10px"})
          ],
            id="description-card",
            style={'marginBottom': 50, 'marginTop': 25, 'marginLeft': 15, 'marginRight': 15}
        ),
        width={"size": 3, "order": "first"},
        align="start"
      ),
      dbc.Col(
        html.Div(
          children=[
            dcc.Tabs(id="tabs", value='tab-1', children=[
                dcc.Tab(label='Descriptive statistics', value='tab-1'),
                dcc.Tab(label='Regression results', value='tab-2'),
            ]),
            html.Div(id='tabs-content')
          ]
        ),
        width={"size": 9, "order": "last"},
        align="center"
      )
    ])
])


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