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
            id="description-card",
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
                html.Div(
                    id='state_label',
                    style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(id="slct_state", clearable=False),
                html.Div(
                    id='year_label',
                    children=[
                        html.Br(),
                        html.P(html.Strong("Select Year"))
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id="year",
                    children=[
                        dcc.Dropdown(
                            id="slct_year",
                            options=[{"label": i, "value": i} for i in year_list],
                            value=year_list[:],
                            multi=True,
                            clearable=False)
                    ],
                    style={'display': 'inline-block'}
                ),
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
                html.Div(
                    id='subgrp_label',
                    children=[
                        html.P(html.Strong("Select Sub group")),
                        dcc.RadioItems(
                            id='slct_subgrp',
                            inputStyle={"margin-right": "10px"}
                        )
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id="Model_elements",
                    children=[
                        html.P(html.Strong("Select Regression Model")),
                        dcc.Dropdown(
                            id='slct_model',
                            options=[
                                {'label': "Multiple Linear", 'value': "Linear"},
                                {'label': "Multivariate Polynomial", 'value': "Poly"},
                                {'label': "Support Vector Machine", 'value': "SVM"},
                                {'label': "Decision Tree", 'value': "Decision"},
                                {'label': "Random Forest", 'value': "Random"}
                            ],
                            value='Linear',
                            clearable=False
                        )
                    ],
                    style={'display': 'block'}
                )
            ],
            style={'marginBottom': 50, 'marginTop': 25, 'marginLeft': 15, 'marginRight': 15}
        ),
        width={"size": 3, "order": "first"},
        align="start"
      ),
      dbc.Col(
        html.Div(
          children=[
            dcc.Tabs(id="tabs", value='tab-2', children=[
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
    [Output(component_id='state_label', component_property='children'),
     Output(component_id='slct_state', component_property='options')],
    [Input(component_id='slct_country', component_property='value')])
def set_states_options(selected_country):
    if selected_country == 'France':
        container = "Select Department"
    elif selected_country == 'Australia':
        container = "Select Region"
    elif selected_country == 'Canada':
        container = "Select Province"
    else:
        container = "Select State"
    return container, [{'label': i, 'value': i} for i in my_dict[selected_country]]


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


@app.callback(
    [Output(component_id='year_label', component_property='style'),
     Output(component_id='year', component_property='style'),
     Output(component_id='subgrp_label', component_property='style'),
     Output(component_id='slct_subgrp', component_property='style')],
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-1':
        return {'display': 'block'}, {'display': 'inline-block'}, {'display': 'block'}, {'display': 'block'}
    if tab == 'tab-2':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output(component_id='Model_elements', component_property='style'),
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-2':
        return {'display': 'block'}
    if tab == 'tab-1':
        return {'display': 'none'}

