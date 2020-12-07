import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app import app

from Data.data_processing import tab12_df, my_df
from Data.helper_functions import label_state

df = tab12_df

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

# Tab 2 variables
tab2_vars = my_df.drop(['Revenue', 'Country', 'State'], axis=1)
tab2_var_list = tab2_vars.columns.tolist()

# Tab 3 variables
my_df = my_df.drop(['Customer_Gender', 'Country', 'State'], axis=1)
tab3_var_list = my_df.columns.tolist()


layout = html.Div(
 children=[
    dbc.Row(children=[
      dbc.Col(
        html.Div(
            id="description-card",
            children=[
                html.H3("Sales Dash Analytics"),
                html.Div(id="intro"),
                html.Br(),
                html.P("Select Country"),
                dcc.Dropdown(id="slct_country",
                                 options=[{'label': c, 'value': c} for c in my_dict.keys()],
                                 multi=False,
                                 value="France",
                                 clearable=False,
                                 searchable=False,
                                 style={'backgroundColor': 'rgba(255, 204, 255, 0.6)'}
                ),
                html.Br(),
                html.Div(
                    id='state_label',
                    style={'font-weight': 'bold', 'margin-bottom': '5px'}
                ),
                dcc.Dropdown(
                    id="slct_state",
                    clearable=False,
                    style={'backgroundColor': 'rgba(0, 204, 153, 0.6)'}
                ),
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
                html.Div(
                 id='subgrp_label',
                 children=[
                     html.Br(),
                     html.P(html.Strong("Select Group")),
                     dcc.Dropdown(
                        id='slct_group',
                        options=[
                            {'label': "Age group", 'value': "Age_Group"},
                            {'label': "Gender", 'value': "Customer_Gender"},
                        ],
                        value='Age_Group',
                        clearable=False),
                     html.Br(),
                     html.P(html.Strong("Select Sub group")),
                     dcc.RadioItems(
                        id='slct_subgrp',
                        inputStyle={"margin-right": "10px", "color": "#ff9966"}
                     )
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id="tab_1_input",
                    children=[
                        html.Br(),
                        html.P(html.B("Input Color Guide")),
                        html.Hr(),
                        html.Span(u'\u25A2', style={'color': 'black', 'backgroundColor': '#ffccff', 'marginRight': 2}),
                        html.P("Input for Bar graph, Table and Map", style={'display': 'inline-block'}),
                        html.Hr(),
                        html.Span(u'\u25A2', style={'color': 'black', 'backgroundColor': '#2f7ed8', 'marginRight': 2}),
                        html.P("Input for Bar graph and Table", style={'display': 'inline-block'}),
                        html.Hr(),
                        html.Span(u'\u25A2', style={'color': 'black', 'backgroundColor': '#00cc99', 'marginRight': 2}),
                        html.P("Input for Map", style={'display': 'inline-block'}),
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id='tab2_variables',
                    children=[
                        html.Br(),
                        html.P(html.Strong("Select Independent Variable")),
                        dcc.Dropdown(
                            id='slct_variable',
                            options=[{"label": i, "value": i} for i in tab2_var_list],
                            value=['Age_Group', 'Customer_Gender', 'Date', 'Day', 'Month', 'Year',
                                   'Customer_Age', 'Product_Category', 'Sub_Category', 'Product'],
                            clearable=False,
                            multi=True
                        ),
                        html.Br()
                    ],
                    style={'display': 'inline-block'}
                ),
                html.Div(
                    id="tab2_models",
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
                            clearable=False,
                            style={'backgroundColor': 'rgba(255, 204, 255, 0.6)'}
                        )
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id="tab_2_input",
                    children=[
                        html.Br(),
                        html.P(html.B("Input Color Guide")),
                        html.Hr(),
                        html.Span(u'\u25A2', style={'color': 'black', 'backgroundColor': '#ffccff', 'marginRight': 2}),
                        html.P("Input for Heat map, Bar plot & Table", style={'display': 'inline-block'})
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id='tab3_variables',
                    children=[
                        html.Br(),
                        html.P(html.Strong("Select Independent Variable")),
                        dcc.Dropdown(
                            id='slct_variables',
                            options=[{"label": i, "value": i} for i in tab3_var_list],
                            value=tab3_var_list[:],
                            clearable=False,
                            multi=True
                        ),
                        html.Br()
                    ],
                    style={'display': 'inline-block'}
                ),
                html.Div(
                    id="tab3_models",
                    children=[
                        html.P(html.Strong("Select Classification Model")),
                        dcc.Dropdown(
                            id='slct_class',
                            options=[
                                {'label': "Logistic Regression", 'value': "Logistic Regression"},
                                {'label': "Support Vector Machine", 'value': "Support Vector Machine"},
                                {'label': "K-Nearest Neighbors", 'value': "K-Nearest Neighbors"},
                                {'label': "Naive Bayes", 'value': "Naive Bayes"},
                                {'label': "Decision Tree", 'value': "Decision Tree"},
                                {'label': "Random Forest", 'value': "Random Forest"}
                            ],
                            value='Logistic Regression',
                            clearable=False,
                            style={'backgroundColor': 'rgba(255, 204, 255, 0.6)'}
                        )
                    ],
                    style={'display': 'block'}
                ),
                html.Div(
                    id="tab_3_input",
                    children=[
                        html.Br(),
                        html.P(html.B("Input Color Guide")),
                        html.Hr(),
                        html.Div(
                            children=[
                                html.Span(u'\u25A2',
                                          style={'color': 'black', 'backgroundColor': '#ffccff', 'marginRight': 2}),
                                "Input for Confusion Matrix, ROC-AUC curve, Accuracy table, Classification table"
                                "and Feature importance plot."
                            ],
                            style={'display': 'inline-block'}
                        )

                    ],
                    style={'display': 'block'}
                ),
                html.Div(style={'marginBottom': 80}),
                #html.Div(
                 #   id='wrap',
                  #  children=[
                       # html.Footer(
                          # children=[
                          #   html.P('Statistical analysis & Data visualisation by'),
                          #   html.Img(
                          #       src=app.get_asset_url("Logo.png"),
                          #       style={'height': '6rem'}
                          #   )
                          # ]
                       # )
                    #],
                #),
            ],
            style={'marginLeft': 15, 'marginRight': 15}
        ),
        width={"size": 3, "order": "first"},
        align="start"
      ),
      dbc.Col(
        html.Div(
          children=[
            dcc.Tabs(id="tabs", value='tab-3', children=[
                dcc.Tab(label='Descriptive statistics', value='tab-1'),
                dcc.Tab(label='Revenue regression', value='tab-2'),
                dcc.Tab(label='Binary classification', value='tab-3'),
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
    Output(component_id='intro', component_property='children'),
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-1':
        return html.Div([html.Em('Exploratory analysis'),
                      ': observe the patterns and key trends in product types and total '
                      'revenues by Country, State, Age group and Gender.'])
    elif tab == 'tab-2':
        return html.Div([
            html.Em('Regression analysis'),
            ': building and comparing predictive models to determine '
            'which variable(s) is/are likely to have an impact on Revenue.'])
    else:
        return html.Div([
            html.Em('Classification analysis'),
            ': differentiate the characteristics between Female and Male '
            'classes in terms of purchasing bikes related products'])


@app.callback(
    [Output(component_id='state_label', component_property='children'),
     Output(component_id='slct_state', component_property='options')],
    [Input(component_id='slct_country', component_property='value')])
def set_states_options(selected_country):
    container = label_state(country=selected_country)
    return container, [{'label': i, 'value': i} for i in my_dict[selected_country]]


@app.callback(
    Output(component_id='slct_state', component_property='value'),
    [Input(component_id='slct_state', component_property='options')])
def set_states_value(country_options):
    return country_options[0]['value']


@app.callback(
    Output(component_id='slct_subgrp', component_property='options'),
    [Input(component_id='slct_group', component_property='value')])
def set_group_options(selected_group):
    return [{'label': i, 'value': i} for i in all_options[selected_group]]
@app.callback(
    Output(component_id='slct_subgrp', component_property='value'),
    [Input(component_id='slct_subgrp', component_property='options')])
def set_subgroup_value(group_option):
    return group_option[0]['value']


@app.callback(
    [Output(component_id='year_label', component_property='style'),
     Output(component_id='year', component_property='style'),
     Output(component_id='subgrp_label', component_property='style'),
     Output(component_id='slct_subgrp', component_property='style'),
     Output(component_id='tab_1_input', component_property='style'),
     Output(component_id='slct_state', component_property='style')
     ],
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-1':
        return {'display': 'block'}, {'display': 'inline-block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, \
               {'backgroundColor': 'rgba(47, 126, 216, 0.5)'}
    if tab == 'tab-2' or tab == 'tab-3':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
               {'backgroundColor': 'rgba(255, 204, 255, 0.6)'}


@app.callback(
    [Output(component_id='tab2_models', component_property='style'),
     Output(component_id='tab2_variables', component_property='style'),
     Output(component_id='tab_2_input', component_property='style')],
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-2':
        return {'display': 'block'}, {'display': 'inline-block'}, {'display': 'block'}
    if tab == 'tab-1' or tab == 'tab-3':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


@app.callback(
    [Output(component_id='tab3_models', component_property='style'),
     Output(component_id='tab3_variables', component_property='style'),
     Output(component_id='tab_3_input', component_property='style'),
     ],
    [Input(component_id='tabs', component_property='value')])
def show_hide_element(tab):
    if tab == 'tab-3':
        return {'display': 'block'}, {'display': 'inline-block'}, {'display': 'block'}
    if tab == 'tab-1' or tab == 'tab-2':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
