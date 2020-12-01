import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table as dt

import plotly.express as px
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

from app import app

from Data.regression_function import update_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

from Data.data_processing import tab12_df

df = tab12_df


np.set_printoptions(precision=2)
pd.options.mode.chained_assignment = None

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            dbc.Row(
              children=[
                html.Div(
                    "Switch between graph and table outputs",
                    style={'display': 'inline-block', 'marginRight': 10}),
                dcc.RadioItems(
                    id='slct_output',
                    options=[
                        {'label': "Graph", 'value': "Graph"},
                        {'label': "Table", 'value': "Table"},
                    ],
                    value='Graph',
                    labelStyle={'display': 'inline-block', 'marginRight': 10}
              )]
            ),
            dbc.Row(
                dbc.Col(
                    children=[
                        html.Div(
                            id="scatter_id", 
                            children=[dcc.Graph(id="scatter")], 
                            style={'display': 'block'}),
                        html.Div(
                            id="table_id",
                            children=[dt.DataTable(
                                    id='table_two',
                                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                                    style_cell={'textAlign': 'center'}
                            )],
                            style={'display': 'block'}
                        )
                    ]
                  )
            )
        ]
    ),
    html.Div(
        id="bike_volume_card",
        children=[
            html.Div(id="r2"),
            html.Hr(),
            dbc.Row(children=[
                dbc.Col(children=[
                    dcc.Graph(
                        id='heatmap'
                    ),
                ], width={"size": 6}
                ),
                dbc.Col(children=[
                    dcc.Graph(
                        id="feature_imp",
                    )],
                    width={"size": 6}
                )
            ])
        ]),
])


@app.callback(
    [Output(component_id='heatmap', component_property='figure'),
     Output(component_id='scatter', component_property='figure'),
     Output(component_id='table_two', component_property='data'),
     Output(component_id='table_two', component_property='columns'),
     Output(component_id='feature_imp', component_property='figure'),
     Output(component_id='r2', component_property='children'),
     Output(component_id='scatter_id', component_property='style'),
     Output(component_id='table_id', component_property='style'),
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_variable', component_property='value'),
     Input(component_id='slct_model', component_property='value'),
     Input(component_id='slct_output', component_property='value')
     ]
)
def output_predict(selected_country, selected_state, selected_variable, selected_model, selected_radio):

    data = df[(df["Country"] == selected_country) & (df["State"] == selected_state)]

    ind_variable = selected_variable.copy()

    try:
        ind_variable.remove("Profit")
    except ValueError:
        pass
    finally:
        data = data.groupby(ind_variable).agg('sum')

    data.reset_index(inplace=True)

    heatmap_data = data.copy()
    heatmap_data = heatmap_data.apply(
        lambda x: pd.factorize(x)[0] if x.name in ['Age_Group', 'Customer_Gender', 'Year'] else x).corr(
        method='pearson', min_periods=1)

    heatmap = ff.create_annotated_heatmap(
        z=heatmap_data.values,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        annotation_text=heatmap_data.round(2).values,
        showscale=True
    )

    heatmap['layout']['xaxis'].update(side='bottom')

    heatmap.update_traces(dict(showscale=False))

    # encode categorical variable
    for i in range(len(selected_variable)):
        if selected_variable[i] != "Profit":
            data[selected_variable[i]] = LabelEncoder().fit_transform(data[selected_variable[i]].values)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values


    # Split the data into train and test sets
    if X.size ==0 and y.size ==0:
        return dash.no_update, dash.no_update, [], [], dash.no_update, [], {}, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # Fit the model on the train set
    poly_reg = PolynomialFeatures(degree=2)
    X_train_poly = poly_reg.fit_transform(X_train)

    if selected_model == "SVM":
        regressor = SVR(kernel='linear', gamma=1e-8)
    elif selected_model == "Decision":
        regressor = DecisionTreeRegressor(random_state=0)
    elif selected_model == "Random":
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    else:
        regressor = LinearRegression()

    if selected_model == "Poly":
        regressor.fit(X_train_poly, y_train)
        y_pred = regressor.predict(poly_reg.fit_transform(X_test))
    else:
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=['Y predict'])
    y_test_df = pd.DataFrame(y_test, columns=['Y test'])
    frames = [y_pred_df, y_test_df]
    results = pd.concat(frames, axis=1)

    results_table = results.copy()

    # SCATTER PLOT
    results['Y_pred'] = results['Y predict'].map('{:,.0f}'.format)
    results['Y_test'] = results['Y test'].map('{:,}'.format)

    results['text'] = 'Y test : ' + results['Y_test'].astype(str) + '<br>' + \
                      'Y predict : ' + results['Y_pred'].astype(str)

    annotations = []
    for i, row in results.iterrows():
        annotations.append(
            dict(x=row["Y test"],
                 y=row["Y predict"],
                 text=row["text"],
                 xref="x",
                 yref="y",
                 showarrow=True,
                 bordercolor='pink',
                 borderpad=4,
                 ax=20,
                 ay=-30,
                 align="right",
                 bgcolor="#abd7eb",
                 opacity=0.8
                 )
        )

    scatter_plt = px.scatter(results, x="Y test", y="Y predict")

    scatter_plt.update_layout(
        annotations=annotations
    )

    # TABLE
    results_table['Y predict'] = results_table['Y predict'].map('{:,.0f}'.format)
    results_table['Y test'] = results_table['Y test'].map('{:,}'.format)

    columns = [{'name': col, 'id': col} for col in results_table.columns]
    data = results_table.to_dict(orient='records')

    # FEATURE IMPORTANCE
    if selected_model == "Linear":
        importance = [round(num, 3) for num in regressor.coef_]
    elif selected_model == "Poly":
        importance = np.delete(regressor.coef_, 0)
        importance = importance[0:len(selected_variable)]
    elif selected_model == "Decision" or selected_model == "Random":
        importance = [round(num, 3) for num in regressor.feature_importances_]
    else:
        importance = list(regressor.coef_.flatten())

    feat_imp = pd.DataFrame({
        'Variable': [x for x in selected_variable],
        'Importance': importance
    })

    feat_imp["Indicator"] = np.where(feat_imp["Importance"] < 0, 'Negative', 'Positive')

    plot_imp = px.bar(feat_imp,
                      x='Variable',
                      y='Importance',
                      color='Indicator',
                      title='Coefficients as Feature Importance'
                      if selected_model == 'Linear' or selected_model == 'Poly' or selected_model == 'SVM'
                      else 'Decision Trees Feature Importance'
    )

    plot_imp.update_layout(title_x=0.5)

    plot_imp.update_xaxes(tickfont=dict(size=11),
                          ticktext=['<b>Year</b>', '<b>Age Group</b>', '<b>Customer Gender</b>', '<b>Profit</b>'],
                          tickvals=selected_variable
                          )

    # R^2 AND ADJUSTED R^2
    r2 = r2_score(y_test.tolist(), y_pred.tolist())
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) /
                              (X_train.shape[0] - X_train.shape[1] - 1)))
    metrics = u'R\u00b2' + ': {}'.format(r2) + ', adjusted ' + u'R\u00b2' + ': {}'.format(adj_r2)

    # mse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)
    # rmse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)

    if selected_radio == 'Table':
        return heatmap, scatter_plt, data, columns, plot_imp, metrics, {'display': 'none'}, {'display': 'block'}
    if selected_radio == 'Graph':
        return heatmap, scatter_plt, data, columns, plot_imp, metrics, {'display': 'block'}, {'display': 'none'}



