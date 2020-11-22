from dash.dependencies import Input, Output
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from app import app
from Data import data_processing

df = data_processing.df
np.set_printoptions(precision=2)
pd.options.mode.chained_assignment = None

layout = html.Div(
    html.Div(children=[
     dbc.Row(children=[
        dbc.Col(children=[
            dt.DataTable(
                id='table_two',
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_cell={'textAlign': 'center'}
            )], width={"size": 4}
        ),
        dbc.Col(children=[
            html.P(
                id="feature_imp",
                style={"border": "1px solid black"}
            )],
            width={"size": 4, "offset": 1}
         )
     ]),
     dbc.Row(children=[
        dbc.Col(html.Div(id="accuracy"))
     ])
    ], style={'marginLeft': 30, 'marginTop': 40})
)


@app.callback(
    [Output(component_id='table_two', component_property='data'),
     Output(component_id='table_two', component_property='columns'),
     Output(component_id='feature_imp', component_property='children'),
     Output(component_id='accuracy', component_property='children')
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_variable', component_property='value'),
     Input(component_id='slct_model', component_property='value')
     ]
)
def output_predict(selected_country, selected_state, selected_variable, selected_model):
    data = df.loc[(df["Country"] == selected_country) & (df["State"] == selected_state)]

    # add Revenue
    selected_variable.append('Revenue')

    data = data[selected_variable]

    # remove Revenue
    selected_variable.pop()

    ind_variable = selected_variable.copy()

    try:
        ind_variable = selected_variable.copy()
        ind_variable.remove("Profit")
    except ValueError:
        pass
    finally:
        data = data.groupby(ind_variable).agg('sum')

    data.reset_index(inplace=True)

    for i in range(len(selected_variable)):
        if selected_variable[i] != "Profit":
            data[selected_variable[i]] = LabelEncoder().fit_transform(data[selected_variable[i]].values)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model on the train set
    poly_reg = PolynomialFeatures(degree=2)
    X_train_poly = poly_reg.fit_transform(X_train)

    if selected_model == "SVM":
        regressor = SVR(kernel='rbf', gamma=1e-8)
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

    results['Y predict'] = results['Y predict'].map('{:,.0f}'.format)
    results['Y test'] = results['Y test'].map('{:,}'.format)

    columns = [{'name': col, 'id': col} for col in results.columns]
    data = results.to_dict(orient='records')

    my_list = []

    if selected_model == "Linear" or selected_model == "Poly":
        importance = [round(num, 3) for num in regressor.coef_]
    elif selected_model == "Decision" or selected_model == "Random":
        importance = [round(num, 3) for num in regressor.feature_importances_]
    else:
        importance = my_list

    feat_imp = pd.DataFrame({
        'cols': [x for x in selected_variable],
        'imps': importance
    })

    print(feat_imp)

    if selected_model == "SVM":
        my_list = "Feature Importance is not available for non-linear kernel"
    else:
        for i, v in enumerate(importance):
            my_list.append('Feature: %0d, Score: %.5f' % (i, v))

    return data, columns, my_list, r2_score(y_test.tolist(), y_pred.tolist())
