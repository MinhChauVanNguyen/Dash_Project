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

from Data.data_processing import my_df

df = my_df

np.set_printoptions(precision=2)
pd.options.mode.chained_assignment = None

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            html.Hr(),
            dbc.Row(
                dbc.Col(
                    children=[
                        dcc.Graph(id='heatmap'),
                    ]
                )
            )
        ]
    ),
    html.Br(),
    html.Div(
        id="bike_volume_card",
        children=[
            html.Div(
                id="r2",
                style={'fontWeight': 'bold'}
            ),
            html.Hr(),
            dbc.Row(children=[
                dbc.Col(children=[
                    dcc.Graph(
                        id="feature_imp"
                    )],
                )
            ])
        ]),
])


@app.callback(
    [Output(component_id='heatmap', component_property='figure'),
     Output(component_id='feature_imp', component_property='figure'),
     Output(component_id='r2', component_property='children')
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_variable', component_property='value'),
     Input(component_id='slct_model', component_property='value')
     ]
)
def output_predict(selected_country, selected_state, selected_variable, selected_model):
    data = df[(df["Country"] == selected_country) & (df["State"] == selected_state)]

    data = data[selected_variable + ['Revenue']]

    data['Day'] = data['Day'].apply(str)
    data['Year'] = data['Year'].apply(str)

    heatmap_data = data.copy()

    heatmap_data=heatmap_data.drop(['Country', 'State'], axis=1)

    heatmap_data = heatmap_data.apply(
        lambda x: pd.factorize(x)[0] if x.dtypes == 'object' else x).corr(
        method='pearson', min_periods=1
    )

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
    X = data.drop('Revenue', axis=1)
    y = data['Revenue']

    print(X)

    label_encoder = LabelEncoder()

    for i, col in enumerate(X):
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Split the data into train and test sets
    if X.size == 0 and y.size == 0:
        return dash.no_update, dash.no_update, []

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

    plot_imp = px.bar(
        feat_imp,
        x='Variable',
        y='Importance',
        color='Indicator',
        title='Coefficients as Feature Importance' if selected_model == 'Linear' or selected_model == 'Poly' or selected_model == 'SVM'
                      else 'Decision Trees Feature Importance'
    )

    plot_imp.update_layout(title_x=0.5)

    plot_imp.update_xaxes(
        tickfont=dict(size=11),
        #ticktext=['<b>Year</b>', '<b>Age Group</b>', '<b>Customer Gender</b>', '<b>Profit</b>'],
        tickvals=selected_variable
    )

    # R^2 AND ADJUSTED R^2
    r2 = r2_score(y_test.tolist(), y_pred.tolist())
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) /
                              (X_train.shape[0] - X_train.shape[1] - 1)))
    metrics = u'R\u00b2' + ': {}'.format(r2) + ', adjusted ' + u'R\u00b2' + ': {}'.format(adj_r2)

    # mse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)
    # rmse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)
    return heatmap, plot_imp, metrics



