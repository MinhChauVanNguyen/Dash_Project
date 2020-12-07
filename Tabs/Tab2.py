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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from Data.data_processing import my_df

df = my_df

np.set_printoptions(precision=2)
pd.options.mode.chained_assignment = None

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            html.Div(
                children='Pearson correlation',
                style={'fontWeight': 'bold'}
            ),
            html.Hr(),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    children=[
                        dcc.Loading(dcc.Graph(id='heatmap')),
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
                children=['Feature Importance bar plot & Regression performance table'],
                style={'fontWeight': 'bold'}
            ),
            html.Hr(),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(children=[
                    dcc.Loading(dcc.Graph(
                        id="feature_imp"
                    ))],
                    width={'size': 6}
                ),
                dbc.Col(children=[
                    html.Div(
                        id='tab2_tbl_title',
                        style={'text-align': 'center'}
                    ),
                    html.Div(
                        id="perf_table",
                        style={'marginRight': 20}
                    )],
                    width={'offset': 1}
                )
            ])
        ]),
])


@app.callback(
    [Output(component_id='heatmap', component_property='figure'),
     Output(component_id='feature_imp', component_property='figure'),
     Output(component_id='perf_table', component_property='children'),
     Output(component_id='tab2_tbl_title', component_property='children')
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

    heatmap_data = heatmap_data.apply(
        lambda x: pd.factorize(x)[0] if x.dtypes == 'object' else x).corr(
        method='pearson', min_periods=1
    )

    heatmap = ff.create_annotated_heatmap(
        z=heatmap_data.values,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        annotation_text=heatmap_data.round(2).values,
        colorscale=['#d5d1cd', '#f9b464', '#afc2ca', '#8ca1a4', '#968a92', '#c4bbbe', '#5dbcd2']
    )

    heatmap['layout']['xaxis'].update(side='bottom')
    heatmap['layout'].update(
        title=f'<b>Feature correlation heatmap for {selected_state} in {selected_country}</b>',
        margin=dict(l=0, r=0, t=40, b=0),
        title_font_size=19,
        title_font_color='#2c8cff',
        title_x=0.5
    )

    heatmap.update_traces(dict(showscale=False))

    # encode categorical variable
    X = data.drop('Revenue', axis=1)
    y = data['Revenue']

    label_encoder = LabelEncoder()

    for i, col in enumerate(X):
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Split the data into train and test sets
    if X.size == 0 and y.size == 0:
        return dash.no_update, dash.no_update, [], []

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
        y='Variable',
        x='Importance',
        orientation='h',
        color='Indicator',
        title='<b>Coefficients as Feature Importance<b>' if selected_model == 'Linear' or selected_model == 'Poly' or selected_model == 'SVM'
                      else '<b>Decision Trees Feature Importance<b>',
        color_discrete_map={
            'Positive': '#e59050',
            'Negative': '#f3d1ae'
        }
    )

    plot_imp.update_layout(
        legend=dict(
            xanchor="center",
            yanchor="top",
            y=-0.15,
            x=0,
            orientation='h'
        ),
        legend_traceorder='reversed',
        legend_title_text='<b>Color Indicator</b>',
        margin=dict(l=0, r=0, t=40, b=0),
        title_font_size=19,
        title_font_color='#2c8cff',
        title_x=0.6,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
        xaxis_range=[-1, 1]
    )

    plot_imp.update_xaxes(
        gridcolor='#d0e1e8',
        zeroline=False
    )

    # R^2 AND ADJUSTED R^2
    r2 = r2_score(y_test.tolist(), y_pred.tolist())
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) /
                              (X_train.shape[0] - X_train.shape[1] - 1)))

    mse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=True)
    rmse = mean_squared_error(y_test.tolist(), y_pred.tolist(), squared=False)

    mae = mean_absolute_error(y_test.tolist(), y_pred.tolist())

    mse_table = pd.DataFrame({
        'Statistic': [u'R\u00b2', 'adjusted ' + u'R\u00b2', 'MSE', 'RMSE', 'MAE'],
        'Definition': ['assumes that every single variable explains the variation in the dependent variable. '
                       ' The higher the value is, the better the model is.',
                       'the percentage of variation explained by only the independent variables '
                       'that actually affect the dependent variable',
                       'the difference between the original and predicted values extracted by squared '
                       'the average difference over the data set.',
                       'the error rate by the square root of MSE',
                       'the difference between the original and predicted values extracted '
                       'by averaged the absolute difference over the data set'
                       ],
        'Value': [r2, adj_r2, mse, rmse, mae]
    })

    mse_table.Value = mse_table.Value.round(4)

    perf_table = html.Div([
        dt.DataTable(
            columns=[{'name': col, 'id': col} for col in mse_table.columns],
            data=mse_table.to_dict(orient='records'),
            style_header={
                'backgroundColor': '#d0e3e8',
                'color': '#2c8cff',
                'fontWeight': 'bold',
                'fontSize': 14
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px'
            },
            style_cell={
                'maxWidth': '200px',
                'font-family': 'Helvetica Neue, sans-serif',
                'textAlign': 'center'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Statistic'},
                 'fontWeight': 'bold',
                 },
                {'if': {'column_id': 'Definition'},
                 'fontSize': 11.5,
                 'textAlign': 'left'
                 }
            ]
        )
    ])

    tab2_table_title = html.Div(html.H5(f'Error metrics table for {selected_model}'))

    return heatmap, plot_imp, perf_table, tab2_table_title



