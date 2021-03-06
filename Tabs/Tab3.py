from app import app
from Data.data_processing import my_df
import pandas as pd
from Data.helper_functions import model_information, select_classification

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_table as dt

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.feature_selection import SelectFromModel

df = my_df
# refactor Customer Gender
df['Customer_Gender'] = df['Customer_Gender'].replace({'F': 0, 'M': 1})

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            html.Div(
                id='model_information',
                style={'display': 'inline-block', 'font-weight': 'bold', 'font-style': 'italic'}
            ),
            html.Div(
                children=[
                    dcc.RadioItems(
                        id='cards_radio',
                        options=[
                            {'label': "Show facts about model", 'value': "show"},
                            {'label': "Hide facts about model", 'value': "hide"}
                        ],
                        value='hide',
                        labelStyle={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 10},
                        inputStyle={"marginRight": 6}
                    )
                ],
                style={'display': 'inline-block'}
            ),
            html.Hr(),
            html.Div(
                id="model_cards",
                style={'display': 'inline-block', 'marginBottom': 25},
                children=[dbc.Row([
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card([
                                dbc.CardHeader(html.H4("Fact #1")),
                                dbc.CardBody(
                                    id="card_1",
                                    children=[
                                        html.Div(id="card_1_text", className="card-text")
                                    ]
                                )], color='primary'
                            )
                        ]
                    ),
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card([
                                dbc.CardHeader(html.H4("Fact #2")),
                                dbc.CardBody(
                                    id="card_2",
                                    children=[
                                        html.Div(id="card_2_text", className="card-text")
                                    ]
                                )], color='primary'
                            )
                        ]
                    ),
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card([
                                dbc.CardHeader(html.H4("Fact #3")),
                                dbc.CardBody(
                                    id="card_3",
                                    children=[
                                        html.Div(id="card_3_text", className="card-text")
                                    ]
                                )
                            ], color='primary'
                            )
                        ]
                    )
                ]
                ),
                    html.Br()
                ]
            ),
            html.Br(),
            html.Div(
                id="hidden",
                children=[
                    dbc.Row(
                        children=[
                            html.Div(children=[
                                html.B("Optimise model by hyperparameters tuning"),
                                html.Small(
                                    " (Please be aware that model tuning can take a long time due to lengthy combination of hyperparameters)")],
                                style={'marginRight': 10, 'marginLeft': 10, 'display': 'inline-block'}),
                            dcc.RadioItems(
                                id='optimise',
                                options=[
                                    {'label': "Don't tune model", 'value': "not_optimised"},
                                    {'label': "Tune model",
                                     'value': "optimised"}
                                ],
                                value='not_optimised',
                                labelStyle={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 30},
                                inputStyle={"marginRight": 6}
                            )
                        ]
                    ),
                    html.Br(),
                    dbc.Row(children=[
                        dbc.Col(
                            children=[
                                dcc.Loading(
                                    id="load1",
                                    children=[dcc.Graph(id="confusion_mat")]
                                )
                            ],
                            width={"size": 5},
                        ),
                        dbc.Col(
                            children=[
                                dcc.Loading(
                                    id="load2",
                                    children=[
                                        html.Div(
                                            children=[dcc.Graph(id='roc_curve')]
                                        )
                                    ]
                                )
                            ],
                            width={"size": 7},
                        )]
                    )],
                style={'display': 'block'}
            ),
            dbc.Row(children=[
                dbc.Col(
                    children=[
                        html.Div(id='error_message')
                    ],
                )
            ]),
            html.Br(),
            html.Br(),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            html.Div(
                                id="accuracy_table_title",
                                children=[html.H5("Accuracy Table")],
                                style={'color': '#2c8cff', 'display': 'block'}
                            ),
                            dcc.Loading(
                                id="load4",
                                children=[
                                    html.Div(
                                        id='accuracy_table',
                                        style={'marginLeft': 15}
                                    )
                                ]
                            )
                        ],
                        width={"size": 5}
                    ),
                    dbc.Col(
                        children=[
                            html.Div(
                                id="class_table_title",
                                children=[html.H5("Classification Table")],
                                style={'color': '#2c8cff', 'display': 'block'}
                            ),
                            dcc.Loading(
                                id="load3",
                                children=[html.Div(id='classification_table')]
                            )
                        ],
                        width={"size": 5, "offset": 1}
                    )
                ]
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            dcc.Loading(
                                id="load5",
                                children=[
                                    html.Div(
                                        id="hide_plt",
                                        children=[dcc.Graph(id='feature_plt')],
                                        style={'display': 'block'}
                                    )
                                ]
                            ),
                            html.Div(id="hide_message")
                        ],
                        width={"size": 10, "offset": 1}
                    )
                ]
            )
        ],
        style={"margin-top": '10px'}
    ),
]
)


@app.callback(
    Output(component_id='model_information', component_property='children'),
    [Input(component_id='slct_class', component_property='value')]
)
def card_output(selected_model):
    return f'{selected_model} Classification'


@app.callback(
    Output(component_id='model_cards', component_property='style'),
    [Input(component_id='cards_radio', component_property='value')]
)
def card_output(selected_radio):
    if selected_radio == 'hide':
        return {'display': 'none'}
    elif selected_radio == 'show':
        return {'display': 'inline-block'}


@app.callback(
    [Output(component_id='card_1_text', component_property='children'),
     Output(component_id='card_2_text', component_property='children'),
     Output(component_id='card_3_text', component_property='children')],
    Input(component_id='slct_class', component_property='value')
)
def card_text_output(selected_model):
    output = model_information(classification=selected_model)
    return output[0], output[1], output[2]


@app.callback(
    [Output(component_id='hidden', component_property='style'),
     Output(component_id='accuracy_table_title', component_property='style'),
     Output(component_id='class_table_title', component_property='style'),
     Output(component_id='confusion_mat', component_property='figure'),
     Output(component_id='roc_curve', component_property='figure'),
     Output(component_id='error_message', component_property='children'),
     Output(component_id='classification_table', component_property='children'),
     Output(component_id='feature_plt', component_property='figure'),
     Output(component_id='hide_plt', component_property='style'),
     Output(component_id='hide_message', component_property='children'),
     Output(component_id='accuracy_table', component_property='children'),
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_variables', component_property='value'),
     Input(component_id='slct_class', component_property='value'),
     Input(component_id='optimise', component_property='value')]
)
def classification(selected_country, selected_state, selected_variable, selected_model, selected_radio):
    data = df[(df["Country"] == selected_country) & (df["State"] == selected_state)]

    data = data[selected_variable + ['Customer_Gender']]

    message = "Target variable only has one class"

    if len(data['Customer_Gender'].unique()) != 2:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, dash.no_update, dash.no_update, \
               message, [], dash.no_update, {'display': 'none'}, [], []

    X = data.drop('Customer_Gender', axis=1)
    y = data['Customer_Gender']

    X['Day'] = X['Day'].apply(str)
    X['Year'] = X['Year'].apply(str)

    # Step 1. Transform categorical variable

    label_encoder = LabelEncoder()

    for i, col in enumerate(X):
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    if X.size == 0 and y.size == 0:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, dash.no_update, dash.no_update, \
               [], [], dash.no_update, {'display': 'none'}, [], []
    else:

    # Step 2. Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Step 3. Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        classifier = select_classification(classification=selected_model)[0]
        tuned_parameters = select_classification(classification=selected_model)[1]

        kfold = model_selection.KFold(n_splits=3)

        # ROC-AUC CURVE
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        roc = go.Figure()

        roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        # If not optimised, fit the model without hyperparamater tuning
        if selected_radio == 'not_optimised':

        # STep 4. Fit the model using train data
            model = classifier.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

        # Step 5. Make prediction using test data
            y_pred = model.predict(X_test)

            # ROC-AUC CURVE
            y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            roc.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    fill='tozeroy',
                    fillcolor='#f3d1ae',
                    # mode='lines',
                    line=dict(color='#e59050')
                )
            )

            roc_title = f'<b>ROC curve (AUC={auc(fpr, tpr):.4f})<b>'
            caption = ''

            selector = SelectFromModel(estimator=classifier).fit(X_train, y_train)

            if selected_model == "Logistic Regression":
                importance = np.round(selector.estimator_.coef_, 2)
            elif selected_model == "Decision Tree" or selected_model == "Random Forest":
                importance = np.round(selector.estimator_.feature_importances_, 2)
            else:
                importance = np.array([])

            error_message = f'Feature Importance is not available for {selected_model}'

    # If optimised, fit the optimal model using the parameters from hyperparameter tuning
        else:
            grid = GridSearchCV(classifier, tuned_parameters, scoring='roc_auc', cv=kfold, n_jobs=-1)

            # Step 4. Fit the model using train data
            model = grid.fit(X_train, y_train)

            # print('Optimized Parameters: {} '.format(model.best_params_))

            train_score = np.mean(
                cross_val_score(model.best_estimator_, X_train, y_train, scoring='accuracy', cv=kfold))

            test_score = np.mean(cross_val_score(model.best_estimator_, X_test, y_test, scoring='accuracy', cv=kfold))

        # Step 5. Make predictions using test data
            y_pred = model.predict(X_test)

            roc_title = f'<b>ROC curve (Mean AUC={grid.best_score_:.4f})'
            caption = f'<b>{str(model.best_params_).replace("{", "").replace("}", "")}<b>'

            i = 0
            for train, test in grid.cv.split(X_train, y_train):
                probas = grid.fit(X_train[train], y_train.iloc[train]).predict_proba(X_train[test])

                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y_train.iloc[test], probas[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                name = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc)

                color = ['#c4bbbe', '#5dbcd2', '#e59050']

                roc.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        name=name, mode='lines',
                        line=dict(color=color[i])
                    )
                )
                i += 1

            #if selected_model in ['Decision Tree', 'Random Forest']:
            if selected_model == "Decision Tree" or selected_model == "Random Forest":
                importance = np.round(model.best_estimator_.feature_importances_, 2)
            else:
                importance = np.array([])

            error_message = f'Feature Importance is not available for optimised {selected_model}'

    # Step 6. Model evaluation

        ## CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)

        cm_labels = np.array([['FN = ', 'TN = '], ['TP = ', 'FP = ']])

        heat_cm = np.array([
            [np.char.add(cm_labels[0, 0], str(cm[0, 0])),
             np.char.add(cm_labels[0, 1], str(cm[0, 1]))],
            [np.char.add(cm_labels[1, 0], str(cm[1, 0])), np.char.add(cm_labels[1, 1], str(cm[1, 1]))]
        ])

        a = ['female', 'male']
        b = ['female', 'male']

        z_text = [[str(b) for b in a] for a in heat_cm]

        z_text[0] = [s + '<b>' for s in z_text[0]]
        z_text[0] = ['<b>' + s for s in z_text[0]]
        z_text[1] = [s + '<b>' for s in z_text[1]]
        z_text[1] = ['<b>' + s for s in z_text[1]]

        confusion = ff.create_annotated_heatmap(
            z=[[1, 0], [0, 1]],
            x=a, y=b,
            annotation_text=z_text,
            colorscale=[[0, '#5dbcd2'], [1, '#f9b464']]
        )

        confusion.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            title_text='<b>Confusion matrix</b>',
            xaxis=dict(title="Predicted Value"),
            yaxis=dict(title="True value"),
            title_x=0.6,
            title_font_size=19,
            title_font_color='#2c8cff'
        )

        confusion['layout']['xaxis'].update(side='bottom')
        for i in range(len(confusion.layout.annotations)):
            confusion.layout.annotations[i].font.size = 13

        ## ROC-CURVE
        roc.update_layout(
            legend=dict(
                x=0.1,
                y=1,
                traceorder='normal'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            title=roc_title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1, mirror=True, ticks='outside', showline=True, linewidth=1,
                       linecolor='black'),
            xaxis=dict(constrain='domain', mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
            title_x=0.5,
            title_font_size=19,
            title_font_color='#2c8cff',
            annotations=[
                dict(xref='paper',
                     yref='paper',
                     x=0.5, y=0.05,
                     showarrow=False,
                     text=caption)
            ],
            plot_bgcolor='rgba(0,0,0,0)'
        )

        ## CLASSIFICATION TABLE
        perf_metrics = classification_report(y_test, y_pred, output_dict=True)

        table = pd.DataFrame(perf_metrics).transpose()
        table.iloc[2]['precision'] = 0
        table.iloc[2]['recall'] = 0

        table.reset_index(inplace=True)

        table = table.drop('support', axis=1)

        table['precision'] = pd.to_numeric(table['precision'])

        for c in table[['precision', 'recall', 'f1-score']]:
            table[c] = pd.to_numeric(table[c])
            table[c] = table[c].map('{:.2f}'.format)

        table[table.eq('0.00')] = ''

        table[table.eq('1')] = 'Male class'
        table[table.eq('0')] = 'Female class'

        #table = table.rename(columns={'index': 'Metric'})

        table.columns = ['Metric', 'Precision', 'Recall', 'F1-score']

        tab3_table = html.Div([
            dt.DataTable(
                columns=[{'name': col, 'id': col} for col in table.columns],
                data=table.to_dict(orient='records'),
                style_header={'fontWeight': 'bold', 'backgroundColor': '#d0dfe1'},
                style_cell={'textAlign': 'center', 'font-family': 'Helvetica Neue, sans-serif'},
                style_data_conditional=([
                    {
                        'if': {
                            'filter_query': '{Metric} eq "Male class"',
                            'column_id': 'Metric'
                        },
                        'fontWeight': 'bold'

                    },
                    {
                        'if': {
                            'filter_query': '{Metric} eq "Female class"',
                            'column_id': 'Metric'
                        },
                        'fontWeight': 'bold'

                    },
                ]
                )
            )
        ])

        ## ACCURACY TABLE
        male_correct_pred = cm[1, 0] / np.sum(cm)
        female_correct_pred = cm[0, 1] / np.sum(cm)

        accuracy_data = {
            'Data': [f'Train data (n_train = {X_train.shape[0]})', f'Test data (n_test = {X_test.shape[0]})',
                     'Class', 'Male', 'Female'],
            'Score': [train_score, test_score, 'Correct prediction', male_correct_pred, female_correct_pred]
        }

        accuracy_df = pd.DataFrame(accuracy_data, columns=['Data', 'Score'])

        accuracy_df = accuracy_df.rename(columns={'Data': f'Full data (N = {X.shape[0]})'})

        row_indexes = [0, 1, 3, 4]

        for i in row_indexes:
            accuracy_df.loc[i, ['Score']] = accuracy_df.loc[i, ['Score']].apply('{:.2%}'.format)

        accuracy_tb = html.Div([
            dt.DataTable(
                columns=[{'name': col, 'id': col} for col in accuracy_df.columns],
                data=accuracy_df.to_dict(orient='records'),
                style_header={'fontWeight': 'bold', 'backgroundColor': '#d0dfe1'},
                style_cell={'textAlign': 'center', 'font-family': 'Helvetica Neue, sans-serif'},
                style_data_conditional=[
                    {
                        'if': {
                            'row_index': 2,
                        },
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgba(243, 209, 174, 0.5)'
                    }
                ],
            )
        ])

        ## FEATURE IMPORTANCE BAR PLOT
        if importance.size == 0:
            return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, confusion, roc, [], \
                   tab3_table, dash.no_update, {'display': 'none'}, error_message, accuracy_tb
        else:
            feat_name = pd.DataFrame([x for x in selected_variable], columns=['Variable'])
            feat_imp = pd.DataFrame(importance.flatten(), columns=['Importance'])
            dat = pd.concat([feat_name, feat_imp], axis=1)

            dat["Indicator"] = np.where(dat["Importance"] < 0, 'Negative', 'Positive')

            dat['Variable'] = dat['Variable'].str.replace('_', ' ')

            imp_bar = px.bar(
                dat,
                y='Variable',
                x='Importance',
                orientation='h',
                color='Indicator',
                title='<b>Feature Importance<b>',
                color_discrete_map={
                    'Positive': '#e59050',
                    'Negative': '#f3d1ae'
                }
            )

            imp_bar.update_layout(
                title_x=0.5,
                margin=dict(l=0, r=0, t=40, b=0),
                title_font_size=19,
                title_font_color='#2c8cff',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
                yaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1, linecolor='black'),
                legend_title_text='<b>Color Indicator</b>'
            )

            imp_bar.update_xaxes(
                gridcolor='#d0e1e8',
                zeroline=False
            )

            return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, confusion, roc, [], \
                   tab3_table, imp_bar, {'display': 'block'}, [], accuracy_tb
