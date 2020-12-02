from app import app
from Data.data_processing import my_df
import pandas as pd
from Data.helper_function import model_information

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
                    labelStyle={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 10}
                 )
                ],
                style={'display': 'inline-block'}
            ),
            html.Hr(),
            html.Div(
                id="model_cards",
                style={'display': 'inline-block'},
                children=[dbc.Row([
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card(
                            dbc.CardBody(
                                id="card_1",
                                children=[
                                    html.H4("Fact #1", className="card-title"),
                                    html.Div(id="card_1_text", className="card-text")
                                ]
                            ))
                        ]
                    ),
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card(
                            dbc.CardBody(
                                id="card_2",
                                children=[
                                    html.H4("Fact #2", className="card-title"),
                                    html.Div(id="card_2_text", className="card-text")
                                ]
                            ))
                        ]
                    ),
                    dbc.Col(
                        children=[
                            html.Br(),
                            dbc.Card(
                             dbc.CardBody(
                                id="card_3",
                                children=[
                                    html.H4("Fact #3", className="card-title"),
                                    html.Div(id="card_3_text", className="card-text")
                                ]
                             )
                            )
                        ]
                    )
                ]
                )]
            ),
            html.Br(),
            html.Div(
                id="hidden",
                children=[
                    dbc.Row(
                        children=[
                            html.Div(children=[
                                html.B("Optimise model by hyperparameters tuning"),
                                html.Small(" (Please be aware that model tuning can take a long time due to lengthy combination of hyperparameters)")],
                                style={'marginRight': 10, 'marginLeft': 10, 'display': 'inline-block'}),
                            dcc.RadioItems(
                                id='optimise',
                                options=[
                                    {'label': "Don't tune model", 'value': "not_optimised"},
                                    {'label': "Tune model",
                                     'value': "optimised"}
                                ],
                                value='not_optimised',
                                labelStyle={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 20}
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
                    children=[html.Div(id='error_message')],
                )
            ]),
            html.Br(),
            html.Br(),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            html.Div(
                                children=[html.H5("Accuracy Table")],
                                style={'text-align': 'center', 'color': '#2c8cff'}
                            ),
                            dcc.Loading(
                                id="load4",
                                children=[html.Div(id='accuracy_table')]
                            )
                        ],
                        width={"size": 5}
                    ),
                    dbc.Col(
                        children=[
                            html.Div(
                                children=[html.H5("Classification Table")],
                                style={'text-align': 'center', 'color': '#2c8cff'}
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
                            html.Div(
                                id="hide_message"
                            )
                        ],
                        width={"size": 10}
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
        return {'display': 'none'}, dash.no_update, dash.no_update, message, [], dash.no_update, {
            'display': 'none'}, [], []

    X = data.drop('Customer_Gender', axis=1)
    y = data['Customer_Gender']

    X['Day'] = X['Day'].apply(str)
    X['Year'] = X['Year'].apply(str)

    label_encoder = LabelEncoder()
    for i, col in enumerate(X):
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    if X.size == 0 and y.size == 0:
        return {'display': 'none'}, dash.no_update, dash.no_update, [], [], dash.no_update, {'display': 'none'}, [], []
    else:
        # Step 1. Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Step 2. Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        if selected_model == "Logistic Regression":
            classifier = LogisticRegression()
            tuned_parameters = {
                'C': np.linspace(.0001, 1000, 200),
                'penalty': ['l2']
            }
        elif selected_model == "Support Vector Machine":
            classifier = SVC(probability=True)
            tuned_parameters = {
                'kernel': ['rbf', 'linear'],
                'gamma': ['auto', 'scale'],
                'degree': [3, 8],
                'C': [1, 10, 100, 1000]
            }
        elif selected_model == "K-Nearest Neighbors":
            classifier = KNeighborsClassifier()
            tuned_parameters = {
                'leaf_size': list(range(1, 50)),
                'n_neighbors': list(range(1, 30)),
                'p': [1, 2]
            }
        elif selected_model == "Naive Bayes":
            classifier = GaussianNB()
            tuned_parameters = {
                'var_smoothing': np.logspace(0, -9, num=100)
            }
        elif selected_model == "Decision Tree":
            classifier = DecisionTreeClassifier()
            tuned_parameters = {
                'max_depth': np.linspace(1, 32, 32, endpoint=True),
                'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
                'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
                # 'max_features': list(range(1, X_train.shape[1]))
            }
        else:
            classifier = RandomForestClassifier()
            tuned_parameters = {
                'min_samples_split': [3, 5, 10],
                'n_estimators': [100, 300],
                'max_depth': [3, 5, 15, 25]
                # 'max_features': list(range(1, X_train.shape[1]))
            }

        # FEATURE IMPORTANCE
        selector = SelectFromModel(estimator=classifier).fit(X_train, y_train)

        if selected_model == "Logistic Regression":
            importance = np.round(selector.estimator_.coef_, 2)
        elif selected_model == "Decision Tree" or selected_model == "Random Forest":
            importance = np.round(selector.estimator_.feature_importances_, 2)
        else:
            importance = np.array([])

        kfold = model_selection.KFold(n_splits=3)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        roc = go.Figure()

        roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        if selected_radio == 'not_optimised':
            model = classifier.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            # Confusion matrix
            y_pred = model.predict(X_test)

            # ROC-CURVE
            y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines')
            )

            roc_title = f'<b>ROC curve (AUC={auc(fpr, tpr):.4f})<b>'
            caption = ''

        else:
            grid = GridSearchCV(classifier, tuned_parameters, scoring='roc_auc', cv=kfold, n_jobs=-1)

            model = grid.fit(X_train, y_train)

            print('Optimized Parameters: {} '.format(model.best_params_))

            train_score = np.mean(
                cross_val_score(model.best_estimator_, X_train, y_train, scoring='accuracy', cv=kfold))

            test_score = np.mean(cross_val_score(model.best_estimator_, X_test, y_test, scoring='accuracy', cv=kfold))

            # Confusion matrix
            y_pred = model.predict(X_test)

            probs = grid.predict_proba(X_test)[:, 1]
            fpr, tpr, threshold = roc_curve(y_test, probs)

            roc_title = f'<b>ROC curve (AUC={auc(fpr, tpr):.4f})'
            caption = f'Hyperparameters: {model.best_params_}'

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

                roc.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=name, mode='lines')
                )
                i += 1

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            roc.add_trace(
                go.Scatter(
                    x=mean_fpr,
                    y=mean_tpr,
                    name=f'Mean ROC (AUC={mean_auc:.2f}&plusmn;{std_auc:.2f})',
                    mode='lines'
                )
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            roc.add_trace(
                go.Scatter(
                    x=mean_fpr,
                    y=tprs_upper,
                    fill='toself',
                    fillcolor="#ff6692",
                    opacity=0.5,
                    name=f'({mean_auc:.2f}-{std_auc:.2f}), ({mean_auc:.2f}+{std_auc:.2f})'
                )
            )

            roc.add_trace(
                go.Scatter(
                    x=mean_fpr,
                    y=tprs_lower,
                    fill='toself',
                    fillcolor="#ff6692",
                    opacity=0.25,
                    name="lower",
                    showlegend=False,
                    mode="none"
                )
            )

        ## CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)

        cm_labels = np.array([['FN = ', 'TN = '], ['TP = ', 'FP = ']])

        heat_cm = np.array([
            [np.char.add(cm_labels[0, 0], str(cm[0, 0])), np.char.add(cm_labels[0, 1], str(cm[0, 1]))],
            [np.char.add(cm_labels[1, 0], str(cm[1, 0])), np.char.add(cm_labels[1, 1], str(cm[1, 1]))]
        ])

        a = ['female', 'male']
        b = ['female', 'male']

        z_text = [[str(b) for b in a] for a in heat_cm]

        confusion = ff.create_annotated_heatmap(
            cm, x=a, y=b, annotation_text=z_text,
            # colorscale=[
            #     ['TP = ', '#7A4579'], 
            #     ['TN = ', '#7A4579'], 
            #     ['FP = ', '#ff99cc'],
            #     ['FN = ', '#ff99cc']
            #     ]
            colorscale=['#7A4579', '#7A4579', '#ff99cc', '#ff99cc', '#ff99cc']
        )

        confusion.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            title_text='<b>Confusion matrix<b>',
            xaxis=dict(title="Predicted Value"),
            yaxis=dict(title="True value"),
            title_x=0.6,
            title_font_size=19,
            title_font_color='#2c8cff'
        )

        confusion['layout']['xaxis'].update(side='bottom')

        ## ROC-CURVE
        roc.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            title=roc_title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            title_x=0.5,
            title_font_size=19,
            title_font_color='#2c8cff',
            annotations=[
                dict(xref='paper',
                     yref='paper',
                     x=0.5, y=0.05,
                     showarrow=False,
                     text=caption)
            ]
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

        table = table.rename(columns={'index': 'Metric'})

        tab3_table = html.Div([
            dt.DataTable(
                columns=[{'name': col, 'id': col} for col in table.columns],
                data=table.to_dict(orient='records'),
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_cell={'textAlign': 'center'},
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

        male_correct_pred = cm[1, 0] / np.sum(cm)
        female_correct_pred = cm[0, 1] / np.sum(cm)

        accuracy_data = {
            'Data': [f'Train data (n_train = {X_train.shape[0]})', f'Test data (n_test = {X_test.shape[0]})',
                     '', 'Class', 'Male', 'Female'],
            'Score': [train_score, test_score, '', 'Correct prediction', male_correct_pred, female_correct_pred]
        }

        accuracy_df = pd.DataFrame(accuracy_data, columns=['Data', 'Score'])

        accuracy_df = accuracy_df.rename(columns={'Data': f'Full data (N = {X.shape[0]})'})

        row_indexes = [0, 1, 4, 5]

        for i in row_indexes:
            accuracy_df.loc[i, ['Score']] = accuracy_df.loc[i, ['Score']].apply('{:.2%}'.format)

        accuracy_tb = html.Div([
            dt.DataTable(
                columns=[{'name': col, 'id': col} for col in accuracy_df.columns],
                data=accuracy_df.to_dict(orient='records'),
                style_header={'fontWeight': 'bold'},
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {
                            'row_index': 3,  # number | 'odd' | 'even'
                        },
                        'fontWeight': 'bold',
                    }
                ],
            )
        ])

        error_message = f'Feature Importance is not available for {selected_model}'

        if importance.size == 0:
            return {'display': 'block'}, confusion, roc, [], tab3_table, dash.no_update, {
                'display': 'none'}, error_message, accuracy_tb
        else:
            feat_name = pd.DataFrame([x for x in selected_variable], columns=['Variable'])
            feat_imp = pd.DataFrame(importance.flatten(), columns=['Importance'])
            dat = pd.concat([feat_name, feat_imp], axis=1)

            dat["Indicator"] = np.where(dat["Importance"] < 0, 'Negative', 'Positive')

            imp_bar = px.bar(
                dat,
                y='Variable',
                x='Importance',
                orientation='h',
                color='Indicator',
                title='<b>Feature Importance<b>'
            )

            imp_bar.update_layout(
                title_x=0.5,
                margin=dict(l=0, r=0, t=40, b=0),
                title_font_size=19,
                title_font_color='#2c8cff'
                # legend=dict(
                #     orientation="h",
                #     yanchor="bottom",
                #     y=1.02,
                #     xanchor="right",
                #     x=1
                # )
            )

            return {'display': 'block'}, confusion, roc, [], tab3_table, imp_bar, {'display': 'block'}, [], accuracy_tb
