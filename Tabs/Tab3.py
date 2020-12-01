from app import app
from Data.data_processing import my_df
import pandas as pd

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

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance


df = my_df

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            html.B("Total bike related products bought by selected country, state and group"),
            html.Hr(),
            html.Div(
                id="hidden",
                children=[
                    dbc.Row(
                        children=[
                            html.Div(
                                html.Strong("Optimise model by hyperparameters tuning"),
                                style={'display': 'inline-block', 'marginRight': 10, 'marginLeft': 5}),
                            dcc.RadioItems(
                                id='optimise',
                                options=[
                                    {'label': "Tune model", 'value': "optimised"},
                                    {'label': "Don't tune model", 'value': "not_optimised"},
                                ],
                                value='not_optimised',
                                labelStyle={'display': 'inline-block', 'marginRight': 10}
                        )]
                    ),
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
                                            children=[dcc.Graph(id='roc_curve')],
                                            style={'marginLeft': -20}
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
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            dcc.Loading(
                                id="load2",
                                children=[html.Div(id='classification_table')]
                                # children=[
                                #     dt.DataTable(
                                #         id='classification_table',
                                #         style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                                #         style_cell={'textAlign': 'center'},
                                        # style_data_conditional= [{
                                        #     'if': {
                                        #         'filter_query': '{Metric} == "Male class" && {Metric} == "Female class"',
                                        #         'column_id': 'Metric'
                                        #     },
                                        #     'backgroundColor': 'hotpink',
                                        #     'color': 'white'
                                        # }]
                                   # )]
                            )
                        ],
                        width={"size": 5}
                    )
                ]
            )
        ],
        style={"margin-top": '10px'}
    ),
]
)


@app.callback(
    [Output(component_id='hidden', component_property='style'),
     Output(component_id='confusion_mat', component_property='figure'),
     Output(component_id='roc_curve', component_property='figure'),
     Output(component_id='error_message', component_property='children'),
     Output(component_id='classification_table', component_property='children')
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

    data['Customer_Gender'] = data['Customer_Gender'].replace({'F': 0, 'M': 1})

    message = "Target variable only has one class"

    if len(data['Customer_Gender'].unique()) != 2:
        return {'display': 'none'}, dash.no_update, dash.no_update, message, []

    X = data.drop('Customer_Gender', axis=1)
    y = data['Customer_Gender']

    X['Day'] = X['Day'].apply(str)
    X['Year'] = X['Year'].apply(str)

    label_encoder = LabelEncoder()
    for i, col in enumerate(X):
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))
    
    if X.size == 0 and y.size == 0:
        return {'display': 'none'}, dash.no_update, dash.no_update, [], []
    else:
        # Step 1. Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Step 2. Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        if selected_model == "Logistic":
            classifier = LogisticRegression()
            tuned_parameters = {
                'C': np.linspace(.0001, 1000, 200),
                'penalty': ['l2']
            }
        elif selected_model == "SVM":
            classifier = SVC(probability=True)
            tuned_parameters = {
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'gamma': ['auto', 'scale'],
                'degree': [3, 8],
                'C': [1, 10, 100, 1000],
                'probability': [True]
            }
        elif selected_model == "KNN":
            classifier = KNeighborsClassifier()
            tuned_parameters = {
                'leaf_size': list(range(1, 50)),
                'n_neighbors': list(range(1, 30)),
                'p': [1, 2]
            }
        elif selected_model == "Naive":
            classifier = GaussianNB()
            tuned_parameters = {
                'var_smoothing': np.logspace(0, -9, num=100)
            }
        elif selected_model == "Decision":
            classifier = DecisionTreeClassifier()
            tuned_parameters = {
                'max_depth': np.linspace(1, 32, 32, endpoint=True),
                'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
                'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
                'max_features': list(range(1, X_train.shape[1]))
            }
        else:
            classifier = RandomForestClassifier()
            tuned_parameters = {
                'min_samples_split': [3, 5, 10],
                'n_estimators': [100, 300],
                'max_depth': [3, 5, 15, 25],
                'max_features': list(range(1, X_train.shape[1]))
            }
        
         # FEATURE IMPORTANCE
        selector = SelectFromModel(estimator=classifier).fit(X_train, y_train)

        if selected_model == "Logistic":
            importance = np.round(selector.estimator_.coef_, 2)
        elif selected_model == "Decision" or selected_model == "Random":
            importance = np.round(selector.estimator_.feature_importances_, 2)
        else: 
            importance = [[]]

        print(type(importance))
        
        kfold = model_selection.KFold(n_splits=5)

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

            #Confusion matrix
            y_pred = model.predict(X_test)

            # ROC-CURVE
            y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines')
            )

        else:
            grid = GridSearchCV(classifier, tuned_parameters, scoring='roc_auc', cv=kfold)
            model = grid.fit(X_train, y_train)

            #Confusion matrix
            y_pred = model.predict(X_test)

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


        ## CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)

        cm_labels = np.array([['FN = ', 'TN = '], ['TP = ', 'FP = ']])

        heat_cm = np.array([
            [np.char.add(cm_labels[0,0], str(cm[0,0])), np.char.add(cm_labels[0,1], str(cm[0,1]))],
            [np.char.add(cm_labels[1,0], str(cm[1,0])), np.char.add(cm_labels[1,1], str(cm[1,1]))]
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
            title_text='<i><b>Confusion matrix<b><i>',
            xaxis=dict(title="Predicted Value"),
            yaxis=dict(title="True value"),
            title_x=0.5
        )

        confusion['layout']['xaxis'].update(side='bottom')


        ## ROC-CURVE
        roc.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            title=f'{selected_model} ROC curve (AUC={auc(fpr, tpr):.4f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            title_x=0.5
        )

        ## CLASSIFICATION TABLE
        perf_metrics = classification_report(y_test, y_pred, output_dict=True)

        table = pd.DataFrame(perf_metrics).transpose()
        table.iloc[2]['precision'] = 0
        table.iloc[2]['recall'] = 0

        table.reset_index(inplace=True)

        table = table.drop('support', axis = 1)

        table['precision'] = pd.to_numeric(table['precision'])

        for c in table[['precision', 'recall', 'f1-score']]:
            table[c] = pd.to_numeric(table[c])
            table[c] = table[c].map('{:.2f}'.format)

        table[table.eq('0.00')] = ''

        table[table.eq('1')] =  'Male class'
        table[table.eq('0')] =  'Female class'

        table = table.rename(columns={'index': 'Metric'})

        minh = html.Div([
            dt.DataTable(
                columns = [{'name': col, 'id': col} for col in table.columns],
                data =  table.to_dict(orient='records'),
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

        # feat_imp = pd.DataFrame({
        #     'Variable': [x for x in selected_variable],
        #     'Importance': importance
        # })

        feat_name = pd.DataFrame([x for x in selected_variable], columns=['Variable'])
        feat_imp = pd.DataFrame(importance.flatten(), columns=['Importance'])
        dat = pd.concat([feat_name, feat_imp], axis=1)

        print(dat)

        return {'display': 'block'}, confusion, roc, [], minh