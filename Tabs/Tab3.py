from app import app
from Data import data_processing
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

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report

df = data_processing.df

layout = html.Div(children=[
    html.Div(
        id="bike_volume_card",
        children=[
            html.B("Total bike related products bought by selected country, state and group"),
            html.Hr(),
            html.Div(
                id='hidden',
                children=[
                    dbc.Row(children=[
                        dbc.Col(
                            children=[dcc.Graph(id="confusion_mat")],
                            width={"size": 5},
                        ),
                        dbc.Col(
                            children=[dcc.Graph(id='roc_curve')],
                            width={"size": 6, "offset": 1},
                        )]
                    )],
                style={'display': 'block'}
            ),
            dbc.Row(children=[
                dbc.Col(
                    children=[html.Div(id='error_message')],
                )
            ]),
            dbc.Row(
                children=[
                    dbc.Col(
                        dt.DataTable(
                            id='classification_table',
                            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                            style_cell={'textAlign': 'center'}
                        )
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
     Output(component_id='classification_table', component_property='columns'),
     Output(component_id='classification_table', component_property='data')
     ],
    [Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_state', component_property='value'),
     Input(component_id='slct_variables', component_property='value'),
     Input(component_id='slct_class', component_property='value')]
)
def classification(selected_country, selected_state, selected_variable, selected_model):
    data = df[(df["Country"] == selected_country) & (df["State"] == selected_state)]

    data = data[selected_variable + ['Customer_Gender']]
    data['Customer_Gender'] = data['Customer_Gender'].replace({'Female': 0, 'Male': 1})

    for i in range(len(selected_variable)):
        if selected_variable != "Profit" or selected_variable != "Revenue":
            data[selected_variable[i]] = LabelEncoder().fit_transform(data[selected_variable[i]].values)

    frequency_table = data['Customer_Gender'].value_counts()

    message = "Target variable only has one class"

    if len(data['Customer_Gender'].unique()) != 2:
        return {'display': 'none'}, dash.no_update, dash.no_update, message, [], []

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if X.size == 0 and y.size == 0:
        return {'display': 'none'}, dash.no_update, dash.no_update, [], [], []
    else:
        # Step 1. Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Step 2. Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # kfold = model_selection.KFold(n_splits=5, random_state=1)

        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        if selected_model == "Logistic":
            classifier = LogisticRegression(random_state=0)
        elif selected_model == "SVM":
            classifier = SVC(kernel='rbf', random_state=0, probability=True)
        elif selected_model == "KNN":
            classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        elif selected_model == "Naive":
            classifier = GaussianNB()
        elif selected_model == "Decision":
            classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        else:
            classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        x = ['female', 'male']
        y = ['female', 'male']

        z_text = [[str(y) for y in x] for x in cm]

        confusion = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

        confusion.update_layout(
            title_text='<i><b>Confusion matrix<b><i>',
            xaxis=dict(title="Predicted Value"),
            yaxis=dict(title="True value"),
            title_x=0.5
        )

        confusion['data'][0]['showscale'] = True
        confusion['layout']['xaxis'].update(side='bottom')

        y_score = classifier.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_score)

        roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )

        roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        roc.update_yaxes(scaleanchor='x', scaleratio=1)
        roc.update_xaxes(constrain='domain')

        perf_metrics = classification_report(y_test, y_pred, output_dict=True)

        #print(classification_report(y_test, y_pred))

        table = pd.DataFrame(perf_metrics).transpose()
        table.iloc[2]['precision'] = 0
        table.iloc[2]['recall'] = 0

        table[table.eq(0)] = ''

        table.reset_index(inplace=True)

        columns = [{'name': col, 'id': col} for col in table.columns]
        dt = table.to_dict(orient='records')

        return {'display': 'block'}, confusion, roc, [], columns, dt
