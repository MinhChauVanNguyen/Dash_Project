"""
Created on Tues, 27 Oct 2020
@author: Minh Chau Van Nguyen
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True


# Read data
df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

df.drop(df.columns.difference(['Age_Group','Product_Category','Order_Quantity', 'State']), 1, inplace=True)
grouped_df = df.groupby(["State", "Age_Group", "Product_Category"])
grouped_df = pd.DataFrame(grouped_df.sum().reset_index())

state_list = grouped_df["State"].unique()


def description_card():

    return html.Div(
        id="description-card",
        children=[
            html.H5("Khabeef forever"),
            html.H3("Welcome to my Dash Dashboard"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
            html.Br(),
            html.P("Select State"),
            dcc.Dropdown(id="slct_state",
                         options=[{"label": i, "value": i} for i in state_list],
                         multi=False,
                         value="British Columbia"
                         ),
        ],
    )

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    id="app-container",

    children=[

        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("Logo.png"))],
        ),

        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card()]
        ),

        # Right column
        html.Div(
            id="right-column",
            className="nine columns",
            children=[
                # State Bar graph
                html.Div(
                    id="bike_volume_card",
                    children=[
                        html.B("Total bike related products bought for various Age groups"),
                        html.Hr(),
                        html.Div(children=[
                            html.Div(
                                children=[dcc.Graph(id="bar_graph")],
                                className="five columns"),
                            html.Div(id='data-table', style={'margin-top': '20px'},
                                     className="four columns offset-by-one")
                        ], className="row")
                    ]
                )
                        #html.Div(id='table-container', className='tableDiv'),
                    ],
                ),
            ],
        )




# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output('bar_graph', 'figure'),
     Output('data-table', 'children')],
    [Input(component_id='slct_state', component_property='value')]
)




def update_output(user_selection):

    df = grouped_df.copy()
    df = df[df["State"] == user_selection]

    name_sort = {'Youth (<25)': 0, 'Young Adults (25-34)': 1, 'Adults (35-64)': 2, 'Seniors (64+)': 3}
    df['name_sort'] = df['Age_Group'].map(name_sort)

    df = df.sort_values(by='name_sort', ascending=True)
    df = df.drop('name_sort', axis=1)

    #Table
    table = html.Div([
        dt.DataTable(
            id='data-table',
            columns=[
                {'name': 'Age Group', 'id': 'Age_Group', 'type': 'text', 'editable': True},
                {'name': 'Product Category', 'id': 'Product_Category', 'type': 'text', 'editable': True},
                {'name': 'Order Quantity', 'id': 'Order_Quantity', 'type': 'numeric', 'editable': True}
            ],
            data=df.to_dict('records'),
            style_table={
                'maxHeight': '20%',
                #'overflowY': 'scroll',
                'width': '30%',
                'minWidth': '10%',
            },
            editable=True,
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'whiteSpace': 'auto', 'height': 'auto', 'width': 'auto'},
            style_cell={'textAlign': 'left'},
            style_data_conditional=([
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Adults (35-64)"',
                    },
                    'backgroundColor': '#FF4136',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Seniors (64+)"',
                    },
                    'backgroundColor': 'hotpink',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Young Adults (25-34)"',
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Age_Group} eq "Youth (<25)"',
                    },
                    'backgroundColor': '#7FDBFF',
                    'color': 'white'
                }
            ])
        )
    ])

    # Plotly Express
    fig = px.bar(
            data_frame=df,
            x='Age_Group',
            y='Order_Quantity',
            color='Product_Category',
            opacity=0.6,
            # category_orders={"Age_Group": ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"],
            #                  "Product_Category": ["Clothing", "Bikes", "Accessories"]},
            color_discrete_map={
                 'Accessories':'#0059b2',
                 'Bikes': '#4ca6ff',
                 'Clothing': '#99ccff'},
            hover_data={'Order_Quantity':':,.0f'},
            labels={'Age_Group':'<b>Age group</b>',
                     'Order_Quantity':'<b>Order quantity</b>',
                     'Product_Category':'<b>Product category</b>'})

    fig.update_layout(
        width=400,
        height=500,
        #plot_bgcolor='rgba(0,0,0,0)',
        legend_traceorder="reversed",
        legend=dict(yanchor="bottom", y=1.02,
                    xanchor= "right", x=1,
                    orientation="h",
                    bordercolor="Black",
                    borderwidth=1.5),
        xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1.5, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True),
        margin=dict(l=50, r=20, t=30, b=20),
        title_x=0.53,
        font=dict(family="Helvetica Neue,  sans-serif"),
        hoverlabel=dict(
            bgcolor="white",
            font_family="Helvetica Neue,  sans-serif"
        )
    )

    return fig, table


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
