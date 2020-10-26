import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
import plotly.express as px

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
# Import and process data
df = pd.read_csv('https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_data.csv')

df.drop(df.columns.difference(['Age_Group','Product_Category','Order_Quantity', 'State']), 1, inplace=True)
grouped_df = df.groupby(["State", "Age_Group", "Product_Category"])
grouped_df = pd.DataFrame(grouped_df.sum().reset_index())

state_list = grouped_df["State"].unique()


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Khabeef forever"),
            html.H3("Welcome to my Dash Dashboard"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
        ],
    )

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select State"),
            dcc.Dropdown(id="slct_state",
                 options=[{"label":i, "value":i} for i in state_list],
                 multi=False,
                 value=state_list[0],
                 style={'width': "300%", 'margin-left': "-85px"}),

             html.Br(),
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
            className="four columns",
            children=[description_card(), generate_control_card()]
        ),

        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # State Bar graph
                html.Div(
                    id="bike_volume_card",
                    children=[
                        html.B("Bike Volume"),
                        html.Hr(),
                        dcc.Graph(id="bar_graph"),
                    ],
                )
            ]
        )
    ]
)



# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id='bar_graph', component_property='figure'),
    [Input(component_id='slct_state', component_property='value')]
)

def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    df = grouped_df.copy()
    df = df[df["State"] == option_slctd]

    # Plotly Express
    fig = px.bar(
            data_frame=df,
            x='Age_Group',
            y='Order_Quantity',
            color='Product_Category',
            title="<b>Number of bike related products bought for various Age groups</b>",
            opacity=0.6,
            category_orders={"Age_Group": ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"],
                                "Product_Category": ["Clothing", "Bikes", "Accessories"]},
            color_discrete_map={
                 'Accessories':'#0059b2',
                 'Bikes': '#4ca6ff',
                 'Clothing': '#99ccff'},
            hover_data={'Order_Quantity':':,.0f'},
            labels={'Age_Group':'<b>Age group</b>',
                     'Order_Quantity':'<b>Order quantity</b>',
                     'Product_Category':'<b>Product category</b>'})

    fig.update_layout(
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        legend_traceorder="reversed",
        legend=dict(yanchor="top", xanchor="right", y=0.95,
                    bordercolor="Black", borderwidth=1.5),
        xaxis=dict(mirror=True, ticks='outside', showline=True, linewidth=1.5, linecolor='black'),
        yaxis=dict(mirror=True, ticks='outside', showline=True),
        margin=dict(l=20, r=20, t=30, b=20),
        title_x=0.53,
        font=dict(family="Courier New, monospace"),
        hoverlabel=dict(
            bgcolor="white",
            font_family="Courier New, monospace"
        )
    )

    return fig

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
