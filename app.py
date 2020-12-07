"""
Created by Minh Chau Van Nguyen
Start date : 23/10/2020
"""

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True  # remove "Id not found in layout" message
)

server = app.server
