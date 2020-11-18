"""
Created on Tues, 27 Oct 2020
@author: Minh Chau Van Nguyen
"""

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
