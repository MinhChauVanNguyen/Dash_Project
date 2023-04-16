"""
Created on Tues, 27 Oct 2020
@author: Minh Chau Van Nguyen
"""

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_html_components as html
from Tabs import Sidebar, Tab1, Tab2, Tab3, Navbar

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True 
)

server = app.server

app.layout = html.Div(
    children=[
        Navbar.layout,
        Sidebar.layout
    ])


@app.callback(Output(component_id='tabs-content', component_property='children'),
              [Input(component_id='tabs', component_property='value')])
def render_content(tab):
    if tab == 'tab-1':
        return Tab1.layout
    elif tab == 'tab-2':
        return Tab2.layout
    elif tab == 'tab-3':
        return Tab3.layout


if __name__ == '__main__':
    app.run_server(debug=True)
