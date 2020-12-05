from dash.dependencies import Input, Output
from app import app
import dash_html_components as html
from Tabs import Sidebar, Tab1, Tab2, Tab3, Navbar
from app import server


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
