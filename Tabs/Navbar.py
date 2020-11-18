import dash_html_components as html
from app import app


layout=html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("Logo.png")),
                html.H5("TITLE")
            ])