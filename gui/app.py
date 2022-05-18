import warnings

warnings.filterwarnings("ignore", category=Warning)

import os
import dash
import webbrowser
from dash import dcc, html, callback
from dash.dependencies import Input, Output

from components import navBar, menuBar
from pages import home, non_exist

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# components
navbar = navBar.navbar
menu_bar = menuBar.menu_bar
url = dcc.Location(id="url")
content = html.Div(id='content')

def serve_layout():
    # 得到最新狀態的 db

    layout = html.Div(
        [
            url,
            navbar,
            menu_bar,
            content,
        ],
    )
    return layout

app.layout = serve_layout

# 透過 url 來決定顯示哪個 page
@callback(
    Output('content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):

    # live update layout
    if pathname in ['/', '/Home']:
        return home.serve_layout()

    # elif pathname == '/Detect':
    #     return discover.serve_layout()

    # elif pathname == '/Transfer':
    #     return security_events.serve_layout()

    return non_exist.serve_layout()

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False) # debug mode
    # pid = os.fork()
    # if pid != 0:
    #     app.run_server()
    # else:
    #     url = "http://127.0.0.1:8050/"
    #     webbrowser.open(url)