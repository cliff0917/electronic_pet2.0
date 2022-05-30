from dash import html
import dash_bootstrap_components as dbc

img_path = './assets/img'
logo = f'{img_path}/logo.png'

navbar = dbc.Navbar(
    [
        # 利用 row, col 來控制排版
        dbc.Row(
            [
                html.A(
                    [
                        dbc.Col(html.Img(src=logo, height="50px", style={'background-color':'white'})),
                    ],
                    href="https://plato.csie.ncku.edu.tw/",
                ),
                html.A(
                    [
                        dbc.Col(dbc.NavbarBrand("Intelligence Evolution Platform(IEP)", className = "ml-2", style={'fontSize': 30})),
                    ],
                    href="/Home"
                ),
            ],
        ),
    ],
    color="dark",
    dark=True,
    sticky='top',
    style={'width':'100%'},
)