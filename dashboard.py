# facial_dashboard.py
from dash import html, dcc
import dash_bootstrap_components as dbc

facial_layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("Facial Recognition Dashboard", className="modern-title"))]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Live Camera Feed", className="card-title"),
                html.Img(id='live-feed', src="", style={'width': '100%'})
            ])
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Detection Logs", className="card-title"),
                html.Div(id='detection-logs', className="detection-logs-scroll")
            ])
        ]), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='confidence-gauge'), width=4),
        dbc.Col(dcc.Graph(id='recognition-pie'), width=4),
        dbc.Col(dcc.Graph(id='detection-bar'), width=4),
    ]),
    dcc.Interval(id='facial-interval', interval=3000, n_intervals=0)
], fluid=True)
