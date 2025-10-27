# behavior_dashboard.py
from dash import html, dcc
import dash_bootstrap_components as dbc

behavior_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Behavior Analysis Dashboard", className="modern-title"), className="title-container")
    ]),

    dbc.Row([
         # ✅ NEW: Throw Count Indicator
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Throw-like Events", className="card-title"),
                dcc.Graph(id='throw-count-indicator')  # <- NEW ID
            ])
        ], className="shadow p-3 mb-4 custom-card"), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                # html.H4("Live Behavior Alerts", className="card-title"),
                html.Div(id='behavior-logs', className="detection-logs-scroll")
            ])
        ]), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Behavior Frequency", className="card-title"),
                dcc.Graph(id='behavior-bar-chart')
            ])
        ]), width=6),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Behavior Summary", className="card-title"),
                dcc.Graph(id='behavior-pie-chart')
            ])
        ]), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Behavior Event Timeline", className="card-title"),
                html.Div("Timeline Placeholder", id='behavior-timeline')
            ])
        ]), width=6),
    ]),

    dcc.Interval(id='behavior-interval', interval=3000, n_intervals=0)
], fluid=True)
