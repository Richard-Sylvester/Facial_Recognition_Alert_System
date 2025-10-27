# app.py
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dashboard import facial_layout
from behavior_dashboard import behavior_layout
from callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Smart Surveillance Dashboard"

app.layout = dbc.Container([
    html.H1("Smart Surveillance Dashboard", className="modern-title my-3"),
    dcc.Tabs([
        dcc.Tab(label="Facial Recognition", children=facial_layout),
        dcc.Tab(label="Behavior Analysis", children=behavior_layout)
    ])
], fluid=True)

# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
