# --- callbacks.py ---
from dash import Input, Output, html
import plotly.graph_objs as go
from collections import Counter
import requests

def register_callbacks(app):
    @app.callback(
        [Output('detection-logs', 'children'),
         Output('confidence-gauge', 'figure'),
         Output('recognition-pie', 'figure'),
         Output('detection-bar', 'figure')],
        Input('facial-interval', 'n_intervals')
    )
    def update_facial_dashboard(n):
        try:
            res = requests.get("http://127.0.0.1:8000/get_logs")
            logs = res.json()["logs"]

            recent_logs = [html.P(f"{log.get('timestamp', 'N/A')} - {log['name']}", className="text-light")
                           for log in logs[-10:][::-1]]

            conf_values = [log['confidence'] for log in logs if log['name'].lower() != "unknown"]
            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_conf,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'deepskyblue'}}
            ))

            name_counts = Counter([log['name'] for log in logs])
            names, counts = zip(*name_counts.items()) if name_counts else ([], [])

            pie = go.Figure(data=[go.Pie(labels=names, values=counts)])
            bar = go.Figure(data=[go.Bar(x=names, y=counts, marker=dict(color='lightblue'))])

            return recent_logs, gauge, pie, bar
        except Exception as e:
            print("Facial callback error:", e)
            return [html.P("No data", className="text-warning")], go.Figure(), go.Figure(), go.Figure()

    @app.callback(
        [Output('throw-count-indicator', 'figure'),   # ✅ NEW
        Output('behavior-logs', 'children')],
        Input('behavior-interval', 'n_intervals')
    )
    def update_behavior_dashboard(n):
        try:
            res = requests.get("http://127.0.0.1:8000/get_behavior_logs")
            logs = res.json()["behavior_logs"]

            # ✅ Count only "throw-like motion"
            throw_count = sum(1 for log in logs if 'throw' in log['event'].lower())

            indicator = go.Figure(go.Indicator(
                mode="number",
                value=throw_count,
                title={'text': "Throw-like Events"},
                number={'font': {'size': 48}}
            ))

            recent = [html.P(f"{log['timestamp']} - {log['event']}", className="text-light")
                      for log in logs[-10:][::-1]]

            return indicator, recent

        except:
            return go.Figure(), ["No data"]
# --- app.py ---
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dashboard import facial_layout
from behavior_dashboard import behavior_layout
from callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Smart Surveillance Dashboard"

app.layout = dbc.Container([
    html.H1("Smart Surveillance Dashboard", className="modern-title my-3 text-center"),
    dcc.Tabs([
        dcc.Tab(label="Facial Recognition", children=facial_layout),
        dcc.Tab(label="Behavior Analysis", children=behavior_layout)
    ])
], fluid=True)

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
