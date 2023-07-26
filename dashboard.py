import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import minitiebout

# Create a function to run the AgentPy simulation and return the results
def run_simulation():
    # Your simulation setup here
    # Example: model = MyModel()
    #          results = model.run()
    #          return results.average("utility"), results.count("moves"), results.history("ties")
    pass

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    dcc.Graph(id='utility-graph'),
    dcc.Graph(id='moves-graph'),
    dcc.Graph(id='ties-graph'),
    html.Button('Run Simulation', id='run-button'),
])

# Define the callback function to update the graphs on button click
@app.callback(
    [Output('utility-graph', 'figure'),
     Output('moves-graph', 'figure'),
     Output('ties-graph', 'figure')],
    [Input('run-button', 'n_clicks')]
)
def update_graphs(n_clicks):
    if n_clicks is not None:
        # Run the AgentPy simulation
        # You can replace this with the actual simulation results
        # For example:
        # avg_utility, num_moves, ties_history = run_simulation()

        # For demonstration purposes, generating random data
        num_agents = 100
        num_steps = 10

        avg_utility = np.random.rand(num_steps)
        num_moves = np.random.randint(5, 20, num_steps)
        ties_history = np.random.randint(0, num_agents, (num_agents, num_steps))

        # Create the figures for the graphs
        utility_figure = go.Figure(data=go.Scatter(x=list(range(num_steps)), y=avg_utility, mode='lines+markers'))
        utility_figure.update_layout(title='Average Utility per Agent')

        moves_figure = go.Figure(data=go.Bar(x=list(range(num_steps)), y=num_moves))
        moves_figure.update_layout(title='Number of Moves per Agent')

        ties_figure = go.Figure(data=go.Heatmap(z=ties_history, colorscale='Viridis'))
        ties_figure.update_layout(title='Agent Ties History')

        return utility_figure, moves_figure, ties_figure

    # Return empty figures before button click
    return go.Figure(), go.Figure(), go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
