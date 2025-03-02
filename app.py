from dash import Dash, html, dcc, callback, Output, Input, callback_context
import plotly.express as px
import pandas as pd
from dash import dcc
import plotly.graph_objs as go
import dash_mantine_components as dmc

from visualizers.lattice import ALattice, ProjectionMethods
from visualizers.array import ArrayInsertionDemo
from fips_204.external_funcs import ml_dsa_key_gen


# Initializ

app = Dash(prevent_initial_callbacks=True, external_stylesheets=dmc.styles.ALL)

temp = ProjectionMethods.VIEW_3D
pk, sk = ml_dsa_key_gen()
lat = ALattice(sk, app)
lat.generate_3d_points(temp)

arr = ArrayInsertionDemo(pk)


# Define the steps in the FIPS process

pages = {
    "Array" : arr,
}
page_names = list(pages.keys()) #TODO: check ordering
current_step_index = 0  # Default starting index
page = pages[page_names[current_step_index]]

# Define the layout of the app
app.layout = dmc.MantineProvider(
    theme={
        # add your colors
        "colors": {
             # add your colors
            "deepBlue": ["#E9EDFC", "#C1CCF6", "#99ABF0" "..."], # 10 colors
            # or replace default theme color
            "blue": ["#E9EDFC", "#C1CCF6", "#99ABF0" "..."],   # 10 colors
        },
        "shadows": {
            # other shadows (xs, sm, lg) will be merged from default theme
            "md": "1px 1px 3px rgba(0,0,0,.25)",
            "xl": "5px 5px 3px rgba(0,0,0,.25)",
        },
        "headings": {
            "fontFamily": "Roboto, sans-serif",
            "sizes": {
                "h1": {"fontSize": '30px'},
            },
        },
    },
    children=[
        html.H1(children='FIPS 204 Visualizer', style={'textAlign': 'center'}),

        # Navigation Controls
        html.Div([
            html.Button('Previous', id='prev-button', n_clicks=0, style={'margin': '10px'}),
            html.Button('Next', id='next-button', n_clicks=0, style={'margin': '10px'}),
        ], style={'textAlign': 'center'}),

        # Dynamic Page Content (updated by callbacks)
        html.Div(id='page-content'),

        dcc.Store(id='register-trigger', data={})
    ]
)

# Define Callback to Update Page Content
@app.callback(
    Output('page-content', 'children'),
    Output("register-trigger", "data"),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks):
    global current_step_index
    ctx = callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'prev-button' and current_step_index > 0:
            current_step_index -= 1
        elif button_id == 'next-button' and current_step_index < len(page_names) - 1:
            current_step_index += 1
    print("Current step index: ", current_step_index)
    # Get the updated page layout based on the current step
    page = pages[page_names[current_step_index]]
    page_layout = page.get_html()
    return page_layout, {"trigger" : True}

@app.callback(
    Output('register-callbacks', 'data'),
    Input('register-trigger', 'data'),
    prevent_initial_call=False
)
def register_page_callbacks(data):
    print("Registering callbacks")
    """ Registers callbacks only after page content is updated """
    page.register_callbacks(app)
    return {}

if __name__ == '__main__':
    app.run(debug=True)