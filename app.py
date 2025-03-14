from dash import Dash, html, dcc, callback, Output, Input, callback_context, no_update
import plotly.express as px
import pandas as pd
from dash import dcc
import plotly.graph_objs as go
import dash_mantine_components as dmc
import numpy as np
from visualizers.display_vars import Display1DArray, Display2DArray
from visualizers.lattice import ALattice, WLattice, ProjectionMethods

from visualizers.variablepage import VariablesList
from fips_204.external_funcs import ml_dsa_key_gen


# Initializ

app = Dash(prevent_initial_callbacks=True)
app.config.suppress_callback_exceptions=True

temp = ProjectionMethods.VIEW_3D
pk, sk = ml_dsa_key_gen()
lat = WLattice(pk, sk, app, "Hello world!")

var1 = Display2DArray(app, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "eee")
var2 = Display1DArray(app, np.array([0,1,2,3,4,5,6,7,8,9]), "ee")

var_list = VariablesList(app, [var1, var2])




# Define the steps in the FIPS process

pages = {
    "Lattice Visualization": lat,
    "Variables" : var_list
    
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
            html.Button('Previous Page', id='prev-button', n_clicks=0, style={'margin': '10px'}),
            html.Button('Next Page', id='next-button', n_clicks=0, style={'margin': '10px'}),
        ], style={'textAlign': 'center'}),

        # Dynamic Page Content (updated by callbacks)
        html.Div(id='page-content', children=pages[page_names[current_step_index]].get_html()),


        dcc.Store(id='register-trigger', data={})
    ]
)

# Define Callback to Update Page Content
@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Output("register-trigger", "data"),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
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

    # Get the updated page layout based on the current step
    page = pages[page_names[current_step_index]]
    page_layout = page.get_html()

    return page_layout, {"trigger" : True}


#after page has rendered, register callbacks.
#must be called after rendering, as the callbacks
#must refer to elements that have already been rendered
@app.callback(
    Output('register-callbacks', 'data'),
    Input('register-trigger', 'data')
)
def register_page_callbacks(data):
    print("Registering callbacks")
    """ Registers callbacks only after page content is updated """
    page.register_callbacks(app)
    return no_update

if __name__ == '__main__':
    app.run(debug=True)