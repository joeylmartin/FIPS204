from typing import Dict, List
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table, no_update
import numpy as np
import plotly.graph_objects as go
from .vis_utils import DemoPage
import dash_mantine_components as dmc
from .display_vars import DisplayVar, Display2DArray, Display1DArray, RoundingRing

class VariablesList(DemoPage):
    def __init__(self, app, variables):
        self.variables : Dict[str, DisplayVar] = {str(var): var for var in variables}
        self.variable_names = list(self.variables.keys())

         #create hash uid to prepend div idsS
        uid = str(hash(self))[:10]
        self.container_div = "var-container" + uid
        self.interactive_div = "interactive-container" + uid
        self.latex_div = "latex-cont" + uid


        self.register_callbacks(app)

        #register var callbacks after they've all been inited
        for n, val in self.variables.items():
            val.register_callbacks(app)

        #index for currently displayed var
        self.current_var_index = 0
        self.current_var = None
        self.set_current_var()

       
    def set_current_var(self):
        self.current_var = self.variables[self.variable_names[self.current_var_index]]
        self.current_var.set_to_selected()


    '''
    def allocate_variable_display_var(self, var, app) -> DisplayVar:

       # Given a var, produce a DisplayVar object for it, 
       # determined by its datatype. Used for UI.

        match var:
            case np.ndarray():
                if var.ndim == 2:
                    return Display2DArray(app, var, str(var))
                elif var.ndim == 1:
                    return Display1DArray(app, var, str(var))
                else:
                    raise ValueError("Unsupported array dimension")
            case str(): #todo allocate diff
                return RoundingRing(app)
                raise ValueError("Unsupported variable type")
            #case int() | float():
            #    return DisplayScalar(var, "Scalar")
    '''

    def get_html(self):
        return html.Div(id=self.container_div,
            children=[
                dmc.Grid(
                [
                    # Left Column: Interactive Representations
                    dmc.Col(
                        html.Div(
                            id=self.interactive_div,
                            children=[
                                html.Div(
                                    var.get_interactive_representation(),
                                    style={"margin-bottom": "20px"}  # Adds spacing between rows
                                )
                                for var in self.variables.values()
                            ],
                            style={
                                "height": "600px",
                                "overflowY": "scroll",
                                "border": "1px solid black",
                                "padding": "10px"
                            }
                        ),
                        span=6  # Half-width
                    ),

                    # Right Column: LaTeX Representations
                    dmc.Col(
                        html.Div(
                            id=self.latex_div,
                            children=[
                                html.Div(
                                    var.get_latex_representation(),
                                    style={"margin-bottom": "20px"}  # Ensures alignment with left column
                                )
                                for var in self.variables.values()
                            ],
                            style={
                                "height": "600px",
                                "overflowY": "scroll",
                                "border": "1px solid black",
                                "padding": "10px"
                            }
                        ),
                        span=6  # Half-width
                    ),
                ],
                gutter="xl"
                ),

                html.Div([
                    html.Button('Previous Index', id='prev-index-button', n_clicks=0, style={'margin': '10px'}),
                    html.Button('Next Index', id='next-index-button', n_clicks=0, style={'margin': '10px'}),
                    html.Button('Previous Variable', id='prev-variable-button', n_clicks=0, style={'margin': '10px'}),
                    html.Button('Next Variable', id='next-variable-button', n_clicks=0, style={'margin': '10px'}),
                ], style={'textAlign': 'center', 'marginTop': '20px'}),

            ]
        )

    def register_callbacks(self, app):
        @app.callback(
            Output(self.container_div, "children", allow_duplicate=True),  
            [Input("prev-index-button", "n_clicks"),
            Input("next-index-button", "n_clicks"),
            Input("prev-variable-button", "n_clicks"),
            Input("next-variable-button", "n_clicks")],
            prevent_initial_call=True  # Ensure it doesn't fire on startup
        )
        def update_display(prev_index, next_index, prev_var, next_var):
            ctx = callback_context
            if not ctx.triggered:
                return no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id in ["prev-index-button", "next-index-button"]:
                change = -1 if button_id == "prev-index-button" else 1
                result = self.current_var.is_valid_index_update(change)

                if result == 0:
                    return self.get_html()
                
                elif 0 <= self.current_var_index + result < len(self.variable_names):
                    self.current_var.set_to_deselected()
                    self.current_var_index += result
                    self.set_current_var()
                    return self.get_html()

            elif button_id in ["prev-variable-button", "next-variable-button"]:
                change = -1 if button_id == "prev-variable-button" else 1

                if 0 <= self.current_var_index + change < len(self.variable_names):
                    self.current_var.set_to_deselected()
                    self.current_var_index += change
                    self.set_current_var()
                    return self.get_html()

            return no_update  # Return no update if nothing changed