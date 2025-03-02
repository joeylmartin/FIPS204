from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import numpy as np
import plotly.graph_objects as go
from .vis_utils import DemoPage
import dash_mantine_components as dmc

# Initialize Dash app
app = Dash(prevent_initial_callbacks=True)

# Array insertion demo class
class ArrayInsertionDemo(DemoPage):
    def __init__(self, sample_array):
        self.sample_array = np.array(sample_array)  # Convert input to NumPy array
        self.rows, self.cols = self._determine_shape(self.sample_array)
        self.array = np.full((self.rows, self.cols), "", dtype=object)  # Empty slots
        self.current_index = [0, 0]  # Current (i, j) position
        self.callbacks_registered = False

    def _determine_shape(self, arr):
        """ Determine shape (rows, cols) based on input array """
        if arr.ndim == 1:  # 1D array case
            return 1, len(arr)  # Single row, multiple columns
        return arr.shape  # Return as-is for 2D arrays

    def get_grid(self):
        """ Construct the visual grid layout using Dash Mantine dmc.Grid """
        grid_items = []

        # Add column labels (j)
        col_labels = [html.Div(str(j), style={"textAlign": "center", "fontWeight": "bold"}) for j in range(self.cols)]
        grid_items.append(dmc.Grid(children=[dmc.Col(html.Div("j →", style={"textAlign": "right", "fontWeight": "bold"}))] + [dmc.Col(label) for label in col_labels]))

        for i in range(self.rows):
            row_items = [dmc.Col(html.Div(str(i), style={"textAlign": "center", "fontWeight": "bold"}))]  # Row label (i)

            for j in range(self.cols):
                color = "red" if [i, j] == self.current_index else "gray"  # Highlight the active slot
                row_items.append(
                    dmc.Col(dmc.Card(
                        html.Div(self.array[i, j], style={"fontSize": "20px", "textAlign": "center"}),
                        style={"backgrounddmc.Color": color, "padding": "10px"}
                    ))
                )

            grid_items.append(dmc.Grid(children=row_items))

        return html.Div(grid_items)

    def get_html(self):
        """ Return HTML layout for the Array Insertion Demo """
        return html.Div([
            html.H2("Array Insertion Demo"),
            html.P("Use the left and right buttons to step through insertion."),
            self.get_grid(),  # Display the grid
            html.Div([
                html.Button("← Left", id="array-left", n_clicks=0),
                html.Button("→ Right", id="array-right", n_clicks=0)
            ], style={"textAlign": "center", "margin": "20px"}),
            dcc.Store(id="array-index", data=self.current_index)  # Store for tracking index
        ])

    def register_callbacks(self, app):
        """ Register Dash callbacks for updating the visualization """
        if self.callbacks_registered:
            return  # Prevent duplicate registration

        @app.callback(
            Output("page-content", "children"),
            Output("array-index", "data"),
            Input("array-left", "n_clicks"),
            Input("array-right", "n_clicks"),
            State("array-index", "data"),
            prevent_initial_call=True
        )
        def update_array(left_clicks, right_clicks, index):
            """ Updates the array when stepping through insertion """
            i, j = index  # Extract current index

            ctx = dash.callback_context
            if not ctx.triggered:
                return self.get_html(), index  # No change

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "array-left":
                if j > 0:
                    j -= 1  # Move left
                elif i > 0:
                    i -= 1
                    j = self.cols - 1  # Move to last column of previous row

            elif button_id == "array-right":
                if j < self.cols - 1:
                    j += 1  # Move right
                elif i < self.rows - 1:
                    i += 1
                    j = 0  # Move to first column of next row

            # Insert the value from the sample array
            flattened_array = self.sample_array.flatten()
            idx_flat = i * self.cols + j  # Compute flat index
            if idx_flat < len(flattened_array):
                self.array[i, j] = flattened_array[idx_flat]  # Assign from input

            # Update visualization with new selection
            self.current_index = [i, j]
            return self.get_html(), [i, j]

        self.callbacks_registered = True  # Mark that callbacks are now registered

# Example input: a 1D array