from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table
import numpy as np
import plotly.graph_objects as go
from .vis_utils import DemoPage
import dash_mantine_components as dmc

# Initialize Dash app
app = Dash(prevent_initial_callbacks=True)

# Array insertion demo class
class ArrayInsertionDemo(DemoPage):
    def __init__(self,app, variables):
        self.register_callbacks(app)
        self.sample_array = np.array(sample_array) 
        self.rows, self.cols = self._determine_shape(np.array(sample_array))
        self.array = np.full((self.rows, self.cols), "", dtype=object)  # Empty slots, working

        self.current_index = [0, 0]  # Current (i, j) position
        self.callbacks_registered = False

    def get_array_ith(self, arr, i):
        if self.rows > 1:
            col = i % self.cols
            row = i // self.cols
            return arr[row, col]
        return arr[i]
    def _determine_shape(self, arr):
        """ Determine shape (rows, cols) based on input array """
        if arr.ndim == 1:  # 1D array case
            return 1, len(arr)  # Single row, multiple columns
        return arr.shape  # Return as-is for 2D arrays
    
    def get_grid(self):
        """ Construct the visual grid layout using Dash Mantine Grid """
        columns = [{"name": f"Col {j}", "id": str(j)} for j in range(self.rows)]
        data = [
            {str(j) : self.sample_array[k + j] for k in range(self.cols)} for j in range(self.rows)
        ]
        return dmc.Grid([
            # Left Pane: Scrollable Data Table
            dmc.Col([
                html.H3("A-Matrix Data Table"),
                dash_table.DataTable(
                    id="a-matrix-table",
                    columns=[{"name": "Row", "id": "Row"}] + columns,
                    data=data,
                    style_table={"height": "400px", "overflowY": "auto"},
                    style_cell={"textAlign": "center", "padding": "10px"},
                    row_selectable="single",
                    cell_selectable=True
                ),
            ], span=6),

            # Right Pane: Math Formula Display
            dmc.Col([
                html.H3("Calculation Breakdown"),
                html.Div(id="formula-display", children=dcc.Markdown("$A_{i,j} = f(...)$", mathjax=True),
                         style={"border": "1px solid black", "padding": "20px", "minHeight": "100px"})
            ], span=6)
        ])
        '''
        normal_style = {
            "border": "2px solid black",
            "textAlign": "center",
            "backgroundColor": "#f0f0f0",
            "padding": "10px",
            "fontSize": "18px",
            "transition": "background-color 0.3s ease-in-out" 
        }
        selected_style = {
            "border": "4px solid black",
            "textAlign": "center",
            "backgroundColor": "#ff5733",  # Highlight selection
            "padding": "10px",
            "fontSize": "18px",
            "transition": "background-color 0.3s ease-in-out"  
        }
        grid_cells = [html.Div(
                dmc.Text(self.sample_array[i + j]),
                style=selected_style if [i, j] == self.current_index else normal_style)
            for i in range(self.rows) for j in range(self.cols)]

        return dmc.SimpleGrid(
            cols=self.cols,  # Fix: Ensure `cols` is set to match array structure
            spacing="md",
            verticalSpacing="md",
            children=grid_cells)
        
        
        # Column labels (j-axis)
        col_labels = [dmc.Text(str(j)) for j in range(self.cols)]
        grid_elements.append(dmc.Grid([dmc.Text("j →")] + col_labels, gutter="xs"))

        # Construct the array grid with row labels
        for i in range(self.rows):
            row_cells = []
            row_cells.append(dmc.Text(str(i)))  # Row label (i)

            for j in range(self.cols):
                is_selected = [i, j] == self.current_index
                cell_style = {"border": "2px solid black", "padding": "15px", "textAlign": "center"}

                if is_selected:
                    cell_style["backgroundColor"] = "red"

                row_cells.append(dmc.Paper(dmc.Text(self.array[i, j], size="lg"), style=cell_style))

            grid_elements.append(dmc.Grid(row_cells, gutter="xs"))

        return html.Div(grid_elements, style={"display": "flex", "flexDirection": "column", "gap": "10px"})
        '''
        ##return grid

    def get_html(self):
        """ Return HTML layout for the Array Insertion Demo"""
        return html.Div([
            html.H2("Array Insertion Demo"),
            self.get_grid(),  # Display the grid
            html.P("Use the left and right buttons to step through insertion."),
            html.Div([
                html.Button("← Left", id="array-left", n_clicks=0),
                html.Button("→ Right", id="array-right", n_clicks=0)
            ], style={"textAlign": "center", "margin": "20px"}),
            dcc.Store(id="array-index", data=self.current_index),  # Store for tracking index
            dcc.Store(id='register-callbacks', data={})  # Hidden trigger for callback registration
        ]) 

    def register_callbacks(self, app):
        """ Register Dash callbacks for updating the visualization """

        @app.callback(
            Output("page-content", "children", allow_duplicate=True),
            Output("array-index", "data"),
            Input("array-left", "n_clicks"),
            Input("array-right", "n_clicks"),
            State("array-index", "data"),
            prevent_initial_call=True)
        def update_array(left_clicks, right_clicks, index):
            """ Updates the array when stepping through insertion """
            i, j = index  # Extract current index
            print(left_clicks)
            print(right_clicks)
            ctx = callback_context
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
            flattened_array = self.sample_array.flatten()
            # Insert the value from the sample array
            idx_flat = i * self.cols + j  # Compute flat index
            if idx_flat < len(flattened_array):
                self.array[i, j] = flattened_array[idx_flat]  # Assign from input

            # Update visualization with new selection
            self.current_index = [i, j]
            return self.get_html(), [i, j]

        self.callbacks_registered = True  # Mark that callbacks are now registered

# Example input: a 1D array