from abc import ABC, abstractmethod
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table
import dash_mantine_components as dmc
import fips_204
import math
import plotly.graph_objects as go
import numpy as np

from fips_204.parametres import D_DROPPED_BITS, Q_MODULUS, K_MATRIX, VECTOR_ARRAY_SIZE


class DisplayVar(ABC):
    @abstractmethod
    def get_interactive_representation(self):
        '''
        Returns a Dash HTML object that represents the variable representation
        '''
        pass

    @abstractmethod
    def is_valid_index_update(self, change: int):
        '''
        When next/prev index buttons are called, returns 0 if current var is still selected.
        Otherwise, returns -1 if below index bounds, and 1 if above. 
        Change is 1 for next index, -1 for previous. 
        Updates selected_index property if still selected.
        '''
        pass

    def get_latex_representation(self):
        '''
        Returns a Latex string that represents the variable value
        '''
        return dcc.Markdown("$A_{i,j} = f(...)$", mathjax=True)

    def register_callbacks(self, app):
        '''
        Registers Dash callbacks for updating the visualization
        '''
        pass

    @abstractmethod
    def get_store_id(self):
        '''
        Store_id used as a trigger to update the visualization
        '''
        pass
  
    @abstractmethod
    def set_to_selected(self):
        '''
        Called by parent VariablePage. Sets the selected_index to init value
        '''
        pass

class Display2DArray(DisplayVar):
    def __init__(self, app, value, name):
        self.array = value
        self.name = name

        #stored as property not globally, but with data rep
        self.selected_index = [None, None]
        self.rows, self.cols = self.array.shape

        self.formatted_data = [  
            {str(i): val for i, val in enumerate(row)} for row in self.array]

        self.table_id = name + "-table"
        self.div_id = self.table_id + "-container"
        self.store_id = self.table_id + "-store"

        self.register_callbacks(app)

    def get_store_id(self):
        return self.store_id
    
    def is_valid_index_update(self, change):
        print(f"Selected index: {self.selected_index}")
        if change == 1:
            if self.selected_index[1] + 1 >= self.cols:
                if self.selected_index[0] + 1 >= self.rows:
                    return 1
                
                self.selected_index[0] += 1
                self.selected_index[1] = 0
                return 0
            
            self.selected_index[1] += 1
            return 0
        
        elif change == -1:
            if self.selected_index[1] - 1 < 0:
                if self.selected_index[0] - 1 < 0:
                    return -1
                
                self.selected_index[0] -= 1
                self.selected_index[1] = self.cols - 1
                return 0
            
            self.selected_index[1] -= 1
            return 0
        
        else:
            raise ValueError("invalid index change")

    def get_table(self):
        """Generates the Dash DataTable with no header row"""

        selected_style = {
            "backgroundColor": "#ff5733",  # Highlighted cell color (orange-red like in your screenshot)
            "color": "white",
            "fontWeight": "bold"
        }
        
        conditional_styles = []
        if self.selected_index and None not in self.selected_index:
            row, col = self.selected_index
            conditional_styles.append({
                "if": {
                    "row_index": row,
                    "column_id": str(col)  # Ensure column ID is a string
                },
                **selected_style
            })

        return dash_table.DataTable(
            id=self.table_id,
            data=[{str(i): val for i, val in enumerate(row)} for row in self.array],
            columns=[{'name': '', 'id': str(i)} for i in range(self.cols)],  # No headers
            style_data_conditional=conditional_styles,
            style_header={"display": "none"}, 
            css=[{"selector": "tbody tr:first-child", "rule": "height: 0px !important; display: none !important;"}],
            style_table={"width": "100%"},
            style_cell={
                "textAlign": "center",
                "padding": "0.5rem",
                "height": "3rem",
                "width": f"{100/self.cols}%",  # Relative width based on number of columns
                "maxWidth": f"{100/self.cols}%"
            },
        )

    def set_to_selected(self):
        self.selected_index = [0, 0]

    def get_interactive_representation(self):
        """Creates a compactly aligned table with row/column indices"""
        
        # The i,j label for the top-left corner
        corner_label = html.Div(
            "i,j",
            style={
                "fontSize": "0.9em",
                "fontWeight": "bold",
                "textAlign": "center",
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }
        )
        
        # Column indices (the j values)
        column_indices = html.Div([
            html.Div(
                f"{j}",
                style={
                    "fontSize": "0.9em",
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "width": f"{100/self.cols}%",
                    "display": "inline-block"
                }
            ) for j in range(self.cols)
        ], style={"width": "100%", "display": "flex", "justifyContent": "space-around"})
        
        # Create the main table
        table = self.get_table()
        
        # Main layout using CSS Grid for perfect alignment
        return html.Div([
            # Top section with grid layout
            html.Div([
                # Left corner with i,j label
                html.Div(corner_label, style={
                    "gridArea": "corner",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                }),
                
                # Top row with column indices
                html.Div(column_indices, style={
                    "gridArea": "cols",
                    "display": "flex",
                    "alignItems": "center"
                })
            ], style={
                "display": "grid",
                "gridTemplateAreas": "'corner cols'",
                "gridTemplateColumns": "3rem 1fr",  # The width of the first column matches the width of row indices
                "marginBottom": "0.5rem"
            }),
            
            # Bottom section with grid layout
            html.Div([
                # Left column with row indices
                html.Div([
                    html.Div(
                        f"{i}",
                        style={
                            "fontSize": "0.9em",
                            "fontWeight": "bold",
                            "textAlign": "center",
                            "height": "3rem",  # Match the height of table cells
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center"
                        }
                    ) for i in range(self.rows)
                ], style={
                    "gridArea": "rows",
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "space-around"
                }),
                
                # Main data table
                html.Div(table, style={
                    "gridArea": "table"
                })
            ], style={
                "display": "grid",
                "gridTemplateAreas": "'rows table'",
                "gridTemplateColumns": "3rem 1fr"  # The width of the first column matches the width of row indices
            }),
            
            # Store component for callbacks
            dcc.Store(id=self.store_id)
        ], id=self.div_id)

    
    def register_callbacks(self, app):
        @app.callback(Output(self.div_id, 'children', allow_duplicate=True),
                Input(self.table_id, 'active_cell'))
        def update_array(active_cell):
            self.selected_index = active_cell #TODO: figure out datatype of ac
            return [self.get_table(),
                      dcc.Store(id=self.store_id)]
        
        @app.callback(Output(self.div_id, "children",allow_duplicate=True), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table(),
                      dcc.Store(id=self.store_id)]
class Display1DArray(DisplayVar):
    def __init__(self, app, value, name):
        self.array = value
        self.name = name

        #stored as property not globally, but with data rep
        self.selected_index : int = None
        self.table_id = name + "-table"
        self.div_id = self.table_id + "-container"
        self.store_id = self.table_id + "-store"

        self.register_callbacks(app)

    def is_valid_index_update(self, change):
        if self.selected_index + change < 0:
            return -1
        
        if self.selected_index + change >= len(self.array):
            return 1
        
        self.selected_index += change
        return 0
    
    def get_store_id(self):
        return self.store_id
    
    def get_table(self):
        normal_style = {
            "border": "2px solid black",
            "textAlign": "center",
            "backgroundColor": "#f0f0f0",
            "padding": "10px",
            "fontSize": "18px"
        }
        selected_style = {
            "border": "4px solid black",
            "textAlign": "center",
            "backgroundColor": "#ff5733",  # Highlight selection
            "padding": "10px",
            "fontSize": "18px"
        }
        grid_cells = [html.Div(
                dmc.Text(self.array[i]),
                style=selected_style if i == self.selected_index else normal_style)
            for i in range(len(self.array))]

        #TODO: add div container for scroll and labels
        return dmc.SimpleGrid(
            id=self.table_id,
            cols=len(self.array),
            spacing="md",
            verticalSpacing="md",
            children=grid_cells)

    def get_interactive_representation(self):
        return html.Div(
            children=[self.get_table(),
                      dcc.Store(id=self.store_id)],
            id=self.div_id  
        )
    
    def set_to_selected(self):
        self.selected_index = 0
    
    
    def register_callbacks(self, app):
        @app.callback(Output(self.div_id, "children"), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table(),
                      dcc.Store(id=self.store_id)]
        

class RoundingRing(DisplayVar):
    def __init__(self, app):
        

        self.graph_id = "t" + "-graph"
        self.div_id = self.graph_id + "-container"
        self.store_id = self.graph_id + "-store"
        
        self.t = fips_204.internal_funcs.global_t.reshape(K_MATRIX * VECTOR_ARRAY_SIZE)
        self.t1 = fips_204.internal_funcs.global_t1.reshape( K_MATRIX * VECTOR_ARRAY_SIZE)
        self.s2 = fips_204.internal_funcs.global_s2.reshape(K_MATRIX * VECTOR_ARRAY_SIZE)
        self.t12d = self.t1 << D_DROPPED_BITS  # T1 multiplied by 2^D


        #gen angles and points
        self.angles = np.linspace(0, 2 * np.pi, 300)
        self.x_circle = np.cos(self.angles)
        self.y_circle = np.sin(self.angles)

        self.selected_index = None
        self.register_callbacks(app)
  
    def get_graph(self):

        #Gen Arc representing T +- error
        arc_start = (self.t[self.selected_index] +(20 *self.s2[self.selected_index])) % Q_MODULUS
        arc_end = (self.t[self.selected_index] - (20*self.s2[self.selected_index])) % Q_MODULUS
        
        

        fig = go.Figure()
    
        # Plot the circle representing Q_MODULUS ring
        fig.add_trace(go.Scatter(x=self.x_circle, y=self.y_circle, mode='lines', name='Ring mod Q'))
        


        def polar_to_cartesian(value, q):
            angle = 2 * np.pi * (value / q)
            return np.cos(angle), np.sin(angle)
        
        x_t, y_t = polar_to_cartesian(self.t[self.selected_index], Q_MODULUS)
        x_t1, y_t1 = polar_to_cartesian(self.t12d[self.selected_index], Q_MODULUS)

        x_arc, y_arc = [], []
        for val in np.linspace(arc_start, arc_end, 30):
            x, y = polar_to_cartesian(val, Q_MODULUS)
            x_arc.append(x)
            y_arc.append(y)
        fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', name='Error band (S2)', line=dict(color='gray', width=2)))
        
        fig.update_layout(
            title='Rounding with High Bits Visualization',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            width=600,
            height=600
        )
        fig.add_trace(go.Scatter(x=[x_t], y=[y_t], mode='markers', name='T (original)', marker=dict(size=10, color='red')))
        fig.add_trace(go.Scatter(x=[x_t1], y=[y_t1], mode='markers', name='T1 * 2^D', marker=dict(size=10, color='blue')))
        return fig

    def register_callbacks(self, app):
        @app.callback(Output(self.div_id, "children",allow_duplicate=True), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [dcc.Graph(id='rounding-graph', figure=self.get_graph()),
                      dcc.Store(id=self.store_id)]

    def is_valid_index_update(self, change):
        if self.selected_index + change < 0 or self.selected_index + change >= (K_MATRIX * VECTOR_ARRAY_SIZE):
            return change
        
        self.selected_index += change
        return 0
    
    def get_latex_representation(self):
        return super().get_latex_representation()

    def get_store_id(self):
        return self.store_id
    
    def set_to_selected(self):
        self.selected_index = 0

    def get_interactive_representation(self):
        return html.Div(
            children=[
                dcc.Graph(id='rounding-graph', figure=self.get_graph()),
                      dcc.Store(id=self.store_id)],
            id=self.div_id  
        )