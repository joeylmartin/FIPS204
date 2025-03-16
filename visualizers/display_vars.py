from abc import ABC, abstractmethod
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table, no_update
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
        Returns a Div, to store text and Latex info
        '''
        return html.Div()

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
        Called by parent VariablePage. Sets the selected_index to init value, to highlight the variable
        '''
        pass

    @abstractmethod
    def set_to_deselected(self):
        '''
        Called by parent VariablePage. Sets the selected_index to Null value
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
        self.table_container_id = self.table_id + "-container"
        self.div_id = self.table_id + "-interactive-container"
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
            columns=[{'name': str(i), 'id': str(i)} for i in range(self.cols)], 
            style_data_conditional=conditional_styles,

            style_header={
                "backgroundColor": "transparent",  # No shading
                "border": "none",  # No border
                "fontSize": "0.9em",  # Smaller text
                "fontWeight": "bold",
                "textAlign": "center",
                "padding": "4px"
            },

            style_cell={
                "textAlign": "center",
                "padding": "0.5rem",
                "height": "3rem",
                "width": f"{100/self.cols}%",  # Ensures proper spacing
                "maxWidth": f"{100/self.cols}%",
                "whiteSpace": "nowrap"  # Prevents wrapping in small cells
            },

            style_table={"width": "100%", "overflowX": "auto"}
        )

    def set_to_selected(self):
        self.selected_index = [0, 0]

    def set_to_deselected(self):
        self.selected_index = [None, None]

    def get_interactive_representation(self):
        corner_label = html.Div(
            "i,j",
            style={
                "fontSize": "0.9em",
                "fontWeight": "bold",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
            }
        )

        row_indices = html.Div([
            html.Div(
                str(i),
                style={
                    "fontSize": "0.9em",
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "height": "3rem",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                }
            ) for i in range(self.rows)
        ], style={"display": "flex", "flexDirection": "column"})

        table = self.get_table()

        return html.Div([
            html.Div(corner_label, style={"gridArea": "corner"}),
            html.Div(table,id=self.table_container_id, style={"gridArea": "table"}),
            html.Div(row_indices, style={"gridArea": "rows"}),

            # Store component for callbacks
            dcc.Store(id=self.store_id)
        ], style={
            "display": "grid",
            "gridTemplateAreas": """
                'corner table'
                'rows table'
            """,
            "gridTemplateColumns": "3rem 1fr",
            "gridTemplateRows": "3rem auto",
            "width": "100%",
            "overflow": "auto",
            "padding": "5px",
            "maxHeight": "20vh",
            "gap": "0px",
        }, id=self.div_id)

    
    def register_callbacks(self, app):
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), Input(self.table_id, 'active_cell'))
        def update_array(active_cell):
            if self.selected_index == [None, None]:
                return no_update
            
            self.selected_index[0] = active_cell['row']
            self.selected_index[1] = active_cell['column']
            return [self.get_table()]
        
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table()]
        
class Display1DArray(DisplayVar):
    def __init__(self, app, value, name):
        self.array = value
        self.name = name

        self.cols = len(self.array)
 
        self.selected_index : int = None
        self.table_id = name + "-table"
        self.table_container_id = self.table_id + "-container"
        self.div_id = self.table_id + "-interactive-container"  #lazy :P
        self.store_id = self.table_id + "-store"

        self.register_callbacks(app)

    def is_valid_index_update(self, change):
        if self.selected_index + change < 0:
            return -1
        
        if self.selected_index + change >= self.cols:
            return 1
        
        self.selected_index += change
        return 0
    
    def get_store_id(self):
        return self.store_id
    
    def get_table(self):
        selected_style = {
            "backgroundColor": "#ff5733",  # Highlighted cell color
            "color": "white",
            "fontWeight": "bold"
        }

        conditional_styles = []
        if self.selected_index is not None:
            col = str(self.selected_index)  # Ensure column_id is a string
            conditional_styles.append({
                "if": {"row_index": 0, "column_id": col},
                **selected_style
            })

        df_d = [{str(i): val for i, val in enumerate(self.array)}]

        return dash_table.DataTable(
            id=self.table_id,
            data=df_d, 
            columns=[{'name': str(i), 'id': str(i)} for i in range(self.cols)], 
            style_data_conditional=conditional_styles,

           
            style_header={
                "backgroundColor": "transparent",  # No shading
                "border": "none",  # No border
                "fontSize": "0.9em",  # Smaller text
                "fontWeight": "bold",
                "textAlign": "center",
                "padding": "4px"
            },

            style_cell={
                "textAlign": "center",
                "padding": "0.5rem",
                "height": "3rem",
                "width": f"{100/self.cols}%",  # Ensures proper spacing
                "maxWidth": f"{100/self.cols}%",
                "whiteSpace": "nowrap"  # Prevents wrapping in small cells
            },

            style_table={"width": "100%", "overflowX": "auto"}
        )

    def set_to_selected(self):
        self.selected_index = 0
    
    def set_to_deselected(self):
        self.selected_index = None

    def get_interactive_representation(self):
        corner_label = html.Div(
            "i",
            style={
                "fontSize": "0.9em",
                "fontWeight": "bold",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
            }
        )

        table = self.get_table()

        return html.Div([
            html.Div(corner_label, style={"gridArea": "corner"}),
            html.Div(table,id=self.table_container_id, style={"gridArea": "table"}),

            # Store component for callbacks
            dcc.Store(id=self.store_id)
        ], style={
            "display": "grid",
            "gridTemplateAreas": """
                'corner table'
            """,
            "gridTemplateColumns": "3rem 1fr",
            "width": "100%",
            "overflow": "auto",
            "padding": "5px",
            "maxHeight": "20vh",
        }, id=self.div_id)

    def register_callbacks(self, app):
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), Input(self.table_id, 'active_cell'))
        def update_array(active_cell):
            if self.selected_index == None:
                return no_update
            
            self.selected_index = active_cell['column']
            return [self.get_table()]
        
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table()]
        
        

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