from abc import ABC, abstractmethod
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table, no_update
import dash_mantine_components as dmc
import fips_204
import math
import plotly.graph_objects as go
import numpy as np
from fips_204.parametres import BYTEORDER, D_DROPPED_BITS, Q_MODULUS, K_MATRIX, VECTOR_ARRAY_SIZE
import random
import app_calc_vals as globals


'''
This is an object to abstract the interface for different types of Variable Display representations (i.e 2D arrays, 3D...)
'''
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
    def __init__(self, value, name, corner_label="i,j", latex_func=None):
        self.array = value
        self.name = name

        #stored as property not globally, but with data rep
        self.selected_index = [None, None]
        self.rows, self.cols = self.array.shape
        
        self.formatted_data = [  
            {str(i): val for i, val in enumerate(row)} for row in self.array]

        self.table_id = name + "-table"
        self.table_container_id = self.table_id + "-container"
        self.store_id = self.table_id + "-store"
        
        self.latex_div = self.name + "-latex-container"

        #label on interactive_representation
        #refers to the concept of its width;
        #i.e some are KxL matrices...
        self.corner_label = corner_label

        #bodge; used by 3D array to point 2d array latex to parent one
        self.parent_latex_func = latex_func

    def get_store_id(self):
        return self.store_id
    
    def is_valid_index_update(self, change):
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
    
    def get_latex_representation(self):

        #bodge; pass in latex function pointer from 3d array
        if self.parent_latex_func:
            return self.parent_latex_func()
        return html.Div()

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
            self.corner_label,
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
        table_and_labels = html.Div([
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
            })
        return html.Div([
            html.Div(f"{self.name}:",  style={"fontSize": "1rem",  "fontWeight": "bold", "marginBottom": "0.5rem"}),
            table_and_labels
        ])
        

     
    def register_callbacks(self, app):
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), 
                      Output(self.latex_div, "children",allow_duplicate=True), 
                      Input(self.table_id, 'active_cell'))
        
        def update_array(active_cell):
            #update array on index select has to also update latex, because it isn't called from parent hierarchy
            if self.selected_index == [None, None]:
                return no_update
            
            self.selected_index[0] = active_cell['row']
            self.selected_index[1] = active_cell['column']
            return [self.get_table()], [self.get_latex_representation()]
        
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True), Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table()]

class Display1DArray(DisplayVar):
    def __init__(self, value, name, corner_label="i"):
        self.array = value
        self.name = name

        self.cols = len(self.array)
 
        self.selected_index : int = None
        self.table_id = name + "-table"
        self.table_container_id = self.table_id + "-container"
        self.store_id = self.table_id + "-store"
        self.latex_div = self.name + "-latex-container"
        
        self.corner_label=corner_label

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
            self.corner_label,
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
        table_and_labels = html.Div([
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
        })
        return html.Div([
            html.Div( f"{self.name}:",  style={"fontSize": "1rem", 
                                               "fontWeight": "bold", "marginBottom": "0.5rem"}),
            table_and_labels
        ])

     
    def register_callbacks(self, app):
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True),
                      Output(self.latex_div, "children",allow_duplicate=True),
                       Input(self.table_id, 'active_cell'))
        def update_array(active_cell):
            if self.selected_index == None:
                return no_update
            
            self.selected_index = active_cell['column']
            return [self.get_table()], [self.get_latex_representation()]
        
        @app.callback(Output(self.table_container_id, "children",allow_duplicate=True),
                       Input(self.store_id, "data"))
        def update_on_index_change(data):
            return [self.get_table()]

class Display3DArray(DisplayVar):
    def __init__(self,  value, name, corner_label="j,k"):
        self.array = value
        self.name = name
        self.selected_var_index = None
        self.subviews = self.array.shape[0]
        self.latex_div = self.name + "-latex-container"
        self.sub_displays = [Display2DArray( self.array[i], f"{self.name}[{i}]", corner_label, latex_func=self.get_latex_representation) for i in range(self.subviews)]

        #override latex_divs to point to parent latex representation:
        for display in self.sub_displays:
            display.latex_div = self.latex_div

    @property
    def selected_index(self):
        """
        Used to get value of selected index across subviews
        """
        selected_var = self.sub_displays[self.selected_var_index]
        return [self.selected_var_index, selected_var.selected_index[0], selected_var.selected_index[1]]
    
    def is_valid_index_update(self, change: int):
        change = self.sub_displays[self.selected_var_index].is_valid_index_update(change)
        if change == 0:
            #index still in subview
            return 0
        elif change == -1:
            #index changing to previous subview
            if self.selected_var_index <= 0:
                return -1
            
            self.selected_var_index -= 1
            self.sub_displays[self.selected_var_index].set_to_selected()

            return 0
        elif change == 1:
            #index changing to next subview
            if self.selected_var_index >= (self.subviews - 1):
                return 1
            
            self.selected_var_index += 1
            self.sub_displays[self.selected_var_index].set_to_selected()

            return 0
        
        raise ValueError("Invalid change value!")

    def get_interactive_representation(self):
        subviews = [html.Div([self.sub_displays[i].get_interactive_representation()]) 
                    for i in range(self.subviews)]
        return html.Div(subviews, style={"display": "flex", "flexDirection": "column"})

    def register_callbacks(self, app):
        for sub_display in self.sub_displays:
            sub_display.register_callbacks(app)


    def get_store_id(self):
        return [sub_display.get_store_id() for sub_display in self.sub_displays]

    def set_to_selected(self):
        self.selected_var_index = 0
        self.sub_displays[self.selected_var_index].set_to_selected()

    def set_to_deselected(self):
        self.sub_displays[self.selected_var_index].set_to_deselected()
        self.selected_var_index = None

class RoundingRing(DisplayVar):
    def __init__(self):
        self.graph_id = "t" + "-graph"
        self.div_id = self.graph_id + "-container"
        self.store_id = self.graph_id + "-store"
        self.slider_id = self.graph_id + "-slider"
        
        self.w = np.array(globals.w).reshape(K_MATRIX * VECTOR_ARRAY_SIZE)
        self.wa = globals.w_a.reshape(K_MATRIX * VECTOR_ARRAY_SIZE)
        
        self.angles = np.linspace(0, 2 * np.pi, 300)
        self.x_circle = np.cos(self.angles)
        self.y_circle = np.sin(self.angles)

        self.selected_index = 0

    def get_graph(self):
        fig = go.Figure()
        
        # Plot the circle representing Q_MODULUS ring
        fig.add_trace(go.Scatter(x=self.x_circle, y=self.y_circle, mode='lines', name='Ring mod Q'))
        
        def polar_to_cartesian(value, q):
            angle = 2 * np.pi * (value / q)
            return np.cos(angle), np.sin(angle)
        
        x_w, y_w = polar_to_cartesian(self.w[self.selected_index], Q_MODULUS)
        x_wa, y_wa = polar_to_cartesian(self.wa[self.selected_index], Q_MODULUS)

        fig.update_layout(
            title='Rounding with High Bits Visualization',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            width=600,
            height=600
        )
        fig.add_trace(go.Scatter(x=[x_w], y=[y_w], mode='markers', name='W (original)', marker=dict(size=10, color='red')))
        fig.add_trace(go.Scatter(x=[x_wa], y=[y_wa], mode='markers', name="W'approx", marker=dict(size=10, color='blue')))
        return fig

    def register_callbacks(self, app):
        @app.callback(
            [Output(self.div_id, "children", allow_duplicate=True),
             Output(self.store_id, "data", allow_duplicate=True)],
            [Input(self.store_id, "data"),
             Input(self.slider_id, "value")]
        )
        def update_on_index_change(data, slider_value):
            self.selected_index = slider_value
            return ([dcc.Graph(id='rounding-graph', figure=self.get_graph()),
                     dcc.Slider(
                         id=self.slider_id,
                         min=0,
                         max=(K_MATRIX * VECTOR_ARRAY_SIZE) - 1,
                         step=1,
                         value=self.selected_index,
                         marks={i: str(i) for i in range(0, (K_MATRIX * VECTOR_ARRAY_SIZE), max(1, (K_MATRIX * VECTOR_ARRAY_SIZE) // 10))},
                     ),
                     dcc.Store(id=self.store_id)],
                    self.selected_index)
    
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

    def set_to_deselected(self):
        self.selected_index = None

    def get_interactive_representation(self):
        return html.Div(
            children=[
                dcc.Graph(id='rounding-graph', figure=self.get_graph()),
                dcc.Slider(
                    id=self.slider_id,
                    min=0,
                    max=(K_MATRIX * VECTOR_ARRAY_SIZE) - 1,
                    step=1,
                    value=self.selected_index,
                    marks={i: str(i) for i in range(0, (K_MATRIX * VECTOR_ARRAY_SIZE), max(1, (K_MATRIX * VECTOR_ARRAY_SIZE) // 10))},
                ),
                dcc.Store(id=self.store_id)
            ],
            id=self.div_id  
        )
    

##KEY GEN

class XiDisplay(Display1DArray):
    def __init__(self):
        seed = random.getrandbits(256) 
        s_b = seed.to_bytes(32, BYTEORDER)
        
        s_arr = np.frombuffer(s_b, dtype=np.uint8)
        super().__init__(s_arr, "xi", "i")

    def get_latex_representation(self):
        latex_str = "𝜉"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("First, we generate a random seed. We take parts of 𝜉 to create seeds 𝜌 and 𝜌′.", 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"},id=self.latex_div)
    

class ADisplay(Display3DArray):
    def __init__(self, value):
        super().__init__(value, "𝐀", "ℓ,256")
    def get_latex_representation(self):
        if self.selected_var_index is not None:
            i, j, _ = self.selected_index
            latex_str = f"𝐀[{i},{j}] = \\text{{RejNTTPoly}}(\\rho + {i} + {j})"
        else:
            latex_str = "𝐀[i,j] = \\text{RejNTTPoly}(\\rho + i + j)"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Where ρ is a randomly sampled seed, and RejNTTPoly samples a 256-order polynomial. 
                     𝐀 consists of a 𝑘(4) x ℓ(4) x 256 matrix. This can be represent a lattice, which consists
                     of a series of basis vectors, with integer combinations to form a pattern of points.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"}, id=self.latex_div)


class S1Display(Display2DArray):
    def __init__(self, value):
        super().__init__( value, "S1", "ℓ,256")

    def get_latex_representation(self):
        if None not in self.selected_index:
            i, j = self.selected_index
            latex_str = f"S1[{i}] = \\text{{RejBoundedPoly}}(\\rho' + {i})"
        else:
            latex_str = "S1[i] = \\text{RejBoundedPoly}(\\rho' + i)"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Where RejBoundedPoly samples a 256-order polynomial, with coefficients in a limited range.
                     S1 is a ℓx256 matrix. This represents a vector: the coefficients of basis vectors to produce
                     a point on the lattice. This is kept secret. 
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"}, id=self.latex_div)

class S2Display(Display2DArray):
    def __init__(self,  value):
        super().__init__( value, "S2", "𝑘,256")

    def get_latex_representation(self):
        if None not in self.selected_index:
            i, j = self.selected_index
            latex_str = f"S2[{i}] = \\text{{RejBoundedPoly}}(\\rho' + {i} + ℓ)"
        else:
            latex_str = "S2[i] = \\text{RejBoundedPoly}(\\rho' + i + ℓ)"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Where RejBoundedPoly samples a 256-order polynomial, with a coefficients in a limited range. 
                     S2 is a 𝑘x256 matrix. This represents a small amount of error that is added to a point on the public
                     value T.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"}, id=self.latex_div)

class TDisplay(Display2DArray):
    def __init__(self, value):
        super().__init__(value, "T", "𝑘,256")

    def get_latex_representation(self):
        latex_str = "T = (A ∘ S1) + S2"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Where (A ∘ S1) represents a point on the lattice, with a small bit of error added from S2.
                     T, alongside A are released as the public key, with S1 and S2 kept secret.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"},id=self.latex_div)
    

##SIGNING

class YDisplay(Display2DArray):
    def __init__(self, value):
        super().__init__(value, "Y", "ℓ,256")

    def get_latex_representation(self):
        if None not in self.selected_index:
            i, _ = self.selected_index
            latex_str = f"\\rho'' = \\text{{hash}}(\\mu+\\text{{hash(msg)}})"
            latex2_str = f"y[{i}] = \\text{{bit_unpack}}(\\text{{hash}}(\\rho''+ {i} +\\kappa, 256 * (1+bitlen(γ₁)))"
        else:
            latex_str = "\\rho'' = \\text{{hash}}(\\mu+\\text{{hash(msg)}})"
            latex2_str = "y[i] = \\text{{bit_unpack}}(\\text{{hash}}(\\rho''+ i +\\kappa, 256 * (1+bitlen(γ₁))))"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            dcc.Markdown(f"${latex2_str}$", mathjax=True),
            html.Div("""Where μ represents a randomly generated number at the beginning of signing.
                     hash() takes data as an input, and deterministically produces an output that
                     would be difficult to reverse. A property of hash functions is that they produce 
                     a uniform spread, centred around 0. This yields a seemingly random, yet short vector.

                    Our final hash, of ρ'' with i and κ is hashed to a bitlength of 1 + γ₁'s bitlength 
                     per polynomial entry; thus each coefficient is in the range of -γ₁+1, γ₁. 
                     The values are extracted with bit_unpack().
                     Variable κ is used for rejection sampling.

                     We then define W as the point on the lattice made from Y: (A ∘ Y).
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"},id=self.latex_div)

class CDisplay(Display1DArray):
    def __init__(self, value):
        super().__init__(value, "c", "i")

    def get_latex_representation(self):
        latex_str = "C = \\text{SampleInBall}(\\text{hash(msg)} + \\text{HighBits}(w))"


        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Where SampleInBall produces a "mask"-- an array of coefficents with a fixed, small about of ones; the rest being 0.
                     The mask is sourced deterministically from hashing the message. W is rounded with HighBits, to make verification valid later.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"},id=self.latex_div)
class ZDisplay(Display2DArray):
    def __init__(self,  value):
        super().__init__( value, "Z", "𝑘,256")

    def get_latex_representation(self):
        latex_str = "Z = y + (c \\cdot S1)"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            html.Div("""Z represents a vector equal to Y, with some of S1 added to it. 
                     Importantly, S1 is not added to it as a point, but to its coefficients.
                    Adding the S1 term turns the vector Z into a challenge; it allows the prover
                     to show their knowledge of S1 to the verifier. And because C is sourced
                     determistically from the message, it makes the process non-interactive. 

                     Importantly, we take an additional step and restart the process (with
                     a increased value for κ) if Z is not a short vector. If Z is long, it
                     removes its alignment with lattice-hardness problems (SVP), and
                     would allow S1 to be reconstructed over many signatures. If Z is short, 
                     the CS1 term would fall within the expected distribution of Y, and
                     be hard to find.

                     As our signature, we release Z, C and the message.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"}, id=self.latex_div)

##VERIFICATION


class WADisplay(Display2DArray):
    def __init__(self, value):
        super().__init__(value, "W'", "𝑘,256")

    def get_latex_representation(self):
        latex_str = """w =A \\cdot Y = A\\cdot(z-c\\cdot s1)"""
        latex2_str = "w' = A\\cdot z - c\\cdot t =A\\cdot z - c\\cdot(A\\cdot s1 + s2)"
        latex3_str = "=A\\cdot(z-c\\cdot s1) +c \\cdot s2"
        latex4_str = "c\\cdot s2\\simeq0 \\therefore w' \\simeq w"
        latex5_str = "\\therefore \\text{HighBits}(w') = \\text{HighBits}(w)"
        latex6_str = "\\therefore c =\\text{SampleInBall}(\\text{hash(msg)} + \\text{HighBits}(w'))"

        return html.Div([
            dcc.Markdown(f"${latex_str}$", mathjax=True),
            dcc.Markdown(f"${latex2_str}$", mathjax=True),
            dcc.Markdown(f"${latex3_str}$", mathjax=True),
            dcc.Markdown(f"${latex4_str}$", mathjax=True),
            dcc.Markdown(f"${latex5_str}$", mathjax=True),
            dcc.Markdown(f"${latex6_str}$", mathjax=True),
            html.Div("""We can define W' as another point on the lattice. It is a reconstruction
                     of W, using only public key values and the signature vector. As seen above, 
                     we can get a very close approximation(only a small error is added), without
                     exposing any of the secret values. 
                     
                     Finding a short vector Z that would satisfy the approximation equations above is 
                     computationally equivalnet to the Short-Integer-Solution (SIS) problem, and
                     is understood as hard against quantum computing.
                     
                     We validate that W' would be close to W by computing the commitment mask. If C' and C are equal,
                     then W' and W must be approximately equal, therefore the signature is valid.
                     """, 
                     style={"fontSize": "0.9rem", "marginTop": "0.5rem"})
        ], style={"marginTop": "1rem", "fontSize": "1.1rem"},id=self.latex_div)