import fips_204.internal_funcs
from fips_204.external_funcs import * #TODO: replace these im
import bitarray
from .vis_utils import sample_lattice_point, center_mod_q, DemoPage
from fips_204.parametres import *
from enum import Enum
from sklearn.manifold import MDS
import pandas as pd
import plotly.express as px
import os
import random

from dash import Dash, html, dcc, callback, Output, Input, State, no_update, MATCH

from abc import ABC, abstractmethod

from fips_204.internal_funcs import signed_kappa
#TODO add aaccess from fontened

mb = b"Hello world!"


ctx_b = os.urandom(255)
ctx = new_bitarray()
ctx.frombytes(ctx_b)

class ProjectionMethods(Enum):
    VIEW_3D = 1
    MDS = 2

temp = ProjectionMethods.VIEW_3D

class ALattice(DemoPage):
    def __str__(self):
        return "A, S1, S2 Lattice Visualization"
    
    def __init__(self,pk, sk, app, extra_vars : dict = {}):

        self.origin_df : pd.DataFrame = pd.DataFrame(np.array([[0, 0, 0]]), columns=['X', 'Y', 'Z'])

        #Dict with 'Var name' -> 'Var value' (being a 1x(K * 256) array).
        #Each generates their own dataframe
        self.vars = extra_vars 
        

        rho, k, tr, s1, s2, t0 = skDecode(sk)
        a = expand_a(rho)
        lattice_points, lattice_with_s2 = sample_lattice_point(a, s2)

        self.vars["Lattice from A"] = lattice_points
        self.vars["Lattice with S2"] = lattice_with_s2

        #default; also way faster
        self.projection_method : ProjectionMethods = ProjectionMethods.VIEW_3D
        #self.get_w(pk, sk)

        
        #This is used in the VIEW_3D projection method, sets the "dimensional slice" we use
        self.step_index = 0

       # self.t = self.flatten_point(np.array(fips_204.internal_funcs.global_t))
        #self.t1 = fips_204.internal_funcs.global_t1 


       # d_p2 = math.pow(2, D_DROPPED_BITS)

        # self.cl = self.t1 * d_p2
        self.register_callbacks(app)
        self.df = None

        self.lattice_plot_id = "w-lattice"

    def generate_3d_points(self):
        dfs = [self.origin_df]

        match self.projection_method:
            case ProjectionMethods.VIEW_3D:

                for name, val in self.vars.items():
                    df = pd.DataFrame(val[:,self.step_index:self.step_index+3], columns=['X', 'Y', 'Z'])
                    df['Type'] = name
                    dfs.append(df)

            case ProjectionMethods.MDS:
                transformed_vals = []

                #get number of data points for each variable
                data_lens = [val.shape[0] for val in self.vars.values()]

                #To pass through PCoA, we have to combine all data together,
                #so we track the data lengths so we can seperate them back out
                mds = MDS(n_components=3)
                temp_inp = np.vstack((*self.vars.values(),))
                temp_out = mds.fit_transform(temp_inp)

                index_counter = 0
                for len in data_lens:
                    transformed_vals.append(temp_out[index_counter:index_counter+len])
                    index_counter += len

                nam_val_pairs = zip(self.vars.keys(), transformed_vals)
                
                for name, val in nam_val_pairs:
                    df = pd.DataFrame(val, columns=['X', 'Y', 'Z'])
                    df['Type'] = name
                    dfs.append(df)


        #join dfs into one
        return pd.concat(dfs, ignore_index=True)

    def flatten_point(self, point: np.ndarray) -> np.ndarray:
        '''
        Flatten point into Kx256-dimension space and modulo to q/2 space
        '''
        eg = point.reshape(1, K_MATRIX * VECTOR_ARRAY_SIZE)
        return center_mod_q(eg)
    
    def set_step_index(self, value):
        """
        Updates 'step' index for "3D slice" viewing method
        """
        self.step_index = value

    def get_figure(self):
        # Create 3D scatter plot
        self.df = self.generate_3d_points()
        fig = px.scatter_3d(self.df, x='X', y='Y', z='Z', color='Type', opacity=0.2, title="3D Lattice Visualization")
        return fig
    
    def get_html(self):
        
        proj_method = dcc.RadioItems(
                id='projection-method',
                options=[{'label': 'View 3D', 'value': ProjectionMethods.VIEW_3D.value},
                         {'label': 'PCoA', 'value': ProjectionMethods.MDS.value}],
                value=self.projection_method.value,
                labelStyle={'display': 'block'}
            )

        return html.Div([
            proj_method,
            html.Div([
                dcc.Graph(id=self.lattice_plot_id, figure=self.get_figure(), style={"width": "90vw", "height": "70vh"}),
            ]),
            html.Div([
                dcc.Slider(
                    id='ind_slider',
                    min=0,
                    max=(VECTOR_ARRAY_SIZE * L_MATRIX - 3),
                    step=3,
                    value=0,
                    marks={i: {"label" : str(i), "style": {"fontSize":"0.5rem"}} for i in range(0, VECTOR_ARRAY_SIZE * L_MATRIX, 12)}
                ),
            ]),

        ], style={"width": "100%", "height" : "100%"})
    
    def register_callbacks(self, app):
        @app.callback(
            Output(self.lattice_plot_id, 'figure', allow_duplicate=True),
            Input('projection-method', 'value')
        )
        def update_projection(projection_method):
            self.projection_method = ProjectionMethods(projection_method)
            return self.get_figure()
        
        @app.callback(
            Output(self.lattice_plot_id, 'figure', allow_duplicate=True),
            Input('ind_slider', 'value')
        )
        def update_plot(value):
            """ Callback function to regenerate the figure when the button is clicked. """
            self.set_step_index(value)
            return self.get_figure()



class WLattice(ALattice):
    def __init__(self, pk, sk, app, m : str):

    
        self.t = super().flatten_point(np.array(fips_204.internal_funcs.global_t))
        self.t1 = fips_204.internal_funcs.global_t1 
        self.cl = self.t1 * math.pow(2, D_DROPPED_BITS)
        self.pk = pk
        self.sk = sk
        self.app = app
        self.register_callbacks(self.app)

        self.lattice_plot_id = "w-lattice"
        self.lattice_plot_container = self.lattice_plot_id + '-container'
    def calc_vals(self, message: bitarray):
        '''
        Gets W, W' and Z from pk, sk and message.
        '''

        sigm = ml_dsa_sign(self.sk, message, ctx) 
        ch, z, h = sigDecode(sigm)
        is_ver = ml_dsa_verify(self.pk, message, sigm, ctx)

        self.w = super().flatten_point(np.array(fips_204.internal_funcs.global_w))
        self.w_a = super().flatten_point(np.array(fips_204.internal_funcs.global_w_a))
        self.z = super().flatten_point(z)

    def sign_message_and_gen_lattice(self, message: bitarray):
        """
        Given a message, sign it and generate the lattice plot.
        """
        self.calc_vals(message)
        extra_vals = {
            "W": self.w,
            "W' approx": self.w_a,
            "Z": self.z
        }
        super().__init__(self.pk, self.sk, self.app, extra_vals)

    def get_html(self):
        return html.Div([
            html.Div([
                dcc.Input(id='message-input', type='text', placeholder='Enter message...', style={'margin-right': '10px'}),
                html.Button('Sign', id='sign-button', n_clicks=0)
            ], style={'margin-bottom': '10px'}),
            
            html.Div(id='message-status', children='(Enter message)', style={'font-style': 'italic'}),
            
            html.Div([
                dcc.Graph(id=self.lattice_plot_id, figure=self.get_figure(), style={"width": "90vw", "height": "70vh"}),
            ],id=self.lattice_plot_container),
            html.Div([
                dcc.Slider(
                    id='ind_slider',
                    min=0,
                    max=(VECTOR_ARRAY_SIZE * L_MATRIX - 3),
                    step=3,
                    value=0,
                    marks={i: {"label" : str(i), "style": {"fontSize":"0.5rem"}} for i in range(0, VECTOR_ARRAY_SIZE * L_MATRIX, 12)}
                ),
            ]),
            dcc.Store(id='lattice-ready', data={'ready': False})

        ], style={"width": "100%", "height" : "100%"})

    def register_callbacks(self, app):
        @app.callback(
            Output(self.lattice_plot_container, 'children'),
            Output('message-status', 'children'),
            Output('lattice-ready', 'data'),
            Input('sign-button', 'n_clicks'),
            State('message-input', 'value')
        )
        def update_lattice(n_clicks, message):
            if n_clicks > 0 and message:
                mb = bytes(message, encoding='utf-8')
                m = new_bitarray()
                m.frombytes(mb)

                self.sign_message_and_gen_lattice(m)

                return (
                    dcc.Graph(id=self.lattice_plot_id, figure=self.get_figure(), style={"width": "90vw", "height": "70vh"}),
                    f'Signed message: "{message}"',
                    {'ready': True}  # Set lattice-ready flag
                )
            return None, '(Enter message)', {'ready': False}

        @app.callback(
            Output({'type': 'lattice-plot', 'index': MATCH}, 'figure', allow_duplicate=True),
            Input('ind_slider', 'value'),
            State('lattice-ready', 'data'),
            prevent_initial_call=True 
        )
        def update_plot(value, lattice_ready):
            """ Updates the lattice figure only after it exists. """
            if lattice_ready and lattice_ready.get('ready'):
                self.set_step_index(value)
                return self.get_figure()
            return no_update
        
