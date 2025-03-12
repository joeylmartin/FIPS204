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

from dash import Dash, html, dcc, callback, Output, Input

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

temp = ProjectionMethods.MDS

class ALattice(DemoPage):
    def __str__(self):
        return "A, S1, S2 Lattice Visualization"
    
    def __init__(self,pk, sk, app, extra_vars : dict):

        self.origin_df : pd.DataFrame = pd.DataFrame(np.array([[0, 0, 0]]), columns=['X', 'Y', 'Z'])

        #Dict with 'Var name' -> 'Var value' (being a 1x(K * 256) array).
        #Each generates their own dataframe
        self.vars = extra_vars 
        

        rho, k, tr, s1, s2, t0 = skDecode(sk)
        a = expand_a(rho)
        lattice_points, lattice_with_s2 = sample_lattice_point(a, s2)

        self.vars["Lattice from A"] = lattice_points
        self.vars["Lattice with S2"] = lattice_with_s2

        #self.get_w(pk, sk)

        
        #This is used in the VIEW_3D projection method, sets the "dimensional slice" we use
        self.step_index = 0

       # self.t = self.flatten_point(np.array(fips_204.internal_funcs.global_t))
        #self.t1 = fips_204.internal_funcs.global_t1 


       # d_p2 = math.pow(2, D_DROPPED_BITS)

        # self.cl = self.t1 * d_p2
        self.register_callbacks(app)


        

    def generate_3d_points(self, projection_method: ProjectionMethods):
        dfs = [self.origin_df]

        match projection_method:
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
        self.df = pd.concat(dfs, ignore_index=True)

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
        self.generate_3d_points(temp)
        fig = px.scatter_3d(self.df, x='X', y='Y', z='Z', color='Type', opacity=0.2, title="3D Lattice Visualization")
        return fig
    
    def get_html(self):
        return html.Div([
            html.Div([
                dcc.Graph(id='lattice-plot', figure=self.get_figure())
            ]),
            html.Div([
                dcc.Slider(
                    id='ind_slider',
                    min=0,
                    max=253,
                    step=3,
                    value=0,
                    marks={i: str(i) for i in range(0, 256, 3)}
                )
            ]),
            dcc.Store(id='register-callbacks', data={})  # Trigger for callback registration
        ], style={"width": "100%", "height" : "100%"})
    
    def register_callbacks(self, app):
        @app.callback(
            Output('lattice-plot', 'figure', allow_duplicate=True),
            Input('ind_slider', 'value')
        )
        def update_plot(value):
            """ Callback function to regenerate the figure when the button is clicked. """
            self.set_step_index(value)
            return self.get_figure()



class WLattice(ALattice):
    def __init__(self, pk, sk, app, m : str):
        

        #set to false, dont show Lattice until message is sent
        self.is_showing_message = False
        #convert message string to bits
        mb = bytes(m, encoding='utf-8')
        self.m: bitarray = new_bitarray()
        self.m.frombytes(mb)

        self.t = super().flatten_point(np.array(fips_204.internal_funcs.global_t))
        self.t1 = fips_204.internal_funcs.global_t1 
        self.cl = self.t1 * math.pow(2, D_DROPPED_BITS)
        self.calc_vals(pk, sk)
        
        extra_vals = {
            "W": self.w,
            "W' approx": self.w_a,
            "Z": self.z
        }
        super().__init__(pk, sk, app, extra_vals)
    
    def calc_vals(self, pk, sk):
        '''
        Gets W, W' and Z from pk and sk. 
        '''

        sigm = ml_dsa_sign(sk, self.m, ctx) 
        ch, z, h = sigDecode(sigm)
        is_ver = ml_dsa_verify(pk, self.m, sigm, ctx)

        self.w = super().flatten_point(np.array(fips_204.internal_funcs.global_w))
        self.w_a = super().flatten_point(np.array(fips_204.internal_funcs.global_w_a))
    
        self.z = super().flatten_point(z)