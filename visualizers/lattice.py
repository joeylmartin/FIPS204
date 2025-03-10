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

from dash import Dash, html, dcc, callback, Output, Input

from abc import ABC, abstractmethod

from fips_204.internal_funcs import signed_kappa
#TODO add aaccess from fontened
mb = b"Hello world!"
m: bitarray = new_bitarray()
m.frombytes(mb)

ctx_b = os.urandom(255)
ctx = new_bitarray()
ctx.frombytes(ctx_b)

class ProjectionMethods(Enum):
    VIEW_3D = 1
    MDS = 2

temp = ProjectionMethods.VIEW_3D

class ALattice(DemoPage):
    def __str__(self):
        return "Lattice Visualization"
    
    def __init__(self,pk, sk, app):
        self.register_callbacks(app)
        self.sk = sk
        self.pk = pk
        rho, self.k, tr, s1, s2, t0 = skDecode(sk)

        self.a = expand_a(rho)
        
        sig = ml_dsa_sign(sk, m, ctx)
        yes = ml_dsa_verify(pk, m, sig, ctx)
        self.lattice_points, self.lattice_with_s2 = sample_lattice_point(self.a, s2)

        self.get_w(pk, sk)

        self.origin_df : pd.DataFrame = pd.DataFrame(np.array([[0, 0, 0]]), columns=['X', 'Y', 'Z'])

        #This is used in the VIEW_3D projection method, sets the "dimensional slice" we use
        self.step_index = 0
    def get_w(self, pk, sk):
        '''
        Gets W, W' and Z.  TODO fix selfs. .
        '''
        m_prime = integer_to_bits(0, 8) + integer_to_bits(int(len(ctx) / 8), 8) + ctx + m
        
        s_b = random.getrandbits(256).to_bytes(32, BYTEORDER)
        rnd = new_bitarray()
        rnd.frombytes(s_b)
        
        _, t1 = pkDecode(pk)
        sigma = ml_dsa_sign_internal(sk, m_prime, rnd)
        ch, z, h = sigDecode(sigma)

        self.w = self.flatten_point(np.array(fips_204.internal_funcs.global_w))
        self.w_a = self.flatten_point(np.array(fips_204.internal_funcs.global_w_a))
        '''c = sample_in_ball(ch)

        #Calc W', determined by C, Z, A, T1
        d_p2 = math.pow(2, D_DROPPED_BITS)
        w_a1 = matrix_vector_ntt(self.a, [NTT(x) for x in z])
        w_a2 = scalar_vector_ntt(NTT(c), [NTT(d_p2 * x) for x in t1])
        
        w_temp_prod = add_vector_ntt(w_a1, -w_a2)
        w_a = np.array([NTT_inv(x) for x in w_temp_prod])
        self.w_a = self.flatten_point(w_a)

        #calculate W, TODO move
        tr = h_shake256(pk.tobytes(), 64 * 8)
    
        message_bytes = (tr + m_prime).tobytes()
        mu = h_shake256(message_bytes, 64 * 8)
        private_seed_bytes = (self.k + rnd + mu).tobytes()
        rho_double_prime = h_shake256(private_seed_bytes, 64 * 8)
       
        kap = fips_204.internal_funcs.signed_kappa
        y = expand_mask(rho_double_prime, kap)
        w_temp_prod = matrix_vector_ntt(self.a, [NTT(x) for x in y])

        w = np.array([NTT_inv(x) for x in w_temp_prod])

        w1_prime = np.zeros((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
        for i in range(K_MATRIX):
            for j in range(VECTOR_ARRAY_SIZE):
                w1_prime[i][j] = use_hint(h[i][j], w_a[i][j])

        self.w = self.flatten_point(w)'''
        self.z = self.flatten_point(z)
        
    def generate_3d_points(self, projection_method: ProjectionMethods):
        
        match projection_method:
            case ProjectionMethods.VIEW_3D:
                #df_z = self.z[:,self.step_index:self.step_index+3]
                df_z =  np.vstack((self.z[:,self.step_index:self.step_index+3],
                                   self.w_a[:,self.step_index:self.step_index+3],
                                   self.w[:,self.step_index:self.step_index+3],))
                df_l = self.lattice_points[:,self.step_index:self.step_index+3]
                df_le = self.lattice_with_s2[:,self.step_index:self.step_index+3]
            case ProjectionMethods.MDS:
                len_data = 625 #TODO: figure out calc for this value!!
                mds = MDS(n_components=3)

                #z is temporarily appended to lattice_points for simplicity
                self.lattice_points += self.z

                #temporarily join error and non-error lattice together for MDS
                temp_inp = np.vstack((self.lattice_points,self.lattice_with_s2))
                temp_out = mds.fit_transform(temp_inp)

                #extract lattice and error lattice from MDS output
                lattice_3d = temp_out[:len_data+1] #z is appended at end
                df_le = temp_out[(len_data+1):]

                df_z = [lattice_3d[-1]]
                df_l = lattice_3d[:-1] #remove z from l3d

        df1 = pd.DataFrame(df_l, columns=['X', 'Y', 'Z'])
        df1['Type'] = 'Lattice Point'
        df2 = pd.DataFrame(df_le, columns=['X', 'Y', 'Z'])
        df2['Type'] = 'Error Lattice Point'
        df3 = pd.DataFrame(df_z, columns=['X', 'Y', 'Z'])
        df3['Type'] = 'Signature'

        #join dfs into one
        self.df = pd.concat([df1, df2, df3, self.origin_df], ignore_index=True)

    def flatten_point(self, point: np.ndarray) -> np.ndarray:
        '''
        Flatten point into Kx256-dimension space and modulo to q/2 space
        '''
        eg = point.reshape(1, K_MATRIX * VECTOR_ARRAY_SIZE)
        return eg
    
    def set_step_index(self, value):
        self.step_index = value

    def get_figure(self):
        # Create 3D scatter plot
        fig = px.scatter_3d(self.df, x='X', y='Y', z='Z', color='Type', opacity=0.2, title="3D Lattice Visualization")
        return fig
    
    def get_html(self):
        return html.Div([
            html.H2("Lattice Visualization"),
            html.P("This page visualizes the lattice and error lattice points in 3D space."),
            html.P("Use the slider to view the lattice points in 3D space."),
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
            html.Button('Regenerate', id='regen-button', n_clicks=0),
            dcc.Store(id='register-callbacks', data={})  # Hidden trigger for callback registration
        ])
    
    def register_callbacks(self, app):
        '''
        @app.callback(
            Output('lattice-plot', 'figure', allow_duplicate=True),
            Input('regen-button', 'n_clicks')
        )
        def update_plot(n_clicks):
            """ Callback function to regenerate the figure when the button is clicked. """
            pk, sk = ml_dsa_key_gen()
            lat = ALattice(sk)
            lat.generate_3d_points(temp)
            return lat.get_figure() # Generates a new plot dynamically
        '''
        @app.callback(
            Output('lattice-plot', 'figure', allow_duplicate=True),
            Input('ind_slider', 'value')
        )
        def update_plot(value):
            """ Callback function to regenerate the figure when the button is clicked. """
            self.set_step_index(value)
            return self.get_figure()
