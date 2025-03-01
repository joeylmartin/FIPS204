from fips import *
import bitarray
from vis_utils import sample_lattice_point, center_mod_q
from parametres import *
from enum import Enum
from sklearn.manifold import MDS
import pandas as pd
import plotly.express as px
import os

from dash import Dash, html, dcc, callback, Output, Input

from abc import ABC, abstractmethod


class DemoPage(ABC):
    @abstractmethod
    def get_html(self):
        '''
        Returns a Dash HTML object that represents the subpage.
        '''
        pass
    @abstractmethod
    def register_callbacks(self, app):
        '''
        Allows us to pass the global Dash app object to the demo page,
        so that we can register the object's local callbacks to the global app.
        '''
        pass



class ProjectionMethods(Enum):
    VIEW_3D = 1
    MDS = 2

temp = ProjectionMethods.VIEW_3D


class ALattice(DemoPage):
    def __str__(self):
        return "Lattice Visualization"
    
    def __init__(self, sk, app):
        self.register_callbacks(app)
        self.sk = sk
        self.rho, k, tr, s1, s2, t0 = skDecode(sk)

        a = expand_a(self.rho)
        self.lattice_points, self.lattice_with_s2 = sample_lattice_point(a, s2)

        self.z = self.get_signature_from_message()

        self.origin_df : pd.DataFrame = pd.DataFrame(np.array([[0, 0, 0]]), columns=['X', 'Y', 'Z'])

        #This is used in the VIEW_3D projection method, sets the "dimensional slice" we use
        self.step_index = 0

    def generate_3d_points(self, projection_method: ProjectionMethods):
        
        match projection_method:
            case ProjectionMethods.VIEW_3D:
                df_z = self.z[:,self.step_index:self.step_index+3]
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
    
    def set_step_index(self, value):
        self.step_index = value

    def get_figure(self):
        # Create 3D scatter plot
        fig = px.scatter_3d(self.df, x='X', y='Y', z='Z', color='Type', opacity=0.2, title="3D Lattice Visualization")
        return fig

    def get_signature_from_message(self, message="Hello world!"):
        m_by = bytes(message, encoding='utf-8')
        m_b = new_bitarray()
        m_b.frombytes(m_by)
        return self.get_z(m_b)

    def get_z(self, m: bitarray):
        '''
        Given a message, creates a signature on the scheme, and converts it
        to flattened array for visualization
        '''
        ctx_b = os.urandom(255)
        ctx = new_bitarray()
        ctx.frombytes(ctx_b)

        sigma = ml_dsa_sign(self.sk, m, ctx) 
        c_hash, z, h = sigDecode(sigma)

        zm = z.reshape(1, K_MATRIX * VECTOR_ARRAY_SIZE)
        return center_mod_q(zm) #TODO, check if reshaping is necessary? thought sig was in q/2 space
    

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
