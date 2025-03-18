import random
from dash import Dash, html, dcc, callback, Output, Input, callback_context, no_update
import plotly.express as px
import pandas as pd
from dash import dcc
import plotly.graph_objs as go
import dash_mantine_components as dmc
import numpy as np
from fips_204.auxiliary_funcs import new_bitarray
from fips_204.parametres import BYTEORDER
from visualizers.display_vars import XiDisplay,ADisplay, S1Display, S2Display, TDisplay, YDisplay, RoundingRing
from visualizers.lattice import ALattice, WLattice, ProjectionMethods

from visualizers.variablepage import VariablesList
from fips_204.external_funcs import ml_dsa_key_gen, ml_dsa_sign, ml_dsa_verify
from fips_204.internal_funcs import skDecode, expand_a, global_y, NTT, NTT_inv
# Initializ
import os

app = Dash(prevent_initial_callbacks=True)
app.config.suppress_callback_exceptions=True


pk, sk = ml_dsa_key_gen()

ctx_b = os.urandom(255)
ctx = new_bitarray()
ctx.frombytes(ctx_b)

seed = random.getrandbits(256) #change to approved RBG
s_b = seed.to_bytes(32, BYTEORDER)
mb = new_bitarray()
mb.frombytes(s_b)

sig = ml_dsa_sign(sk, mb, ctx)
ver = ml_dsa_verify(pk, mb, sig, ctx)
print(ver)


rho, k, tr, s1, s2, t0 = skDecode(sk)
a = expand_a(rho)






def page1():
    var0 = RoundingRing()
    var1 = ADisplay( a[:,:,:20])
    var2 = S1Display(s1[:,:30])
    var3 = S2Display( s2[:,:30])
    var4 = TDisplay(s2[:,:30])
    return VariablesList(app, [var0, var1, var2, var3, var4])

def page2():
    return ALattice(pk, sk,app)

#def page3():
#    return VariablesList(app, [YDisplay(global_y)])

def page4():
    return WLattice(pk, sk,app)
# Define the steps in the FIPS process

pages = {
    #"Page 0" : VariablesList(app, [XiDisplay]),
    "Page 1" : page1(),
    "Page 2" : page2(),
  #  "Page 3" : page3(),
    "Page 4" : page4(),
}
page_names = list(pages.keys()) #TODO: check ordering
current_step_index = 0  # Default starting index
page = pages[page_names[current_step_index]]

# Define the layout of the app
app.layout = dmc.MantineProvider(
    
    theme={
        # add your colors
        "colors": {
             # add your colors
            "deepBlue": ["#E9EDFC", "#C1CCF6", "#99ABF0" "..."], # 10 colors
            # or replace default theme color
            "blue": ["#E9EDFC", "#C1CCF6", "#99ABF0" "..."],   # 10 colors
        },
        "shadows": {
            # other shadows (xs, sm, lg) will be merged from default theme
            "md": "1px 1px 3px rgba(0,0,0,.25)",
            "xl": "5px 5px 3px rgba(0,0,0,.25)",
        },
        "headings": {
            "fontFamily": "Roboto, sans-serif",
            "sizes": {
                "h1": {"fontSize": '30px'},
            },
        },
    },
    children=[
        html.H1(children='FIPS 204 Visualizer Demo', style={'textAlign': 'center'}),

        # Navigation Controls
        html.Div([
            html.Button('Previous Page', id='prev-button', n_clicks=0, style={'margin': '10px'}),
            html.Button('Next Page', id='next-button', n_clicks=0, style={'margin': '10px'}),
        ],id="dummy-header", style={'textAlign': 'center'}),

        # Dynamic Page Content (updated by callbacks)
        html.Div(id='page-content', children=pages[page_names[current_step_index]].get_html()),


        dcc.Store(id='register-trigger', data={})
    ]
)

# Define Callback to Update Page Content
@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Output("register-trigger", "data"),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
)
def navigate_steps(prev_clicks, next_clicks):
    global current_step_index
    ctx = callback_context

    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'prev-button' and current_step_index > 0:
            current_step_index -= 1
        elif button_id == 'next-button' and current_step_index < len(page_names) - 1:
            current_step_index += 1
  
    # Get the updated page layout based on the current step
    page = pages[page_names[current_step_index]]
    page_layout = page.get_html()

    return page_layout, {"trigger" : True}


#after page has rendered, register callbacks.
#must be called after rendering, as the callbacks
#must refer to elements that have already been rendered
@app.callback(
    Output('dummy-header', 'children'),
    Input('register-trigger', 'data')
)
def register_page_callbacks(data):
    print("Registering callbacks")
    """ Registers callbacks only after page content is updated """
    page.register_callbacks(app)

    return no_update#dummy, need update for sm reason

if __name__ == '__main__':
    app.run(debug=True)