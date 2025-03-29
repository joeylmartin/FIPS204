
from dash import Dash, html, dcc, callback, Output, Input, callback_context, no_update
import plotly.express as px
import pandas as pd
from dash import dcc
import plotly.graph_objs as go
import dash_mantine_components as dmc
import numpy as np

from visualizers.display_vars import WADisplay, XiDisplay,ADisplay, S1Display, S2Display, TDisplay, YDisplay, RoundingRing, CDisplay, ZDisplay
from visualizers.lattice import ALattice, WLattice, ProjectionMethods

from visualizers.variablepage import VariablesList
from fips_204.parametres import BYTEORDER, K_MATRIX, VECTOR_ARRAY_SIZE

import app_calc_vals as globals

from visualizers.vis_utils import center_mod_q

app = Dash(prevent_initial_callbacks=True)
app.config.suppress_callback_exceptions=True





def flatten_point(point: np.ndarray) -> np.ndarray:
    '''
    Flatten point into Kx256-dimension space and modulo to q/2 space
    '''
    eg = point.reshape(1, K_MATRIX * VECTOR_ARRAY_SIZE)
    return center_mod_q(eg)

def page1():
    var0 = XiDisplay()
    var1 = ADisplay(globals.a[:,:,:20])
    var2 = S1Display(globals.s1[:,:30])
    var3 = S2Display(globals.s2[:,:30])
    var4 = TDisplay(globals.t[:,:30])
    return VariablesList(app, [var0, var1, var2, var3, var4])

def page2():
    return ALattice(app)

def page3():
    var0 = YDisplay(globals.y)
    var1 = CDisplay(globals.c)
    var2 = ZDisplay(globals.z)
    return VariablesList(app, [var0, var1, var2])

def page4():
    return WLattice(app)

def page5():
    var0 = WADisplay(globals.w_a)
    var1 = RoundingRing()
    return VariablesList(app, [var0, var1])

def page6():

    extra_vals = {
            "W": flatten_point(np.array(globals.w)),
            "W' approx": flatten_point(np.array(globals.w_a)),
        }
    return ALattice( app, extra_vars=extra_vals)
# Define the steps in the FIPS process

pages = {
    "Page 1" : page1(),
    "Page 2" : page2(),
    "Page 3" : page3(),
    "Page 4" : page4(),
    "Page 5" : page5(),
    "Page 6" : page6()
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
    Output("register-trigger", "data", allow_duplicate=True),
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
    Output('dummy-header', 'children', allow_duplicate=True),
    Input('register-trigger', 'data')
)
def register_page_callbacks(data):
    print("Registering callbacks")
    """ Registers callbacks only after page content is updated """
    page.register_callbacks(app)

    return no_update#dummy, need update for sm reason

if __name__ == '__main__':
    app.run(debug=True)