from abc import ABC, abstractmethod
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context, dash_table
import dash_mantine_components as dmc


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
        return dash_table.DataTable(data=self.formatted_data,
            columns=[{'name': str(i), 'id': str(i)} for i in range(self.cols)],
            id=self.table_id)

    def set_to_selected(self):
        self.selected_index = [0, 0]

    def get_interactive_representation(self):
        return html.Div(
            children=[self.get_table(),
                      dcc.Store(id=self.store_id)],
            id=self.div_id  
        )
    
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
                dmc.Text(self.sample_array[i]),
                style=selected_style if i == self.current_index else normal_style)
            for i in range(len(self.data))]

        #TODO: add div container for scroll and labels
        return dmc.SimpleGrid(
            id=self.table_id,
            cols=len(self.data),  
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