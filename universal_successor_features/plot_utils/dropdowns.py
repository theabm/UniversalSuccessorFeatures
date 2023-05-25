import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import universal_successor_features.envs as envs
import universal_successor_features.agents as agents

from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc



dropdown_menu_policy_goal = dcc.Dropdown(
    # The id allows me to later specify it in the callback
    # if i dont have a variable
    id="goal_position",
    # The options of my drop down menu
    options=[
        # Options can be declared as a list, however, using this syntax with
        # dictionaries, we can specify even more options
        # The label is what the user sees
        # The value is the value used inside the program
        # Disabling makes that option not available for selection
        {"label": "(1,0)", "value": "(1,0)", "disabled": False},
        {"label": "(2,0)", "value": "(2,0)"},
        {"label": "(3,0)", "value": "(3,0)"},
    ],
    # The initial value it will have at start up time. In the case of dropdown
    # with single selection, it is one value, for multiselection, it can be
    # a list of values
    value="(1,0)",
    # Whether multi selection is supported
    multi=False,
    # Whether it can be erased and stay blank
    clearable=False,
    # The default text user sees when no value is selected
    # this is only valid if clearable is True
    # placeholder="",
    # Whether the user can search inside the menu
    searchable=True,
)

# dropdown_menu_agent_position = dcc.Dropdown(
#     id="agent_position",
#     options=[f"({i},{j})" for i in range(env.rows) for j in range(env.columns)],
#     value="(0,0)",
#     clearable=False,
#     searchable=True,
# )
#
# dropdown_menu_color_scheme = dcc.Dropdown(
#     id="color_scheme",
#     options=px.colors.named_colorscales(),
#     value="portland",
#     clearable=False,
# )
