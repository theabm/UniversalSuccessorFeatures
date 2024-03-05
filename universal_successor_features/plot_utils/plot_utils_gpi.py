import torch
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import os
import exputils.data.logging as log
import universal_successor_features as usf

from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

agent_name = "agent.pt"
env_name = "env"

# If we want to use bootstrap componets we HAVE to declare a theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# experiments_path = (
#     "/scratch/pictor/abermeom/projects/universalSuccessorFeatures/experiments/"
# )
experiments_path = "/home/andres/inria/projects/universalSuccessorFeatures/experiments/"

dropdown_menu_campaign = dcc.Dropdown(
    id="campaign",
    # label: "directory_name", value: "experiment_path/directory_name"
    options=[
        {"label": f.name, "value": f.path}
        for f in sorted(os.scandir(experiments_path), key=lambda e: e.name)
        if f.is_dir()
    ],
    # default campaign is the first it finds in the list (random order)
    value=experiments_path + os.listdir(experiments_path)[0],
)

dropdown_menu_experiment_number = dcc.Dropdown(
    id="experiment_num",
    options=[
        {"label": f.name, "value": f.path}
        for f in sorted(
            os.scandir(dropdown_menu_campaign.value + "/experiments"),
            key=lambda e: e.name,
        )
        if f.is_dir()
    ],
    # default value is first in list
    value=dropdown_menu_campaign.value
    + "/experiments/"
    + os.listdir(dropdown_menu_campaign.value + "/experiments")[0],
)

dropdown_menu_experiment_rep = dcc.Dropdown(
    id="experiment_rep",
    options=[
        {"label": f.name, "value": f.path}
        for f in sorted(
            os.scandir(dropdown_menu_experiment_number.value), key=lambda e: e.name
        )
        if f.is_dir()
    ],
    # default value is first in list
    value=dropdown_menu_experiment_number.value
    + "/"
    + os.listdir(dropdown_menu_experiment_number.value)[0],
)

dropdown_menu_color_scheme = dcc.Dropdown(
    id="color_scheme",
    options=px.colors.named_colorscales(),
    value="blues",
    clearable=False,
)

dropdown_menu_trajectory = dcc.Dropdown(
    id="trajectory_menu", placeholder="Select trajectory to explore ..."
)
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Visualizing Successor Features",
                    style={"textAlign": "center", "color": "white"},
                ),
                width={"size": 12, "offset": 0, "order": 1},
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Campaign",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_campaign,
                        html.Label(
                            "Experiment Number",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_experiment_number,
                    ],
                    width={"size": 5, "offset": 0, "order": 1},
                ),
                dbc.Col(
                    [
                        html.Label(
                            "Experiment Repetition",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_experiment_rep,
                        html.Label(
                            "Trajectory",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_trajectory,
                    ],
                    width={"size": 5, "offset": 0, "order": 1},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [dcc.Graph(id="fig", figure={})],
                    width={"size": 5, "offset": 0, "order": 1},
                )
            ]
        ),
    ],
    # fluid: If False (default), we have some space in the margins
    # if True, no space in the margins and we can utilize all 12 columns
    # note that if fluid = False, we still "think" with 12 columns, we just
    # will be using less
    fluid=True,
)


@app.callback(
    Output(dropdown_menu_experiment_number, "options"),
    Output(dropdown_menu_experiment_number, "value"),
    Input(dropdown_menu_campaign, "value"),
)
def update_experiment_number_dropdown(campaign_path):
    # Changing the campaign resets the possible experiments options available
    # and the experiment reps available (since they may not be the same across
    # campaigns)

    # os.list dir lists the directories at the path
    # /path_to_campaign_selected/experiments
    options_experiment_number = [
        {"label": f.name, "value": f.path}
        for f in sorted(
            os.scandir(campaign_path + "/experiments"), key=lambda e: e.name
        )
        if f.is_dir()
    ]

    return (
        options_experiment_number,
        options_experiment_number[0]["value"],
    )


@app.callback(
    Output(dropdown_menu_experiment_rep, "options"),
    Output(dropdown_menu_experiment_rep, "value"),
    Input(dropdown_menu_experiment_number, "value"),
)
def update_experiment_repetition_dropdown(experiment_number_path):
    options_experiment_rep = [
        {"label": f.name, "value": f.path}
        for f in sorted(os.scandir(experiment_number_path), key=lambda e: e.name)
        if f.is_dir()
    ]

    return (
        options_experiment_rep,
        options_experiment_rep[0]["value"],
    )


@app.callback(
    Output(dropdown_menu_trajectory, "options"),
    Output(dropdown_menu_trajectory, "value"),
    Input(dropdown_menu_experiment_rep, "value"),
)
def update_trajectory_menu(
    experiment_rep_path,
):
    new_path = experiment_rep_path + "/data/"
    log.set_directory(new_path)

    trajectory_info = log.load_single_object("trajectory_info")

    len_trajectories = len(trajectory_info)

    options_trajectory_menu = [i for i in range(len_trajectories)]

    return (
        options_trajectory_menu,
        options_trajectory_menu[0],
    )

@app.callback(
        Output("fig", "figure"),
        Input(dropdown_menu_trajectory, "value"),
        Input(dropdown_menu_experiment_rep, "value"),
        )
def display_trajectory(trajectory_num, experiment_rep_path):

    new_path = experiment_rep_path + "/data/"
    log.set_directory(new_path)
    trajectory_info = log.load_single_object("trajectory_info")

    env = usf.envs.GridWorld.load_from_checkpoint(
            new_path + "env.cfg"
        )
    
    idx = trajectory_num
    trajectory = trajectory_info[idx]
    print(trajectory)

    start_agent_position, goal_position = trajectory[0]
    agent_i, agent_j = get_i_j(start_agent_position)
    goal_i, goal_j = get_i_j(goal_position)

    positions = {
        "agent_i": agent_i,
        "agent_j": agent_j,
        "policy_goal_i": goal_i,
        "policy_goal_j": goal_j,
    }

    obs, *_ = env.reset(start_agent_position = start_agent_position, goal_position = goal_position)

    data = np.full((env.rows, env.columns), -100)

    for action, goal in trajectory[2:]:
        obs, *_ = env.step(action)
        data[get_i_j(obs["agent_position"])] = goal*10
        
    fig = make_heap_map_figure(data, positions, env, "Blues", "Trajectory")

    return fig

def get_i_j(arr):
    return arr[0][0], arr[0][1]

def make_heap_map_figure(data, positions, env, colorscale, title):
    fig = go.Figure(
        data=go.Heatmap(z=data, visible=True, colorscale=colorscale),
    )
    fig = add_rectangles_to_figure(
        fig,
        positions["agent_i"],
        positions["agent_j"],
        positions["policy_goal_i"],
        positions["policy_goal_j"],
    )
    fig = color_forbidden_cells(fig, fill_color="green", border_color="red", env=env)

    fig = modify_layout_of_figure(fig, title=title)
    return fig


def color_forbidden_cells(fig, fill_color, border_color, env):
    try:
        for i, j in env.forbidden_cells:
            fig.add_shape(
                type="rect",
                x0=j - 0.5,
                y0=i - 0.5,
                x1=j + 0.5,
                y1=i + 0.5,
                line=dict(color=border_color),
                fillcolor=fill_color,
            )
        return fig
    except:
        return fig


def add_rectangles_to_figure(fig, agent_i, agent_j, policy_goal_i, policy_goal_j):
    fig.add_shape(
        type="rect",
        x0=agent_j - 0.5,
        y0=agent_i - 0.5,
        x1=agent_j + 0.5,
        y1=agent_i + 0.5,
        line=dict(color="Yellow"),
        fillcolor="Yellow",
    )

    fig.add_shape(
        type="rect",
        x0=policy_goal_j - 0.5,
        y0=policy_goal_i - 0.5,
        x1=policy_goal_j + 0.5,
        y1=policy_goal_i + 0.5,
        line=dict(color="Purple"),
        fillcolor="Purple",
    )
    return fig


def modify_layout_of_figure(fig, title):
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title={"text": title, "x": 0.5})
    return fig


def list_of_goals_to_list_of_strings(list_of_goals):
    return [f"({goal[0][0]},{goal[0][1]})" for goal in list_of_goals]


if __name__ == "__main__":
    # app.run_server(debug=True, port=9000)
    app.run_server(debug=True, port=8000)
