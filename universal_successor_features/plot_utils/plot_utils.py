import torch
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import universal_successor_features.envs as envs
import universal_successor_features.agents as agents
import universal_successor_features.plot_utils.dropdowns as drop
import os
import exputils.data.logging as log

from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

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
        for f in os.scandir(experiments_path)
        if f.is_dir()
    ],
    # default campaign is the first it finds in the list (random order)
    value=experiments_path + os.listdir(experiments_path)[0],
)

dropdown_menu_experiment_number = dcc.Dropdown(
    id="experiment_num",
    options=[
        {"label": f.name, "value": f.path}
        for f in os.scandir(dropdown_menu_campaign.value + "/experiments")
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
        for f in os.scandir(dropdown_menu_experiment_number.value)
        if f.is_dir()
    ],
    # default value is first in list
    value=dropdown_menu_experiment_number.value
    + "/"
    + os.listdir(dropdown_menu_experiment_number.value)[0],
)

dropdown_menu_policy_goal = dcc.Dropdown(
    id="goal_position", placeholder="Select Experiment Number ..."
)


dropdown_menu_agent_position = dcc.Dropdown(
    id="agent_position", placeholder="Select Experiment Number ..."
)

dropdown_menu_color_scheme = dcc.Dropdown(
    id="color_scheme",
    options=px.colors.named_colorscales(),
    value="blues",
    clearable=False,
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
                        html.H2(
                            "Experiment Details",
                            style={"textAlign": "center", "color": "white"},
                        ),
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
                        html.Label(
                            "Experiment Repetition",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_experiment_rep,
                        dcc.Graph(id="fig0", figure={}),
                        dcc.Graph(id="fig2", figure={}),
                    ],
                    width={"size": 5, "offset": 0, "order": 1},
                ),
                dbc.Col(
                    [
                        html.H2(
                            "Agent Details",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        html.Label(
                            "Agent Position",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_agent_position,
                        html.Label(
                            "Goal Position",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_policy_goal,
                        html.Label(
                            "Color Scheme",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_color_scheme,
                        dcc.Graph(id="fig1", figure={}),
                        dcc.Graph(id="fig3", figure={}),
                    ],
                    width={"size": 5, "offset": 0, "order": 2},
                ),
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
def update_experiment_number_dropdown(
    campaign_path
):
    # Changing the campaign resets the possible experiments options available
    # and the experiment reps available (since they may not be the same across
    # campaigns)

    # os.list dir lists the directories at the path
    # /path_to_campaign_selected/experiments
    options_experiment_number = [
        {"label": f.name, "value": f.path}
        for f in os.scandir(campaign_path + "/experiments")
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
def update_experiment_repetition_dropdown(
    experiment_number_path
):

    options_experiment_rep = [
        {"label": f.name, "value": f.path}
        for f in os.scandir(experiment_number_path)
        if f.is_dir()
    ]

    return (
        options_experiment_rep,
        options_experiment_rep[0]["value"],
    )


# # Note order is top to bottom
@app.callback(
    Output(dropdown_menu_policy_goal, "options"),
    Output(dropdown_menu_policy_goal, "value"),
    Output(dropdown_menu_agent_position, "options"),
    Output(dropdown_menu_agent_position, "value"),
    Input(dropdown_menu_experiment_rep, "value"),
)
def update_agent_pos_and_goal_dropdown(experiment_rep_path):
    # something like /scratch/.../experiment0000001/repetition_000000
    new_path = experiment_rep_path + "/data/"

    log.set_directory(new_path)
    env = log.load_single_object("env")

    options_agent_position = [
        {"label": f"({i},{j})", "value": f"({i},{j})"}
        for i in range(env.rows)
        for j in range(env.columns)
        if (i, j) not in env.forbidden_cells
    ]

    options_agent_goal = [
        {
            "label": f"({goal[0][0]},{goal[0][1]})",
            "value": f"({goal[0][0]},{goal[0][1]})",
        }
        for goal in env.goal_list_source_tasks
        + env.goal_list_target_tasks
        + env.goal_list_evaluation_tasks
    ]

    return (
        options_agent_goal,
        options_agent_goal[0]["value"],
        options_agent_position,
        options_agent_position[0]["value"],
    )


@app.callback(
    Output("fig0", "figure"),
    Output("fig1", "figure"),
    Output("fig2", "figure"),
    Output("fig3", "figure"),
    Input(dropdown_menu_experiment_rep, "value"),
    Input(dropdown_menu_policy_goal, "value"),
    Input(dropdown_menu_agent_position, "value"),
    Input(dropdown_menu_color_scheme, "value"),
)
def display_successor_features(
    experiment_rep_path, policy_goal_position, agent_position, colors
):
    new_path = experiment_rep_path + "/data/"
    log.set_directory(new_path)
    env = log.load_single_object("env")
    agent = log.load_single_object("agent")

    agent_position = np.array([[int(agent_position[1]), int(agent_position[3])]])

    agent_i = agent_position[0][0]
    agent_j = agent_position[0][1]

    policy_goal_position = np.array(
        [[int(policy_goal_position[1]), int(policy_goal_position[3])]]
    )
    policy_goal_i = policy_goal_position[0][0]
    policy_goal_j = policy_goal_position[0][1]
    positions = {
        "agent_i": agent_i,
        "agent_j": agent_j,
        "policy_goal_i": policy_goal_i,
        "policy_goal_j": policy_goal_j,
    }

    obs = {
        "agent_position": agent_position,
        "agent_position_features": env._get_agent_position_features_at(agent_position),
        "goal_position": policy_goal_position,
        "goal_weights": env._get_goal_weights_at(policy_goal_position),
    }

    obs_dict = agent._build_arguments_from_obs(obs)

    with torch.no_grad():
        q, sf, *_ = agent.policy_net(
            policy_goal_position=torch.tensor(policy_goal_position)
            .to(torch.float)
            .to(agent.device),
            **obs_dict,
        )
        sf = (
            sf.squeeze()
            .reshape(agent.action_space, env.rows, env.columns)
            .cpu()
            .numpy()
        )
    #########################################################################################

    fig0 = make_heap_map_figure(
        data=sf[0], positions=positions, env=env, colorscale=colors, title="UP"
    )

    #########################################################################################

    fig1 = make_heap_map_figure(
        data=sf[1], positions=positions, env=env, colorscale=colors, title="DOWN"
    )

    #########################################################################################

    fig2 = make_heap_map_figure(
        data=sf[2], positions=positions, env=env, colorscale=colors, title="RIGHT"
    )

    #########################################################################################

    fig3 = make_heap_map_figure(
        data=sf[3], positions=positions, env=env, colorscale=colors, title="LEFT"
    )

    #########################################################################################

    return fig0, fig1, fig2, fig3


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


def add_rectangles_to_figure(fig, agent_i, agent_j, policy_goal_i, policy_goal_j):
    fig.add_shape(
        type="rect",
        x0=agent_j - 0.5,
        y0=agent_i - 0.5,
        x1=agent_j + 0.5,
        y1=agent_i + 0.5,
        line=dict(color="Yellow"),
    )

    fig.add_shape(
        type="rect",
        x0=policy_goal_j - 0.5,
        y0=policy_goal_i - 0.5,
        x1=policy_goal_j + 0.5,
        y1=policy_goal_i + 0.5,
        line=dict(color="Red"),
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
