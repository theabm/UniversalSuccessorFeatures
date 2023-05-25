import torch
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import universal_successor_features.envs as envs
import universal_successor_features.agents as agents
import universal_successor_features.plot_utils.dropdowns as drop

from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

# If we want to use bootstrap componets we HAVE to declare a theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

dropdown_menu_agent_type = dcc.Dropdown(
    id="agent_type",
    options=[
        {
            "label": "FGW USF",
            "value": "FeatureGoalWeightAgent",
        },
        {
            "label": "FG USF",
            "value": "FeatureGoalAgent",
        },
    ],
    placeholder="Please Select...",
)

dropdown_menu_policy_goal = dcc.Dropdown(
    id="goal_position", placeholder="Select Agent Type..."
)


dropdown_menu_agent_position = dcc.Dropdown(
    id="agent_position", placeholder="Select Agent Type..."
)

dropdown_menu_color_scheme = dcc.Dropdown(
    id="color_scheme",
    options=px.colors.named_colorscales(),
    value="blues",
    clearable=False,
)

dropdown_menu_graph_type = dcc.Dropdown(
    id="graph_type",
    options=[
        {"label": "Pixels", "value": "Heatmap"},
        {"label": "Contour Lines", "value": "Contour"},
        {"label": "Smooth", "value": "Heatmap"},
    ],
    value="Heatmap",
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
                        html.Label(
                            "Agent Type",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_agent_type,
                    ],
                    width=5,
                )
            ],
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Agent Position",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_agent_position,
                        html.Label(
                            "Color Scheme",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_color_scheme,
                        dcc.Graph(id="fig0", figure={}),
                        dcc.Graph(id="fig2", figure={}),
                    ],
                    # We can set width as width = 5
                    # However, here we show another way which gives many more
                    # options
                    # size -> how many columns wide
                    # offset -> how many empty columns before content
                    # order -> by default, order is write order, but here we can
                    # specify in case we want to switch the order.
                    # Note, we must always be sure that size+offset for all columns
                    # is <= 12
                    width={"size": 5, "offset": 0, "order": 1},
                ),
                dbc.Col(
                    [
                        html.Label(
                            "Goal Position",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_policy_goal,
                        html.Label(
                            "Graph Type",
                            style={"textAlign": "center", "color": "white"},
                        ),
                        dropdown_menu_graph_type,
                        dcc.Graph(id="fig1", figure={}),
                        dcc.Graph(id="fig3", figure={}),
                    ],
                    width={"size": 5, "offset": 0, "order": 2},
                ),
            ],
            justify="around"  # values can be
            #'start', 'center', 'end', 'around', 'between', 'evenly'
        ),
    ],
    # fluid: If False (default), we have some space in the margins
    # if True, no space in the margins and we can utilize all 12 columns
    # note that if fluid = False, we still "think" with 12 columns, we just
    # will be using less
    fluid=True,
)


# Note order is top to bottom
@app.callback(
    Output(dropdown_menu_policy_goal, "options"),
    Output(dropdown_menu_policy_goal, "value"),
    Output(dropdown_menu_agent_position, "options"),
    Output(dropdown_menu_agent_position, "value"),
    Input(dropdown_menu_agent_type, "value"),
)
def update_dropdown(agent_type):
    if agent_type is None:
        return [], None, [], None
    base_path = "/home/andres/inria/projects/universalSuccessorFeatures/agent_checkpoints_for_testing/"

    if agent_type == "FeatureGoalWeightAgent":
        env = envs.RoomGridWorld.load_from_checkpoint(
            base_path + "env_config.cfg",
        )
        agent = agents.FeatureGoalWeightAgent.load_from_checkpoint(
            env,
            base_path + agent_type + "_checkpoint.pt",
        )
    else:
        env = envs.RoomGridWorld.load_from_checkpoint(
            base_path + "env_config1.cfg",
        )
        agent = agents.FeatureGoalAgent.load_from_checkpoint(
            env,
            base_path + agent_type + "_checkpoint.pt",
        )

    options_agent_position = [
        f"({i},{j})" for i in range(env.rows) for j in range(env.columns)
    ]

    options_agent_goal = [
        f"({goal[0][0]},{goal[0][1]})" for goal in env.goal_list_source_tasks
    ]

    return (
        options_agent_goal,
        options_agent_goal[0],
        options_agent_position,
        options_agent_position[0],
    )


# @app.callback(Output("fig"))
# def display_successor_features_UP():
#     pass


@app.callback(
    Output("fig0", "figure"),
    Output("fig1", "figure"),
    Output("fig2", "figure"),
    Output("fig3", "figure"),
    Input(dropdown_menu_agent_type, "value"),
    Input(dropdown_menu_policy_goal, "value"),
    Input(dropdown_menu_agent_position, "value"),
    Input(dropdown_menu_color_scheme, "value"),
    Input(dropdown_menu_graph_type, "value"),
)
def display_successor_features(
    agent_type, policy_goal_position, agent_position, colors, graph_type
):
    if policy_goal_position is None or agent_position is None:
        return {}, {}, {}, {}

    base_path = "/home/andres/inria/projects/universalSuccessorFeatures/agent_checkpoints_for_testing/"
    if agent_type == "FeatureGoalWeightAgent":
        env = envs.RoomGridWorld.load_from_checkpoint(
            base_path + "env_config.cfg",
        )
        agent = agents.FeatureGoalWeightAgent.load_from_checkpoint(
            env,
            base_path + agent_type + "_checkpoint.pt",
        )
    else:
        env = envs.RoomGridWorld.load_from_checkpoint(
            base_path + "env_config1.cfg",
        )
        agent = agents.FeatureGoalAgent.load_from_checkpoint(
            env,
            base_path + agent_type + "_checkpoint.pt",
        )

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
        data=sf[0], positions=positions, colorscale=colors, title="UP"
    )

    #########################################################################################

    fig1 = make_heap_map_figure(
        data=sf[1], positions=positions, colorscale=colors, title="DOWN"
    )

    #########################################################################################

    fig2 = make_heap_map_figure(
        data=sf[2], positions=positions, colorscale=colors, title="RIGHT"
    )

    #########################################################################################

    fig3 = make_heap_map_figure(
        data=sf[3], positions=positions, colorscale=colors, title="LEFT"
    )

    #########################################################################################

    return fig0, fig1, fig2, fig3


def make_heap_map_figure(data, colorscale, positions, title):
    fig = go.Figure(
        data=go.Heatmap(z=data, visible=True, colorscale=colorscale),
    )
    fig = add_shapes_to_figure(
        fig,
        positions["agent_i"],
        positions["agent_j"],
        positions["policy_goal_i"],
        positions["policy_goal_j"],
    )
    fig = modify_layout_of_figure(fig, title=title)
    return fig


def add_shapes_to_figure(fig, agent_i, agent_j, policy_goal_i, policy_goal_j):
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
    app.run_server(debug=True, port=8054)
