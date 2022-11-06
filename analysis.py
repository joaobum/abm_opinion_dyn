import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from configuration import *
import networkx as nx


class Analysis:
    def __init__(self, simulation_info=None, load_from_path=None) -> None:
        if simulation_info is not None:
            self.simulation_info = simulation_info
        elif load_from_path is not None:
            self.load_from_file(load_from_path)

        self.snapshots = self.simulation_info['snapshots']
        self.n_snapshots = len(self.snapshots)

        # Store array of epochs
        self.epochs = np.array(
            [snapshot['epoch'] for snapshot in self.snapshots]
        )

        # Store group opinions and metrics
        self.opinions = np.array(
            [snapshot['group_opinions'] for snapshot in self.snapshots]
        )
        self.vote_polls = np.array(
            [snapshot['poll'] for snapshot in self.snapshots]
        )

        # Store graphs and metrics
        self.graphs = [
            nx.from_numpy_matrix(self.snapshots[i]['adjacency'])
            for i in range(self.n_snapshots)
        ]
        self.graphs_densities = np.array(
            [nx.density(graph) for graph in self.graphs]
        )
        self.max_density = np.max(self.graphs_densities)
        self.max_degree = 0

    def save_to_file(self):
        # Save snaphsots to file
        timestamp = datetime.now().strftime('%m-%d-%H.%M')
        filename = DATA_DIR + timestamp + \
            f'-run-({AGENTS_COUNT}|{N_EPOCHS}|{POLICIES_COUNT}|{N_SNAPSHOTS}|{INIT_EMOTIONS_MEAN:.2f}|{INIT_EMOTIONS_STD:.2f}|{NOISE_MEAN}|{NOISE_STD}).dat'
        print(f'Saving snapshots data to: {filename}')
        pickle.dump(self.simulation_info, open(filename, 'wb'))

    def load_from_file(self, file_path):
        simulation_info = pickle.load(open(file_path, 'rb'))
        self.simulation_info = simulation_info

    def get_graph_network_traces(self, step=0):
        graph = self.graphs[step]
        opinions = self.opinions[step]
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in graph.edges():
            x0, y0, z0 = opinions[edge[0]]
            x1, y1, z1 = opinions[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter3d(
            x=opinions[:, 0], y=opinions[:, 1], z=opinions[:, 2],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=False,
                color=[],
                size=10,
                # colorbar=dict(
                #     thickness=15,
                #     title='Node Connections',
                #     xanchor='left',
                #     titleside='right'
                # ),
                line_width=2
            )
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('# of connections: '+str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return [edge_trace, node_trace]

    def get_graph_histogram_trace(self, step=0):
        graph = self.graphs[step]
        histogram = nx.degree_histogram(graph)
        histogram_trace = go.Histogram(x=histogram)
        if np.max(histogram) > self.max_degree:
            self.max_degree = np.max(histogram)
        return [histogram_trace]

    def get_graph_density_traces(self, step=0):
        density_trace = go.Scatter(
            x=self.epochs,
            y=self.graphs_densities,
            mode='lines'
        )
        epoch_reference = go.Scatter(
            x=[self.epochs[step], self.epochs[step]],
            y=[0, 1],
            mode='lines',
            line={
                'color': 'grey',
                'dash': 'dash'
            }
        )

        return [density_trace, epoch_reference]

    def get_vote_polls_traces(self, step=0):
        candidates_traces = []
        for i in range(CANDIDATES_COUNT):
            candidates_traces.append(
                go.Scatter(
                    x=self.epochs,
                    y=self.vote_polls[:, i] * 100,
                    mode='lines',
                    marker=dict(color=[i]),
                    name=f'Candidate {i}'
                )
            )
        epoch_reference = go.Scatter(
            x=[self.epochs[step], self.epochs[step]],
            y=[0, 100],
            mode='lines',
            line={
                'color': 'grey',
                'dash': 'dash'
            }
        )
        candidates_traces.append(epoch_reference)
        return candidates_traces

    def get_mean_opinions_traces(self, step=0):
        mean_opinions_array = np.array(
            [np.mean(opinions) for opinions in self.opinions]
        )
        mean_opinions_trace = go.Scatter(
            x=self.epochs, y=mean_opinions_array, mode='lines')
        return [mean_opinions_trace]

    def plot_full_analysis(self):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )

        # Set up menus

        layout = dict(
            xaxis1={
                # 'domain': [0.0, 0.45],
                'anchor': 'y1',
                'title': 'Epoch',
                'range': [0, self.epochs[-1]]
            },
            yaxis1={
                'anchor': 'x1',
                'title': 'Graph Density',
                'range': [0, 1]
            },
            xaxis2={
                # 'domain': [0.55, 1.0],
                'anchor': 'y2',
                'title': 'Node Degree',
                'range': [0, 20]
            },
            yaxis2={
                # 'domain': [0.58, 0.98],
                'anchor': 'x2',
                'title': 'Count',
                'range': [0, 20]
            },
            xaxis3={
                # 'domain': [0.0, 0.42],
                'anchor': 'y3',
                'title': 'Epoch',
                'range': [0, self.epochs[-1]]
            },
            yaxis3={
                # 'domain': [0.0, 0.40],
                'anchor': 'x3',
                'title': 'Vote probability %',
                'range': [0, 100]
            },
            yaxis6={
                # 'domain': [0.0, 0.42],
                'anchor': 'x3',
                'overlaying': 'y3',
                'title': 'Mean opinion',
                'range': [0, 1],
                'side': 'right'
            },
            margin={
                't': 50,
                'b': 50,
                'l': 50,
                'r': 50
            },
            updatemenus=[{
                'buttons': [
                    {
                        'args': [
                            [str(i) for i in range(self.n_snapshots)],
                            {
                                'frame': {
                                    'duration': 500.0,
                                    'redraw': True
                                },
                                'fromcurrent': True,
                                'transition': {
                                    'duration': 500,
                                    'easing': 'linear'
                                }
                            }
                        ],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [
                            [None],
                            {
                                'frame': {
                                    'duration': 0,
                                    'redraw': True
                                },
                                'mode': 'immediate',
                                'transition': {
                                    'duration': 0
                                }
                            }
                        ],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {
                    'r': 10,
                    't': 85
                },
                'showactive': True,
                'type': 'buttons',
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top'
            }],
            sliders=[{
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {
                        'size': 16
                    },
                    'prefix': 'Epoch: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'pad': {
                    'b': 10,
                    't': 50
                },
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [
                            [str(i)],
                            {
                                'frame': {
                                    'duration': 500.0,
                                    'easing': 'linear',
                                    'redraw': False
                                },
                                'transition': {
                                    'duration': 0,
                                    'easing': 'linear'
                                }
                            }
                        ],
                        'label': str(self.epochs[i]),
                        'method': 'animate'
                    }
                    for i in range(self.n_snapshots)
                ]
            }]
        )

        # Create initial plots, the same order here needs to be followed in the frames array
        plot_rows = [1, 1, 2, 1, 1, 2, 2, 2, 2]
        plot_cols = [1, 1, 1, 2, 2, 2, 2, 2, 2]
        secondary_ys = [False, False, False, False,
                        False, False, False, False, False]
        fig.add_traces((self.get_graph_network_traces() +
                        self.get_graph_histogram_trace() +
                        self.get_graph_density_traces() +
                        self.get_vote_polls_traces() + 
                        self.get_mean_opinions_traces()),
                       rows=plot_rows,
                       cols=plot_cols,
                       secondary_ys=secondary_ys
                       )

        # Then add all animation frames
        frames = [dict(
            name=str(step),
            # Tracing needs to be in the same order as the initial figures
            data=(self.get_graph_network_traces(step) +
                  self.get_graph_histogram_trace(step) +
                  self.get_graph_density_traces(step) +
                  self.get_vote_polls_traces(step) +
                  self.get_mean_opinions_traces(step)
                  ),
            # Using the plot rows from the initial figure guarantees consistency
            traces=list(range(len(plot_rows)))
        ) for step in range(self.n_snapshots)]
        fig.update_layout(layout)
        fig.update(frames=frames)
        fig.show()

# def plot_full_analysis(simulation_info):
#     '''
#     Generate an animated plot over epochs with 4 subplots: a 2-PCA analysis, the population orientation as a
#     scatter and histogram, and a timeline of agents orientation majority.
#     '''
#     snapshots_df=pd.DataFrame(simulation_info)
#     epochs=snapshots_df.epoch.unique()

#     # Generate the dataframe for the graph edges
#     for index, row in snapshots_df.iterrows():
#         graph=nx.from_numpy_array(row.adjacency)
#         edges_x=[]
#         edges_y=[]
#         edges_z=[]
#         for edge in graph.edges():
#             x0, y0, z0=row.group_opinions[edge[0]]
#             x1, y1, z1=row.group_opinions[edge[1]]
#             edges_x.append(x0)
#             edges_x.append(x1)
#             edges_x.append(None)
#             edges_y.append(y0)
#             edges_y.append(y1)
#             edges_y.append(None)
#             edges_z.append(z0)
#             edges_z.append(z1)
#             edges_z.append(None)

#         snapshots_df.at[index, 'edges_x']=np.array(edges_x)
#         snapshots_df.at[index, 'edges_y']=np.array(edges_y)
#         snapshots_df.at[index, 'edges_z']=np.array(edges_z)

#     graph_df=snapshots_df[['epoch', 'edges_x', 'edges_y', 'edges_z']].explode(
#         ['edges_x', 'edges_y', 'edges_z'])

#     opinions_df=snapshots_df[['epoch', 'group_opinions']]

#     # pca_df = snapshots_df.explode('pca')
#     # pca_df[['pca1', 'pca2']] = pd.DataFrame(
#     #     pca_df.pca.tolist(), index=pca_df.index)

#     # histogram_df = pd.DataFrame(
#     #     columns=['epoch', 'histogram_x', 'histogram_y'])
#     # for index, row in snapshots_df.iterrows():
#     #     histogram = np.histogram(
#     #         row.orientations, 200, range=(-1, 1), density=True)
#     #     for i in range(len(histogram[0])):
#     #         histogram_df = pd.concat([
#     #             histogram_df,
#     #             pd.DataFrame(
#     #                 {'epoch': row.epoch, 'histogram_x': histogram[1][i], 'histogram_y': histogram[0][i]}, index=[0])
#     #         ], ignore_index=True)

#     # orientations_df = snapshots_df.explode('orientations').reset_index()
#     # orientations_df['agent_id'] = pd.DataFrame(
#     #     [i % self.n_agents for i in range(len(orientations_df))])
#     fig=make_subplots(rows = 1, cols = 1,
#                         specs = [[{'type': 'scene'}]])

#     fig.update_layout(dict(
#             width=1200,
#             height=900,
#             title='analysis',
#             # title=f'Analysis over {self.n_epochs} epochs: {self.n_agents} agents, {self.n_policies} policies (neg={self.policies_neg}, μ={self.policies_mean}|σ={self.policies_std}), ' +
#             # f'emotions (μ={self.emotions_mean},σ={self.emotions_std}), noise (μ={self.noise_intensity_mean}, σ={self.noise_intensity_std})',
#             xaxis1={
#                 'domain': [0.0, 0.45],
#                 'anchor': 'y1',
#                 'title': 'policy 1'
#             },
#             yaxis1={
#                 'domain': [0.58, 0.98],
#                 'anchor': 'x1',
#                 'title': 'policy 2'
#             },
#             xaxis2={
#                 'domain': [0.55, 1.0],
#                 'anchor': 'y2',
#                 'title': 'orientation interval',
#                 'range': [-1, 1]
#             },
#             yaxis2={
#                 'domain': [0.58, 0.98],
#                 'anchor': 'x2',
#                 'title': 'ratio',
#                 'range': [0, 5]
#             },
#             xaxis3={
#                 'domain': [0.0, 0.42],
#                 'anchor': 'y3',
#                 'title': 'agent id',
#                 'range': [0, AGENTS_COUNT]
#             },
#             yaxis3={
#                 'domain': [0.0, 0.40],
#                 'anchor': 'x3',
#                 'title': 'orientation',
#                 'range': [-1, 1]
#             },
#             xaxis4={
#                 'domain': [0.55, 1.0],
#                 'anchor': 'y4',
#                 'title': 'epoch',
#                 'range': [0, N_EPOCHS]
#             },
#             yaxis4={
#                 'domain': [0.0, 0.42],
#                 'anchor': 'x4',
#                 'title': '% negative oriented',
#                 'range': [0, 100]
#             },
#             margin={
#                 't': 50,
#                 'b': 50,
#                 'l': 50,
#                 'r': 50
#             },
#             updatemenus=[
#                 {
#                     'buttons': [
#                         {
#                             'args': [
#                                 [str(epoch) for epoch in epochs],
#                                 {
#                                     'frame': {
#                                         'duration': 500.0,
#                                         'redraw': False
#                                     },
#                                     'fromcurrent': True,
#                                     'transition': {
#                                         'duration': 500,
#                                         'easing': 'linear'
#                                     }
#                                 }
#                             ],
#                             'label': 'Play',
#                             'method': 'animate'
#                         },
#                         {
#                             'args': [
#                                 [None],
#                                 {
#                                     'frame': {
#                                         'duration': 0,
#                                         'redraw': False
#                                     },
#                                     'mode': 'immediate',
#                                     'transition': {
#                                         'duration': 0
#                                     }
#                                 }
#                             ],
#                             'label': 'Pause',
#                             'method': 'animate'
#                         }
#                     ],
#                     'direction': 'left',
#                     'pad': {
#                         'r': 10,
#                         't': 85
#                     },
#                     'showactive': True,
#                     'type': 'buttons',
#                     'x': 0.1,
#                     'y': 0,
#                     'xanchor': 'right',
#                     'yanchor': 'top'
#                 }
#             ],
#             sliders=[
#                 {
#                     'yanchor': 'top',
#                     'xanchor': 'left',
#                     'currentvalue': {
#                         'font': {
#                             'size': 16
#                         },
#                         'prefix': 'Epoch: ',
#                         'visible': True,
#                         'xanchor': 'right'
#                     },
#                     'transition': {
#                         'duration': 500.0,
#                         'easing': 'linear'
#                     },
#                     'pad': {
#                         'b': 10,
#                         't': 50
#                     },
#                     'len': 0.9,
#                     'x': 0.1,
#                     'y': 0,
#                     'steps': [
#                         {
#                             'args': [
#                                 [str(epoch)],
#                                 {
#                                     'frame': {
#                                         'duration': 500.0,
#                                         'easing': 'linear',
#                                         'redraw': False
#                                     },
#                                     'transition': {
#                                         'duration': 0,
#                                         'easing': 'linear'
#                                     }
#                                 }
#                             ],
#                             'label': str(epoch),
#                             'method': 'animate'
#                         }
#                         for epoch in epochs
#                     ]
#                 }],
#             # annotations=[
#             #     {
#             #         'font': {
#             #             'size': 16
#             #         },
#             #         'showarrow': False,
#             #         'text': 'Social network distributed on opinions',
#             #         'x': 0.225,
#             #         'xanchor': 'center',
#             #         'xref': 'paper',
#             #         'y': 0.98,
#             #         'yanchor': 'bottom',
#             #         'yref': 'paper'
#             #     },
#             #     {
#             #         'font': {
#             #             'size': 16
#             #         },
#             #         'showarrow': False,
#             #         'text': "Agent's orientation histogram",
#             #         'x': 0.775,
#             #         'xanchor': 'center',
#             #         'xref': 'paper',
#             #         'y': 0.98,
#             #         'yanchor': 'bottom',
#             #         'yref': 'paper'
#             #     },
#             #     {
#             #         'font': {
#             #             'size': 16
#             #         },
#             #         'showarrow': False,
#             #         'text': 'Orientation per agent',
#             #         'x': 0.225,
#             #         'xanchor': 'center',
#             #         'xref': 'paper',
#             #         'y': 0.45,
#             #         'yanchor': 'bottom',
#             #         'yref': 'paper'
#             #     },
#             #     {
#             #         'font': {
#             #             'size': 16
#             #         },
#             #         'showarrow': False,
#             #         'text': 'Ratio of negative orientation agents',
#             #         'x': 0.775,
#             #         'xanchor': 'center',
#             #         'xref': 'paper',
#             #         'y': 0.45,
#             #         'yanchor': 'bottom',
#             #         'yref': 'paper'
#             #     }
#             # ],
#         ))

#     # We need to build the figure by hand to have subplots sharing an animation


#         data=[
#             {
#                 'type': 'scatter3d',
#                 'name': 'Social network',
#                 'x': graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_x'],
#                 'y': graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_y'],
#                 'z': graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_z'],
#                 'line': {
#                     'color': 'indigo'
#                 },
#                 'mode': 'lines',
#                 'showlegend': False,
#                 'xaxis': 'x1',
#                 'yaxis': 'y1',
#                 'zaxis': 'z1'
#             },
#             {
#                 'type': 'bar',
#                 'name': 'Orientations histogram',
#                 'x': histogram_df.loc[histogram_df['epoch'] == epochs[0]]['histogram_x'],
#                 'y': histogram_df.loc[histogram_df['epoch'] == epochs[0]]['histogram_y'],
#                 'showlegend': False,
#                 'xaxis': 'x2',
#                 'yaxis': 'y2'
#             },
#             {
#                 'type': 'scatter',
#                 'name': 'Orientations',
#                 'x': orientations_df.loc[orientations_df['epoch'] == epochs[0]]['agent_id'],
#                 'y': orientations_df.loc[orientations_df['epoch'] == epochs[0]]['orientations'],
#                 'line': {
#                     'color': 'teal'
#                 },
#                 'mode':
#                     'markers',
#                     'showlegend': False,
#                     'xaxis': 'x3',
#                     'yaxis': 'y3'
#             },
#             {
#                 'type': 'scatter',
#                 'name': 'Negative ratio',
#                 'x': snapshots_df['epoch'],
#                 'y': snapshots_df['neg_ratio'],
#                 'line': {
#                     'color': 'black'
#                 },
#                 'mode': 'lines',
#                 'showlegend': False,
#                 'xaxis': 'x4',
#                 'yaxis': 'y4'
#             },
#             {
#                 'type': 'scatter',
#                 'name': 'Negative ratio',
#                 'x': snapshots_df['epoch'],
#                 'y': [50 for epoch in epochs],
#                 'line': {
#                     'color': 'darkgreen',
#                     'dash': 'dot'
#                 },
#                 'mode': 'lines',
#                 'showlegend': False,
#                 'xaxis': 'x4',
#                 'yaxis': 'y4'
#             },
#             {
#                 'type': 'scatter',
#                 'name': 'Negative ratio',
#                 'x': [0 for i in range(0, 100)],
#                 'y': [y for y in range(0, 100)],
#                 'line': {
#                     'color': 'red'
#                 },
#                 'mode': 'lines',
#                 'showlegend': False,
#                 'xaxis': 'x4',
#                 'yaxis': 'y4'
#             },
#         ],

#         frames=[
#             {
#                 'name': str(epoch), 'layout': {},
#                 'data': [
#                     {
#                         'type': 'scatter3d',
#                         'name': 'Social network',
#                         'x': graph_df.loc[graph_df['epoch'] == epoch]['edges_x'],
#                         'y': graph_df.loc[graph_df['epoch'] == epoch]['edges_y'],
#                         'z': graph_df.loc[graph_df['epoch'] == epoch]['edges_z'],
#                         'line': {
#                             'color': 'indigo'
#                         },
#                         'mode': 'lines',
#                         'showlegend': False,
#                         'xaxis': 'x1',
#                         'yaxis': 'y1'
#                     },
#                     # {
#                     #     'type': 'bar',
#                     #     'name': 'Orientations histogram',
#                     #     'x': histogram_df.loc[histogram_df['epoch'] == epoch]['histogram_x'],
#                     #     'y': histogram_df.loc[histogram_df['epoch'] == epoch]['histogram_y'],
#                     #     'showlegend': False,
#                     #     'xaxis': 'x2',
#                     #     'yaxis': 'y2'
#                     # },
#                     # {
#                     #     'type': 'scatter',
#                     #     'name': 'Orientations',
#                     #     'x': orientations_df.loc[orientations_df['epoch'] == epoch]['agent_id'],
#                     #     'y': orientations_df.loc[orientations_df['epoch'] == epoch]['orientations'],
#                     #     'line': {
#                     #         'color': 'teal'
#                     #     },
#                     #     'mode': 'markers',
#                     #     'showlegend': False,
#                     #     'xaxis': 'x3',
#                     #     'yaxis': 'y3'
#                     # },
#                     # {
#                     #     'type': 'scatter',
#                     #     'name': 'Negative ratio',
#                     #     'x': snapshots_df['epoch'],
#                     #     'y': snapshots_df['neg_ratio'],
#                     #     'line': {
#                     #         'color': 'black'
#                     #     },
#                     #     'mode': 'lines',
#                     #     'showlegend': False,
#                     #     'xaxis': 'x4',
#                     #     'yaxis': 'y4'
#                     # },
#                     # {
#                     #     'type': 'scatter',
#                     #     'name': 'Negative ratio',
#                     #     'x': snapshots_df['epoch'],
#                     #     'y': [50 for epoch in epochs],
#                     #     'line': {
#                     #         'color': 'darkgreen',
#                     #         'dash': 'dot'
#                     #     },
#                     #     'mode': 'lines',
#                     #     'showlegend': False,
#                     #     'xaxis': 'x4',
#                     #     'yaxis': 'y4'
#                     # },
#                     # {
#                     #     'type': 'scatter',
#                     #     'name': 'Negative ratio',
#                     #     'x': [epoch for i in range(0, 100)],
#                     #     'y': [y for y in range(0, 100)],
#                     #     'line': {
#                     #         'color': 'red'
#                     #     },
#                     #     'mode': 'lines',
#                     #     'showlegend': False,
#                     #     'xaxis': 'x4',
#                     #     'yaxis': 'y4'
#                     # },
#                 ],

#             }
#             for epoch in epochs
#         ]
#     # )
#     # fig = go.Figure(fig)
#     # fig.add_trace(
#     #     go.Scatter3d(
#     #         x=graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_x'],
#     #         y=graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_y'],
#     #         z=graph_df.loc[graph_df['epoch'] == epochs[0]]['edges_z'],
#     #         mode='lines',
#     #         marker=dict(color="red")
#     #     ), row=1, col=1)
#     # frames = [go.Frame(
#     #     dict(
#     #         name=str(epoch),
#     #         data=[
#     #             go.Scatter3d(
#     #                 x=graph_df.loc[graph_df['epoch'] == epoch]['edges_x'],
#     #                 y=graph_df.loc[graph_df['epoch'] == epoch]['edges_y'],
#     #                 z=graph_df.loc[graph_df['epoch'] == epoch]['edges_z'],
#     #                 mode='lines',
#     #                 marker=dict(color="red")
#     #             )
#     #         ],
#     #         traces=[0])
#     # ) for epoch in epochs]
#     # fig.add_trace(
#     #     go.Scatter3d(
#     #         x=snapshots_df.loc[snapshots_df['epoch'] == epochs[0]]['group_opinions'][0][:, 0],
#     #         y=snapshots_df.loc[snapshots_df['epoch'] == epochs[0]]['group_opinions'][0][:, 1],
#     #         z=snapshots_df.loc[snapshots_df['epoch'] == epochs[0]]['group_opinions'][0][:, 2],
#     #         mode='markers',
#     #         marker=dict(color="blue")
#     #     ), row=1, col=1)
#     # frames = [go.Frame(
#     #     dict(
#     #         name=str(epochs[i]),
#     #         data=[
#     #             go.Scatter3d(
#     #                 x=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 0],
#     #                 y=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 1],
#     #                 z=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 2],
#     #                 mode='markers',
#     #                 marker=dict(color="blue")
#     #             )
#     #         ],
#     #         traces=[0])
#     # ) for i in range(len(epochs))]
#     # fig.update(frames=frames)

#     # # print(fig.frames[0].data)
#     # # print(fig.frames[1].data)
#     # # print(fig.frames[2].data)

#     # # fig.show()

#     # fig = go.Figure(
#     #     data=[go.Scatter3d(x=[0, 1], y=[0, 1], z=[0,1])],
#     #     layout=go.Layout(
#     #         scene=dict(
#     #             xaxis=dict(range=[-1, 1], autorange=False),
#     #             yaxis=dict(range=[-1, 1], autorange=False),
#     #             zaxis=dict(range=[-1, 1], autorange=False),
#     #         ),
#     #         width=500,
#     #         height=500,
#     #         updatemenus=[dict(
#     #             type="buttons",
#     #             buttons=[dict(label="Play",
#     #                         method="animate",
#     #                         args=[None])])]

#     #     ),
#     #     frames=[go.Frame(data=[go.Scatter3d(x=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 0],
#     #                 y=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 1],
#     #                 z=snapshots_df.loc[snapshots_df['epoch'] == epochs[i]]['group_opinions'][i][:, 2])])
#     #             for i in range(len(epochs))
#     #     ]
#     # )

#     fig.show()
