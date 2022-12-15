import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from configuration import *
import networkx as nx
from sklearn.decomposition import PCA

class Analysis:
    def __init__(self, model_data=None, load_from_path=None) -> None:
        if model_data is not None:
            self.data = model_data
        elif load_from_path is not None:
            self.load_from_file(load_from_path)

        self.snapshots = self.data['snapshots']
        self.n_snapshots = len(self.snapshots)

        # Store array of epochs
        self.epochs = np.array(
            [snapshot['epoch'] for snapshot in self.snapshots]
        )

        # Unpack group opinions and get metrics
        self.opinions = np.array(
            [snapshot['group_opinions'] for snapshot in self.snapshots]
        )
        self.mean_opinions = np.array(
            [np.mean(opinions) for opinions in self.opinions]
        )
        self.max_mean_opinions = np.max(self.mean_opinions)
        self.min_mean_opinions = np.min(self.mean_opinions)
        self.n_agents = len(self.opinions[0])
    
        # Unpack polls and get metrics 
        self.vote_polls = np.array(
            [snapshot['poll'] for snapshot in self.snapshots]
        )
        self.max_vote_prob = np.max(self.vote_polls)
        self.min_vote_prob = np.min(self.vote_polls)
        

        # Get graphs and metrics
        self.graphs = [
            nx.from_numpy_matrix(self.snapshots[i]['adjacency'])
            for i in range(self.n_snapshots)
        ]
        self.graphs_densities = np.array(
            [nx.density(graph) for graph in self.graphs]
        )
        self.max_density = np.max(self.graphs_densities)
        self.max_degree = 0
        self.max_degree_count = 0
        self.get_max_degree()

    def save_to_file(self):
        # Save snaphsots to file
        timestamp = datetime.now().strftime('%m-%d-%H.%M')
        filename = DATA_DIR + timestamp + \
            f'-run-({AGENTS_COUNT}ag|{N_EPOCHS}ep|{POLICIES_COUNT}po|or(σ={self.data["orientations_std"]})|em(μ={self.data["emotions_mean"]}σ={self.data["emotions_std"]})|me(μ={self.data["media_conformities_mean"]}σ={self.data["media_conformities_std"]})|ba={self.data["connections_created"] - self.data["connections_destroyed"]}.dat'
        print(f'Saving snapshots data to: {filename}')
        pickle.dump(self.data, open(filename, 'wb'))
        
        title=f'{AGENTS_COUNT} agents for {N_EPOCHS} epochs conn({self.data["init_connections"]}), orientations (σ={self.data["orientations_std"]}), emotions (μ={self.data["emotions_mean"]},σ={self.data["emotions_std"]}), media (μ={self.data["media_conformities_mean"]},σ={self.data["media_conformities_std"]}) balance {self.data["connections_created"] - self.data["connections_destroyed"]}',

    def load_from_file(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        self.data = data

    def get_graph_network_traces(self, step=0):
        graph = self.graphs[step]
        opinions = self.opinions[step]
        # If our opinion is more than 3d, then get a PCA
        if opinions.shape[1] >= 3:
            pca = PCA(n_components=3, random_state=0, svd_solver="auto")
            opinions = pca.fit_transform(opinions)
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
            line=dict(width=0.3, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
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
                size=4,
                # colorbar=dict(
                #     thickness=15,
                #     title='Node Connections',
                #     xanchor='left',
                #     titleside='right'
                # ),
                line_width=2
            ),
            showlegend=False
        )

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'connections: {len(adjacencies[1])}\nopinions: {self.opinions[step][node]}')
            # node_text.append(f'opinions: {self.opinions[step][node]}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return [edge_trace, node_trace]

    def get_max_degree(self):
        for graph in self.graphs:
            histogram = nx.degree_histogram(graph)
            if len(histogram) - 1 > self.max_degree:
                self.max_degree = len(histogram) - 1
            if np.max(histogram) > self.max_degree_count:
                self.max_degree_count = np.max(histogram) 
                
    def get_agents_mean_opinion_trace(self, step=0):
        mean_opinions = np.mean(self.opinions[step], axis=1)
        opinions_trace = go.Scatter(
            x=np.arange(0, len(mean_opinions)),
            y=mean_opinions, 
            mode='markers',
            # hoverinfo='text',
            # marker=dict(
            #     showscale=False,
            #     colorscale='YlGnBu',
            #     reversescale=False,
            #     color=[],
            #     size=4,
            #     # colorbar=dict(
            #     #     thickness=15,
            #     #     title='Node Connections',
            #     #     xanchor='left',
            #     #     titleside='right'
            #     # ),
            #     line_width=2
            # ),
            showlegend=False
        )
        
        return [opinions_trace]


    def get_graph_histogram_trace(self, step=0):
        graph = self.graphs[step]
        histogram = nx.degree_histogram(graph)
        converted_histogram = []
        for degree in range(len(histogram)):
            degree_count = histogram[degree]
            converted_histogram.extend([degree for i in range(degree_count)])
            
        histogram_trace = go.Histogram(
            x=converted_histogram,
            histnorm='percent',
            showlegend=False,
            xbins={
                'start': 0,
                'end': self.max_degree,
                'size': 1
            }
        )

        return [histogram_trace]

    def get_graph_density_traces(self, step=0):
        density_trace = go.Scatter(
            x=self.epochs,
            y=self.graphs_densities,
            mode='lines',
            line={
                'color': 'black'
            },
            showlegend=False
        )
        epoch_reference = go.Scatter(
            x=[self.epochs[step], self.epochs[step]],
            y=[0, 1],
            mode='lines',
            line={
                'color': 'grey',
                'dash': 'dash'
            },
            showlegend=False
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
            },
            showlegend=False
        )
        candidates_traces.append(epoch_reference)
        return candidates_traces

    def get_mean_opinions_traces(self, step=0):
        
        mean_opinions_trace = go.Scatter(
            x=self.epochs,
            y=self.mean_opinions,
            mode='lines',
            line={
                'color': 'black'
            },
            showlegend=False
        )
        return [mean_opinions_trace]
    
    def get_pca_snapshot(self):
        opinions_list = [self.agents[i].opinions for i in range(self.n_agents)]
        opinion_array = np.array(opinions_list)
        pca = PCA(n_components=2, random_state=0, svd_solver='full')
        components = pca.fit_transform(opinion_array)
        return components, pca.explained_variance_ratio_

    def plot_full_analysis(self):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy', 'secondary_y': True}, {'type': 'bar'}]]
        )

        # For complex figures with custom rations and axes,
        # layout must be set in low-level
        layout = dict(
            title=f'{AGENTS_COUNT} agents for {N_EPOCHS} epochs conn({self.data["init_connections"]}), orientations (σ={self.data["orientations_std"]}), emotions (μ={self.data["emotions_mean"]},σ={self.data["emotions_std"]}), media (μ={self.data["media_conformities_mean"]},σ={self.data["media_conformities_std"]}) balance {self.data["connections_created"] - self.data["connections_destroyed"]}',
            # scene={
            #     'xaxis': {
            #         'title': 'Policy 1',
            #         'range': [-1, 1]
            #     },
            #     'yaxis': {
            #         'title': 'Policy 2',
            #         'range': [-1, 1]
            #     },
            #     'zaxis': {
            #         'title': 'Policy 3',
            #         'range': [-1, 1]
            #     },
            #     'domain_x': [0, 0.5],
            #     'domain_y': [0, 1]
            # },
            # scene_aspectmode='cube',
            xaxis1={
                # 'domain': [0, 0.45],
                'anchor': 'y1',
                'range': [0, self.n_agents],
                'title': 'Agent'
            },
            yaxis1={
                # 'domain': [0.55, 1],
                'anchor': 'x1',
                'title': 'Mean opinion',
                'range': [-1, 1]
            },
            xaxis2={
                # 'domain': [0.55, 1],
                'anchor': 'y2',
                'range': [0, self.epochs[-1]],
                'title': 'Epoch'
            },
            yaxis2={
                # 'domain': [0.55, 1],
                'anchor': 'x2',
                'title': 'Graph Density',
                'range': [0, self.max_density + 0.2]
            },
            xaxis3={
                # 'domain': [0.0, 0.45],
                'anchor': 'y3',
                'title': 'Epoch',
                'range': [0, self.epochs[-1]],
                'visible': False
            },
            yaxis3={
                # 'domain': [0, 0.45],
                'anchor': 'x3',
                'title': 'Vote probability [%]',
                'range': [self.min_vote_prob*100*0.9, self.max_vote_prob*100*1.1]
            },
            yaxis5={
                'anchor': 'x3',
                'overlaying': 'y3',
                'title': 'Mean opinion',
                'range': [self.min_mean_opinions*0.9, self.max_mean_opinions*1.1],
                'side': 'right'
            },
            xaxis4={
                # 'domain': [0.55, 1],
                'anchor': 'y4',
                'title': 'Node Degree',
                'range': [0, self.max_degree]
            },
            yaxis4={
                # 'domain': [0.0, 0.45],
                'anchor': 'x4',
                'title': 'Ratio [%]',
                'range': [0, self.max_degree_count/self.n_agents * 100 + 10]
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
                                    'redraw': True
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
        plot_rows = [1, 2, 2, 2, 2, 1, 1, 2]
        plot_cols = [1, 1, 1, 1, 1, 2, 2, 2]
        secondary_ys = [False, False, False, False,
                        True, False, False, False, False]
        fig.add_traces((self.get_agents_mean_opinion_trace() +
                        self.get_vote_polls_traces() +
                        self.get_mean_opinions_traces() +
                        self.get_graph_density_traces()+
                        self.get_graph_histogram_trace()),
                       rows=plot_rows,
                       cols=plot_cols,
                       secondary_ys=secondary_ys
                       )

        # Then add all animation frames
        frames = [dict(
            name=str(step),
            # Tracing needs to be in the same order as the initial figures
            data=(self.get_agents_mean_opinion_trace(step) +
                  self.get_vote_polls_traces(step) +
                  self.get_mean_opinions_traces(step) +
                  self.get_graph_density_traces(step)+
                  self.get_graph_histogram_trace(step)),
            # Using the plot rows from the initial figure guarantees consistency
            traces=list(range(len(plot_rows)))
        ) for step in range(self.n_snapshots)]
        fig.update_layout(layout)
        fig.update(frames=frames)
        fig.show()
