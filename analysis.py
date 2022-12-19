import glob
import numpy as np
from multiprocessing import Pool
import os

import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from configuration import *
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation, KMeans


class Analysis:
    def __init__(self, model_data=None, load_from_path=None) -> None:
        if model_data is not None:
            self.data = model_data

        elif load_from_path is not None:
            print(f'Loading from file: {load_from_path}')
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
        # self.compute_cluster_count()
        # print(self.cluster_count)

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
        self.file_path = DATA_DIR + \
            f'run-(ag={self.n_agents}|ep={N_EPOCHS}|po={self.data["n_policies"]}|ss={self.data["social_sparsity"]}|ir={self.data["interaction_ratio"]}|or={self.data["orientations_std"]}|em(μ={self.data["emotions_mean"]}σ={self.data["emotions_std"]})|me(μ={self.data["media_conformities_mean"]}σ={self.data["media_conformities_std"]})|ba={self.data["connections_balance"]}|ca={str(self.data["candidates_opinions"]).replace(" ", "")}|se={self.data["seed"]}.dat'
        print(f'Saving snapshots data to: {self.file_path}')
        pickle.dump(self.data, open(self.file_path, 'wb'))

    def load_from_file(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        self.data = data
        self.file_path = file_path
        
    def compute_cluster_count(self):
        
        # Create an empty dictionary to save the within cluster sum of square values
        self.cluster_count = []
        for opinions_snapshot in self.opinions:
            
            # We initialise the within cluster sum of squares array
            wcss = np.zeros(10)
            # Look through the number of clusters
            for i in range(10):
                # Run kmeans model
                kmeans = KMeans(n_clusters=i+1, random_state=0).fit(opinions_snapshot)
                #Sum of squared distances of samples to their closest cluster center.
                wcss[i] = (kmeans.inertia_)
                
            # We are interested in the biggest decay of wcss, so we take the diff
            diff = np.diff(wcss)
            weighted_diff = 0
            # And now caculate the index by weighting the i+2 index by the diffs
            for i in range(len(diff)):
                weighted_diff += (i + 1) * diff[i]
            
            
            # clustering = AffinityPropagation(preference=0.01,random_state=SEED).fit(opinions_snapshot)
            self.cluster_count.append(np.argmin(diff) + 1)
            

    def get_graph_network_traces(self, step=0):
        graph = self.graphs[step]
        opinions = self.opinions[step]
        # If our opinion is more than 3d, then get a PCA
        if opinions.shape[1] >= 3:
            pca = PCA(n_components=3, svd_solver="full")
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
            node_text.append(
                f'connections: {len(adjacencies[1])}\nopinions: {self.opinions[step][node]}')
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
    
    def get_cluster_count_trace(self, step=0):
        
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

    def get_mean_opinions_trace(self, step=0):
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

    def plot_full_analysis(self, show=True, save=False):
        if show and save:
            print('Plotting and saving full analysis...')
        elif show:
            print('Plotting full analysis...')
        elif save:
            print('Saving full analysis...')

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'is_3d': True}, {'type': 'xy'}],
                   [{'type': 'bar'}, {'type': 'xy', 'secondary_y': True}]]
        )

        # For complex figures with custom rations and axes,
        # layout must be set in low-level
        layout = dict(
            title=f'{self.n_agents} agents, {self.data["n_policies"]} policies, sparsity={self.data["social_sparsity"]}, interact={self.data["interaction_ratio"]}, conn={self.data["init_connections"]}, orientations=(σ={self.data["orientations_std"]}), emotions (μ={self.data["emotions_mean"]},σ={self.data["emotions_std"]}), media (μ={self.data["media_conformities_mean"]},σ={self.data["media_conformities_std"]}) balance {self.data["connections_balance"]}',
            scene={
                'xaxis': {
                    'title': 'Policy 1',
                    'range': [-1, 1]
                },
                'yaxis': {
                    'title': 'Policy 2',
                    'range': [-1, 1]
                },
                'zaxis': {
                    'title': 'Policy 3',
                    'range': [-1, 1]
                },
                'domain_x': [0, 0.5],
                'domain_y': [0, 1]
            },
            scene_aspectmode='cube',
            xaxis1={
                'domain': [0.6, 0.95],
                'anchor': 'y1',
                'range': [0, self.epochs[-1]],
                'title': 'Epoch'
            },
            yaxis1={
                'domain': [0.4, 0.69],
                'anchor': 'x1',
                'title': 'Graph Density',
                'range': [0, self.max_density + 0.2]
            },
            xaxis2={
                'domain': [0.6, 0.95],
                'anchor': 'y2',
                'title': 'Node Degree',
                'range': [0, self.max_degree]
            },
            yaxis2={
                'domain': [0.0, 0.3],
                'anchor': 'x2',
                'title': 'Ratio [%]',
                'range': [0, self.max_degree_count/self.n_agents * 100 + 10]
            },
            xaxis3={
                'domain': [0.6, 0.95],
                'anchor': 'y3',
                'title': 'Epoch',
                'range': [0, self.epochs[-1]],
                'visible': False
            },
            yaxis3={
                'domain': [0.71, 1.0],
                'anchor': 'x3',
                'title': 'Vote probability [%]',
                'range': [self.min_vote_prob*100*0.9, self.max_vote_prob*100*1.1]
            },
            yaxis4={
                'anchor': 'x3',
                'overlaying': 'y3',
                'title': 'Mean opinion',
                'range': [self.min_mean_opinions*0.9, self.max_mean_opinions*1.1],
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
        fig.update_layout(layout)

        # Create initial plots, the same order here needs to be followed in the frames array
        
        plot_rows = [1, 1, 2, 1, 1, 2, 2]
        plot_cols = [1, 1, 1, 2, 2, 2, 2]
        secondary_ys = [False, False, False, False,
                        False, True]
        # As we have a variable number of candidates, we need to 
        # manually extend the lists
        candidates_plots = [2] * len(self.data['candidates_opinions'])
        plot_rows.extend(candidates_plots)
        plot_cols.extend(candidates_plots)
        candidates_ys = [False] * len(self.data['candidates_opinions'])
        secondary_ys.extend(candidates_ys)
        
        fig.add_traces((self.get_graph_network_traces() +
                        self.get_graph_histogram_trace() +
                        self.get_graph_density_traces() +
                        self.get_mean_opinions_trace() +
                        self.get_vote_polls_traces()),
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
                  self.get_mean_opinions_trace(step) +
                  self.get_vote_polls_traces(step)
                  ),
            # Using the plot rows from the initial figure guarantees consistency
            traces=list(range(len(plot_rows)))
        ) for step in range(self.n_snapshots)]
        fig.update(frames=frames)

        if save:
            html_file = DATA_DIR + '/html/' + \
                self.file_path.split('/')[-1].replace('.dat', '.html')
            print(f'Writing html to: {html_file}')
            fig.write_html(html_file, auto_play=False, )
        if show:
            print('Showing plot...')
            fig.show()

    # def plot_social_network(self):
    #     print('Plotting social network...')
    #     fig = go.Figure(
    #         data=self.get_graph_network_traces(),
    #         layout=go.Layout(
    #             title=f'{self.n_agents} agents, {self.data["n_policies"]} policies, sparsity={self.data["social_sparsity"]}, interact={self.data["interaction_ratio"]}, conn={self.data["init_connections"]}, orientations=(σ={self.data["orientations_std"]}), emotions (μ={self.data["emotions_mean"]},σ={self.data["emotions_std"]}), media (μ={self.data["media_conformities_mean"]},σ={self.data["media_conformities_std"]}) balance {self.data["connections_balance"]}',
    #             margin={
    #                 't': 50,
    #                 'b': 50,
    #                 'l': 50,
    #                 'r': 50
    #             },
    #             updatemenus=[{
    #                 'buttons': [
    #                     {
    #                         'args': [
    #                             [str(i) for i in range(self.n_snapshots)],
    #                             {
    #                                 'frame': {
    #                                     'duration': 500.0,
    #                                     'redraw': True
    #                                 },
    #                                 'fromcurrent': True,
    #                                 'transition': {
    #                                     'duration': 500,
    #                                     'easing': 'linear'
    #                                 }
    #                             }
    #                         ],
    #                         'label': 'Play',
    #                         'method': 'animate'
    #                     },
    #                     {
    #                         'args': [
    #                             [None],
    #                             {
    #                                 'frame': {
    #                                     'duration': 0,
    #                                     'redraw': True
    #                                 },
    #                                 'mode': 'immediate',
    #                                 'transition': {
    #                                     'duration': 0
    #                                 }
    #                             }
    #                         ],
    #                         'label': 'Pause',
    #                         'method': 'animate'
    #                     }
    #                 ],
    #                 'direction': 'left',
    #                 'pad': {
    #                     'r': 10,
    #                     't': 85
    #                 },
    #                 'showactive': True,
    #                 'type': 'buttons',
    #                 'x': 0.1,
    #                 'y': 0,
    #                 'xanchor': 'right',
    #                 'yanchor': 'top'
    #             }],
    #             sliders=[{
    #                 'yanchor': 'top',
    #                 'xanchor': 'left',
    #                 'currentvalue': {
    #                     'font': {
    #                         'size': 16
    #                     },
    #                     'prefix': 'Epoch: ',
    #                     'visible': True,
    #                     'xanchor': 'right'
    #                 },
    #                 'pad': {
    #                     'b': 10,
    #                     't': 50
    #                 },
    #                 'len': 0.9,
    #                 'x': 0.1,
    #                 'y': 0,
    #                 'steps': [
    #                     {
    #                         'args': [
    #                             [str(i)],
    #                             {
    #                                 'frame': {
    #                                     'duration': 500.0,
    #                                     'easing': 'linear',
    #                                     'redraw': True
    #                                 },
    #                                 'transition': {
    #                                     'duration': 0,
    #                                     'easing': 'linear'
    #                                 }
    #                             }
    #                         ],
    #                         'label': str(self.epochs[i]),
    #                         'method': 'animate'
    #                     }
    #                     for i in range(self.n_snapshots)
    #                 ]
    #             }]
    #         ),
    #         frames=[dict(
    #             name=str(step),
    #             # Tracing needs to be in the same order as the initial figures
    #             data=self.get_graph_network_traces(step)
    #         ) for step in range(self.n_snapshots)]
    #     )

    #     fig.show()


def plot_analysis(file_path: str):
    analysis = Analysis(load_from_path=file_path)
    analysis.plot_full_analysis(show=True, save=False)


def save_analysis(file_path: str):
    analysis = Analysis(load_from_path=file_path)
    analysis.plot_full_analysis(show=False, save=True)


def run_data_analyser(data_dir: str):
    user_input = ''
    file_filter = None
    while True:
        if file_filter is not None:
            file_list = glob.glob(f'{data_dir}/*{file_filter}*.dat')
        else:
            file_list = glob.glob(f'{data_dir}/*.dat')

        print('\n********************************************************************************')
        user_input = input(f'There are {len(file_list)} data files. Choose from the following:\n'
                           f'\tlist: List the files filtered by active filter\n'
                           f'\tset filename_filter: Sets the file name filter\n'
                           f'\tplot index_min-index_max: Plots the files in the interval[index_min:index_max]\n'
                           f'\tsave index_min-index_max: Saves the html export for the files in the interval[index_min:index_max]\n'
                           f'\tquit: Quits\n'
                           '\nOption: ')

        if user_input == 'quit':
            break

        if user_input == 'list':
            print(f'\nShowing file list for filter: {file_filter}\n')
            for i in range(len(file_list)):
                print(f'[{i}] : {file_list[i].split("/")[-1]}')
            continue

        if user_input.startswith('set'):
            file_filter = user_input.split(' ')[1]
            file_list = glob.glob(f'{data_dir}/*{file_filter}*.dat')
            for i in range(len(file_list)):
                print(f'[{i}] : {file_list[i].split("/")[-1]}')
            continue

        if user_input.startswith('plot'):
            plot_range = user_input.split(' ')[1]
            start = int(plot_range.split('-')[0])
            end = int(plot_range.split('-')[1])
            plot_list = file_list[start:end]
            processes = max(8, len(plot_list))

            with Pool(processes) as pool:
                pool.map(plot_analysis, plot_list)
            continue

        if user_input.startswith('save'):
            plot_range = user_input.split(' ')[1]
            start = int(plot_range.split('-')[0])
            end = int(plot_range.split('-')[1])
            save_list = file_list[start:end]
            processes = min(8, len(save_list))

            with Pool(processes) as pool:
                pool.map(save_analysis, save_list)
            continue


if __name__ == "__main__":
    # plot_analysis('/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data/run-(50ag|1000ep|7po|0.65ss|0.2ir|or=0.15|em(μ=0.35σ=0.15)|me(μ=0σ=0)|ba=-5|ca=[-0.85,-0.35,0.4,0.65]|se=69.dat')
    run_data_analyser(
        '/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data')
