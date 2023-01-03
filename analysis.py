###############################################################################
#   University of Sussex - Department of Informatics
#   MSc in Artificial Intelligence and Adaptive Systems
#
#   Project title: A co-evolution model for opinions in a social network
#   Candidate number: 229143
#
###############################################################################

# Standard libraries
from multiprocessing import Pool
import glob
import os
import time
import pickle
# External packages
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.decomposition import PCA
# Internal modules
from configuration import *


class ModelAnalysis:
    """
    Defines the model analysis class utilised in plotting and saving results 
    for single-run simulations.
    """

    def __init__(self,
                 model_data: dict = None,
                 load_from_path: str = None
                 ) -> None:
        """
        Initialises an instance of the ModelAnalysis class.
        Analysis data can either be loaded from the resulting dictionary from 
        the Model::run method, or be loaded from a file path.

        Keyword Arguments:
            model_data {dict} -- Argument to be used in case instantiating the class
            with the result dictionary from Model::run (default: {None})
            load_from_path {str} -- Argument to be used in case instantiating the class
            by loading simulation data from file (default: {None})
        """
        print(f'PID {os.getpid()} generating data analysis')
        if model_data is not None:
            self.data = model_data

        elif load_from_path is not None:
            print(f'Loading from file: {load_from_path}')
            self.load_from_file(load_from_path)

        self.snapshots = self.data['snapshots']
        self.n_snapshots = len(self.snapshots)

        # Store array of epoch values
        self.epochs = np.array(
            [snapshot['epoch'] for snapshot in self.snapshots]
        )

        # Unpack group opinions and get metrics
        self.opinions = np.array(
            [snapshot['group_opinions'] for snapshot in self.snapshots]
        )
        self.n_agents = len(self.opinions[0])
        self.mean_opinions = np.array(
            [np.mean(opinions) for opinions in self.opinions]
        )
        self.max_mean_opinions = np.max(self.mean_opinions)
        self.min_mean_opinions = np.min(self.mean_opinions)
        self.mean_opinions_margin = abs(
            (self.max_mean_opinions + self.min_mean_opinions) / 2) * 0.1
        # Initialise min and max to set plot boundaries (will be tweaked in case of PCA)
        self.o_max = [1, 1, 1]
        self.o_min = [-1, -1, -1]

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
            [nx.average_clustering(graph) for graph in self.graphs]
        )

        self.graphs_densities = np.array(
            [nx.average_clustering(graph) for graph in self.graphs]
        )
        self.compute_communities()
        self.max_density = np.max(self.graphs_densities)
        self.max_degree = 0
        self.max_degree_count = 0
        self.get_max_degree()

    def save_to_file(self) -> None:
        """
        Save the simulation data dictionary to a file with stanrdardised name 
        constructed from simulation parameters.
        """
        # Save snaphsots to file
        self.file_path = DATA_DIR + \
            f'run-(ag={self.n_agents}|ep={N_EPOCHS}|po={self.data["n_policies"]}|ss={self.data["social_sparsity"]}|ir={self.data["interaction_ratio"]}|or={self.data["orientations_std"]}|em(μ={self.data["emotions_mean"]}σ={self.data["emotions_std"]})|me(μ={self.data["media_conformities_mean"]}σ={self.data["media_conformities_std"]})|ba={self.data["connections_balance"]}|ca={str(self.data["candidates_opinions"]).replace(" ", "")}|se={self.data["seed"]}|co={self.communities_count[-1]}|ou={int(self.community_outliers[-1])}.dat'
        print(f'Saving snapshots data to: {self.file_path}')
        pickle.dump(self.data, open(self.file_path, 'wb'))

    def load_from_file(self, file_path: str) -> None:
        """
        Loads the simulation data from the file in file_path

        Arguments:
            file_path {str} -- Path to a .dat file containing simulation data.
        """
        data = pickle.load(open(file_path, 'rb'))
        self.data = data
        self.file_path = file_path

    def compute_communities(self):
        """
        Helper method to calculate the number of communities and outliers in
        the array of social graphs for each epoch. A community is a group of
        agents with more than 5% of the total population count in which there
        exists a path between all agents in that group.
        """
        self.min_community_size = int(self.n_agents * 0.05)
        self.communities_count = []
        self.community_outliers = []
        for graph in self.graphs:
            communities = [len(c)
                           for c in list(nx.connected_components(graph))]
            filtered_in = list(filter(
                lambda community_size: community_size >= self.min_community_size, communities))
            filtered_out = list(filter(
                lambda community_size: community_size < self.min_community_size, communities))
            self.communities_count.append(len(filtered_in))
            self.community_outliers.append(np.sum(filtered_out))

        self.max_communities = max(self.communities_count)

    def get_graph_network_traces(self, step: int = 0) -> list:
        """
        Retrieves a list containing a trace for edges and one for nodes
        in a go.Scatter3d plot respective to the step value.

        Keyword Arguments:
            step {int} -- Time epoch value for which the trace is
            to be generated (default: {0})

        Returns:
            list -- A list containing the edge and node traces of go.Scatter3d 
            type
        """
        graph = self.graphs[step]
        opinions = self.opinions[step]
        # If our opinion is more than 3d, then get a PCA
        if opinions.shape[1] > 3:
            pca = PCA(n_components=3, svd_solver="full")
            opinions = pca.fit_transform(opinions)
            self.o_max = [max(self.o_max[i], np.max(opinions[:, i]))
                          for i in range(3)]
            self.o_min = [min(self.o_min[i], np.min(opinions[:, i]))
                          for i in range(3)]

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

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return [edge_trace, node_trace]

    def get_max_degree(self):
        """
        Helper function to update the max degree value used for setting plot
        boundaries.
        """
        for graph in self.graphs:
            histogram = nx.degree_histogram(graph)
            if len(histogram) - 1 > self.max_degree:
                self.max_degree = len(histogram) - 1
            if np.max(histogram) > self.max_degree_count:
                self.max_degree_count = np.max(histogram)

    def get_agents_mean_opinion_trace(self, step: int = 0) -> list:
        """
        Retrieves a list containing the mean opinions trace for
        the given step.

        Keyword Arguments:
            step {int} -- Epoch time step for which the plot should 
            be generated (default: {0})

        Returns:
            list -- Containing a single element representing a go.Scatter plot
        """
        mean_opinions = np.mean(self.opinions[step], axis=1)
        opinions_trace = go.Scatter(
            x=np.arange(0, len(mean_opinions)),
            y=mean_opinions,
            mode='markers',
            showlegend=False
        )

        return [opinions_trace]

    def get_graph_histogram_trace(self, step: int = 0):
        """
        Retrieves the node connections degree histogram for the agent's 
        population for the given time step.

        Keyword Arguments:
            step {int} -- Epoch time step for which the plot should 
            be generated (default: {0})

        Returns:
            _type_ -- List containing a single item of go.Histrogram plot type
        """
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


    def get_community_traces(self, step: int = 0) -> list:
        """
        Generates the plots for communities and outliers, as well as 
        a dashed vertical line trace to serve as epoch reference

        Keyword Arguments:
            step {int} -- Epoch time step for which the plot should 
            be generated (default: {0})

        Returns:
            list -- List containing the communities, outliers and epoch 
            reference go.Scatter traces.
        """
        communities_trace = go.Scatter(
            x=self.epochs,
            y=self.communities_count,
            mode='lines',
            line={
                'color': 'darkgreen'
            },
            showlegend=False
        )
        outliers_trace = go.Scatter(
            x=self.epochs,
            y=self.community_outliers,
            mode='lines',
            line={
                'color': 'red'
            },
            showlegend=False
        )
        epoch_reference = go.Scatter(
            x=[self.epochs[step], self.epochs[step]],
            y=[0, self.max_communities],
            mode='lines',
            line={
                'color': 'grey',
                'dash': 'dash'
            },
            showlegend=False
        )

        return [communities_trace, epoch_reference, outliers_trace]

    def get_vote_polls_traces(self, step: int = 0) -> list:
        """
        Generates the traces for all candidates vote intentions, as well as
        an epoch reference

        Keyword Arguments:
            step {int} -- Epoch time step for which the plot should 
            be generated (default: {0})

        Returns:
            list -- List containing an epoch reference go.Scatter trace and one vote
            intention go.Scatter trace for each political candidate.
        """
        candidates_traces = []
        for i in range(CANDIDATES_COUNT):
            candidates_traces.append(
                go.Scatter(
                    x=self.epochs,
                    y=self.vote_polls[:, i] * 100,
                    mode='lines',
                    marker=dict(color=[i]),
                    name=f'Cd σ={self.data["candidates_opinions"][i]}'
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

    def get_mean_opinions_trace(self) -> list:
        """
        Generates the population's mean opinion trace

        Returns:
            list -- List containing a single go.Scatter trace item for
            the population mean opinion.
        """
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

    def get_pca_snapshot(self) -> tuple:
        """
        Helper function to generate a PCA decomposition in case the opinion
        dimensionality is higher than 3 (so it can be plotted in Scatter3d)

        Returns:
            tuple -- A tuple containing a list of primary components, and
            the explained variance.
        """

        opinions_list = [self.agents[i].opinions for i in range(self.n_agents)]
        opinion_array = np.array(opinions_list)
        pca = PCA(n_components=2, random_state=0, svd_solver='full')
        components = pca.fit_transform(opinion_array)
        return components, pca.explained_variance_ratio_

    def plot_model_analysis(self, show: bool = True, save: bool = False) -> None:
        """
        Generate the single-run full analysis multi-plot.
        The plot is generated as animated frames for each of the epochs 
        in which snapshots were taken.

        Keyword Arguments:
            show {bool} -- Displays the interactive generated plot in a browser 
            window (default: {True})
            save {bool} -- Saves the interactive generated plot to an html file
            (default: {False})
        """
        if show and save:
            print('Plotting and saving full analysis...')
        elif show:
            print('Plotting full analysis...')
        elif save:
            print('Saving full analysis...')

        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{'is_3d': True},    {'type': 'xy', 'secondary_y': True}],
                   [None,               {'type': 'xy', 'secondary_y': True}],
                   [None,               {'type': 'xy', 'secondary_y': True}]]
        )

        # Create initial plots, the same order here needs to be followed in the frames array
        plot_rows = [
            # Social network
            1, 1,
            # Histogram
            3,
            # Graph
            2, 2, 2,
            # Voting
            1, 1]
        plot_cols = [
            # Social network
            1, 1,
            # Histogram
            2,
            # Graph
            2, 2, 2,
            # Voting
            2, 2]
        secondary_ys = [
            # Social network
            False, False,
            # Histogram
            False,
            # Graph
            False, False, True,
            # Voting
            True, False]
        # As we have a variable number of candidates, we need to
        # manually extend the lists
        candidates_rows = [1] * len(self.data['candidates_opinions'])
        plot_rows.extend(candidates_rows)
        candidates_cols = [2] * len(self.data['candidates_opinions'])
        plot_cols.extend(candidates_cols)
        candidates_ys = [False] * len(self.data['candidates_opinions'])
        secondary_ys.extend(candidates_ys)

        fig.add_traces((self.get_graph_network_traces() +
                        self.get_graph_histogram_trace() +
                        self.get_community_traces() +
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
                  self.get_community_traces(step) +
                  self.get_mean_opinions_trace(step) +
                  self.get_vote_polls_traces(step)
                  ),
            # Using the plot rows from the initial figure guarantees consistency
            traces=list(range(len(plot_rows)))
        ) for step in range(self.n_snapshots)]
        fig.update(frames=frames)

        # For complex figures with custom rations and axes,
        # layout must be set in low-level
        layout = dict(
            title=f'{self.n_agents} agents, {self.data["n_policies"]} policies, sparsity={self.data["social_sparsity"]}, interact={self.data["interaction_ratio"]}, conn={self.data["init_connections"]}, orientations=(σ={self.data["orientations_std"]}), emotions (μ={self.data["emotions_mean"]},σ={self.data["emotions_std"]}), media (μ={self.data["media_conformities_mean"]},σ={self.data["media_conformities_std"]}) balance {self.data["connections_balance"]}',
            scene={
                'xaxis': {
                    'title': 'Policy 1',
                    'range': [self.o_min[0], self.o_max[0]]
                },
                'yaxis': {
                    'title': 'Policy 2',
                    'range': [self.o_min[1], self.o_max[1]]
                },
                'zaxis': {
                    'title': 'Policy 3',
                    'range': [self.o_min[2], self.o_max[2]]
                },
                'domain_x': [0, 0.5],
                'domain_y': [0, 1]
            },
            scene_aspectmode='cube',
            xaxis1={
                # 'domain': [0.6, 0.95],
                'anchor': 'y1',
                'title': 'Epoch',
                'range': [0, self.epochs[-1]],
                'visible': False
            },
            yaxis1={
                'domain': [0.705, 1.0],
                'anchor': 'x1',
                'title': 'Vote probability [%]',
                'range': [self.min_vote_prob*100*0.9, self.max_vote_prob*100*1.1]
            },
            yaxis2={
                'anchor': 'x1',
                'title': 'Mean opinion',
                'range': [self.min_mean_opinions - self.mean_opinions_margin, self.max_mean_opinions + self.mean_opinions_margin]
            },
            xaxis2={
                # 'domain': [0.6, 0.95],
                'anchor': 'y3',
                'range': [0, self.epochs[-1]],
                'title': 'Epoch'
            },
            yaxis3={
                'domain': [0.4, 0.695],
                'anchor': 'x3',
                'title': 'Communities',
                # 'range': [0, self.max_density + 0.2]
            },
            yaxis4={
                'anchor': 'x3',
                'overlaying': 'y3',
                'title': 'Outliers',
                # 'range': [self.min_mean_opinions - self.mean_opinions_margin, self.max_mean_opinions + self.mean_opinions_margin],
                'side': 'right'
            },
            xaxis3={
                # 'domain': [0.6, 0.95],
                'anchor': 'y5',
                'title': 'Node Degree',
                'range': [0, self.max_degree]
            },
            yaxis5={
                # 'domain': [0.0, 0.3],
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
        fig.update_layout(layout)

        if save:
            html_file = DATA_DIR + '/html/' + \
                self.file_path.split('/')[-1].replace('.dat', '.html')
            print(f'Writing html to: {html_file}')
            fig.write_html(html_file, auto_play=False, )
        if show:
            print('Showing plot...')
            fig.show()


class StatisticalAnalysis:
    def __init__(self, 
                 models_results: list, 
                 parameter_name: str, 
                 parameter_values: list
                 ) -> None:
        """
        Instantiate a StatisticalAnalysis object to calculate adn plot statistics
        related to multiple model simulations whilst sweeping through a single
        parameter.

        Arguments:
            models_results {list} -- List of model results dictionaries. The list
            will be broken in result segments based on the different values of
            the parameter parameter_name.
            parameter_name {str} -- Name of the parameter that the statistical 
            analysis will sweep through
            parameter_values {list} -- Values that the parameter under analysis has
            assumed in the model simulations
        """
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.min_community_size = len(models_results[0].agents) * 0.05
        self.n_communities_mean = []
        self.n_communities_max = []
        self.n_communities_min = []
        self.n_outliers_mean = []
        self.n_outliers_max = []
        self.n_outliers_min = []
        self.mean_opinions_mean = []
        self.mean_opinions_max = []
        self.mean_opinions_min = []
        self.var_opinions_mean = []
        self.var_opinions_max = []
        self.var_opinions_min = []

        # Break down the model results in segments of same parameter value
        results_segments = []
        for parameter_value in parameter_values:
            result_segment = []
            for model_result in models_results:
                if model_result.data[parameter_name] == parameter_value:
                    result_segment.append(model_result)
            results_segments.append(result_segment)

        for result_segment in results_segments:
            self.compute_communities(result_segment)
            self.compute_mean_opinions(result_segment)

    def compute_communities(self, result_segment: list) -> None:
        """
        Calculates the mean, min and max values for the number of communities
        and outliers in the received result segment.

        Arguments:
            result_segment {list} -- Sub-list of dictionaries of model run iterations
            for the same parameter values.
        """
        n_communities = []
        n_outliers = []

        for result in result_segment:
            graph = nx.from_numpy_matrix(result.adjacency_matrix)
            communities = [len(c)
                           for c in list(nx.connected_components(graph))]
            filtered_in = list(filter(
                lambda community_size: community_size >= self.min_community_size, communities))
            filtered_out = list(
                filter(lambda community_size: community_size < self.min_community_size, communities))

            n_communities.append(len(filtered_in))
            n_outliers.append(np.sum(filtered_out))

        self.n_communities_mean.append(np.mean(n_communities))
        self.n_communities_max.append(np.max(n_communities))
        self.n_communities_min.append(np.min(n_communities))
        self.n_outliers_mean.append(np.mean(n_outliers))
        self.n_outliers_max.append(np.max(n_outliers))
        self.n_outliers_min.append(np.min(n_outliers))

    def compute_mean_opinions(self, result_segment: list) -> None:
        """
        Calculates the mean, min and max values for the population's mean 
        opinions and variance in the received result segment.

        Arguments:
            result_segment {list} -- Sub-list of dictionaries of model run iterations
            for the same parameter values.
        """
        mean_opinions = []
        mean_opinions_var = []

        for result in result_segment:
            opinions_array = np.array(
                [snapshot['group_opinions']
                    for snapshot in result.data['snapshots']]
            )
            mean_opinions_array = np.array(
                [np.mean(opinions) for opinions in opinions_array]
            )

            mean_opinions.append(mean_opinions_array[-1])
            mean_opinions_var.append(np.var(mean_opinions_array))

        self.mean_opinions_mean.append(np.mean(mean_opinions))
        self.mean_opinions_max.append(np.max(mean_opinions))
        self.mean_opinions_min.append(np.min(mean_opinions))
        self.var_opinions_mean.append(np.mean(mean_opinions_var))
        self.var_opinions_max.append(np.max(mean_opinions_var))
        self.var_opinions_min.append(np.min(mean_opinions_var))

    def plot_statistical_analysis(self, 
                                  show: bool = True, 
                                  save: bool = False, 
                                  prefix: str = 'normal'
                                  ) -> None:
        """
        Generates the multi-plot static analysis for the statistics
        of multiple parameter-sweep simulations. It can both display the results
        in a browser window, and/or save the static image to a .png file

        Keyword Arguments:
            show {bool} -- Whether to show the result in a browser window 
            (default: {True})
            save {bool} -- Whether to save the static image to a .png file 
            (default: {False})
            prefix {str} -- Prefix that should be appended to the file name
            (default: {'normal'})
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[
                [
                    {'type': 'xy', 'secondary_y': True},
                    {'type': 'xy', 'secondary_y': True}
                ]
            ],
            subplot_titles=[
                'Community analysis',
                'Mean opinion analysis'
            ]
        )
        x_padding = (self.parameter_values[-1] - self.parameter_values[0])/20
        layout = dict(
            title=f'Statistical analysis for varying values of {self.parameter_name}',
            height=500,
            width=1100,
            xaxis1={
                'anchor': 'y1',
                'title': self.parameter_name,
                'range': [self.parameter_values[0] - x_padding, self.parameter_values[-1] + x_padding],
                'tickmode': 'array',
                'tickvals': self.parameter_values[::2] if len(self.parameter_values) > 6 else self.parameter_values
            },
            yaxis1={
                'anchor': 'x1',
                'title': 'final number of communities'
            },
            yaxis2={
                'anchor': 'x1',
                'title': 'final number of outliers',
                'side': 'right'
            },
            xaxis2={
                'anchor': 'y3',
                'title': self.parameter_name,
                'range': [self.parameter_values[0] - x_padding, self.parameter_values[-1] + x_padding],
                'tickmode': 'array',
                'tickvals': self.parameter_values[::2] if len(self.parameter_values) > 6 else self.parameter_values
            },
            yaxis3={
                'anchor': 'x2',
                'title': 'final mean opinions',
            },
            yaxis4={
                'anchor': 'x2',
                'title': 'mean opinions variance',
                'side': 'right'
            },
        )
        fig.update_layout(layout)

        fig.add_trace(
            go.Scatter(
                name='n_outliers',
                x=self.parameter_values,
                y=self.n_outliers_mean,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=np.subtract(self.n_outliers_max,
                                      self.n_outliers_mean),
                    arrayminus=np.subtract(
                        self.n_outliers_mean, self.n_outliers_min),
                    visible=True
                )
            ),
            row=1,
            col=1,
            secondary_y=True)

        fig.add_trace(
            go.Scatter(
                name='n_communities',
                x=self.parameter_values,
                y=self.n_communities_mean,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=np.subtract(self.n_communities_max,
                                      self.n_communities_mean),
                    arrayminus=np.subtract(
                        self.n_communities_mean, self.n_communities_min),
                    visible=True
                )
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                name='mean opinions var',
                x=self.parameter_values,
                y=self.var_opinions_mean,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=np.subtract(self.var_opinions_max,
                                      self.var_opinions_mean),
                    arrayminus=np.subtract(
                        self.var_opinions_mean, self.var_opinions_min),
                    visible=True
                )
            ),
            row=1,
            col=2,
            secondary_y=True)

        fig.add_trace(
            go.Scatter(
                name='mean opinions',
                x=self.parameter_values,
                y=self.mean_opinions_mean,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=np.subtract(self.mean_opinions_max,
                                      self.mean_opinions_mean),
                    arrayminus=np.subtract(
                        self.mean_opinions_mean, self.mean_opinions_min),
                    visible=True
                )
            ),
            row=1,
            col=2)

        if save:
            svg_file = DATA_DIR + \
                f'svg/{self.parameter_name}-{prefix}-{int(time.time())}.svg'
            print(f'Writing svg to: {svg_file}')
            fig.write_image(svg_file)

            png_file = DATA_DIR + \
                f'png/{self.parameter_name}-{prefix}-{int(time.time())}.png'
            print(f'Writing png to: {png_file}')
            fig.write_image(png_file)
        if show:
            print('Showing plot...')
            fig.show()


def plot_analysis(file_path: str) -> None:
    """
    Helper function to plot an interactive single-run analysis for the
    simulation data contained in file_path.
    Fucntion to be used by the CLI single-run analysis helper.

    Arguments:
        file_path {str} -- Path to the file containing simulation data
    """
    analysis = ModelAnalysis(load_from_path=file_path)
    analysis.plot_model_analysis(show=True, save=False)


def save_analysis(file_path: str) -> None:
    """
    Helper function to save an interactive single-run analysis for the
    simulation data contained in file_path to an html file.
    Fucntion to be used by the CLI single-run analysis helper.

    Arguments:
        file_path {str} -- Path to the file containing simulation data
    """
    analysis = ModelAnalysis(load_from_path=file_path)
    analysis.plot_model_analysis(show=False, save=True)


def run_data_analyser(data_dir: str) -> None:
    """
    Initiates a CLI tool to assist in results filtering, plotting and saving to html

    Arguments:
        data_dir {str} -- Folder in the file-system that simulation data files
        can be found.
    """
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
            processes = min(8, len(plot_list))

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
    # plot_analysis(
    #     '/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data/run-(50ag|1000ep|7po|0.65ss|0.2ir|or=0.15|em(μ=0.35σ=0.15)|me(μ=0σ=0)|ba=-5|ca=[-0.85,-0.35,0.4,0.65]|se=69.dat')
    run_data_analyser(
        '/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data')
