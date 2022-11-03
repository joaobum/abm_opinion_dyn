import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go


################################################################################
#       DATA ANALYSER CLASS DEFINITION
################################################################################
class Analysis:
    '''
    Datas analyse class, can be loaded from file or from dictionary and creates relevant analysis plots
    '''

    def __init__(self, simulation_info=None, filepath=None):
        '''
        Initialises the data analyser class. Can either be done from file or from simulation dictionary.
        In case both arguments are passed, file path takes priority

        Args:
            snapshots ([type], optional): [description]. Defaults to None.
            filepath ([type], optional): [description]. Defaults to None.
        '''
        if filepath:
            simulation_info = pickle.load(open(filepath, 'rb'))

        self.snapshots = simulation_info['snapshots']
        self.n_agents = simulation_info['n_agents']
        self.n_policies = simulation_info['n_policies']
        self.n_epochs = simulation_info['n_epochs']
        self.emotions_mean = simulation_info['emotions_mean']
        self.emotions_std = simulation_info['emotions_std']
        self.noise_intensity_mean = simulation_info['noise_intensity_mean']
        self.noise_intensity_std = simulation_info['noise_intensity_std']


    def plot_full_analysis(self):
        '''
        Generate an animated plot over epochs with 4 subplots: a 2-PCA analysis, the population orientation as a 
        scatter and histogram, and a timeline of agents orientation majority.
        '''
        snapshots_df = pd.DataFrame(self.snapshots)
        epochs = snapshots_df.epoch.unique()

        pca_df = snapshots_df.explode('pca')
        pca_df[['pca1', 'pca2']] = pd.DataFrame(
            pca_df.pca.tolist(), index=pca_df.index)

        histogram_df = pd.DataFrame(
            columns=['epoch', 'histogram_x', 'histogram_y'])
        for index, row in snapshots_df.iterrows():
            histogram = np.histogram(
                row.orientations, 200, range=(-1, 1), density=True)
            for i in range(len(histogram[0])):
                histogram_df = pd.concat([
                    histogram_df,
                    pd.DataFrame(
                        {'epoch': row.epoch, 'histogram_x': histogram[1][i], 'histogram_y': histogram[0][i]}, index=[0])
                ], ignore_index=True)

        orientations_df = snapshots_df.explode('orientations').reset_index()
        orientations_df['agent_id'] = pd.DataFrame(
            [i % self.n_agents for i in range(len(orientations_df))])

        # We need to build the figure by hand to have subplots sharing an animation
        fig = dict(
            layout=dict(
                width=1200,
                height=900,
                xaxis1={'domain': [0.0, 0.45],
                        'anchor': 'y1', 'title': 'pca1'},
                yaxis1={'domain': [0.58, 0.98],
                        'anchor': 'x1', 'title': 'pca2'},
                xaxis2={'domain': [0.55, 1.0], 'anchor': 'y2',
                        'title': 'orientation interval', 'range': [-1, 1]},
                yaxis2={'domain': [0.58, 0.98], 'anchor': 'x2',
                        'title': 'ratio', 'range': [0, 5]},
                xaxis3={'domain': [0.0, 0.42], 'anchor': 'y3',
                        'title': 'agent id', 'range': [0, self.n_agents]},
                yaxis3={'domain': [0.0, 0.40], 'anchor': 'x3',
                        'title': 'orientation', 'range': [-1, 1]},
                xaxis4={'domain': [0.55, 1.0], 'anchor': 'y4',
                        'title': 'epoch', 'range': [0, self.n_epochs]},
                yaxis4={'domain': [0.0, 0.42], 'anchor': 'x4',
                        'title': '% negative oriented', 'range': [0, 100]},
                title=f'Analysis over {self.n_epochs} epochs: {self.n_agents} agents, {self.n_policies} policies (neg={self.policies_neg}, μ={self.policies_mean}|σ={self.policies_std}), ' +
                f'emotions (μ={self.emotions_mean},σ={self.emotions_std}), noise (μ={self.noise_intensity_mean}, σ={self.noise_intensity_std})',
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
                updatemenus=[{'buttons': [{'args': [[str(epoch) for epoch in epochs], {'frame': {'duration': 500.0, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 500, 'easing': 'linear'}}], 'label': 'Play', 'method': 'animate'}, {'args': [[None], {'frame': {
                    'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 85}, 'showactive': True, 'type': 'buttons', 'x': 0.1, 'y': 0, 'xanchor': 'right', 'yanchor': 'top'}],
                sliders=[{'yanchor': 'top', 'xanchor': 'left', 'currentvalue': {'font': {'size': 16}, 'prefix': 'Epoch: ', 'visible': True, 'xanchor': 'right'}, 'transition': {'duration': 500.0, 'easing': 'linear'}, 'pad': {'b': 10, 't': 50}, 'len': 0.9, 'x': 0.1, 'y': 0,
                          'steps': [{'args': [[str(epoch)], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False}, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': str(epoch), 'method': 'animate'} for epoch in snapshots_df.epoch.unique()]
                          }],
                annotations=[{'font': {'size': 16}, 'showarrow': False, 'text': "Agent's opinions 2-PCA analysis", 'x': 0.225, 'xanchor': 'center', 'xref': 'paper', 'y': 0.98, 'yanchor': 'bottom', 'yref': 'paper'}, {'font': {'size': 16}, 'showarrow': False, 'text': "Agent's orientation histogram", 'x': 0.775, 'xanchor': 'center', 'xref': 'paper', 'y': 0.98, 'yanchor': 'bottom', 'yref': 'paper'}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Orientation per agent', 'x': 0.225, 'xanchor': 'center', 'xref': 'paper', 'y': 0.45, 'yanchor': 'bottom', 'yref': 'paper'}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Ratio of negative orientation agents', 'x': 0.775, 'xanchor': 'center', 'xref': 'paper', 'y': 0.45, 'yanchor': 'bottom', 'yref': 'paper'}],
            ),

            data=[
                {'type': 'scatter', 'name': 'PCA', 'x': pca_df.loc[pca_df['epoch'] == epochs[0]]['pca1'], 'y': pca_df.loc[pca_df['epoch'] == epochs[0]]['pca2'], 'line': {
                    'color': 'indigo'}, 'mode': 'markers', 'showlegend': False, 'xaxis': 'x1', 'yaxis': 'y1'},
                {'type': 'bar', 'name': 'Orientations histogram', 'x': histogram_df.loc[histogram_df['epoch'] == epochs[0]]['histogram_x'],
                    'y': histogram_df.loc[histogram_df['epoch'] == epochs[0]]['histogram_y'], 'showlegend': False, 'xaxis': 'x2', 'yaxis': 'y2'},
                {'type': 'scatter', 'name': 'Orientations', 'x': orientations_df.loc[orientations_df['epoch'] == epochs[0]]['agent_id'], 'y': orientations_df.loc[
                    orientations_df['epoch'] == epochs[0]]['orientations'], 'line': {'color': 'teal'}, 'mode': 'markers', 'showlegend': False, 'xaxis': 'x3', 'yaxis': 'y3'},
                {'type': 'scatter', 'name': 'Negative ratio', 'x': snapshots_df['epoch'], 'y': snapshots_df['neg_ratio'], 'line': {
                    'color': 'black'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
                {'type': 'scatter', 'name': 'Negative ratio', 'x': snapshots_df['epoch'], 'y': [50 for epoch in epochs], 'line': {
                    'color': 'darkgreen', 'dash': 'dot'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
                {'type': 'scatter', 'name': 'Negative ratio', 'x': [0 for i in range(0, 100)], 'y': [y for y in range(
                    0, 100)], 'line': {'color': 'red'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
            ],

            frames=[
                {
                    'name': str(epoch), 'layout': {},
                    'data': [
                        {'type': 'scatter', 'name': 'PCA', 'x': pca_df.loc[pca_df['epoch'] == epoch]['pca1'], 'y': pca_df.loc[pca_df['epoch'] == epoch]['pca2'], 'line': {
                            'color': 'indigo'}, 'mode': 'markers', 'showlegend': False, 'xaxis': 'x1', 'yaxis': 'y1'},
                        {'type': 'bar', 'name': 'Orientations histogram', 'x': histogram_df.loc[histogram_df['epoch'] == epoch]['histogram_x'],
                            'y': histogram_df.loc[histogram_df['epoch'] == epoch]['histogram_y'], 'showlegend': False, 'xaxis': 'x2', 'yaxis': 'y2'},
                        {'type': 'scatter', 'name': 'Orientations', 'x': orientations_df.loc[orientations_df['epoch'] == epoch]['agent_id'], 'y': orientations_df.loc[
                            orientations_df['epoch'] == epoch]['orientations'], 'line': {'color': 'teal'}, 'mode': 'markers', 'showlegend': False, 'xaxis': 'x3', 'yaxis': 'y3'},
                        {'type': 'scatter', 'name': 'Negative ratio', 'x': snapshots_df['epoch'], 'y': snapshots_df['neg_ratio'], 'line': {
                            'color': 'black'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
                        {'type': 'scatter', 'name': 'Negative ratio', 'x': snapshots_df['epoch'], 'y': [50 for epoch in epochs], 'line': {
                            'color': 'darkgreen', 'dash': 'dot'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
                        {'type': 'scatter', 'name': 'Negative ratio', 'x': [epoch for i in range(0, 100)], 'y': [y for y in range(
                            0, 100)], 'line': {'color': 'red'}, 'mode': 'lines', 'showlegend': False, 'xaxis': 'x4', 'yaxis': 'y4'},
                    ],

                }
                for epoch in epochs
            ]
        )
        fig = go.Figure(fig)
        fig.show()
