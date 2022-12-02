from datetime import datetime

import numpy as np
import pickle
from sklearn.decomposition import PCA
import networkx.generators.random_graphs as random_graphs
import networkx as nx

from configuration import *
from agent import Agent
import time


class Model:

    def __init__(self,
                 init_connections,
                 orientations_std,
                 emotions_mean,
                 emotions_std,
                 media_conformities_mean,
                 media_conformities_std):
        # Initialise social network
        self.init_connections_prob = init_connections
        social_graph = random_graphs.erdos_renyi_graph(
            AGENTS_COUNT, init_connections, seed=SEED)
        self.adjacency_matrix = nx.to_numpy_array(social_graph)

        # Initialise agents
        # Draw orientation, emotion and media conformity values from
        # normal distribution for all agents, then make sure to clip them to
        # boundaries
        agents_orientations = np.random.normal(
            INIT_ORIENTATIONS_MEAN, orientations_std, AGENTS_COUNT)
        agents_orientations = np.clip(
            agents_orientations, ORIENTATION_MIN, ORIENTATION_MAX)
        # We store the values for emotions and conformities
        self.agents_emotions = np.random.normal(
            emotions_mean, emotions_std, AGENTS_COUNT)
        self.agents_emotions = np.clip(
            self.agents_emotions, EMOTION_MIN, EMOTION_MAX)
        self.media_conformities = np.random.normal(
            media_conformities_mean, media_conformities_std, AGENTS_COUNT)
        self.media_conformities = np.clip(
            self.media_conformities, MEDIA_CONFORMITY_MIN, MEDIA_CONFORMITY_MAX)

        # Populate the agents array. The agents initilisation draws values
        # for their opinion array based on initial orientation and emotion.
        self.agents = [
            Agent(i, agents_orientations[i], self.agents_emotions[i],
                  self.adjacency_matrix[i], self.media_conformities[i])
            for i in range(AGENTS_COUNT)
        ]

        # Initialise dynamic aggregation arrays
        self.refresh_group_opinions()
        self.update_group_similarities()
        self.refresh_media_opinions()

        # Set model data collection dictionary
        self.data = {
            'init_connections': init_connections, 
            'orientations_std': orientations_std, 
            'emotions_mean': emotions_mean, 
            'emotions_std': emotions_std, 
            'media_conformities_mean': media_conformities_mean, 
            'media_conformities_std': media_conformities_std,
            'connections_created': 0,
            'connections_destroyed': 0,
            'snapshots': []
        }
        self.connections_created = 0
        self.connections_destroyed = 0

    def step(self):
        # We first select a random smaple of the agent's pool without replacement
        active_agents = np.random.choice(AGENTS_COUNT, int(AGENTS_COUNT * INTERACTION_RATIO), False)
        #######################################################################
        #   1.  Trust matrices update
        #######################################################################
        tic = time.perf_counter()
        # Based on the media and group opinions, the trust is updated
        for i in active_agents:
            self.agents[i].update_agents_trust(self.group_similarities)
            self.agents[i].update_media_trust(self.media_opinions)

        # And local arrays refreshed
        self.refresh_group_trust()
        self.refresh_media_trust()
        print(f'1  ->  {time.perf_counter() - tic}')
        
        #######################################################################
        #   2.  Opinion interaction
        #######################################################################
        tic = time.perf_counter()
        # We got all we need to calculate opinion influences
        # Get the total group and total media influence based on trust in each element
        group_influence = np.matmul(self.group_trust, self.group_opinions)
        media_influence = np.matmul(self.media_trust, self.media_opinions)

        # Then calculate total external influence, weighted by media conformity
        external_influence = (group_influence.T * (1 - self.media_conformities)).T \
            + (media_influence.T * self.media_conformities).T

        # Get noise from communications innacuracies
        comms_noise = np.random.normal(
            NOISE_MEAN, NOISE_STD, (AGENTS_COUNT, POLICIES_COUNT))
        noise_influence = (comms_noise.T * (1 - self.agents_emotions)).T

        # Print state if required
        if VERBOSITY & V_MODEL:
            print('\n* Group influence:')
            print_array(group_influence)
            print('\n* Media influence:')
            print_array(media_influence)
            print('\n* Media conformities:')
            print_array(self.media_conformities)
            print('\n* Group influence updated:')
            print_array((group_influence.T * (1 - self.media_conformities)).T)
            print('\n* Media influence updated:')
            print_array((media_influence.T * self.media_conformities).T)
            print('\n* External influence:')
            print_array(external_influence)
            print('\n* Noise influence:')
            print_array(noise_influence)

        # Now update each agent's opinion by making it interact with the calculated
        # external influence. The opinions will also suffer some noise that represents
        # communication innacuracies.
        for i in active_agents:
            self.agents[i].interact(external_influence[i])
            self.agents[i].add_noise(noise_influence[i])

        # Refresh local opinion arrays
        self.refresh_group_opinions()
        self.update_group_similarities()
        self.refresh_media_opinions()
        self.update_media_similarities()
        print(f'2  ->  {time.perf_counter() - tic}')

        #######################################################################
        #   3.  Social network update
        #######################################################################
        tic = time.perf_counter()
        # Connections can be created or ended based on the opinion
        # similarity and emotional affectiveness
        for i in active_agents:
            create_connections, end_connections = self.agents[i].update_social_network(
                self.group_similarities, self.group_opinion_strengths)
            # End the connections symmetrically
            for connection in end_connections:
                self.agents[i].adjacency[connection] = 0
                self.agents[connection].adjacency[i] = 0
                self.data['connections_destroyed'] += 1
            # Create new connections symmetrically
            for connection in create_connections:
                self.agents[i].adjacency[connection] = 1
                self.agents[connection].adjacency[i] = 1
                self.data['connections_created'] += 1

        # Update the adjacency matrix
        self.refresh_adjacency_matrix()
        print(f'3  ->  {time.perf_counter() - tic}')

        

    def refresh_group_opinions(self):
        self.group_opinions = np.array(
            [agent.opinions for agent in self.agents]
        )
        self.group_opinion_strengths = np.linalg.norm(self.group_opinions, axis=1) / np.sqrt(POLICIES_COUNT)
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup opinions refreshed:')
            print_array(self.group_opinions)
            print_array(self.group_opinion_strengths)

    def update_group_similarities(self):
        norms = np.linalg.norm(self.group_opinions, axis=1)
        cos = np.clip(
            np.dot(self.group_opinions, self.group_opinions.T) /
            np.outer(norms, norms),
            -1, 1)
        self.group_similarities = 1 - (np.arccos(cos) / np.pi)

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup similarities updated:')
            print_array(self.group_similarities)

    def refresh_group_trust(self):
        self.group_trust = np.array(
            [agent.agents_trust for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup trust refreshed:')
            print_array(self.group_trust)

    def refresh_media_opinions(self):
        self.media_opinions = np.array(
            [
                np.random.normal(
                    MEDIA_OUTLETS_MEANS[i],
                    MEDIA_OUTLETS_STD[i],
                    POLICIES_COUNT
                ) for i in range(MEDIA_OUTLETS_COUNT)
            ]
        )
        self.media_opinions = np.clip(
            self.media_opinions, OPINION_MIN, OPINION_MAX)

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia opinions refreshed:')
            print_array(self.media_opinions)

    def update_media_similarities(self):
        norms_group = np.linalg.norm(self.group_opinions, axis=1)
        norms_media = np.linalg.norm(self.media_opinions, axis=1)
        cos = np.clip(
            np.dot(self.group_opinions, self.media_opinions.T) /
            np.outer(norms_group, norms_media),
            -1, 1)
        self.media_similarities = 1 - (np.arccos(cos) / np.pi)

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\Media similarities updated:')
            print_array(self.media_similarities)

    def refresh_media_trust(self):
        self.media_trust = np.array(
            [agent.media_trust for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia trust refreshed:')
            print_array(self.media_trust)

    def refresh_adjacency_matrix(self):
        self.adjacency_matrix = np.array(
            [agent.adjacency for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tAdjacency matrix refreshed:')
            print_array(self.adjacency_matrix)

    def get_election_poll(self):
        # Initialise a zero array
        poll = np.zeros(CANDIDATES_COUNT)
        # Sum the normalised intentions of all agents
        for agent in self.agents:
            poll += agent.get_vote_intention()
        # Then normalise the sum
        poll = poll / np.sum(poll)
        return poll

    def run(self, n_epochs: int):
        snapshot_epochs = np.linspace(0, n_epochs, N_SNAPSHOTS).astype(int)
        print('\n********************************************************************************\n')
        print(f'Starting model run for {n_epochs} epochs.\n')

        # Start epoch from 0 so we save the initial state before stepping
        for epoch in range(0, n_epochs + 1):
            # Print status on each 10%
            if epoch % (n_epochs / 10) == 0:
                print(f'Run progress:\t{int(epoch/n_epochs * 100)}%')

            # Store snapshots for plotting
            if epoch in snapshot_epochs:
                poll = self.get_election_poll()
                self.data['snapshots'].append({
                    'epoch': epoch,
                    'group_opinions': self.group_opinions.copy(),
                    'media_opinions': self.media_opinions.copy(),
                    'adjacency': self.adjacency_matrix.copy(),
                    'poll': poll
                })

            # Step the model
            self.step()

    
